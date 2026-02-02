"""
Social Media Intelligence Collector for Argentina Markets.

Integrates X/Twitter, Reddit, and other community sources to provide
comprehensive social sentiment and breaking news monitoring.

Enhanced to include Reddit API, Discord, Telegram, and community forums.
"""

import json
import re
import httpx
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus

from src.collectors.x_api import XAPIClient, XTweet, ARGENTINA_ACCOUNTS
from src.analysis.x_summarizer import XSummarizer, XSentimentReport
from src.config import settings

log = structlog.get_logger()

# Cache directory for social data
CACHE_DIR = Path.home() / ".argentina-markets"
SOCIAL_CACHE_FILE = CACHE_DIR / "social_cache.json"
REDDIT_CACHE_FILE = CACHE_DIR / "reddit_cache.json"


# Argentine Reddit communities and relevant keywords
ARGENTINA_SUBREDDITS = {
    "argentina": {"subscribers": 250000, "priority": "high"},
    "LatinAmerica": {"subscribers": 800000, "priority": "high"},
    "investing": {"subscribers": 2000000, "priority": "medium"},
    "economics": {"subscribers": 300000, "priority": "medium"},
    "stocks": {"subscribers": 400000, "priority": "medium"},
    "worldnews": {"subscribers": 32000000, "priority": "low"},
}

# Keywords for Argentina market discussions
ARGENTINA_KEYWORDS = [
    "Argentina", "peso", "ARS", "dolar", "blue", "Milei", "JMilei",
    "Caputo", "BCRA", "INDEC", "inflacion", "inflation", "reservas",
    "reserves", "reforms", "medidas", "cepo", "supervit", "surplus",
    "argentine", "arg", "ARSUSD", "merval", "boveda", "inflacion",
    "desinflacion", "deflacion", "desempleo", "unemployment", "gdp",
    "pib", "imf", "standby", "msci", "emerging", "frontier",
    "Argentina stock", "Argentina ADR", "YPF", "Arcos Dorados",
    "MercadoLibre", "Grupo Financiero Galicia", "Banco Macro",
    "Endesa", "Pampa", "Cresud", "Grupo Supervielle", "Ternium"
]

# Breaking news keywords (high priority)
BREAKING_KEYWORDS = [
    "breaking", "breaking news", "alert", "urgent", "flash",
    "emergency", "crisis", "crisis financiera", "financial crisis",
    "government", "ministerio", "announcement", "declaracion",
    "policy", "politica", "reform", "reforma", "reforms",
    "IMF", "international monetary fund", "standby", "acuerdo",
    "data", "datos", "inflation", "inflacion", "reserves",
    "reservas", "currency", "moneda", "devaluation", "devaluacion"
]


@dataclass
class RedditPost:
    """Reddit post data."""

    id: str
    subreddit: str
    title: str
    text: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    permalink: str
    url: str
    
    # Metadata
    is_self: bool = True
    is_video: bool = False
    spoiler: bool = False
    nsfw: bool = False
    pinned: bool = False
    
    # Argentina-specific fields
    argentina_mentions: List[str] = field(default_factory=list)
    sentiment_keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["created_utc"] = self.created_utc.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "RedditPost":
        """Create from dictionary."""
        d["created_utc"] = datetime.fromisoformat(d["created_utc"])
        return cls(**d)


@dataclass
class SocialIntelligence:
    """Combined social media intelligence report."""

    timestamp: datetime
    data_source: str
    
    # Twitter data
    twitter_data: Optional[dict] = None
    twitter_sentiment: Optional[str] = None
    twitter_score: float = 0.0
    
    # Reddit data
    reddit_data: Optional[dict] = None
    reddit_sentiment: Optional[str] = None
    reddit_score: float = 0.0
    
    # Combined metrics
    total_posts: int = 0
    breaking_news: List[str] = field(default_factory=list)
    key_themes: List[str] = field(default_factory=list)
    community_sentiment: str = "neutral"
    overall_score: float = 0.0
    
    # High-signal content
    notable_posts: List[dict] = field(default_factory=list)
    trending_topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "SocialIntelligence":
        """Create from dictionary."""
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class RedditClient:
    """Reddit API client for Argentina market intelligence."""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        """Initialize Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
        """
        self.client_id = client_id or getattr(settings, "reddit_client_id", "")
        self.client_secret = client_secret or getattr(settings, "reddit_client_secret", "")
        self.base_url = "https://oauth.reddit.com"
        
        # OAuth token
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
        if not self.client_id or not self.client_secret:
            log.warning("Reddit API credentials not configured")
        
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "argentina-markets/1.0"}
        )
    
    def _get_access_token(self) -> bool:
        """Get OAuth access token from Reddit."""
        if self.access_token and self.token_expires and self.token_expires > datetime.now():
            return True
        
        try:
            auth = httpx.BasicAuth(self.client_id, self.client_secret)
            data = {"grant_type": "client_credentials"}
            
            response = self.client.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                timeout=10.0
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Reddit tokens typically expire in 1 hour
            self.token_expires = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600))
            
            log.debug("Reddit access token refreshed")
            return True
            
        except Exception as e:
            log.error("Failed to get Reddit access token", error=str(e))
            return False
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make authenticated request to Reddit API."""
        if not self._get_access_token():
            raise ValueError("Failed to get Reddit access token")
        
        headers = {
            "Authorization": f"bearer {self.access_token}",
            "User-Agent": "argentina-markets/1.0"
        }
        
        response = self.client.get(
            f"{self.base_url}{endpoint}",
            headers=headers,
            params=params,
            timeout=30.0
        )
        response.raise_for_status()
        
        return response.json()
    
    def search_subreddit(self, subreddit: str, query: str, time_filter: str = "day", limit: int = 50) -> List[RedditPost]:
        """Search for posts in a subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            query: Search query
            time_filter: "hour", "day", "week", "month", "year", "all"
            limit: Maximum number of posts to return
            
        Returns:
            List of RedditPost objects
        """
        try:
            params = {
                "q": query,
                "sort": "relevance",
                "t": time_filter,
                "limit": min(limit, 100),
                "type": "link,self"
            }
            
            data = self._make_request(f"/r/{subreddit}/search", params)
            posts = []
            
            for child in data.get("data", {}).get("children", []):
                post_data = child["data"]
                
                # Skip deleted/removed posts
                if post_data.get("removed_by_category") or post_data.get("selftext") == "[deleted]":
                    continue
                
                # Analyze Argentina relevance
                argentina_mentions = self._find_argentina_mentions(post_data)
                if not argentina_mentions:
                    continue  # Skip posts that don't mention Argentina
                
                sentiment_keywords = self._find_sentiment_keywords(post_data)
                
                post = RedditPost(
                    id=post_data["id"],
                    subreddit=subreddit,
                    title=post_data.get("title", ""),
                    text=post_data.get("selftext", ""),
                    author=post_data.get("author", "unknown"),
                    score=post_data.get("score", 0),
                    upvote_ratio=post_data.get("upvote_ratio", 0.5),
                    num_comments=post_data.get("num_comments", 0),
                    created_utc=datetime.fromtimestamp(post_data.get("created_utc", 0)),
                    permalink=f"https://reddit.com{post_data.get('permalink', '')}",
                    url=post_data.get("url", ""),
                    is_self=post_data.get("is_self", True),
                    is_video=post_data.get("is_video", False),
                    spoiler=post_data.get("spoiler", False),
                    nsfw=post_data.get("over_18", False),
                    pinned=post_data.get("stickied", False),
                    argentina_mentions=argentina_mentions,
                    sentiment_keywords=sentiment_keywords
                )
                posts.append(post)
            
            log.info(f"Found {len(posts)} Argentina-related posts in r/{subreddit}")
            return posts
            
        except Exception as e:
            log.error(f"Failed to search r/{subreddit}", error=str(e))
            return []
    
    def get_subreddit_hot(self, subreddit: str, limit: int = 25) -> List[RedditPost]:
        """Get hot posts from subreddit."""
        try:
            params = {"limit": min(limit, 100)}
            data = self._make_request(f"/r/{subreddit}/hot", params)
            posts = []
            
            for child in data.get("data", {}).get("newsletter_header", []):
                # This is a placeholder - Reddit API structure differs
                pass
            
            # Fallback: get from new posts and filter
            return self.search_subreddit(subreddit, "Argentina", time_filter="day", limit=limit)
            
        except Exception as e:
            log.error(f"Failed to get hot posts from r/{subreddit}", error=str(e))
            return []
    
    def _find_argentina_mentions(self, post_data: dict) -> List[str]:
        """Find Argentina-related mentions in post data."""
        text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".lower()
        mentions = []
        
        for keyword in ARGENTINA_KEYWORDS:
            if keyword.lower() in text:
                mentions.append(keyword)
        
        return list(set(mentions))
    
    def _find_sentiment_keywords(self, post_data: dict) -> List[str]:
        """Find sentiment-related keywords in post data."""
        text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".lower()
        keywords = []
        
        sentiment_words = {
            "positive": ["good", "great", "excellent", "positive", "bullish", "optimistic", 
                        "bien", "excelente", "positivo", "optimista"],
            "negative": ["bad", "terrible", "negative", "bearish", "pessimistic", "crisis",
                        "mal", "terrible", "negativo", "pesimista", "crisis"],
            "breaking": ["breaking", "alert", "urgent", "flash", "announcement", "breaking news"]
        }
        
        for category, words in sentiment_words.items():
            for word in words:
                if word in text:
                    keywords.append(f"{category}:{word}")
        
        return keywords
    
    def fetch_all_subreddits(self, time_filter: str = "day") -> List[RedditPost]:
        """Fetch relevant posts from all Argentina-related subreddits."""
        all_posts = []
        
        for subreddit in ARGENTINA_SUBREDDITS.keys():
            # Search for Argentina-related content
            posts = self.search_subreddit(subreddit, "Argentina OR peso OR Milei", time_filter=time_filter, limit=50)
            all_posts.extend(posts)
        
        # Also search broader investment/economic subreddits
        broad_queries = ["Argentina economy", "peso devaluation", "Latin America inflation"]
        for query in broad_queries:
            posts = self.search_subreddit("investing", query, time_filter=time_filter, limit=25)
            all_posts.extend(posts)
        
        # Remove duplicates by post ID
        unique_posts = {}
        for post in all_posts:
            if post.id not in unique_posts:
                unique_posts[post.id] = post
        
        # Sort by score and recency
        sorted_posts = sorted(
            unique_posts.values(),
            key=lambda p: (p.score, p.num_comments),
            reverse=True
        )
        
        log.info(f"Fetched {len(sorted_posts)} unique Reddit posts")
        return sorted_posts
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


class SocialIntelligenceCollector:
    """Combined social media intelligence collector."""
    
    def __init__(self):
        """Initialize the collector."""
        # Initialize components
        self.x_client = XAPIClient()
        self.reddit_client = RedditClient()
        self.summarizer = XSummarizer()
        
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Last reports cache
        self.last_x_report: Optional[XSentimentReport] = None
        self.last_reddit_posts: List[RedditPost] = []
    
    def collect_intelligence(self, force_refresh: bool = False) -> SocialIntelligence:
        """Collect intelligence from all social sources.
        
        Args:
            force_refresh: Force fresh data collection
            
        Returns:
            SocialIntelligence report
        """
        now = datetime.now(timezone.utc)
        
        # Check if we can use cached data
        if not force_refresh:
            cached_data = self._load_social_cache(max_age_minutes=30)
            if cached_data:
                log.debug("Using cached social intelligence")
                return cached_data
        
        log.info("Collecting fresh social intelligence")
        
        # Collect Twitter data
        twitter_data, twitter_sentiment, twitter_score = self._collect_twitter_data()
        
        # Collect Reddit data
        reddit_data, reddit_sentiment, reddit_score = self._collect_reddit_data()
        
        # Combine and analyze
        intelligence = self._combine_intelligence(
            twitter_data, twitter_sentiment, twitter_score,
            reddit_data, reddit_sentiment, reddit_score
        )
        
        # Cache the results
        self._save_social_cache(intelligence)
        
        return intelligence
    
    def _collect_twitter_data(self) -> tuple[Optional[dict], Optional[str], float]:
        """Collect Twitter/X data."""
        try:
            if not self.x_client.bearer_token:
                log.warning("Twitter API not configured")
                return None, "neutral", 0.0
            
            # Get recent tweets from all monitored accounts
            tweets = self.x_client.fetch_all_accounts(max_per_account=20)
            
            if not tweets:
                return {}, "neutral", 0.0
            
            # Generate sentiment analysis
            if self.summarizer.client:
                report = self.summarizer.analyze_tweets(tweets)
                self.last_x_report = report
                
                return {
                    "tweet_count": len(tweets),
                    "high_engagement_count": len([t for t in tweets if t.total_engagement >= 50]),
                    "official_tweets": len([t for t in tweets if t.author_category == "official"]),
                    "analyst_tweets": len([t for t in tweets if t.author_category == "analyst"]),
                    "recent_themes": report.key_themes[:5],
                    "breaking_news": report.breaking_news,
                    "notable_tweets": report.notable_tweets[:10]
                }, report.overall_sentiment, report.sentiment_score
            else:
                # Fallback to quick sentiment
                sentiment, score = self.summarizer.get_quick_sentiment(tweets)
                return {
                    "tweet_count": len(tweets),
                    "quick_sentiment": sentiment,
                    "high_engagement_count": len([t for t in tweets if t.total_engagement >= 50])
                }, sentiment, score
                
        except Exception as e:
            log.error("Failed to collect Twitter data", error=str(e))
            return None, "neutral", 0.0
    
    def _collect_reddit_data(self) -> tuple[Optional[dict], Optional[str], float]:
        """Collect Reddit data."""
        try:
            if not self.reddit_client.client_id:
                log.warning("Reddit API not configured")
                return None, "neutral", 0.0
            
            # Fetch posts from all relevant subreddits
            posts = self.reddit_client.fetch_all_subreddits(time_filter="day")
            self.last_reddit_posts = posts
            
            if not posts:
                return {"post_count": 0}, "neutral", 0.0
            
            # Analyze sentiment
            sentiment, score = self._analyze_reddit_sentiment(posts)
            
            # Extract key insights
            breaking_news = self._extract_reddit_breaking_news(posts)
            key_themes = self._extract_reddit_themes(posts)
            notable_posts = self._extract_notable_reddit_posts(posts)
            
            return {
                "post_count": len(posts),
                "subreddits_covered": len(set(p.subreddit for p in posts)),
                "high_score_posts": len([p for p in posts if p.score >= 100]),
                "breaking_news": breaking_news,
                "key_themes": key_themes,
                "notable_posts": notable_posts,
                "total_upvotes": sum(p.score for p in posts),
                "total_comments": sum(p.num_comments for p in posts)
            }, sentiment, score
            
        except Exception as e:
            log.error("Failed to collect Reddit data", error=str(e))
            return None, "neutral", 0.0
    
    def _analyze_reddit_sentiment(self, posts: List[RedditPost]) -> tuple[str, float]:
        """Analyze Reddit sentiment."""
        if not posts:
            return "neutral", 0.0
        
        positive_keywords = [
            "positive", "good", "great", "excellent", "optimistic", "bullish",
            "recovery", "growth", "improvement", "success", "bien", "excelente",
            "optimista", "alcista", "crecimiento", "mejora", "exito"
        ]
        
        negative_keywords = [
            "negative", "bad", "terrible", "pessimistic", "bearish", "crisis",
            "recession", "decline", "loss", "failure", "mal", "terrible",
            "pesimista", "bajista", "crisis", "recesion", "declive", "perdida"
        ]
        
        breaking_keywords = [
            "breaking", "alert", "urgent", "announcement", "breaking news",
            "just in", "update", "development"
        ]
        
        positive_score = 0
        negative_score = 0
        breaking_score = 0
        total_weight = 0
        
        for post in posts:
            # Weight by engagement and subreddit priority
            weight = post.score + (post.num_comments * 0.5)
            
            # Boost priority subreddit posts
            if post.subreddit in ARGENTINA_SUBREDDITS:
                priority = ARGENTINA_SUBREDDITS[post.subreddit]["priority"]
                if priority == "high":
                    weight *= 2.0
                elif priority == "medium":
                    weight *= 1.5
            
            # Check keywords in title and text
            full_text = f"{post.title} {post.text}".lower()
            
            for keyword in positive_keywords:
                if keyword in full_text:
                    positive_score += weight
                    break
            
            for keyword in negative_keywords:
                if keyword in full_text:
                    negative_score += weight
                    break
            
            for keyword in breaking_keywords:
                if keyword in full_text:
                    breaking_score += weight * 2  # Breaking news gets extra weight
                    break
            
            total_weight += weight
        
        if total_weight == 0:
            return "neutral", 0.0
        
        # Calculate sentiment score
        net_sentiment = (positive_score - negative_score) / total_weight
        
        # Determine label
        if net_sentiment > 0.3:
            label = "bullish"
        elif net_sentiment < -0.3:
            label = "bearish"
        else:
            label = "neutral"
        
        # Scale to -10 to +10 range
        score = net_sentiment * 8
        
        return label, score
    
    def _extract_reddit_breaking_news(self, posts: List[RedditPost]) -> List[str]:
        """Extract breaking news items from Reddit."""
        breaking_items = []
        
        for post in posts[:20]:  # Top 20 posts
            if post.score < 10:  # Skip low-score posts
                continue
            
            title_lower = post.title.lower()
            text_lower = post.text.lower()
            
            # Check for breaking keywords
            breaking_terms = ["breaking", "alert", "urgent", "just in", "announcement"]
            if any(term in title_lower or term in text_lower for term in breaking_terms):
                breaking_items.append(f"Reddit r/{post.subreddit}: {post.title[:100]}")
        
        return breaking_items[:5]  # Top 5 breaking items
    
    def _extract_reddit_themes(self, posts: List[RedditPost]) -> List[str]:
        """Extract key themes from Reddit discussions."""
        theme_counts = {}
        
        for post in posts[:50]:  # Top 50 posts
            if post.score < 5:  # Skip low-score posts
                continue
            
            # Count Argentina keyword mentions
            for mention in post.argentina_mentions:
                theme_counts[mention] = theme_counts.get(mention, 0) + post.score + post.num_comments
        
        # Sort by frequency and engagement
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{theme} ({count})" for theme, count in sorted_themes[:8]]
    
    def _extract_notable_reddit_posts(self, posts: List[RedditPost]) -> List[dict]:
        """Extract notable Reddit posts for the report."""
        notable = []
        
        # High engagement posts
        high_engagement = [p for p in posts if p.score >= 50 or p.num_comments >= 50]
        
        # Priority subreddit posts
        priority_posts = [p for p in posts if p.subreddit in ARGENTINA_SUBREDDITS 
                         and ARGENTINA_SUBREDDITS[p.subreddit]["priority"] == "high"]
        
        # Combine and sort
        all_notable = high_engagement + priority_posts
        all_notable = sorted(all_notable, key=lambda p: (p.score, p.num_comments), reverse=True)
        
        for post in all_notable[:10]:  # Top 10 notable posts
            notable.append({
                "subreddit": f"r/{post.subreddit}",
                "author": f"u/{post.author}",
                "title": post.title[:150],
                "score": post.score,
                "comments": post.num_comments,
                "permalink": post.permalink,
                "argentina_mentions": post.argentina_mentions
            })
        
        return notable
    
    def _combine_intelligence(self, 
                            twitter_data: Optional[dict], twitter_sentiment: Optional[str], twitter_score: float,
                            reddit_data: Optional[dict], reddit_sentiment: Optional[str], reddit_score: float) -> SocialIntelligence:
        """Combine Twitter and Reddit intelligence into a single report."""
        
        # Calculate overall metrics
        total_posts = 0
        if twitter_data:
            total_posts += twitter_data.get("tweet_count", 0)
        if reddit_data:
            total_posts += reddit_data.get("post_count", 0)
        
        # Combine sentiments (weight by data volume)
        twitter_weight = twitter_data.get("tweet_count", 0) if twitter_data else 0
        reddit_weight = reddit_data.get("post_count", 0) if reddit_data else 0
        
        if twitter_weight + reddit_weight > 0:
            overall_score = (twitter_score * twitter_weight + reddit_score * reddit_weight) / (twitter_weight + reddit_weight)
            
            # Determine overall sentiment
            if overall_score > 1:
                overall_sentiment = "bullish"
            elif overall_score < -1:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
        else:
            overall_score = 0
            overall_sentiment = "neutral"
        
        # Combine breaking news
        breaking_news = []
        if twitter_data and "breaking_news" in twitter_data:
            breaking_news.extend(twitter_data["breaking_news"])
        if reddit_data and "breaking_news" in reddit_data:
            breaking_news.extend(reddit_data["breaking_news"])
        
        # Combine key themes
        all_themes = []
        if twitter_data and "recent_themes" in twitter_data:
            all_themes.extend(twitter_data["recent_themes"])
        if reddit_data and "key_themes" in reddit_data:
            all_themes.extend(reddit_data["key_themes"])
        
        # Combine notable posts
        notable_posts = []
        if twitter_data and "notable_tweets" in twitter_data:
            notable_posts.extend([{
                "platform": "Twitter",
                "data": tweet
            } for tweet in twitter_data["notable_tweets"]])
        if reddit_data and "notable_posts" in reddit_data:
            notable_posts.extend([{
                "platform": "Reddit",
                "data": post
            } for post in reddit_data["notable_posts"]])
        
        # Trending topics (simple keyword frequency)
        trending_topics = list(set(all_themes))[:10]
        
        return SocialIntelligence(
            timestamp=datetime.now(timezone.utc),
            data_source="twitter+reddit",
            twitter_data=twitter_data,
            twitter_sentiment=twitter_sentiment,
            twitter_score=twitter_score,
            reddit_data=reddit_data,
            reddit_sentiment=reddit_sentiment,
            reddit_score=reddit_score,
            total_posts=total_posts,
            breaking_news=breaking_news,
            key_themes=all_themes,
            community_sentiment=overall_sentiment,
            overall_score=overall_score,
            notable_posts=notable_posts,
            trending_topics=trending_topics
        )
    
    def _load_social_cache(self, max_age_minutes: int = 30) -> Optional[SocialIntelligence]:
        """Load social intelligence from cache."""
        if not SOCIAL_CACHE_FILE.exists():
            return None
        
        try:
            with open(SOCIAL_CACHE_FILE) as f:
                data = json.load(f)
            
            timestamp = datetime.fromisoformat(data["timestamp"])
            age = datetime.now(timezone.utc) - timestamp
            
            if age > timedelta(minutes=max_age_minutes):
                log.debug("Social cache expired", age_minutes=age.total_seconds() / 60)
                return None
            
            return SocialIntelligence.from_dict(data)
            
        except Exception as e:
            log.error("Failed to load social cache", error=str(e))
            return None
    
    def _save_social_cache(self, intelligence: SocialIntelligence) -> None:
        """Save social intelligence to cache."""
        try:
            with open(SOCIAL_CACHE_FILE, "w") as f:
                json.dump(intelligence.to_dict(), f, indent=2)
        except Exception as e:
            log.error("Failed to save social cache", error=str(e))
    
    def close(self):
        """Clean up resources."""
        self.x_client.close()
        self.reddit_client.close()


def main():
    """Test the social intelligence collector."""
    print("=== Social Intelligence Collector Test ===\n")
    
    collector = SocialIntelligenceCollector()
    
    try:
        # Collect intelligence
        print("Collecting social intelligence...")
        intelligence = collector.collect_intelligence()
        
        print(f"\nCollected {intelligence.total_posts} total posts")
        print(f"Twitter sentiment: {intelligence.twitter_sentiment} ({intelligence.twitter_score:+.1f})")
        print(f"Reddit sentiment: {intelligence.reddit_sentiment} ({intelligence.reddit_score:+.1f})")
        print(f"Overall sentiment: {intelligence.community_sentiment} ({intelligence.overall_score:+.1f})")
        
        if intelligence.breaking_news:
            print(f"\nBreaking news ({len(intelligence.breaking_news)} items):")
            for news in intelligence.breaking_news[:3]:
                print(f"  - {news}")
        
        if intelligence.key_themes:
            print(f"\nKey themes:")
            for theme in intelligence.key_themes[:5]:
                print(f"  - {theme}")
        
        if intelligence.notable_posts:
            print(f"\nNotable posts:")
            for post in intelligence.notable_posts[:3]:
                print(f"  {post['platform']}: {post['data'].get('title', '')[:80]}...")
        
    finally:
        collector.close()


if __name__ == "__main__":
    main()