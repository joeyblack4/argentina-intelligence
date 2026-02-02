"""
X (Twitter) API v2 client for Argentine financial intelligence.

Official X API with pay-per-use pricing - replaces fragile Nitter RSS.
"""

import json
import httpx
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Simple translation function for now
def translate_text(text):
    """Simple translation placeholder - in full version would use translation service."""
    if not text:
        return ""
    return text  # For now, return original text
try:
    from .config import settings
except ImportError:
    # Fallback if config is not properly set up
    class Settings:
        x_bearer_token = None
        anthropic_api_key = None
        reddit_client_id = None
        reddit_client_secret = None
    
    settings = Settings()

log = structlog.get_logger()

# Cache directory for tweet storage
CACHE_DIR = Path.home() / ".argentina-markets"
CACHE_FILE = CACHE_DIR / "x_cache.json"


# Key Argentine financial X accounts
ARGENTINA_ACCOUNTS = {
    # Analysts/Economists - high signal
    "fernaborzel": {"name": "Ferna Borzel", "category": "analyst", "weight": 1.5},
    "infabordelois": {"name": "Info Bordelois", "category": "analyst", "weight": 1.5},
    "econlobo": {"name": "Econ Lobo", "category": "analyst", "weight": 1.3},
    "lucaborzellifin": {"name": "Luca Borzelli", "category": "analyst", "weight": 1.2},

    # Government Officials - critical for policy signals
    "MinEconomia_Ar": {"name": "Ministerio de Economia", "category": "official", "weight": 2.0},
    "BancoCentral_AR": {"name": "BCRA", "category": "official", "weight": 2.0},
    "JMilei": {"name": "Javier Milei", "category": "official", "weight": 1.8},
    "LuisCaputoAR": {"name": "Luis Caputo", "category": "official", "weight": 1.8},
    "INDECArgentina": {"name": "INDEC", "category": "official", "weight": 1.5},

    # Financial Media
    "BloombergLinea": {"name": "Bloomberg Linea", "category": "media", "weight": 1.3},
    "Reuters": {"name": "Reuters", "category": "media", "weight": 1.2},
    "Abordelois": {"name": "A Bordelois", "category": "media", "weight": 1.0},

    # Market Participants
    "argabordelois": {"name": "Arg Bordelois", "category": "market", "weight": 1.0},
    "GaliciaFinanzas": {"name": "Galicia Finanzas", "category": "market", "weight": 1.2},
}


@dataclass
class XTweet:
    """Tweet from X API with full metadata."""

    id: str
    author_username: str
    author_name: str
    author_followers: int
    text: str
    text_en: str  # Translated to English
    created_at: datetime

    # Engagement metrics
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    quotes: int = 0

    # Context
    is_reply: bool = False
    reply_to_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Media
    has_media: bool = False
    media_urls: list[str] = field(default_factory=list)

    # Metadata
    author_category: str = "unknown"
    author_weight: float = 1.0

    @property
    def total_engagement(self) -> int:
        """Total engagement score."""
        return self.likes + self.retweets * 2 + self.replies + self.quotes * 2

    @property
    def weighted_engagement(self) -> float:
        """Engagement weighted by author importance."""
        return self.total_engagement * self.author_weight

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "XTweet":
        """Create from dictionary."""
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


@dataclass
class XConversation:
    """Full conversation thread."""

    root_tweet: XTweet
    replies: list[XTweet] = field(default_factory=list)

    @property
    def total_engagement(self) -> int:
        """Total engagement across thread."""
        total = self.root_tweet.total_engagement
        for reply in self.replies:
            total += reply.total_engagement
        return total

    @property
    def reply_count(self) -> int:
        """Number of replies in thread."""
        return len(self.replies)


class XAPIClient:
    """Official X API v2 client for Argentina markets."""

    def __init__(self, bearer_token: str = None):
        """Initialize the X API client.

        Args:
            bearer_token: X API bearer token. Uses settings if not provided.
        """
        self.bearer_token = bearer_token or getattr(settings, "x_bearer_token", "")
        self.base_url = "https://api.x.com/2"

        if not self.bearer_token:
            log.warning("X API bearer token not configured")

        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "argentina-markets/1.0",
            },
        )

        # User ID cache (username -> user_id)
        self._user_cache: dict[str, str] = {}

        # API call counter for cost tracking
        self.api_calls = 0

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make authenticated request to X API.

        Args:
            endpoint: API endpoint (e.g., "/users/by/username/JMilei")
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        if not self.bearer_token:
            raise ValueError("X API bearer token not configured")

        url = f"{self.base_url}{endpoint}"
        self.api_calls += 1

        log.debug("X API request", endpoint=endpoint, params=params, call_num=self.api_calls)

        response = self.client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username.

        Args:
            username: X username (without @)

        Returns:
            User ID string or None if not found
        """
        if username in self._user_cache:
            return self._user_cache[username]

        try:
            data = self._make_request(f"/users/by/username/{username}")
            user_id = data.get("data", {}).get("id")
            if user_id:
                self._user_cache[username] = user_id
            return user_id
        except Exception as e:
            log.error("Failed to get user ID", username=username, error=str(e))
            return None

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 20,
        since_id: str = None,
    ) -> list[XTweet]:
        """Fetch recent tweets from a user.

        Args:
            username: X username (without @)
            max_results: Maximum tweets to return (5-100)
            since_id: Only return tweets newer than this ID

        Returns:
            List of XTweet objects
        """
        user_id = self.get_user_id(username)
        if not user_id:
            return []

        # Get account metadata
        account_info = ARGENTINA_ACCOUNTS.get(username, {})
        author_category = account_info.get("category", "unknown")
        author_weight = account_info.get("weight", 1.0)
        author_name = account_info.get("name", username)

        # Basic tier compatible params (no expansions)
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,conversation_id,in_reply_to_user_id",
        }

        if since_id:
            params["since_id"] = since_id

        try:
            data = self._make_request(f"/users/{user_id}/tweets", params)
        except Exception as e:
            log.error("Failed to fetch tweets", username=username, error=str(e))
            return []

        tweets = []
        raw_tweets = data.get("data", [])

        for raw in raw_tweets:
            try:
                # Get engagement metrics
                metrics = raw.get("public_metrics", {})

                # Parse timestamp
                created_at = datetime.fromisoformat(raw["created_at"].replace("Z", "+00:00"))

                # Translate text
                text = raw.get("text", "")
                try:
                    text_en = translate_text(text) if text else ""
                except Exception:
                    text_en = text

                tweet = XTweet(
                    id=raw["id"],
                    author_username=username,
                    author_name=author_name,
                    author_followers=0,  # Not available on Basic tier
                    text=text,
                    text_en=text_en,
                    created_at=created_at,
                    likes=metrics.get("like_count", 0),
                    retweets=metrics.get("retweet_count", 0),
                    replies=metrics.get("reply_count", 0),
                    quotes=metrics.get("quote_count", 0),
                    is_reply=raw.get("in_reply_to_user_id") is not None,
                    reply_to_id=raw.get("in_reply_to_user_id"),
                    conversation_id=raw.get("conversation_id"),
                    has_media=False,  # Not available on Basic tier
                    media_urls=[],
                    author_category=author_category,
                    author_weight=author_weight,
                )
                tweets.append(tweet)

            except Exception as e:
                log.error("Failed to parse tweet", tweet_id=raw.get("id"), error=str(e))
                continue

        log.info("Fetched user tweets", username=username, count=len(tweets))
        return tweets

    def get_conversation(self, tweet_id: str) -> Optional[XConversation]:
        """Fetch full conversation thread including replies.

        Args:
            tweet_id: ID of the root tweet

        Returns:
            XConversation with root tweet and replies, or None on error
        """
        params = {
            "query": f"conversation_id:{tweet_id}",
            "max_results": 100,
            "tweet.fields": "created_at,public_metrics,conversation_id,in_reply_to_user_id,author_id",
            "expansions": "author_id",
            "user.fields": "name,username,public_metrics",
        }

        try:
            data = self._make_request("/tweets/search/recent", params)
        except Exception as e:
            log.error("Failed to fetch conversation", tweet_id=tweet_id, error=str(e))
            return None

        raw_tweets = data.get("data", [])
        if not raw_tweets:
            return None

        includes = data.get("includes", {})
        users_by_id = {u["id"]: u for u in includes.get("users", [])}

        # Parse tweets
        tweets = []
        root_tweet = None

        for raw in raw_tweets:
            author_id = raw.get("author_id", "")
            author_data = users_by_id.get(author_id, {})
            username = author_data.get("username", "unknown")

            account_info = ARGENTINA_ACCOUNTS.get(username, {})

            # Translate
            text = raw.get("text", "")
            try:
                text_en = translate_text(text) if text else ""
            except Exception:
                text_en = text

            metrics = raw.get("public_metrics", {})

            tweet = XTweet(
                id=raw["id"],
                author_username=username,
                author_name=author_data.get("name", username),
                author_followers=author_data.get("public_metrics", {}).get("followers_count", 0),
                text=text,
                text_en=text_en,
                created_at=datetime.fromisoformat(raw["created_at"].replace("Z", "+00:00")),
                likes=metrics.get("like_count", 0),
                retweets=metrics.get("retweet_count", 0),
                replies=metrics.get("reply_count", 0),
                quotes=metrics.get("quote_count", 0),
                is_reply=raw.get("in_reply_to_user_id") is not None,
                conversation_id=raw.get("conversation_id"),
                author_category=account_info.get("category", "unknown"),
                author_weight=account_info.get("weight", 1.0),
            )

            if raw["id"] == tweet_id:
                root_tweet = tweet
            else:
                tweets.append(tweet)

        if not root_tweet and tweets:
            # Root tweet might be older than search window, use first tweet
            root_tweet = tweets.pop(0)

        if root_tweet:
            return XConversation(root_tweet=root_tweet, replies=tweets)

        return None

    def search_tweets(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[XTweet]:
        """Search for relevant tweets.

        Args:
            query: Search query (e.g., "Argentina economia OR peso OR Milei")
            max_results: Maximum tweets to return (10-100)

        Returns:
            List of XTweet objects
        """
        params = {
            "query": f"{query} -is:retweet lang:es OR lang:en",
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,conversation_id,in_reply_to_user_id,author_id",
            "expansions": "author_id",
            "user.fields": "name,username,public_metrics",
        }

        try:
            data = self._make_request("/tweets/search/recent", params)
        except Exception as e:
            log.error("Failed to search tweets", query=query, error=str(e))
            return []

        raw_tweets = data.get("data", [])
        includes = data.get("includes", {})
        users_by_id = {u["id"]: u for u in includes.get("users", [])}

        tweets = []
        for raw in raw_tweets:
            author_id = raw.get("author_id", "")
            author_data = users_by_id.get(author_id, {})
            username = author_data.get("username", "unknown")

            account_info = ARGENTINA_ACCOUNTS.get(username, {})

            text = raw.get("text", "")
            try:
                text_en = translate_text(text) if text else ""
            except Exception:
                text_en = text

            metrics = raw.get("public_metrics", {})

            tweet = XTweet(
                id=raw["id"],
                author_username=username,
                author_name=author_data.get("name", username),
                author_followers=author_data.get("public_metrics", {}).get("followers_count", 0),
                text=text,
                text_en=text_en,
                created_at=datetime.fromisoformat(raw["created_at"].replace("Z", "+00:00")),
                likes=metrics.get("like_count", 0),
                retweets=metrics.get("retweet_count", 0),
                replies=metrics.get("reply_count", 0),
                quotes=metrics.get("quote_count", 0),
                is_reply=raw.get("in_reply_to_user_id") is not None,
                conversation_id=raw.get("conversation_id"),
                author_category=account_info.get("category", "unknown"),
                author_weight=account_info.get("weight", 1.0),
            )
            tweets.append(tweet)

        log.info("Search completed", query=query[:50], count=len(tweets))
        return tweets

    def fetch_all_accounts(self, max_per_account: int = 10) -> list[XTweet]:
        """Fetch recent tweets from all monitored accounts.

        Args:
            max_per_account: Max tweets per account

        Returns:
            All tweets sorted by weighted engagement (highest first)
        """
        all_tweets = []

        for username in ARGENTINA_ACCOUNTS.keys():
            tweets = self.get_user_tweets(username, max_results=max_per_account)
            all_tweets.extend(tweets)

        # Sort by weighted engagement
        all_tweets.sort(key=lambda t: t.weighted_engagement, reverse=True)

        log.info(
            "Fetched all accounts",
            accounts=len(ARGENTINA_ACCOUNTS),
            total_tweets=len(all_tweets),
            api_calls=self.api_calls,
        )

        return all_tweets

    def get_high_engagement_tweets(
        self,
        tweets: list[XTweet],
        min_engagement: int = 50,
    ) -> list[XTweet]:
        """Filter for high-engagement tweets.

        Args:
            tweets: List of tweets to filter
            min_engagement: Minimum total engagement

        Returns:
            Tweets with engagement above threshold
        """
        return [t for t in tweets if t.total_engagement >= min_engagement]

    def get_api_stats(self) -> dict:
        """Get API usage statistics.

        Returns:
            Dict with call counts and estimated costs
        """
        # X API v2 pay-per-use pricing (approximate)
        cost_per_call = 0.01  # $0.01 per request

        return {
            "total_calls": self.api_calls,
            "estimated_cost_usd": self.api_calls * cost_per_call,
            "accounts_monitored": len(ARGENTINA_ACCOUNTS),
        }

    def save_cache(self, tweets: list[XTweet]) -> None:
        """Save tweets to cache file.

        Args:
            tweets: Tweets to cache
        """
        cache_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tweets": [t.to_dict() for t in tweets],
        }

        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)

        log.debug("Saved tweet cache", count=len(tweets))

    def load_cache(self, max_age_minutes: int = 30) -> list[XTweet]:
        """Load tweets from cache if fresh.

        Args:
            max_age_minutes: Maximum cache age in minutes

        Returns:
            Cached tweets or empty list if stale/missing
        """
        if not CACHE_FILE.exists():
            return []

        try:
            with open(CACHE_FILE) as f:
                cache_data = json.load(f)

            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            age = datetime.now(timezone.utc) - timestamp

            if age > timedelta(minutes=max_age_minutes):
                log.debug("Cache expired", age_minutes=age.total_seconds() / 60)
                return []

            tweets = [XTweet.from_dict(d) for d in cache_data["tweets"]]
            log.debug("Loaded cache", count=len(tweets), age_minutes=age.total_seconds() / 60)
            return tweets

        except Exception as e:
            log.error("Failed to load cache", error=str(e))
            return []

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


def main():
    """Test the X API client."""
    print("=== X API Client Test ===\n")

    client = XAPIClient()

    if not client.bearer_token:
        print("X_BEARER_TOKEN not configured. Set it in .env file.")
        return

    try:
        # Fetch from one account
        print("Fetching tweets from @LuisCaputoAR...")
        tweets = client.get_user_tweets("LuisCaputoAR", max_results=5)

        if tweets:
            print(f"\nFound {len(tweets)} tweets:\n")
            for tweet in tweets:
                print(f"@{tweet.author_username} ({tweet.created_at.strftime('%Y-%m-%d %H:%M')})")
                print(f"  EN: {tweet.text_en[:100]}..." if len(tweet.text_en) > 100 else f"  EN: {tweet.text_en}")
                print(f"  Engagement: {tweet.total_engagement} (likes: {tweet.likes}, RT: {tweet.retweets})")
                print()
        else:
            print("No tweets fetched.")

        # Show API stats
        stats = client.get_api_stats()
        print(f"\nAPI Stats: {stats['total_calls']} calls, est. ${stats['estimated_cost_usd']:.4f}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
