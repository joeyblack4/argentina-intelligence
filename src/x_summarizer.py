"""
X/Twitter Sentiment Summarizer using Claude.

Analyzes tweets about Argentina markets and generates:
- Overall sentiment score (-10 to +10)
- Key themes and insights
- Breaking news alerts
- Actionable investment signals
"""

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.collectors.x_api import XTweet, XConversation, ARGENTINA_ACCOUNTS
from src.config import settings

log = structlog.get_logger()


@dataclass
class XSentimentReport:
    """Claude-generated sentiment analysis report."""

    timestamp: datetime
    tweet_count: int

    # Sentiment
    overall_sentiment: str  # "bullish", "bearish", "neutral", "mixed"
    sentiment_score: float  # -10 to +10
    confidence: float  # 0-100

    # Insights
    key_themes: list[str] = field(default_factory=list)
    notable_tweets: list[dict] = field(default_factory=list)  # Simplified tweet data
    breaking_news: list[str] = field(default_factory=list)

    # Narrative
    summary: str = ""
    actionable_insights: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    # Metadata
    accounts_analyzed: int = 0
    high_engagement_count: int = 0
    official_tweets_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "XSentimentReport":
        """Create from dictionary."""
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


# Prompt for Claude to analyze tweets
X_ANALYSIS_PROMPT = """You are analyzing X/Twitter content about Argentina's economy and markets for an investment platform focused on Argentine ADRs.

## TWEET DATA
{tweet_data}

## HIGH-ENGAGEMENT CONVERSATIONS
{conversation_data}

## ACCOUNT CATEGORIES
- **official**: Government/BCRA accounts - highest signal for policy changes
- **analyst**: Economists/analysts - high signal for market interpretation
- **media**: Financial news - breaking news and market coverage
- **market**: Market participants - trading sentiment

## YOUR TASK

Analyze this X content and provide a comprehensive sentiment report for Argentina ADR investors.

**Output Format (JSON):**
```json
{{
    "overall_sentiment": "bullish|bearish|neutral|mixed",
    "sentiment_score": <float -10 to +10>,
    "confidence": <int 0-100>,
    "key_themes": [
        "Theme 1: Brief description",
        "Theme 2: Brief description"
    ],
    "notable_tweets": [
        {{
            "author": "@username",
            "summary": "Key point from tweet",
            "why_notable": "Why this matters"
        }}
    ],
    "breaking_news": [
        "Time-sensitive item 1",
        "Time-sensitive item 2"
    ],
    "summary": "2-3 paragraph narrative summary of Argentine FinTwit sentiment...",
    "actionable_insights": [
        "Specific investment insight 1",
        "Specific investment insight 2"
    ],
    "risk_flags": [
        "Risk or concern being raised"
    ]
}}
```

**Scoring Guidelines:**
- +8 to +10: Extremely bullish - major positive catalyst, policy wins
- +4 to +7: Bullish - positive sentiment, constructive outlook
- +1 to +3: Slightly bullish - cautiously optimistic
- -1 to +1: Neutral - mixed signals, wait-and-see
- -3 to -1: Slightly bearish - some concerns emerging
- -7 to -4: Bearish - negative sentiment, risk-off
- -10 to -8: Extremely bearish - crisis concerns, major risks

**Focus Areas:**
1. Policy signals from official accounts (BCRA, Economy Ministry, Milei, Caputo)
2. Reserve accumulation / currency dynamics
3. IMF negotiations and fiscal targets
4. Market sentiment on Argentine ADRs
5. Breaking economic data or news
6. Political developments affecting markets

Be specific and actionable. Argentine FinTwit often provides early signals before mainstream news."""


class XSummarizer:
    """Use Claude to analyze and summarize X content."""

    def __init__(self):
        """Initialize the summarizer."""
        if ANTHROPIC_AVAILABLE and settings.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            self.client = None
            if not ANTHROPIC_AVAILABLE:
                log.warning("Anthropic not available - X summarization disabled")
            elif not settings.anthropic_api_key:
                log.warning("ANTHROPIC_API_KEY not set - X summarization disabled")

    def _format_tweets(self, tweets: list[XTweet]) -> str:
        """Format tweets for the prompt."""
        if not tweets:
            return "No tweets available."

        lines = []

        # Group by category for better analysis
        by_category = {}
        for tweet in tweets:
            cat = tweet.author_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(tweet)

        for category in ["official", "analyst", "media", "market", "unknown"]:
            if category not in by_category:
                continue

            category_tweets = by_category[category]
            lines.append(f"\n### {category.upper()} ACCOUNTS ({len(category_tweets)} tweets)")

            for tweet in category_tweets[:15]:  # Limit per category
                engagement_str = f"[{tweet.likes}L/{tweet.retweets}RT]"
                time_str = tweet.created_at.strftime("%m/%d %H:%M")

                lines.append(f"\n@{tweet.author_username} {engagement_str} ({time_str})")
                # Use translated text
                text = tweet.text_en if tweet.text_en else tweet.text
                lines.append(f"  {text[:300]}")

        return "\n".join(lines)

    def _format_conversations(self, conversations: list[XConversation]) -> str:
        """Format conversations for the prompt."""
        if not conversations:
            return "No high-engagement conversations."

        lines = []
        for conv in conversations[:5]:  # Top 5 conversations
            root = conv.root_tweet
            lines.append(f"\n**Thread by @{root.author_username}** ({conv.reply_count} replies, {conv.total_engagement} engagement)")
            lines.append(f"  Root: {root.text_en[:200]}")

            if conv.replies:
                lines.append("  Key replies:")
                for reply in conv.replies[:3]:
                    lines.append(f"    - @{reply.author_username}: {reply.text_en[:100]}")

        return "\n".join(lines)

    def analyze_tweets(
        self,
        tweets: list[XTweet],
        conversations: list[XConversation] = None,
    ) -> XSentimentReport:
        """Generate comprehensive sentiment report.

        Args:
            tweets: List of tweets to analyze
            conversations: Optional high-engagement conversations

        Returns:
            XSentimentReport with sentiment analysis
        """
        if not tweets:
            return XSentimentReport(
                timestamp=datetime.now(timezone.utc),
                tweet_count=0,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0,
                summary="No tweets available for analysis.",
            )

        if not self.client:
            log.error("Anthropic client not available")
            return XSentimentReport(
                timestamp=datetime.now(timezone.utc),
                tweet_count=len(tweets),
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0,
                summary="AI analysis unavailable - Anthropic API not configured.",
                risk_flags=["AI summarization disabled"],
            )

        # Format data for prompt
        tweet_data = self._format_tweets(tweets)
        conversation_data = self._format_conversations(conversations or [])

        prompt = X_ANALYSIS_PROMPT.format(
            tweet_data=tweet_data,
            conversation_data=conversation_data,
        )

        try:
            log.info("Calling Claude for X sentiment analysis", tweet_count=len(tweets))

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_response = response.content[0].text

            # Parse the JSON response
            report = self._parse_response(raw_response, tweets, conversations)

            log.info(
                "Generated X sentiment report",
                sentiment=report.overall_sentiment,
                score=report.sentiment_score,
                themes=len(report.key_themes),
            )

            return report

        except Exception as e:
            log.error("Failed to generate sentiment report", error=str(e))
            return XSentimentReport(
                timestamp=datetime.now(timezone.utc),
                tweet_count=len(tweets),
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0,
                summary=f"Error generating analysis: {str(e)}",
                risk_flags=[str(e)],
            )

    def _parse_response(
        self,
        raw: str,
        tweets: list[XTweet],
        conversations: list[XConversation] = None,
    ) -> XSentimentReport:
        """Parse Claude's response into structured report."""
        report = XSentimentReport(
            timestamp=datetime.now(timezone.utc),
            tweet_count=len(tweets),
            overall_sentiment="neutral",
            sentiment_score=0.0,
            confidence=50,
        )

        # Calculate metadata
        report.accounts_analyzed = len(set(t.author_username for t in tweets))
        report.high_engagement_count = len([t for t in tweets if t.total_engagement >= 50])
        report.official_tweets_count = len([t for t in tweets if t.author_category == "official"])

        try:
            # Extract JSON from response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                log.warning("No JSON found in response")
                report.summary = raw[:500]
                return report

            json_str = raw[json_start:json_end]
            data = json.loads(json_str)

            # Parse sentiment
            report.overall_sentiment = data.get("overall_sentiment", "neutral")
            report.sentiment_score = float(data.get("sentiment_score", 0))
            report.confidence = int(data.get("confidence", 50))

            # Parse insights
            report.key_themes = data.get("key_themes", [])
            report.notable_tweets = data.get("notable_tweets", [])
            report.breaking_news = data.get("breaking_news", [])

            # Parse narrative
            report.summary = data.get("summary", "")
            report.actionable_insights = data.get("actionable_insights", [])
            report.risk_flags = data.get("risk_flags", [])

            # Clamp score to valid range
            report.sentiment_score = max(-10, min(10, report.sentiment_score))
            report.confidence = max(0, min(100, report.confidence))

        except json.JSONDecodeError as e:
            log.warning("Failed to parse JSON response", error=str(e))
            report.summary = raw[:500]
            report.risk_flags.append("Failed to parse AI response")

        return report

    def get_quick_sentiment(self, tweets: list[XTweet]) -> tuple[str, float]:
        """Get quick sentiment without full analysis.

        Uses simple heuristics for fast results.

        Args:
            tweets: List of tweets

        Returns:
            Tuple of (sentiment_label, score)
        """
        if not tweets:
            return "neutral", 0.0

        # Bullish keywords (Spanish and English)
        bullish_words = [
            "sube", "subiendo", "positivo", "bien", "excelente", "record",
            "rally", "bullish", "up", "gains", "strong", "growth",
            "reservas", "reserves", "acumulacion", "superavit", "surplus",
        ]

        # Bearish keywords
        bearish_words = [
            "baja", "bajando", "negativo", "mal", "crisis", "riesgo",
            "bearish", "down", "losses", "weak", "decline", "fall",
            "devaluacion", "devaluation", "deficit", "inflacion", "inflation",
        ]

        bullish_count = 0
        bearish_count = 0
        total_weight = 0

        for tweet in tweets:
            text = f"{tweet.text} {tweet.text_en}".lower()
            weight = tweet.author_weight

            for word in bullish_words:
                if word in text:
                    bullish_count += weight
                    break

            for word in bearish_words:
                if word in text:
                    bearish_count += weight
                    break

            total_weight += weight

        if total_weight == 0:
            return "neutral", 0.0

        # Calculate score
        net_sentiment = bullish_count - bearish_count
        score = (net_sentiment / total_weight) * 5  # Scale to -5 to +5 range

        if score > 2:
            return "bullish", min(score, 10)
        elif score < -2:
            return "bearish", max(score, -10)
        else:
            return "neutral", score


def main():
    """Test the X summarizer."""
    print("=== X Summarizer Test ===\n")

    # Create mock tweets for testing
    from src.collectors.x_api import XTweet

    mock_tweets = [
        XTweet(
            id="1",
            author_username="LuisCaputoAR",
            author_name="Luis Caputo",
            author_followers=500000,
            text="Seguimos acumulando reservas. El programa sigue su curso.",
            text_en="We continue accumulating reserves. The program continues on track.",
            created_at=datetime.now(timezone.utc),
            likes=5000,
            retweets=1000,
            author_category="official",
            author_weight=1.8,
        ),
        XTweet(
            id="2",
            author_username="fernaborzel",
            author_name="Ferna Borzel",
            author_followers=100000,
            text="Los datos de reservas muestran una tendencia muy positiva.",
            text_en="Reserve data shows a very positive trend.",
            created_at=datetime.now(timezone.utc),
            likes=500,
            retweets=100,
            author_category="analyst",
            author_weight=1.5,
        ),
    ]

    summarizer = XSummarizer()

    # Test quick sentiment
    sentiment, score = summarizer.get_quick_sentiment(mock_tweets)
    print(f"Quick sentiment: {sentiment} ({score:+.1f})")

    # Test full analysis (if API available)
    if summarizer.client:
        print("\nGenerating full analysis...")
        report = summarizer.analyze_tweets(mock_tweets)
        print(f"\nSentiment: {report.overall_sentiment} ({report.sentiment_score:+.1f})")
        print(f"Confidence: {report.confidence}%")
        print(f"\nSummary: {report.summary[:300]}...")
        print(f"\nKey themes: {report.key_themes}")
    else:
        print("\nSkipping full analysis - Anthropic API not configured")


if __name__ == "__main__":
    main()
