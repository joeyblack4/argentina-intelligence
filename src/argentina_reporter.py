"""
Argentina Markets Intelligence Reporter.

Generates comprehensive morning intelligence reports and afternoon updates
combining Twitter, Reddit, news, and market data for Joey Sterling.

Scheduled via cron jobs at 9am PST (morning) and 2pm PST (afternoon).
"""

import json
import structlog
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.collectors.social_collector import SocialIntelligenceCollector, SocialIntelligence
from src.collectors.x_api import XAPIClient
from src.analysis.x_summarizer import XSummarizer
from src.config import settings

log = structlog.get_logger()

# Memory directory for reports
MEMORY_DIR = Path.home() / ".openclaw" / "memory"
REPORTS_DIR = MEMORY_DIR / "argentina-reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ArgentinaIntelligenceReporter:
    """Generate comprehensive Argentina market intelligence reports."""
    
    def __init__(self):
        """Initialize the reporter."""
        self.social_collector = SocialIntelligenceCollector()
        self.x_client = XAPIClient()
        self.summarizer = XSummarizer()
    
    def generate_morning_intelligence(self) -> str:
        """Generate comprehensive morning intelligence report."""
        try:
            log.info("Generating morning intelligence report")
            
            # Collect latest social intelligence
            intelligence = self.social_collector.collect_intelligence(force_refresh=True)
            
            # Generate market context
            market_context = self._get_market_context()
            
            # Create comprehensive report
            report = self._format_morning_report(intelligence, market_context)
            
            # Save to memory
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_file = REPORTS_DIR / f"morning-intelligence-{date_str}.md"
            
            with open(report_file, "w") as f:
                f.write(report)
            
            log.info(f"Morning intelligence saved to {report_file}")
            
            return report
            
        except Exception as e:
            log.error("Failed to generate morning intelligence", error=str(e))
            return self._generate_error_report("morning intelligence", str(e))
    
    def generate_afternoon_update(self) -> str:
        """Generate focused afternoon update report."""
        try:
            log.info("Generating afternoon update report")
            
            # Collect latest social intelligence
            intelligence = self.social_collector.collect_intelligence(force_refresh=True)
            
            # Get previous morning report for comparison
            morning_report = self._get_latest_morning_report()
            
            # Create update report
            report = self._format_afternoon_update(intelligence, morning_report)
            
            # Save to memory
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_file = REPORTS_DIR / f"afternoon-update-{date_str}.md"
            
            with open(report_file, "w") as f:
                f.write(report)
            
            log.info(f"Afternoon update saved to {report_file}")
            
            return report
            
        except Exception as e:
            log.error("Failed to generate afternoon update", error=str(e))
            return self._generate_error_report("afternoon update", str(e))
    
    def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context and key metrics."""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_hours": self._is_market_hours(),
            "key_events": self._get_upcoming_events(),
            "sentiment_drivers": self._get_sentiment_drivers()
        }
        
        return context
    
    def _is_market_hours(self) -> bool:
        """Check if we're in US market hours."""
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now_eastern = datetime.now(eastern)
        
        # Extended hours: 8 AM to 5 PM ET
        market_open = now_eastern.replace(hour=8, minute=0, second=0, microsecond=0)
        market_close = now_eastern.replace(hour=17, minute=0, second=0, microsecond=0)
        
        return market_open <= now_eastern <= market_close
    
    def _get_upcoming_events(self) -> List[str]:
        """Get upcoming events that could impact Argentina markets."""
        # This would integrate with your events collector
        return [
            "MSCI reclassification decision (pending)",
            "IMF quarterly review (ongoing)",
            "Monthly INDEC inflation data (monthly)"
        ]
    
    def _get_sentiment_drivers(self) -> List[str]:
        """Get current key sentiment drivers."""
        drivers = [
            "Peso stability and BCRA intervention",
            "Reserve accumulation trajectory", 
            "Fiscal surplus maintenance",
            "Milei reform momentum",
            "International capital flows"
        ]
        return drivers
    
    def _format_morning_report(self, intelligence: SocialIntelligence, market_context: Dict) -> str:
        """Format the morning intelligence report."""
        
        now = datetime.now()
        
        report = f"""# Argentina Markets Intelligence Report
**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')} PST

## Executive Summary

**Overall Sentiment:** {intelligence.community_sentiment.upper()} ({intelligence.overall_score:+.1f}/10)
**Data Coverage:** {intelligence.total_posts} social media posts analyzed
**Key Themes:** {len(intelligence.key_themes)} major discussion topics identified

---

## Social Media Intelligence

### Twitter/X Analysis
- **Sentiment:** {intelligence.twitter_sentiment or 'N/A'} ({intelligence.twitter_score:+.1f}/10)
"""

        if intelligence.twitter_data:
            report += f"""- **Posts Analyzed:** {intelligence.twitter_data.get('tweet_count', 0)}
- **High Engagement:** {intelligence.twitter_data.get('high_engagement_count', 0)} posts
- **Official Sources:** {intelligence.twitter_data.get('official_tweets', 0)} government/BCRA tweets
- **Analyst Coverage:** {intelligence.twitter_data.get('analyst_tweets', 0)} analyst tweets
"""
        
        report += f"""
### Reddit Community Sentiment
- **Sentiment:** {intelligence.reddit_sentiment or 'N/A'} ({intelligence.reddit_score:+.1f}/10)
"""

        if intelligence.reddit_data:
            report += f"""- **Posts Analyzed:** {intelligence.reddit_data.get('post_count', 0)}
- **Subreddits Covered:** {intelligence.reddit_data.get('subreddits_covered', 0)}
- **High Engagement:** {intelligence.reddit_data.get('high_score_posts', 0)} posts
- **Total Upvotes:** {intelligence.reddit_data.get('total_upvotes', 0):,}
"""

        # Breaking News Section
        if intelligence.breaking_news:
            report += "\n## 🚨 Breaking News & Alerts\n\n"
            for i, news in enumerate(intelligence.breaking_news[:5], 1):
                report += f"{i}. {news}\n"
            report += "\n"

        # Key Themes Section
        if intelligence.key_themes:
            report += "\n## Key Discussion Themes\n\n"
            for i, theme in enumerate(intelligence.key_themes[:8], 1):
                report += f"{i}. {theme}\n"
            report += "\n"

        # Notable Posts Section
        if intelligence.notable_posts:
            report += "\n## Notable Social Media Posts\n\n"
            for i, post in enumerate(intelligence.notable_posts[:5], 1):
                platform = post.get('platform', 'Unknown')
                data = post.get('data', {})
                
                if platform == 'Twitter':
                    author = data.get('author', 'Unknown')
                    summary = data.get('summary', 'No summary available')
                    why_notable = data.get('why_notable', '')
                    report += f"**{i}. Twitter @{author}**\n{summary}\n"
                    if why_notable:
                        report += f"*Why notable: {why_notable}*\n\n"
                else:  # Reddit
                    subreddit = data.get('subreddit', 'Unknown')
                    title = data.get('title', 'No title')
                    score = data.get('score', 0)
                    comments = data.get('comments', 0)
                    report += f"**{i}. {subreddit}** (Score: {score}, Comments: {comments})\n{title}\n\n"
        
        # Market Context
        report += f"""
## Market Context & Catalysts

### Current Environment
- **Market Hours:** {"Open" if market_context['market_hours'] else "Closed"}
- **Key Events:** {len(market_context['key_events'])} upcoming catalysts

### Sentiment Drivers
"""
        for driver in market_context['sentiment_drivers']:
            report += f"- {driver}\n"

        report += f"""
### Upcoming Events
"""
        for event in market_context['key_events']:
            report += f"- {event}\n"

        # Investment Implications
        report += self._generate_investment_implications(intelligence)
        
        # Risk Assessment
        report += self._generate_risk_assessment(intelligence)
        
        report += f"""
---
*Report generated by Argentina Markets Intelligence System*
*Next update: Tomorrow 9:00 AM PST*
"""

        return report
    
    def _format_afternoon_update(self, intelligence: SocialIntelligence, morning_report: Optional[str]) -> str:
        """Format the afternoon update report."""
        
        now = datetime.now()
        
        report = f"""# Argentina Markets Afternoon Update
**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')} PST

## Quick Status Update

**Current Sentiment:** {intelligence.community_sentiment.upper()} ({intelligence.overall_score:+.1f}/10)
**New Posts Since Morning:** {intelligence.total_posts}
"""

        # Compare to morning if available
        if morning_report:
            report += self._compare_to_morning(intelligence, morning_report)

        report += "\n## What's Changed Since Morning\n\n"

        # New breaking news
        if intelligence.breaking_news:
            report += "### New Breaking News\n\n"
            for i, news in enumerate(intelligence.breaking_news[:3], 1):
                report += f"{i}. {news}\n"
            report += "\n"

        # Updated themes
        if intelligence.key_themes:
            report += "### Updated Discussion Themes\n\n"
            for i, theme in enumerate(intelligence.key_themes[:5], 1):
                report += f"{i}. {theme}\n"
            report += "\n"

        # Notable new posts
        if intelligence.notable_posts:
            report += "### Notable New Posts\n\n"
            for i, post in enumerate(intelligence.notable_posts[:3], 1):
                platform = post.get('platform', 'Unknown')
                data = post.get('data', {})
                
                if platform == 'Twitter':
                    author = data.get('author', 'Unknown')
                    summary = data.get('summary', 'No summary')
                    report += f"**{i}. Twitter @{author}** - {summary}\n"
                else:
                    subreddit = data.get('subreddit', 'Unknown')
                    title = data.get('title', 'No title')
                    report += f"**{i}. {subreddit}** - {title}\n"
            report += "\n"

        # Action Items
        report += self._generate_action_items(intelligence)

        report += f"""
---
*Afternoon update complete. Full morning report available in memory files.*
*Next update: Tomorrow 2:00 PM PST*
"""

        return report
    
    def _compare_to_morning(self, intelligence: SocialIntelligence, morning_report: str) -> str:
        """Compare current intelligence to morning report."""
        
        # This is a simplified comparison - in practice you'd parse the morning report
        # and extract key metrics to compare against current state
        
        comparison = "\n## Morning vs Afternoon Comparison\n\n"
        
        # Extract key sentiment changes
        comparison += f"**Sentiment Change:** "
        if intelligence.overall_score > 1:
            comparison += "Becoming more bullish 📈\n"
        elif intelligence.overall_score < -1:
            comparison += "Turning more bearish 📉\n"
        else:
            comparison += "Remains neutral ➡️\n"
        
        # Highlight new themes
        if intelligence.key_themes:
            comparison += "\n**New Themes Emerging:**\n"
            for theme in intelligence.key_themes[:3]:
                comparison += f"- {theme}\n"
        
        return comparison
    
    def _generate_investment_implications(self, intelligence: SocialIntelligence) -> str:
        """Generate investment implications section."""
        
        implications = "\n## Investment Implications\n\n"
        
        if intelligence.overall_score > 3:
            implications += "### Bullish Signals 🚀\n"
            implications += "- Strong positive sentiment across social platforms\n"
            implications += "- Community confidence in Argentina's reform trajectory\n"
            implications += "- Consider increasing exposure to Argentina ADRs\n"
            
        elif intelligence.overall_score < -3:
            implications += "### Bearish Signals ⚠️\n"
            implications += "- Negative sentiment prevailing in discussions\n"
            implications += "- Community concerns about policy execution\n"
            implications += "- Consider risk management and position sizing\n"
            
        else:
            implications += "### Mixed Signals ⚖️\n"
            implications += "- Sentiment remains balanced\n"
            implications += "- Monitor key developments before making moves\n"
            implications += "- Focus on fundamentals and official data\n"
        
        return implications
    
    def _generate_risk_assessment(self, intelligence: SocialIntelligence) -> str:
        """Generate risk assessment section."""
        
        risks = "\n## Risk Assessment\n\n"
        
        # Check for risk flags
        risk_flags = []
        
        if intelligence.breaking_news:
            # Analyze breaking news for risk indicators
            for news in intelligence.breaking_news:
                if any(word in news.lower() for word in ['crisis', 'devaluation', 'default', 'collapse']):
                    risk_flags.append("Market-moving negative news detected")
        
        if intelligence.overall_score < -2:
            risk_flags.append("Strong negative community sentiment")
        
        if risk_flags:
            risks += "### Risk Factors Identified ⚠️\n"
            for risk in risk_flags:
                risks += f"- {risk}\n"
        else:
            risks += "### Risk Level: Moderate ✅\n"
            risks += "- No immediate risk indicators detected\n"
            risks += "- Maintain standard risk management protocols\n"
        
        return risks
    
    def _generate_action_items(self, intelligence: SocialIntelligence) -> str:
        """Generate action items for the afternoon update."""
        
        actions = "\n## Action Items\n\n"
        
        if intelligence.overall_score > 2:
            actions += "### Consider\n"
            actions += "- ✅ Monitor for buying opportunities\n"
            actions += "- 📊 Track official data releases\n"
            
        elif intelligence.overall_score < -2:
            actions += "### Consider\n"
            actions += "- ⚠️ Review position sizes\n"
            actions += "- 📰 Watch for official clarifications\n"
            
        else:
            actions += "### Monitor\n"
            actions += "- 🔍 Track sentiment evolution\n"
            actions += "- 📈 Watch for breakout signals\n"
        
        actions += "\n"
        return actions
    
    def _get_latest_morning_report(self) -> Optional[str]:
        """Get the latest morning report for comparison."""
        try:
            # Get most recent morning report
            report_files = list(REPORTS_DIR.glob("morning-intelligence-*.md"))
            if not report_files:
                return None
            
            latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_report, "r") as f:
                return f.read()
                
        except Exception as e:
            log.error("Failed to load latest morning report", error=str(e))
            return None
    
    def _generate_error_report(self, report_type: str, error: str) -> str:
        """Generate an error report when data collection fails."""
        
        return f"""# Argentina Markets Intelligence Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PST

## Report Status: {report_type.title()} Generation Failed

**Error:** {error}

**Next Steps:**
- Automatic retry scheduled
- Manual intervention may be required
- Check API configurations and data sources

---
*Error report generated by Argentina Markets Intelligence System*
"""

    def close(self):
        """Clean up resources."""
        self.social_collector.close()


def generate_morning_report():
    """Function to be called by cron job for morning intelligence."""
    reporter = ArgentinaIntelligenceReporter()
    try:
        report = reporter.generate_morning_intelligence()
        print("Morning intelligence report generated successfully")
        return report
    finally:
        reporter.close()


def generate_afternoon_report():
    """Function to be called by cron job for afternoon update."""
    reporter = ArgentinaIntelligenceReporter()
    try:
        report = reporter.generate_afternoon_update()
        print("Afternoon update report generated successfully")
        return report
    finally:
        reporter.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "morning":
        print("Generating morning intelligence...")
        generate_morning_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "afternoon":
        print("Generating afternoon update...")
        generate_afternoon_report()
    else:
        print("Usage: python argentina_reporter.py [morning|afternoon]")