#!/usr/bin/env python3
"""
Argentina Markets Intelligence - Entry Point

Quick CLI for running intelligence reports and tests.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.argentina_reporter import ArgentinaIntelligenceReporter


def main():
    parser = argparse.ArgumentParser(description="Argentina Markets Intelligence")
    parser.add_argument(
        "action",
        choices=["morning", "afternoon", "test", "collect"],
        help="Action to perform"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh data collection"
    )
    
    args = parser.parse_args()
    
    reporter = ArgentinaIntelligenceReporter()
    
    try:
        if args.action == "morning":
            print("Generating pre-market intelligence report...")
            report = reporter.generate_morning_intelligence()
            print("Report generated successfully!")
            print("\n" + "="*50)
            print(report[:500] + "..." if len(report) > 500 else report)
            
        elif args.action == "afternoon":
            print("Generating post-market update...")
            report = reporter.generate_afternoon_update()
            print("Update generated successfully!")
            print("\n" + "="*50)
            print(report[:500] + "..." if len(report) > 500 else report)
            
        elif args.action == "test":
            print("Testing intelligence collection...")
            intelligence = reporter.social_collector.collect_intelligence(force_refresh=True)
            print(f"Collected {intelligence.total_posts} posts")
            print(f"Overall sentiment: {intelligence.community_sentiment} ({intelligence.overall_score:+.1f})")
            
        elif args.action == "collect":
            print("Collecting social intelligence...")
            intelligence = reporter.social_collector.collect_intelligence(force_refresh=args.force_refresh)
            print(f"Total posts: {intelligence.total_posts}")
            print(f"Twitter sentiment: {intelligence.twitter_sentiment} ({intelligence.twitter_score:+.1f})")
            print(f"Reddit sentiment: {intelligence.reddit_sentiment} ({intelligence.reddit_score:+.1f})")
            print(f"Overall: {intelligence.community_sentiment} ({intelligence.overall_score:+.1f})")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        reporter.close()


if __name__ == "__main__":
    main()