# Argentina Markets Intelligence

Comprehensive social media intelligence system for Argentina market analysis, combining Twitter/X, Reddit, and community-driven insights for investment decision support.

## 🎯 Overview

This system provides:
- **Pre-market intelligence** (6:00 AM PST daily) - Comprehensive sentiment analysis
- **Post-market updates** (2:00 PM PST daily) - What's changed since morning
- **Real-time monitoring** of key Argentina market accounts
- **Breaking news detection** and sentiment scoring
- **Investment implications** and risk assessment

## 📊 Data Sources

### Twitter/X
- 15+ key Argentina accounts with weighted importance
- Government officials (BCRA, Economy Ministry, Milei) - highest signal
- Top analysts and financial media
- Breaking policy announcements

### Reddit
- r/argentina, r/LatinAmerica, r/investing, r/economics
- Argentina-specific keyword filtering
- Community sentiment analysis
- High-engagement post prioritization

### Key Metrics
- Sentiment scores (-10 to +10)
- Breaking news alerts
- Investment implications
- Risk assessments
- Actionable insights

## 🚀 Features

- **Automated Reporting**: Scheduled intelligence reports
- **Sentiment Analysis**: AI-powered sentiment scoring
- **Breaking News Detection**: Real-time alert system
- **Investment Insights**: Actionable market intelligence
- **Risk Management**: Comprehensive risk assessment

## 📈 Investment Focus Areas

- Peso stability and BCRA intervention
- Reserve accumulation trajectory
- Fiscal surplus maintenance
- Milei reform momentum
- International capital flows
- MSCI reclassification
- IMF negotiations

## 🛠️ Setup

1. Configure API keys:
   - X/Twitter Bearer Token
   - Reddit API credentials
   - Anthropic API key for AI analysis

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the intelligence collector:
   ```bash
   python argentina_reporter.py morning
   ```

## 📋 Reports

### Morning Intelligence (6:00 AM PST)
- Comprehensive social sentiment analysis
- Key themes and discussions
- Breaking news and alerts
- Investment implications
- Risk assessment

### Afternoon Update (2:00 PM PST)
- Changes since morning
- New developments
- Sentiment shifts
- Action items

## 🔧 Architecture

```
argentina-intelligence/
├── argentina_reporter.py     # Main reporting system
├── social_collector.py       # Social media data collection
├── x_api.py                 # Twitter/X API client
├── x_summarizer.py          # AI-powered sentiment analysis
├── requirements.txt         # Dependencies
├── config.py               # Configuration settings
└── README.md              # This file
```

## 📊 Report Output

Reports are generated as:
- **Markdown files** for human-readable format
- **JSON data** for programmatic access
- **Automated delivery** via scheduled jobs

## 🎯 Use Cases

- **Pre-market positioning** - Get intelligence before trading
- **Risk management** - Monitor sentiment shifts
- **Opportunity identification** - Spot emerging themes
- **Investment thesis validation** - Track community sentiment

---

*Built for Joey Sterling's Argentina investment strategy*
