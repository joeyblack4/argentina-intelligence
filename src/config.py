"""
Configuration settings for Argentina Markets Intelligence system.
"""

from typing import Optional


class Settings:
    """Application settings."""
    
    # X/Twitter API
    x_bearer_token: Optional[str] = None
    
    # Reddit API
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    
    # Anthropic API for AI analysis
    anthropic_api_key: Optional[str] = None
    
    # Database (optional)
    database_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    # Cache settings
    cache_dir: str = "~/.argentina-intelligence"
    
    # API rate limiting
    x_rate_limit: int = 300  # requests per 15 minutes
    reddit_rate_limit: int = 60  # requests per minute
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# API Key Configuration Template
X_API_BEARER_TOKEN = """
# X/Twitter API v2 Bearer Token
# Get from: https://developer.x.com/
X_BEARER_TOKEN=your_x_bearer_token_here
"""

REDDIT_API_CONFIG = """
# Reddit API Credentials
# Get from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
"""

ANTHROPIC_API_CONFIG = """
# Anthropic API Key for AI Analysis
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""

EXAMPLE_ENV_FILE = f"""# Argentina Markets Intelligence Configuration
# Copy this file to .env and fill in your API keys

# X/Twitter API (optional - system works without)
{X_API_BEARER_TOKEN.strip()}

# Reddit API (optional - system works without)
{REDDIT_API_CONFIG.strip()}

# Anthropic API for AI analysis (optional - works with basic sentiment)
{ANTHROPIC_API_CONFIG.strip()}

# Optional: Database for data persistence
# DATABASE_URL=postgresql://user:pass@localhost/argentina

# Optional: Logging level
LOG_LEVEL=INFO
"""
