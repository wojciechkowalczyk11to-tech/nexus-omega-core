"""
Telegram bot configuration.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class BotSettings(BaseSettings):
    """Bot settings from environment variables."""

    # Telegram
    telegram_bot_token: str = Field(..., description="Telegram bot token")

    # Backend API
    backend_url: str = Field(
        default="http://backend:8000",
        description="Backend API base URL",
    )

    # Redis
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL",
    )

    # Rate limiting
    rate_limit_requests: int = Field(default=30, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Window in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = BotSettings()
