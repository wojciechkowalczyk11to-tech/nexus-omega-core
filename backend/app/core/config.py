"""
Configuration module using Pydantic Settings.
Loads all environment variables from .env file.
"""

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Telegram Bot ===
    telegram_bot_token: str = Field(..., description="Telegram Bot API token")
    telegram_mode: str = Field(default="polling", description="Bot mode: polling or webhook")
    webhook_url: str = Field(default="", description="Webhook URL for webhook mode")
    webhook_port: int = Field(default=8443, description="Webhook port")
    webhook_path: str = Field(default="webhook", description="Webhook path")
    webhook_secret_token: str = Field(default="", description="Webhook secret token")

    # === Access Control ===
    allowed_user_ids: list[int] = Field(
        default_factory=list, description="Allowed Telegram user IDs"
    )
    admin_user_ids: list[int] = Field(default_factory=list, description="Admin Telegram user IDs")
    full_telegram_ids: str = Field(default="", description="Comma-separated FULL access user IDs")
    demo_telegram_ids: str = Field(default="", description="Comma-separated DEMO access user IDs")

    # === Auth ===
    demo_unlock_code: str = Field(..., description="Code to unlock DEMO access")
    bootstrap_admin_code: str = Field(..., description="Code to bootstrap admin user")
    jwt_secret_key: str = Field(..., description="JWT secret key (256-bit)")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_hours: int = Field(default=24, description="JWT expiration in hours")

    # === Backend ===
    backend_url: str = Field(default="http://backend:8000", description="Backend service URL")
    database_url: str = Field(..., description="PostgreSQL connection URL")
    redis_url: str = Field(default="redis://redis:6379/0", description="Redis connection URL")
    postgres_password: str = Field(..., description="PostgreSQL password")

    # === AI Providers ===
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    xai_api_key: str = Field(default="", description="xAI Grok API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic Claude API key")

    # === Vertex AI ===
    vertex_project_id: str = Field(default="", description="Google Cloud project ID")
    vertex_location: str = Field(default="us-central1", description="Vertex AI location")
    vertex_search_datastore_id: str = Field(default="", description="Vertex AI Search datastore ID")

    # === Web Search ===
    brave_search_api_key: str = Field(default="", description="Brave Search API key")

    # === GitHub (Devin-mode) ===
    github_app_id: str = Field(default="", description="GitHub App ID")
    github_private_key_path: str = Field(default="", description="Path to GitHub private key")
    github_webhook_secret: str = Field(default="", description="GitHub webhook secret")
    github_token: str = Field(
        default="", description="GitHub Personal Access Token for API operations"
    )

    # === Provider Policy ===
    provider_policy_json: str = Field(
        default='{"default":{"providers":{"gemini":{"enabled":true},"deepseek":{"enabled":true},"groq":{"enabled":true}}}}',
        description="Provider policy configuration as JSON string",
    )

    # === Feature Flags ===
    voice_enabled: bool = Field(default=True, description="Enable voice features")
    inline_enabled: bool = Field(default=True, description="Enable inline queries")
    image_gen_enabled: bool = Field(default=True, description="Enable image generation")
    github_enabled: bool = Field(default=False, description="Enable GitHub Devin-mode")
    vertex_enabled: bool = Field(default=True, description="Enable Vertex AI Search")
    payments_enabled: bool = Field(default=False, description="Enable Telegram Stars payments")

    # === Logging ===
    log_level: str = Field(default="INFO", description="Logging level")
    log_json: bool = Field(default=True, description="Use JSON logging format")

    # === CORS ===
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=list,
        description="Comma-separated list of allowed CORS origins",
    )
    environment: str = Field(default="development", description="Runtime environment")

    # === Limits ===
    demo_grok_daily: int = Field(default=5, description="Daily Grok calls for DEMO users")
    demo_web_daily: int = Field(default=5, description="Daily web search calls for DEMO users")
    demo_smart_credits_daily: int = Field(
        default=20, description="Daily smart credits for DEMO users"
    )
    demo_deepseek_daily: int = Field(default=50, description="Daily DeepSeek calls for DEMO users")
    full_daily_usd_cap: float = Field(
        default=5.0, description="Daily USD spending cap for FULL users"
    )
    rate_limit_per_minute: int = Field(default=30, description="Rate limit per user per minute")

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v: Any) -> list[int]:
        """Parse allowed_user_ids from JSON string or list."""
        if isinstance(v, str):
            if not v or v == "[]":
                return []
            try:
                parsed = json.loads(v)
                return [int(uid) for uid in parsed]
            except (json.JSONDecodeError, ValueError):
                return []
        return v if isinstance(v, list) else []

    @field_validator("admin_user_ids", mode="before")
    @classmethod
    def parse_admin_user_ids(cls, v: Any) -> list[int]:
        """Parse admin_user_ids from JSON string or list."""
        if isinstance(v, str):
            if not v or v == "[]":
                return []
            try:
                parsed = json.loads(v)
                return [int(uid) for uid in parsed]
            except (json.JSONDecodeError, ValueError):
                return []
        return v if isinstance(v, list) else []

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if not v or v == "[]":
                return ["*"]
            try:
                parsed = json.loads(v)
                return [str(origin) for origin in parsed]
            except json.JSONDecodeError:
                return [o.strip() for o in v.split(",") if o.strip()]
        return v if isinstance(v, list) else ["*"]

    def get_provider_policy(self) -> dict[str, Any]:
        """Parse provider policy JSON."""
        try:
            return json.loads(self.provider_policy_json)
        except json.JSONDecodeError:
            return {"default": {"providers": {}}}

    def get_full_user_ids(self) -> list[int]:
        """Parse FULL access user IDs from comma-separated string."""
        if not self.full_telegram_ids:
            return []
        try:
            return [int(uid.strip()) for uid in self.full_telegram_ids.split(",") if uid.strip()]
        except ValueError:
            return []

    def get_demo_user_ids(self) -> list[int]:
        """Parse DEMO access user IDs from comma-separated string."""
        if not self.demo_telegram_ids:
            return []
        try:
            return [int(uid.strip()) for uid in self.demo_telegram_ids.split(",") if uid.strip()]
        except ValueError:
            return []


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
