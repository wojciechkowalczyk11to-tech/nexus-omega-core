"""
User cache service using Redis.
"""

import json
from typing import Any

import redis.asyncio as redis

from telegram_bot.config import settings


class UserCache:
    """Redis cache for user data and tokens."""

    def __init__(self, redis_url: str | None = None) -> None:
        """
        Initialize user cache.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or settings.redis_url
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        """Close Redis connection."""
        await self.redis.aclose()

    async def get_user_token(self, telegram_id: int) -> str | None:
        """
        Get cached JWT token for user.

        Args:
            telegram_id: Telegram user ID

        Returns:
            JWT token or None
        """
        key = f"user_token:{telegram_id}"
        return await self.redis.get(key)

    async def set_user_token(
        self, telegram_id: int, token: str, ttl: int = 86400
    ) -> None:
        """
        Cache JWT token for user.

        Args:
            telegram_id: Telegram user ID
            token: JWT token
            ttl: Time to live in seconds (default 24h)
        """
        key = f"user_token:{telegram_id}"
        await self.redis.setex(key, ttl, token)

    async def get_user_data(self, telegram_id: int) -> dict[str, Any] | None:
        """
        Get cached user data.

        Args:
            telegram_id: Telegram user ID

        Returns:
            User data dict or None
        """
        key = f"user_data:{telegram_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set_user_data(
        self, telegram_id: int, user_data: dict[str, Any], ttl: int = 3600
    ) -> None:
        """
        Cache user data.

        Args:
            telegram_id: Telegram user ID
            user_data: User data dict
            ttl: Time to live in seconds (default 1h)
        """
        key = f"user_data:{telegram_id}"
        await self.redis.setex(key, ttl, json.dumps(user_data))

    async def get_user_mode(self, telegram_id: int) -> str | None:
        """
        Get user's current mode.

        Args:
            telegram_id: Telegram user ID

        Returns:
            Mode (eco, smart, deep) or None
        """
        key = f"user_mode:{telegram_id}"
        return await self.redis.get(key)

    async def set_user_mode(self, telegram_id: int, mode: str) -> None:
        """
        Set user's current mode.

        Args:
            telegram_id: Telegram user ID
            mode: Mode (eco, smart, deep)
        """
        key = f"user_mode:{telegram_id}"
        await self.redis.set(key, mode)

    async def increment_rate_limit(
        self, telegram_id: int, window: int = 60
    ) -> int:
        """
        Increment rate limit counter.

        Args:
            telegram_id: Telegram user ID
            window: Window in seconds

        Returns:
            Current count
        """
        key = f"rate_limit:{telegram_id}"
        count = await self.redis.incr(key)

        if count == 1:
            await self.redis.expire(key, window)

        return count
