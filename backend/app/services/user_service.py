"""
User service for CRUD operations on User model.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UserNotFoundError
from app.db.models.user import User


class UserService:
    """Service for managing users."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_telegram_id(self, telegram_id: int) -> User:
        """
        Get user by Telegram ID.

        Args:
            telegram_id: Telegram user ID

        Returns:
            User instance

        Raises:
            UserNotFoundError: If user not found
        """
        result = await self.db.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()

        if not user:
            raise UserNotFoundError(telegram_id)

        return user

    async def get_by_id(self, user_id: int) -> User:
        """
        Get user by internal ID.

        Args:
            user_id: Internal user ID

        Returns:
            User instance

        Raises:
            UserNotFoundError: If user not found
        """
        result = await self.db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise UserNotFoundError(user_id)

        return user

    async def create(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        role: str = "DEMO",
    ) -> User:
        """
        Create a new user.

        Args:
            telegram_id: Telegram user ID
            username: Telegram username
            first_name: User's first name
            last_name: User's last name
            role: User role (DEMO, FULL_ACCESS, ADMIN)

        Returns:
            Created user instance
        """
        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            role=role,
            authorized=False,
            verified=False,
        )

        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)

        return user

    async def update_role(self, telegram_id: int, role: str) -> User:
        """
        Update user role.

        Args:
            telegram_id: Telegram user ID
            role: New role

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
        """
        user = await self.get_by_telegram_id(telegram_id)
        user.role = role
        await self.db.flush()
        await self.db.refresh(user)

        return user

    async def authorize(self, telegram_id: int) -> User:
        """
        Authorize a user.

        Args:
            telegram_id: Telegram user ID

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
        """
        user = await self.get_by_telegram_id(telegram_id)
        user.authorized = True
        await self.db.flush()
        await self.db.refresh(user)

        return user

    async def update_subscription(
        self,
        telegram_id: int,
        subscription_tier: str,
        expires_at: datetime,
    ) -> User:
        """
        Update user subscription.

        Args:
            telegram_id: Telegram user ID
            subscription_tier: Subscription tier name
            expires_at: Subscription expiration datetime

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
        """

        user = await self.get_by_telegram_id(telegram_id)
        user.subscription_tier = subscription_tier
        user.subscription_expires_at = expires_at
        await self.db.flush()
        await self.db.refresh(user)

        return user

    async def update_settings(self, telegram_id: int, settings: dict) -> User:
        """
        Update user settings.

        Args:
            telegram_id: Telegram user ID
            settings: Settings dictionary

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
        """
        user = await self.get_by_telegram_id(telegram_id)
        user.settings = settings
        await self.db.flush()
        await self.db.refresh(user)

        return user

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[User]:
        """
        List all users with pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip

        Returns:
            List of users
        """
        result = await self.db.execute(
            select(User).limit(limit).offset(offset).order_by(User.created_at.desc())
        )
        return list(result.scalars().all())
