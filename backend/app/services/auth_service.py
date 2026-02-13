"""
Authentication service for user registration, unlock, and JWT management.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    UserNotFoundError,
)
from app.core.security import create_access_token, hash_invite_code
from app.db.models.user import User
from app.services.invite_service import InviteService
from app.services.user_service import UserService


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.user_service = UserService(db)
        self.invite_service = InviteService(db)

    async def register(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> tuple[User, str]:
        """
        Register a new user with DEMO role.

        Args:
            telegram_id: Telegram user ID
            username: Telegram username
            first_name: User's first name
            last_name: User's last name

        Returns:
            Tuple of (User instance, JWT token)
        """
        # Check if user already exists
        try:
            existing_user = await self.user_service.get_by_telegram_id(telegram_id)
            # User exists, return with new token
            token = self.create_jwt_token(existing_user)
            return existing_user, token
        except UserNotFoundError:
            pass

        # Create new user with DEMO role
        user = await self.user_service.create(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            role="DEMO",
        )

        await self.db.commit()

        # Create JWT token
        token = self.create_jwt_token(user)

        return user, token

    async def unlock(self, telegram_id: int, unlock_code: str) -> User:
        """
        Unlock DEMO access for a user using unlock code.

        Args:
            telegram_id: Telegram user ID
            unlock_code: DEMO unlock code

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
            AuthenticationError: If unlock code is invalid
        """
        # Verify unlock code
        if unlock_code != settings.demo_unlock_code:
            raise AuthenticationError("Nieprawidłowy kod odblokowania")

        # Get user
        user = await self.user_service.get_by_telegram_id(telegram_id)

        # Authorize user
        user.authorized = True
        await self.db.flush()
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def bootstrap(self, telegram_id: int, bootstrap_code: str) -> User:
        """
        Bootstrap admin user using admin code.

        Args:
            telegram_id: Telegram user ID
            bootstrap_code: Admin bootstrap code

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
            AuthenticationError: If bootstrap code is invalid
        """
        # Verify bootstrap code
        if bootstrap_code != settings.bootstrap_admin_code:
            raise AuthenticationError("Nieprawidłowy kod administratora")

        # Get user
        user = await self.user_service.get_by_telegram_id(telegram_id)

        # Upgrade to ADMIN role
        user.role = "ADMIN"
        user.authorized = True
        user.verified = True
        await self.db.flush()
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def consume_invite(self, telegram_id: int, invite_code: str) -> User:
        """
        Consume an invite code to upgrade user role.

        Args:
            telegram_id: Telegram user ID
            invite_code: Plain text invite code

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user not found
            InvalidInviteCodeError: If invite code is invalid
        """
        # Get user
        user = await self.user_service.get_by_telegram_id(telegram_id)

        # Validate and consume invite code
        code_hash = hash_invite_code(invite_code)
        invite = await self.invite_service.validate(code_hash)

        # Update user role
        user.role = invite.role
        user.authorized = True
        user.verified = True

        # Consume invite code
        await self.invite_service.consume(code_hash, telegram_id)

        await self.db.commit()
        await self.db.refresh(user)

        return user

    def create_jwt_token(self, user: User) -> str:
        """
        Create JWT access token for user.

        Args:
            user: User instance

        Returns:
            JWT token string
        """
        payload = {
            "telegram_id": user.telegram_id,
            "role": user.role,
            "username": user.username,
        }

        token = create_access_token(payload)
        return token

    async def get_current_user(self, telegram_id: int) -> User:
        """
        Get current user by Telegram ID.

        Args:
            telegram_id: Telegram user ID

        Returns:
            User instance

        Raises:
            UserNotFoundError: If user not found
        """
        return await self.user_service.get_by_telegram_id(telegram_id)
