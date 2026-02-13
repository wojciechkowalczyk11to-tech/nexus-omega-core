"""
Invite code service for managing user invitations.
"""

import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import InvalidInviteCodeError
from app.core.security import hash_invite_code
from app.db.models.invite_code import InviteCode


class InviteService:
    """Service for managing invite codes."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_code(
        self,
        created_by: int,
        role: str = "DEMO",
        uses_left: int = 1,
        expires_at: datetime | None = None,
    ) -> tuple[str, InviteCode]:
        """
        Create a new invite code.

        Args:
            created_by: Telegram ID of creator
            role: Role to grant (DEMO, FULL_ACCESS)
            uses_left: Number of uses allowed
            expires_at: Optional expiration datetime

        Returns:
            Tuple of (plain code, InviteCode instance)
        """
        # Generate random code
        plain_code = secrets.token_urlsafe(16)
        code_hash = hash_invite_code(plain_code)

        # Create invite code record
        invite = InviteCode(
            code_hash=code_hash,
            role=role,
            expires_at=expires_at,
            uses_left=uses_left,
            created_by=created_by,
        )

        self.db.add(invite)
        await self.db.flush()
        await self.db.refresh(invite)

        return plain_code, invite

    async def validate(self, code_hash: str) -> InviteCode:
        """
        Validate an invite code.

        Args:
            code_hash: SHA-256 hash of invite code

        Returns:
            InviteCode instance

        Raises:
            InvalidInviteCodeError: If code is invalid, expired, or exhausted
        """
        result = await self.db.execute(select(InviteCode).where(InviteCode.code_hash == code_hash))
        invite = result.scalar_one_or_none()

        if not invite:
            raise InvalidInviteCodeError("Kod zaproszenia nie istnieje")

        # Check expiration
        if invite.expires_at and invite.expires_at < datetime.now(UTC):
            raise InvalidInviteCodeError("Kod zaproszenia wygasł")

        # Check uses
        if invite.uses_left <= 0:
            raise InvalidInviteCodeError("Kod zaproszenia został wykorzystany")

        return invite

    async def consume(self, code_hash: str, consumed_by: int) -> InviteCode:
        """
        Consume an invite code.

        Args:
            code_hash: SHA-256 hash of invite code
            consumed_by: Telegram ID of consumer

        Returns:
            Updated InviteCode instance

        Raises:
            InvalidInviteCodeError: If code is invalid
        """
        invite = await self.validate(code_hash)

        # Decrement uses
        invite.uses_left -= 1

        # Set consumption details if first use
        if invite.consumed_by is None:
            invite.consumed_by = consumed_by
            invite.consumed_at = datetime.now(UTC)

        await self.db.flush()
        await self.db.refresh(invite)

        return invite

    async def list_active(self, created_by: int | None = None) -> list[InviteCode]:
        """
        List active invite codes.

        Args:
            created_by: Optional filter by creator

        Returns:
            List of active InviteCode instances
        """
        query = select(InviteCode).where(InviteCode.uses_left > 0)

        if created_by:
            query = query.where(InviteCode.created_by == created_by)

        result = await self.db.execute(query.order_by(InviteCode.created_at.desc()))
        return list(result.scalars().all())

    async def revoke(self, code_hash: str) -> InviteCode:
        """
        Revoke an invite code by setting uses_left to 0.

        Args:
            code_hash: SHA-256 hash of invite code

        Returns:
            Updated InviteCode instance

        Raises:
            InvalidInviteCodeError: If code not found
        """
        result = await self.db.execute(select(InviteCode).where(InviteCode.code_hash == code_hash))
        invite = result.scalar_one_or_none()

        if not invite:
            raise InvalidInviteCodeError("Kod zaproszenia nie istnieje")

        invite.uses_left = 0
        await self.db.flush()
        await self.db.refresh(invite)

        return invite
