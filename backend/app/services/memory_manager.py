"""
Memory manager for sessions, snapshots, and absolute user memory.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import SessionNotFoundError
from app.db.models.message import Message
from app.db.models.session import ChatSession
from app.db.models.user_memory import UserMemory


class MemoryManager:
    """Manager for conversation memory and sessions."""

    # Snapshot creation threshold
    SNAPSHOT_MESSAGE_THRESHOLD = 20

    # Context window size (messages to retrieve)
    CONTEXT_WINDOW_SIZE = 10

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_create_session(
        self,
        user_id: int,
        session_name: str = "Default Session",
        mode: str = "eco",
    ) -> ChatSession:
        """
        Get active session or create new one.

        Args:
            user_id: User's Telegram ID
            session_name: Session name
            mode: AI mode (eco, smart, deep)

        Returns:
            ChatSession instance
        """
        # Try to get active session
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.active == True)
            .order_by(ChatSession.updated_at.desc())
        )
        session = result.scalar_one_or_none()

        if session:
            return session

        # Create new session
        session = ChatSession(
            user_id=user_id,
            name=session_name,
            mode=mode,
            active=True,
            message_count=0,
        )

        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)

        return session

    async def get_session_by_id(self, session_id: int) -> ChatSession:
        """
        Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            ChatSession instance

        Raises:
            SessionNotFoundError: If session not found
        """
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(session_id)

        return session

    async def get_context_messages(
        self, session_id: int, limit: int | None = None
    ) -> list[dict[str, str]]:
        """
        Get context messages for a session.

        Returns snapshot + last N messages.

        Args:
            session_id: Session ID
            limit: Optional message limit (default: CONTEXT_WINDOW_SIZE)

        Returns:
            List of message dicts with role and content
        """
        session = await self.get_session_by_id(session_id)
        messages = []

        # Add snapshot if exists
        if session.snapshot_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"[Podsumowanie poprzedniej konwersacji]\n{session.snapshot_text}",
                }
            )

        # Get recent messages
        message_limit = limit or self.CONTEXT_WINDOW_SIZE
        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(message_limit)
        )
        recent_messages = list(result.scalars().all())

        # Reverse to chronological order
        recent_messages.reverse()

        # Add to context
        for msg in recent_messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def persist_message(
        self,
        session_id: int,
        user_id: int,
        role: str,
        content: str,
        content_type: str = "text",
        metadata: dict | None = None,
    ) -> Message:
        """
        Persist a message to the database.

        Args:
            session_id: Session ID
            user_id: User's Telegram ID
            role: Message role (user, assistant, system)
            content: Message content
            content_type: Content type (text, image, etc.)
            metadata: Optional metadata dict

        Returns:
            Created Message instance
        """
        message = Message(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            content_type=content_type,
            msg_metadata=metadata or {},
        )

        self.db.add(message)
        await self.db.flush()

        # Increment session message count
        session = await self.get_session_by_id(session_id)
        session.message_count += 1
        await self.db.flush()

        await self.db.refresh(message)
        return message

    async def maybe_create_snapshot(self, session_id: int) -> bool:
        """
        Create snapshot if message threshold reached.

        Compresses conversation history using LLM (placeholder for now).

        Args:
            session_id: Session ID

        Returns:
            True if snapshot created, False otherwise
        """
        session = await self.get_session_by_id(session_id)

        # Check if threshold reached
        if session.message_count < self.SNAPSHOT_MESSAGE_THRESHOLD:
            return False

        # Get all messages since last snapshot
        query = select(Message).where(Message.session_id == session_id)

        if session.snapshot_at:
            query = query.where(Message.created_at > session.snapshot_at)

        result = await self.db.execute(query.order_by(Message.created_at))
        messages = list(result.scalars().all())

        if not messages:
            return False

        # Create snapshot text (simple concatenation for now)
        # TODO: In Phase 2, use LLM to compress this
        snapshot_parts = []
        for msg in messages:
            snapshot_parts.append(f"{msg.role}: {msg.content[:200]}")

        snapshot_text = "\n".join(snapshot_parts)

        # Update session snapshot
        session.snapshot_text = snapshot_text
        session.snapshot_at = datetime.now(timezone.utc)
        await self.db.flush()

        return True

    async def get_absolute_memory(self, user_id: int, key: str) -> str | None:
        """
        Get absolute memory value by key.

        Args:
            user_id: User's Telegram ID
            key: Memory key

        Returns:
            Memory value or None if not found
        """
        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id, UserMemory.key == key
            )
        )
        memory = result.scalar_one_or_none()

        return memory.value if memory else None

    async def set_absolute_memory(self, user_id: int, key: str, value: str) -> UserMemory:
        """
        Set absolute memory key-value pair.

        Args:
            user_id: User's Telegram ID
            key: Memory key
            value: Memory value

        Returns:
            UserMemory instance
        """
        # Try to get existing memory
        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id, UserMemory.key == key
            )
        )
        memory = result.scalar_one_or_none()

        if memory:
            # Update existing
            memory.value = value
        else:
            # Create new
            memory = UserMemory(user_id=user_id, key=key, value=value)
            self.db.add(memory)

        await self.db.flush()
        await self.db.refresh(memory)

        return memory

    async def delete_absolute_memory(self, user_id: int, key: str) -> bool:
        """
        Delete absolute memory by key.

        Args:
            user_id: User's Telegram ID
            key: Memory key

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.user_id == user_id, UserMemory.key == key
            )
        )
        memory = result.scalar_one_or_none()

        if not memory:
            return False

        await self.db.delete(memory)
        await self.db.flush()

        return True

    async def list_absolute_memories(self, user_id: int) -> list[UserMemory]:
        """
        List all absolute memories for a user.

        Args:
            user_id: User's Telegram ID

        Returns:
            List of UserMemory instances
        """
        result = await self.db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.key)
        )
        return list(result.scalars().all())

    async def list_sessions(
        self, user_id: int, active_only: bool = False
    ) -> list[ChatSession]:
        """
        List sessions for a user.

        Args:
            user_id: User's Telegram ID
            active_only: Only return active sessions

        Returns:
            List of ChatSession instances
        """
        query = select(ChatSession).where(ChatSession.user_id == user_id)

        if active_only:
            query = query.where(ChatSession.active == True)

        result = await self.db.execute(query.order_by(ChatSession.updated_at.desc()))
        return list(result.scalars().all())

    async def delete_session(self, session_id: int) -> bool:
        """
        Delete a session and all its messages.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        try:
            session = await self.get_session_by_id(session_id)
            await self.db.delete(session)
            await self.db.flush()
            return True
        except SessionNotFoundError:
            return False
