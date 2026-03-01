"""
Backend API client for Telegram bot.
"""

from __future__ import annotations

from typing import Any

import httpx

from telegram_bot.config import settings

_instance: BackendClient | None = None


def get_backend_client() -> BackendClient:
    """Get or create singleton BackendClient instance with connection pooling."""
    global _instance
    if _instance is None:
        _instance = BackendClient()
    return _instance


class BackendClient:
    """Client for communicating with backend API."""

    def __init__(self, base_url: str | None = None) -> None:
        """
        Initialize backend client.

        Args:
            base_url: Backend API base URL
        """
        self.base_url = base_url or settings.backend_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def register_user(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Register or get existing user.

        Args:
            telegram_id: Telegram user ID
            username: Telegram username
            first_name: First name
            last_name: Last name

        Returns:
            User data with JWT token
        """
        response = await self.client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": telegram_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
            },
        )
        response.raise_for_status()
        return response.json()

    async def unlock_demo(self, telegram_id: int, unlock_code: str) -> dict[str, Any]:
        """
        Unlock DEMO access.

        Args:
            telegram_id: Telegram user ID
            unlock_code: Unlock code

        Returns:
            User data
        """
        response = await self.client.post(
            "/api/v1/auth/unlock",
            json={"telegram_id": telegram_id, "unlock_code": unlock_code},
        )
        response.raise_for_status()
        return response.json()

    async def get_user(self, token: str) -> dict[str, Any]:
        """
        Get current user.

        Args:
            token: JWT token

        Returns:
            User data
        """
        response = await self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        token: str,
        query: str,
        session_id: int | None = None,
        mode: str | None = None,
        deep_confirmed: bool = False,
    ) -> dict[str, Any]:
        """
        Send chat message.

        Args:
            token: JWT token
            query: User query
            session_id: Optional session ID
            mode: Optional mode override
            deep_confirmed: DEEP mode confirmation

        Returns:
            Chat response
        """
        response = await self.client.post(
            "/api/v1/chat/chat",
            json={
                "query": query,
                "session_id": session_id,
                "mode": mode,
                "deep_confirmed": deep_confirmed,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    async def upload_rag_document(
        self, token: str, filename: str, content: bytes
    ) -> dict[str, Any]:
        """
        Upload RAG document.

        Args:
            token: JWT token
            filename: File name
            content: File content

        Returns:
            Upload response
        """
        files = {"file": (filename, content)}
        response = await self.client.post(
            "/api/v1/rag/upload",
            files=files,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    async def list_rag_documents(self, token: str) -> dict[str, Any]:
        """
        List RAG documents.

        Args:
            token: JWT token

        Returns:
            List of documents
        """
        response = await self.client.get(
            "/api/v1/rag/list",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    async def delete_rag_document(self, token: str, item_id: int) -> dict[str, Any]:
        """
        Delete RAG document.

        Args:
            token: JWT token
            item_id: Document ID

        Returns:
            Delete response
        """
        response = await self.client.delete(
            f"/api/v1/rag/{item_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()
