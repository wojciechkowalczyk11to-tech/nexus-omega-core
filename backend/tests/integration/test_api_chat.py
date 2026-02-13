"""
Integration tests for chat API endpoints.
"""

import pytest
from app.main import app
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_unauthorized():
    """Test chat without authorization."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/chat",
            json={"query": "Hello"},
        )

        assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_with_valid_token():
    """Test chat with valid JWT token."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register user
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 666666,
                "username": "chattest",
            },
        )
        token = register_response.json()["token"]

        # Unlock DEMO access (assuming DEMO_UNLOCK_CODE is set)
        # For test, we'll skip this and expect 403 or provider error

        # Send chat message
        response = await client.post(
            "/api/v1/chat/chat",
            json={"query": "Hello, test query"},
            headers={"Authorization": f"Bearer {token}"},
        )

        # May fail due to missing provider API keys in test env
        # Just check that it's not 401 (unauthorized)
        assert response.status_code != 401


@pytest.mark.asyncio
async def test_get_providers():
    """Test getting available providers."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register user
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 555555,
                "username": "providertest",
            },
        )
        token = register_response.json()["token"]

        # Get providers
        response = await client.get(
            "/api/v1/chat/providers",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
