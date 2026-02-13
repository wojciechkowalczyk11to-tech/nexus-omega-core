"""
Integration tests for auth API endpoints.
"""

import pytest
from app.main import app
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_new_user():
    """Test registering a new user."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 999999,
                "username": "testuser",
                "first_name": "Test",
                "last_name": "User",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["telegram_id"] == 999999
        assert data["role"] == "DEMO"
        assert data["authorized"] is False
        assert "token" in data


@pytest.mark.asyncio
async def test_register_existing_user():
    """Test registering an existing user returns existing data."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register first time
        response1 = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 888888,
                "username": "existing",
            },
        )
        assert response1.status_code == 200
        token1 = response1.json()["token"]

        # Register again
        response2 = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 888888,
                "username": "existing",
            },
        )
        assert response2.status_code == 200
        token2 = response2.json()["token"]

        # Tokens should be different (new JWT)
        assert token1 != token2


@pytest.mark.asyncio
async def test_get_current_user():
    """Test getting current user with JWT."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register user
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "telegram_id": 777777,
                "username": "jwttest",
            },
        )
        token = register_response.json()["token"]

        # Get current user
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["telegram_id"] == 777777
        assert data["username"] == "jwttest"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Test getting current user with invalid token."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == 401
