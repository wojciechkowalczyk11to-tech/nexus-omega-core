"""
Unit tests for authentication service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.exceptions import AuthenticationError, UserNotFoundError
from app.db.models.user import User
from app.services.auth_service import AuthService


@pytest.mark.asyncio
async def test_register_new_user():
    """Test registering a new user."""
    db_mock = AsyncMock()
    user_service_mock = AsyncMock()
    
    # Mock user_service.get_by_telegram_id to raise UserNotFoundError
    user_service_mock.get_by_telegram_id.side_effect = UserNotFoundError(123456)
    
    # Mock user_service.create to return new user
    new_user = User(
        id=1,
        telegram_id=123456,
        username="testuser",
        role="DEMO",
        authorized=False,
    )
    user_service_mock.create.return_value = new_user
    
    auth_service = AuthService(db_mock)
    auth_service.user_service = user_service_mock
    
    user, token = await auth_service.register(
        telegram_id=123456,
        username="testuser",
    )
    
    assert user.telegram_id == 123456
    assert user.role == "DEMO"
    assert token is not None
    assert len(token) > 0


@pytest.mark.asyncio
async def test_register_existing_user():
    """Test registering an existing user returns user and new token."""
    db_mock = AsyncMock()
    user_service_mock = AsyncMock()
    
    # Mock user_service.get_by_telegram_id to return existing user
    existing_user = User(
        id=1,
        telegram_id=123456,
        username="testuser",
        role="DEMO",
        authorized=True,
    )
    user_service_mock.get_by_telegram_id.return_value = existing_user
    
    auth_service = AuthService(db_mock)
    auth_service.user_service = user_service_mock
    
    user, token = await auth_service.register(
        telegram_id=123456,
        username="testuser",
    )
    
    assert user.telegram_id == 123456
    assert token is not None


@pytest.mark.asyncio
async def test_unlock_with_valid_code():
    """Test unlocking DEMO access with valid code."""
    db_mock = AsyncMock()
    user_service_mock = AsyncMock()
    
    user = User(
        id=1,
        telegram_id=123456,
        role="DEMO",
        authorized=False,
    )
    user_service_mock.get_by_telegram_id.return_value = user
    
    auth_service = AuthService(db_mock)
    auth_service.user_service = user_service_mock
    
    # Mock settings
    from unittest.mock import patch
    with patch("app.services.auth_service.settings") as settings_mock:
        settings_mock.demo_unlock_code = "VALID_CODE"
        
        unlocked_user = await auth_service.unlock(123456, "VALID_CODE")
        
        assert unlocked_user.authorized is True


@pytest.mark.asyncio
async def test_unlock_with_invalid_code():
    """Test unlocking with invalid code raises error."""
    db_mock = AsyncMock()
    user_service_mock = AsyncMock()
    
    user = User(
        id=1,
        telegram_id=123456,
        role="DEMO",
        authorized=False,
    )
    user_service_mock.get_by_telegram_id.return_value = user
    
    auth_service = AuthService(db_mock)
    auth_service.user_service = user_service_mock
    
    from unittest.mock import patch
    with patch("app.services.auth_service.settings") as settings_mock:
        settings_mock.demo_unlock_code = "VALID_CODE"
        
        with pytest.raises(AuthenticationError):
            await auth_service.unlock(123456, "WRONG_CODE")


@pytest.mark.asyncio
async def test_bootstrap_with_valid_code():
    """Test bootstrapping admin with valid code."""
    db_mock = AsyncMock()
    user_service_mock = AsyncMock()
    
    user = User(
        id=1,
        telegram_id=123456,
        role="DEMO",
        authorized=False,
        verified=False,
    )
    user_service_mock.get_by_telegram_id.return_value = user
    
    auth_service = AuthService(db_mock)
    auth_service.user_service = user_service_mock
    
    from unittest.mock import patch
    with patch("app.services.auth_service.settings") as settings_mock:
        settings_mock.bootstrap_admin_code = "ADMIN_CODE"
        
        admin_user = await auth_service.bootstrap(123456, "ADMIN_CODE")
        
        assert admin_user.role == "ADMIN"
        assert admin_user.authorized is True
        assert admin_user.verified is True


@pytest.mark.asyncio
async def test_create_jwt_token():
    """Test JWT token creation."""
    db_mock = AsyncMock()
    
    user = User(
        id=1,
        telegram_id=123456,
        username="testuser",
        role="DEMO",
    )
    
    auth_service = AuthService(db_mock)
    token = auth_service.create_jwt_token(user)
    
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Verify token can be decoded
    from app.core.security import verify_access_token
    payload = verify_access_token(token)
    
    assert payload["telegram_id"] == 123456
    assert payload["role"] == "DEMO"
    assert payload["username"] == "testuser"
