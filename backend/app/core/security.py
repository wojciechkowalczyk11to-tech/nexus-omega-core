"""
Security utilities for JWT token creation/verification and password hashing.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.app.core.config import settings
from backend.app.core.exceptions import AuthenticationError

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return encoded_jwt


def verify_access_token(token: str) -> dict[str, Any]:
    """
    Verify and decode a JWT access token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Nieprawidłowy token: {str(e)}") from e


def hash_invite_code(code: str) -> str:
    """
    Hash an invite code using SHA-256.

    Args:
        code: Plain text invite code

    Returns:
        SHA-256 hash of the code
    """
    return hashlib.sha256(code.encode()).hexdigest()


def verify_invite_code(plain_code: str, hashed_code: str) -> bool:
    """
    Verify an invite code against its hash.

    Args:
        plain_code: Plain text invite code
        hashed_code: SHA-256 hash to verify against

    Returns:
        True if code matches hash, False otherwise
    """
    return hash_invite_code(plain_code) == hashed_code


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracing.

    Returns:
        Unique request ID string
    """
    import uuid

    return str(uuid.uuid4())
