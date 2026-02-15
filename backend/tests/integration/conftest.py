"""
Integration test fixtures.
"""

import os
from unittest.mock import AsyncMock

import pytest_asyncio
from app.api.deps import get_db, get_redis_pool
from app.db.base import Base
from app.db.models import *  # noqa: F401,F403 — register all models with metadata
from app.main import app
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Render PostgreSQL-specific types as SQLite-compatible for testing
try:
    from pgvector.sqlalchemy import Vector

    @compiles(Vector, "sqlite")
    def _compile_vector_sqlite(type_, compiler, **kw):
        return "TEXT"

except ImportError:
    pass


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(JSON(), **kw)


# Integration test database (file-based SQLite for multi-connection support)
_INTEGRATION_DB_URL = "sqlite+aiosqlite:////tmp/integration_test.db"

_engine = create_async_engine(
    _INTEGRATION_DB_URL,
    poolclass=NullPool,
    echo=False,
)
_TestSessionLocal = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _setup_integration_db():
    """Create tables, override deps, tear down after each test."""
    # Clean up any previous test database
    db_path = "/tmp/integration_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Override get_db
    async def _override_get_db():
        async with _TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Override Redis with a mock
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=True)

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_redis_pool] = lambda: mock_redis

    yield

    app.dependency_overrides.clear()

    # Drop tables and clean up
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    if os.path.exists(db_path):
        os.remove(db_path)
