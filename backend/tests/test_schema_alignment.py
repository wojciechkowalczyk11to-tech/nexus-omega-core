"""
Schema alignment tests.

Verifies that SQLAlchemy models and Alembic migrations stay consistent.
Uses in-memory SQLite (matching existing test infra) to assert that all
expected tables and columns exist after ``Base.metadata.create_all``.

Note: conftest.py already registers JSONB→JSON and Vector→TEXT compilers
for the SQLite dialect, so we rely on those registrations here.
"""

import pytest
import pytest_asyncio
from app.db.base import Base
from app.db.models import (  # noqa: F401 — ensure all models are registered
    AgentTrace,
    AuditLog,
    ChatSession,
    InviteCode,
    Message,
    Payment,
    RagChunk,
    RagItem,
    ToolCounter,
    UsageLedger,
    User,
    UserMemory,
)
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Expected schema specification
# ---------------------------------------------------------------------------

EXPECTED_TABLES: dict[str, list[str]] = {
    "users": [
        "id",
        "created_at",
        "updated_at",
        "telegram_id",
        "username",
        "first_name",
        "last_name",
        "role",
        "authorized",
        "verified",
        "subscription_tier",
        "subscription_expires_at",
        "credits_balance",
        "default_mode",
        "cost_preference",
        "settings",
    ],
    "chat_sessions": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "name",
        "mode",
        "provider_pref",
        "active",
        "snapshot_text",
        "snapshot_at",
        "message_count",
    ],
    "messages": [
        "id",
        "created_at",
        "updated_at",
        "session_id",
        "user_id",
        "role",
        "content",
        "content_type",
        "metadata",
    ],
    "usage_ledger": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "session_id",
        "provider",
        "model",
        "profile",
        "difficulty",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "tool_costs",
        "latency_ms",
        "fallback_used",
    ],
    "tool_counters": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "date",
        "grok_calls",
        "web_calls",
        "smart_credits_used",
        "vertex_queries",
        "deepseek_calls",
        "total_cost_usd",
    ],
    "audit_logs": [
        "id",
        "created_at",
        "updated_at",
        "actor_telegram_id",
        "action",
        "target",
        "details",
        "ip_address",
    ],
    "invite_codes": [
        "id",
        "created_at",
        "updated_at",
        "code_hash",
        "role",
        "expires_at",
        "uses_left",
        "created_by",
        "consumed_by",
        "consumed_at",
    ],
    "rag_items": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "scope",
        "source_type",
        "source_url",
        "filename",
        "stored_path",
        "chunk_count",
        "status",
        "metadata",
    ],
    "rag_chunks": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "rag_item_id",
        "content",
        "chunk_index",
        "embedding",
        "metadata",
    ],
    "user_memories": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "key",
        "value",
    ],
    "payments": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "telegram_payment_charge_id",
        "product_id",
        "plan",
        "amount_stars",
        "stars_amount",
        "credits_granted",
        "currency",
        "provider_payment_charge_id",
        "status",
        "expires_at",
    ],
    "agent_traces": [
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "message_id",
        "iteration",
        "action",
        "thought",
        "tool_name",
        "tool_args",
        "tool_result",
        "correction_reason",
        "timestamp_ms",
    ],
}

# Runtime-critical tables that MUST exist
RUNTIME_CRITICAL_TABLES = [
    "users",
    "payments",
    "rag_items",
    "rag_chunks",
    "agent_traces",
    "chat_sessions",
    "messages",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def schema_engine():
    """Create a clean in-memory DB engine with all tables."""
    engine = create_async_engine(
        TEST_DB_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_expected_tables_exist(schema_engine):
    """Every expected table must be created by Base.metadata.create_all."""
    async with schema_engine.connect() as conn:
        table_names = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())

    for table in EXPECTED_TABLES:
        assert table in table_names, f"Missing table: {table}"


@pytest.mark.asyncio
async def test_runtime_critical_tables_exist(schema_engine):
    """Runtime-critical tables must be present."""
    async with schema_engine.connect() as conn:
        table_names = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())

    for table in RUNTIME_CRITICAL_TABLES:
        assert table in table_names, f"Runtime-critical table missing: {table}"


@pytest.mark.asyncio
async def test_expected_columns_exist(schema_engine):
    """All expected columns must exist for every table."""
    async with schema_engine.connect() as conn:
        for table_name, expected_cols in EXPECTED_TABLES.items():
            columns = await conn.run_sync(
                lambda sync_conn, t=table_name: [
                    c["name"] for c in inspect(sync_conn).get_columns(t)
                ]
            )
            for col in expected_cols:
                assert col in columns, (
                    f"Column '{col}' missing from table '{table_name}'. Existing columns: {columns}"
                )


@pytest.mark.asyncio
async def test_users_credits_balance_column(schema_engine):
    """users.credits_balance must exist (used by PaymentService)."""
    async with schema_engine.connect() as conn:
        columns = await conn.run_sync(
            lambda sync_conn: {c["name"]: c for c in inspect(sync_conn).get_columns("users")}
        )
    assert "credits_balance" in columns, "users.credits_balance is required by PaymentService"
    assert columns["credits_balance"]["nullable"] is False


@pytest.mark.asyncio
async def test_users_cost_preference_column(schema_engine):
    """users.cost_preference must exist (used by SLM router)."""
    async with schema_engine.connect() as conn:
        columns = await conn.run_sync(
            lambda sync_conn: {c["name"]: c for c in inspect(sync_conn).get_columns("users")}
        )
    assert "cost_preference" in columns, "users.cost_preference is required by SLM router"
    assert columns["cost_preference"]["nullable"] is False


@pytest.mark.asyncio
async def test_payments_product_id_column(schema_engine):
    """payments.product_id must exist (used by PaymentService)."""
    async with schema_engine.connect() as conn:
        columns = await conn.run_sync(
            lambda sync_conn: {c["name"]: c for c in inspect(sync_conn).get_columns("payments")}
        )
    assert "product_id" in columns
    assert "amount_stars" in columns
    assert "credits_granted" in columns
    assert "provider_payment_charge_id" in columns


@pytest.mark.asyncio
async def test_model_metadata_table_count():
    """SQLAlchemy metadata must know about all expected tables."""
    model_tables = set(Base.metadata.tables.keys())
    expected = set(EXPECTED_TABLES.keys())
    missing = expected - model_tables
    assert not missing, f"Models missing for tables: {missing}"


@pytest.mark.asyncio
async def test_no_extra_model_tables_without_spec():
    """Every table defined in models should be listed in EXPECTED_TABLES."""
    model_tables = set(Base.metadata.tables.keys())
    expected = set(EXPECTED_TABLES.keys())
    extra = model_tables - expected
    assert not extra, (
        f"Tables {extra} exist in models but not in EXPECTED_TABLES spec. "
        "Add them to the test to prevent future drift."
    )
