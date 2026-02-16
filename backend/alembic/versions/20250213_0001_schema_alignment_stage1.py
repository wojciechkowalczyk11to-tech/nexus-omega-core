"""Schema alignment stage 1 – add missing tables and columns.

Adds:
  - users.cost_preference (String(50), NOT NULL, default 'balanced')
  - users.credits_balance (Integer, NOT NULL, default 0)
  - payments.product_id (String(100), NOT NULL, default '')
  - payments.amount_stars (Integer, NOT NULL, default 0)
  - payments.credits_granted (Integer, NOT NULL, default 0)
  - payments.provider_payment_charge_id (String(255), nullable)
  - rag_chunks table (with pgvector embedding column)
  - agent_traces table

Revision ID: 002_schema_align
Revises: 001_initial
Create Date: 2025-02-13 00:01:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_schema_align"
down_revision: str = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- pgvector extension (idempotent) ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- users: add missing columns ---
    op.add_column(
        "users",
        sa.Column(
            "cost_preference",
            sa.String(length=50),
            nullable=False,
            server_default="balanced",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "credits_balance",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    # --- payments: add missing columns ---
    op.add_column(
        "payments",
        sa.Column(
            "product_id",
            sa.String(length=100),
            nullable=False,
            server_default="",
        ),
    )
    op.create_index(op.f("ix_product_id"), "payments", ["product_id"], unique=False)

    op.add_column(
        "payments",
        sa.Column(
            "amount_stars",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "payments",
        sa.Column(
            "credits_granted",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "payments",
        sa.Column(
            "provider_payment_charge_id",
            sa.String(length=255),
            nullable=True,
        ),
    )

    # --- rag_chunks table ---
    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("rag_item_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column(
            "embedding",
            postgresql.ARRAY(sa.Float(), dimensions=1),
            nullable=False,
            comment="pgvector 384-dim; raw SQL uses ::vector",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.telegram_id"],
            name=op.f("fk_rag_chunks_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rag_item_id"],
            ["rag_items.id"],
            name=op.f("fk_rag_chunks_rag_item_id_rag_items"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_chunks")),
    )
    op.create_index(op.f("ix_rag_chunks_user_id"), "rag_chunks", ["user_id"], unique=False)
    op.create_index(op.f("ix_rag_chunks_rag_item_id"), "rag_chunks", ["rag_item_id"], unique=False)

    # Convert embedding column to proper pgvector type
    op.execute(
        "ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)"
    )

    # --- agent_traces table ---
    op.create_table(
        "agent_traces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("thought", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.String(length=100), nullable=True),
        sa.Column("tool_args", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tool_result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("correction_reason", sa.Text(), nullable=True),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.telegram_id"],
            name=op.f("fk_agent_traces_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_agent_traces_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_traces")),
    )
    op.create_index(op.f("ix_agent_traces_user_id"), "agent_traces", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_agent_traces_message_id"), "agent_traces", ["message_id"], unique=False
    )


def downgrade() -> None:
    op.drop_table("agent_traces")
    op.drop_table("rag_chunks")

    op.drop_column("payments", "provider_payment_charge_id")
    op.drop_column("payments", "credits_granted")
    op.drop_column("payments", "amount_stars")
    op.drop_index(op.f("ix_product_id"), table_name="payments")
    op.drop_column("payments", "product_id")

    op.drop_column("users", "credits_balance")
    op.drop_column("users", "cost_preference")
