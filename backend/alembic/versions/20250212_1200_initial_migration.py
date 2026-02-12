"""Initial migration with 11 tables

Revision ID: 001_initial
Revises: 
Create Date: 2025-02-12 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('first_name', sa.String(length=255), nullable=True),
        sa.Column('last_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('authorized', sa.Boolean(), nullable=False),
        sa.Column('verified', sa.Boolean(), nullable=False),
        sa.Column('subscription_tier', sa.String(length=50), nullable=True),
        sa.Column('subscription_expires_at', sa.DateTime(), nullable=True),
        sa.Column('default_mode', sa.String(length=50), nullable=False),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('telegram_id', name=op.f('uq_users_telegram_id'))
    )
    op.create_index(op.f('ix_telegram_id'), 'users', ['telegram_id'], unique=False)
    op.create_index(op.f('ix_role'), 'users', ['role'], unique=False)

    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('mode', sa.String(length=50), nullable=False),
        sa.Column('provider_pref', sa.String(length=50), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False),
        sa.Column('snapshot_text', sa.Text(), nullable=True),
        sa.Column('snapshot_at', sa.DateTime(), nullable=True),
        sa.Column('message_count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_chat_sessions_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_sessions'))
    )
    op.create_index(op.f('ix_user_id'), 'chat_sessions', ['user_id'], unique=False)

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_type', sa.String(length=50), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name=op.f('fk_messages_session_id_chat_sessions'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_messages_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_messages'))
    )
    op.create_index(op.f('ix_session_id'), 'messages', ['session_id'], unique=False)
    op.create_index(op.f('ix_messages_user_id'), 'messages', ['user_id'], unique=False)

    # Create usage_ledger table
    op.create_table(
        'usage_ledger',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('profile', sa.String(length=50), nullable=False),
        sa.Column('difficulty', sa.String(length=50), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('cost_usd', sa.Float(), nullable=False),
        sa.Column('tool_costs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('fallback_used', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_usage_ledger_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_usage_ledger'))
    )
    op.create_index(op.f('ix_ledger_user_id'), 'usage_ledger', ['user_id'], unique=False)
    op.create_index(op.f('ix_provider'), 'usage_ledger', ['provider'], unique=False)
    op.create_index(op.f('ix_ledger_session_id'), 'usage_ledger', ['session_id'], unique=False)

    # Create tool_counters table
    op.create_table(
        'tool_counters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('grok_calls', sa.Integer(), nullable=False),
        sa.Column('web_calls', sa.Integer(), nullable=False),
        sa.Column('smart_credits_used', sa.Integer(), nullable=False),
        sa.Column('vertex_queries', sa.Integer(), nullable=False),
        sa.Column('deepseek_calls', sa.Integer(), nullable=False),
        sa.Column('total_cost_usd', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_tool_counters_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_tool_counters')),
        sa.UniqueConstraint('user_id', 'date', name=op.f('uq_tool_counter_user_date'))
    )
    op.create_index(op.f('ix_counter_user_id'), 'tool_counters', ['user_id'], unique=False)
    op.create_index(op.f('ix_date'), 'tool_counters', ['date'], unique=False)

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('actor_telegram_id', sa.BigInteger(), nullable=False),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('target', sa.String(length=255), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.ForeignKeyConstraint(['actor_telegram_id'], ['users.telegram_id'], name=op.f('fk_audit_logs_actor_telegram_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_logs'))
    )
    op.create_index(op.f('ix_actor_telegram_id'), 'audit_logs', ['actor_telegram_id'], unique=False)
    op.create_index(op.f('ix_action'), 'audit_logs', ['action'], unique=False)

    # Create invite_codes table
    op.create_table(
        'invite_codes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('code_hash', sa.String(length=64), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('uses_left', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.BigInteger(), nullable=False),
        sa.Column('consumed_by', sa.BigInteger(), nullable=True),
        sa.Column('consumed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['consumed_by'], ['users.telegram_id'], name=op.f('fk_invite_codes_consumed_by_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.telegram_id'], name=op.f('fk_invite_codes_created_by_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_invite_codes')),
        sa.UniqueConstraint('code_hash', name=op.f('uq_invite_codes_code_hash'))
    )
    op.create_index(op.f('ix_code_hash'), 'invite_codes', ['code_hash'], unique=False)
    op.create_index(op.f('ix_created_by'), 'invite_codes', ['created_by'], unique=False)
    op.create_index(op.f('ix_consumed_by'), 'invite_codes', ['consumed_by'], unique=False)

    # Create rag_items table
    op.create_table(
        'rag_items',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('scope', sa.String(length=50), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('stored_path', sa.String(length=512), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_rag_items_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_rag_items'))
    )
    op.create_index(op.f('ix_rag_user_id'), 'rag_items', ['user_id'], unique=False)
    op.create_index(op.f('ix_status'), 'rag_items', ['status'], unique=False)

    # Create user_memories table
    op.create_table(
        'user_memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_user_memories_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_user_memories')),
        sa.UniqueConstraint('user_id', 'key', name=op.f('uq_user_memory_user_key'))
    )
    op.create_index(op.f('ix_memory_user_id'), 'user_memories', ['user_id'], unique=False)
    op.create_index(op.f('ix_key'), 'user_memories', ['key'], unique=False)

    # Create payments table
    op.create_table(
        'payments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('telegram_payment_charge_id', sa.String(length=255), nullable=False),
        sa.Column('plan', sa.String(length=50), nullable=False),
        sa.Column('stars_amount', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=10), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.telegram_id'], name=op.f('fk_payments_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_payments')),
        sa.UniqueConstraint('telegram_payment_charge_id', name=op.f('uq_payments_telegram_payment_charge_id'))
    )
    op.create_index(op.f('ix_payment_user_id'), 'payments', ['user_id'], unique=False)
    op.create_index(op.f('ix_telegram_payment_charge_id'), 'payments', ['telegram_payment_charge_id'], unique=False)
    op.create_index(op.f('ix_payment_status'), 'payments', ['status'], unique=False)


def downgrade() -> None:
    op.drop_table('payments')
    op.drop_table('user_memories')
    op.drop_table('rag_items')
    op.drop_table('invite_codes')
    op.drop_table('audit_logs')
    op.drop_table('tool_counters')
    op.drop_table('usage_ledger')
    op.drop_table('messages')
    op.drop_table('chat_sessions')
    op.drop_table('users')
