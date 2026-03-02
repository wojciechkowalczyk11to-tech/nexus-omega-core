"""Remove duplicate stars_amount column from payments table.

The stars_amount column was an alias for amount_stars. amount_stars is kept as the canonical column.

Revision ID: 003_remove_stars_amount
Revises: 002_schema_align
Create Date: 2025-03-01 00:01:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003_remove_stars_amount"
down_revision: str = "002_schema_align"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("payments", "stars_amount")


def downgrade() -> None:
    op.add_column(
        "payments",
        sa.Column("stars_amount", sa.Integer(), nullable=False, server_default="0"),
    )
