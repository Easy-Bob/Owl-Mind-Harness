"""Topics for each course document, shown to students as what they can ask about.

Revision ID: 0014
Revises: 0013
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("topics", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        schema="materials",
    )


def downgrade() -> None:
    op.drop_column("documents", "topics", schema="materials")
