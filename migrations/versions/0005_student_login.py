"""Add password hashes and notice acceptance to students.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("students", sa.Column("password_hash", sa.String(255)), schema="app")
    op.add_column(
        "students", sa.Column("notice_accepted_at", sa.DateTime(timezone=True)), schema="app"
    )


def downgrade() -> None:
    op.drop_column("students", "notice_accepted_at", schema="app")
    op.drop_column("students", "password_hash", schema="app")
