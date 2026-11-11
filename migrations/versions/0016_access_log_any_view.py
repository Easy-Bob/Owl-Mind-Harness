"""Let the access log record views that aren't one run's, such as a student's usage on Users.

Revision ID: 0016
Revises: 0015
"""

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("access_log", "run_id", nullable=True, schema="ops")


def downgrade() -> None:
    op.execute("DELETE FROM ops.access_log WHERE run_id IS NULL")
    op.alter_column("access_log", "run_id", nullable=False, schema="ops")
