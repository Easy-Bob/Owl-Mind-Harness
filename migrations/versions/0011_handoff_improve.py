"""Count each draft's AI rewrites, and link a draft to the run that made it.

Revision ID: 0011
Revises: 0010
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE app.handoffs
            ADD COLUMN run_id uuid REFERENCES app.runs (id),
            ADD COLUMN improvements integer NOT NULL DEFAULT 0
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE app.handoffs DROP COLUMN run_id, DROP COLUMN improvements")
