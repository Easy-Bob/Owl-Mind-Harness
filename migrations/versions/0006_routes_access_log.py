"""Record each run's route and task kind, and add the access log.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("route", sa.String(20)), schema="app")
    op.add_column("runs", sa.Column("task_kind", sa.String(20)), schema="app")
    op.execute(
        """
        CREATE TABLE ops.access_log (
            id uuid PRIMARY KEY,
            viewer_id uuid NOT NULL,
            student_id uuid NOT NULL,
            run_id uuid NOT NULL,
            reason text NOT NULL,
            viewed_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_access_log_student ON ops.access_log (student_id)")


def downgrade() -> None:
    op.execute("DROP TABLE ops.access_log")
    op.drop_column("runs", "task_kind", schema="app")
    op.drop_column("runs", "route", schema="app")
