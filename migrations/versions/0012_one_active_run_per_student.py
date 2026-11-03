"""One unfinished run per student, across all their threads.

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Copied from the thread, because a unique index can't reach through a join.
    op.add_column(
        "runs",
        sa.Column("student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id")),
        schema="app",
    )
    op.execute(
        "UPDATE app.runs r SET student_id = t.student_id "
        "FROM app.threads t WHERE t.id = r.thread_id"
    )
    op.alter_column("runs", "student_id", nullable=False, schema="app")
    # Runs left unfinished by a crash are failed at startup. Do the same here, so that old rows
    # can't block the new index.
    op.execute(
        """
        UPDATE app.runs SET status = 'failed'
        WHERE status IN ('queued', 'running') AND id NOT IN (
            SELECT DISTINCT ON (student_id) id FROM app.runs
            WHERE status IN ('queued', 'running')
            ORDER BY student_id, created_at DESC
        )
        """
    )
    op.create_index(
        "ix_app_runs_one_active_per_student",
        "runs",
        ["student_id"],
        unique=True,
        schema="app",
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("ix_app_runs_one_active_per_student", table_name="runs", schema="app")
    op.drop_column("runs", "student_id", schema="app")
