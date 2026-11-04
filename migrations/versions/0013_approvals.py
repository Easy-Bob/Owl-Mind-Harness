"""Approvals for changes to a student's plan or progress, and waiting runs that hold a slot.

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_BUSY = "status IN ('queued', 'running', 'waiting_approval')"
_ACTIVE = "status IN ('queued', 'running')"


def _busy_indexes(where: str) -> None:
    for name, column in (
        ("ix_app_runs_one_active_per_thread", "thread_id"),
        ("ix_app_runs_one_active_per_student", "student_id"),
    ):
        op.drop_index(name, table_name="runs", schema="app")
        op.create_index(
            name, "runs", [column], unique=True, schema="app", postgresql_where=sa.text(where)
        )


def upgrade() -> None:
    op.create_table(
        "approvals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("app.runs.id"), nullable=False),
        sa.Column(
            "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), nullable=False
        ),
        sa.Column("thread_id", UUID(as_uuid=True), sa.ForeignKey("app.threads.id"), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("items", JSONB(), nullable=False),
        # The task kind and tutors, so the run can be built again to resume.
        sa.Column("setup", JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("resume_run_id", UUID(as_uuid=True), sa.ForeignKey("app.runs.id")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'expired')", name="ck_approvals_status"
        ),
        schema="app",
    )
    op.create_index(
        "ix_app_approvals_one_pending_per_student",
        "approvals",
        ["student_id"],
        unique=True,
        schema="app",
        postgresql_where=sa.text("status = 'pending'"),
    )
    _busy_indexes(_BUSY)


def downgrade() -> None:
    op.execute("UPDATE app.runs SET status = 'stopped' WHERE status = 'waiting_approval'")
    _busy_indexes(_ACTIVE)
    op.drop_index("ix_app_approvals_one_pending_per_student", table_name="approvals", schema="app")
    op.drop_table("approvals", schema="app")
