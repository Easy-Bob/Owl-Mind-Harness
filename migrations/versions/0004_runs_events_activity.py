"""Add run events, one active run per thread, and activity tables.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_events",
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("app.runs.id"), primary_key=True),
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        schema="app",
    )
    op.create_index(
        "ix_app_runs_one_active_per_thread",
        "runs",
        ["thread_id"],
        unique=True,
        schema="app",
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
    op.execute(
        """
        CREATE TABLE ops.activity_events (
            event_id uuid PRIMARY KEY,
            student_id uuid NOT NULL,
            thread_id uuid NOT NULL,
            event_type text NOT NULL CHECK (event_type IN ('session_started', 'message_sent')),
            occurred_at timestamptz NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX ix_activity_events_student ON ops.activity_events (student_id)")
    op.execute(
        """
        CREATE TABLE ops.activity_daily (
            student_id uuid NOT NULL,
            day date NOT NULL,
            sessions_started integer NOT NULL DEFAULT 0,
            messages_sent integer NOT NULL DEFAULT 0,
            PRIMARY KEY (student_id, day)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE ops.activity_daily")
    op.execute("DROP TABLE ops.activity_events")
    op.drop_index("ix_app_runs_one_active_per_thread", table_name="runs", schema="app")
    op.drop_table("run_events", schema="app")
