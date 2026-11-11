"""Make runs safe to repeat: a message ID per run, its agent setup, and posted quizzes kept.

Revision ID: 0017
Revises: 0016
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Sent by the web with each message, so a repeat returns the same run.
    op.add_column("runs", sa.Column("client_message_id", UUID(as_uuid=True)), schema="app")
    # The task kind and tutors, so a worker taking the run over can resume it.
    op.add_column("runs", sa.Column("setup", JSONB()), schema="app")
    op.create_index(
        "ix_app_runs_client_message",
        "runs",
        ["student_id", "client_message_id"],
        unique=True,
        schema="app",
        postgresql_where=sa.text("client_message_id IS NOT NULL"),
    )
    op.create_table(
        "quiz_posts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("app.runs.id"), nullable=False),
        sa.Column(
            "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), nullable=False
        ),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("quiz", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        schema="app",
    )
    op.create_index("ix_app_quiz_posts_run_id", "quiz_posts", ["run_id"], schema="app")


def downgrade() -> None:
    op.drop_table("quiz_posts", schema="app")
    op.drop_index("ix_app_runs_client_message", table_name="runs", schema="app")
    op.drop_column("runs", "setup", schema="app")
    op.drop_column("runs", "client_message_id", schema="app")
