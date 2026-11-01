"""Create the five schemas, pgvector and the first app tables.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SCHEMAS = ("lg_checkpoint", "lg_store", "app", "materials", "ops")


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    for schema in SCHEMAS:
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")

    op.create_table(
        "students",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        _created_at(),
        schema="app",
    )
    op.create_table(
        "threads",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), nullable=False
        ),
        sa.Column("title", sa.String(200), nullable=False),
        _created_at(),
        schema="app",
    )
    op.create_index("ix_app_threads_student_id", "threads", ["student_id"], schema="app")
    op.create_table(
        "runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("thread_id", UUID(as_uuid=True), sa.ForeignKey("app.threads.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        _created_at(),
        schema="app",
    )
    op.create_index("ix_app_runs_thread_id", "runs", ["thread_id"], schema="app")


def downgrade() -> None:
    # Only the app tables are removed. The schemas stay because the saver and store own tables
    # inside them, and the version table lives in app.
    op.drop_table("runs", schema="app")
    op.drop_table("threads", schema="app")
    op.drop_table("students", schema="app")
