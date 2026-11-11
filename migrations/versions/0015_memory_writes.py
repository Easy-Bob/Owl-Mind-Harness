"""When each chat's memory was last written, so the job writes only after new turns.

Revision ID: 0015
Revises: 0014
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_writes",
        sa.Column(
            "thread_id",
            UUID(as_uuid=True),
            sa.ForeignKey("app.threads.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), nullable=False
        ),
        # A hash of what the writer saw, so an unchanged chat costs no call.
        sa.Column("input_hash", sa.Text()),
        sa.Column("written_at", sa.DateTime(timezone=True)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        schema="app",
    )


def downgrade() -> None:
    op.drop_table("memory_writes", schema="app")
