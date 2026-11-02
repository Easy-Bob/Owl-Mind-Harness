"""Create the handoff drafts and forum posts tables.

Revision ID: 0008
Revises: 0007
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE app.handoffs (
            id uuid PRIMARY KEY,
            student_id uuid NOT NULL REFERENCES app.students (id),
            thread_id uuid NOT NULL REFERENCES app.threads (id),
            status text NOT NULL CHECK (
                status IN ('draft', 'posting', 'posted', 'retrying', 'fallback', 'cancelled')
            ),
            title text NOT NULL,
            body text NOT NULL,
            graded boolean NOT NULL DEFAULT false,
            category text NOT NULL,
            url text,
            attempts integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_handoffs_student_thread ON app.handoffs (student_id, thread_id)")
    op.execute(
        """
        CREATE TABLE app.forum_posts (
            handoff_id uuid PRIMARY KEY REFERENCES app.handoffs (id),
            url text NOT NULL,
            node_id text NOT NULL,
            posted_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE app.forum_posts")
    op.execute("DROP TABLE app.handoffs")
