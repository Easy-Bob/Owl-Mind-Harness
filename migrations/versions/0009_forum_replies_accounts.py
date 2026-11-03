"""Track forum posts for replies, save the replies, and store connected GitHub accounts.

Revision ID: 0009
Revises: 0008
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE app.handoffs ADD COLUMN posted_as text")
    op.execute(
        """
        ALTER TABLE app.forum_posts
            ADD COLUMN state text NOT NULL DEFAULT 'open'
                CHECK (state IN ('open', 'done', 'gone')),
            ADD COLUMN next_poll_at timestamptz,
            ADD COLUMN failures integer NOT NULL DEFAULT 0,
            ADD COLUMN lease_until timestamptz
        """
    )
    op.execute("UPDATE app.forum_posts SET next_poll_at = posted_at + interval '5 minutes'")
    op.execute(
        "CREATE INDEX ix_forum_posts_due ON app.forum_posts (next_poll_at) WHERE state = 'open'"
    )
    op.execute(
        """
        CREATE TABLE app.forum_replies (
            id text PRIMARY KEY,
            handoff_id uuid NOT NULL REFERENCES app.forum_posts (handoff_id) ON DELETE CASCADE,
            author text NOT NULL,
            body text NOT NULL,
            created_at timestamptz NOT NULL,
            is_answer boolean NOT NULL DEFAULT false,
            seen boolean NOT NULL DEFAULT false,
            saved_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_forum_replies_handoff ON app.forum_replies (handoff_id)")
    # Tokens are encrypted by the app before they get here.
    op.execute(
        """
        CREATE TABLE app.forum_accounts (
            student_id uuid PRIMARY KEY REFERENCES app.students (id) ON DELETE CASCADE,
            github_login text NOT NULL,
            access_token bytea NOT NULL,
            refresh_token bytea,
            access_expires_at timestamptz,
            refresh_expires_at timestamptz,
            needs_reconnect boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE app.forum_accounts")
    op.execute("DROP TABLE app.forum_replies")
    op.execute("DROP INDEX app.ix_forum_posts_due")
    op.execute(
        "ALTER TABLE app.forum_posts DROP COLUMN state, DROP COLUMN next_poll_at,"
        " DROP COLUMN failures, DROP COLUMN lease_until"
    )
    op.execute("ALTER TABLE app.handoffs DROP COLUMN posted_as")
