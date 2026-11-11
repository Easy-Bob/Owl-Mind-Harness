"""Release operations (M4-G): a cohort tag per account, and usage kept after an account goes.

Revision ID: 0018
Revises: 0017
"""

import sqlalchemy as sa
from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A hash of the invite code the account signed up with, never the code, so live agent runs
    # can be turned on for one cohort at a time.
    op.add_column("students", sa.Column("invite_hash", sa.String(64)), schema="app")
    # What a deleted account used, as counts per day with nothing that names anyone.
    op.execute(
        """
        CREATE TABLE ops.deleted_usage (
            day date NOT NULL,
            accounts integer NOT NULL DEFAULT 0,
            runs integer NOT NULL DEFAULT 0,
            model_calls integer NOT NULL DEFAULT 0,
            tokens bigint NOT NULL DEFAULT 0,
            cost_usd numeric(14, 6) NOT NULL DEFAULT 0,
            PRIMARY KEY (day)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE ops.deleted_usage")
    op.drop_column("students", "invite_hash", schema="app")
