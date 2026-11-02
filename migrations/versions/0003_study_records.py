"""Create the tables behind the study tools.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _id() -> sa.Column:
    return sa.Column("id", UUID(as_uuid=True), primary_key=True)


def _student() -> sa.Column:
    return sa.Column(
        "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), nullable=False
    )


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "review_schedule",
        sa.Column(
            "student_id", UUID(as_uuid=True), sa.ForeignKey("app.students.id"), primary_key=True
        ),
        sa.Column("topic", sa.String(200), primary_key=True),
        sa.Column("ease", sa.Float(), nullable=False),
        sa.Column("interval_days", sa.Integer(), nullable=False),
        sa.Column("repetitions", sa.Integer(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=False),
        schema="app",
    )
    op.create_index(
        "ix_app_review_schedule_due_date", "review_schedule", ["due_date"], schema="app"
    )
    op.create_table(
        "quiz_specs",
        _id(),
        _student(),
        sa.Column("spec", JSONB(), nullable=False),
        _created_at(),
        schema="app",
    )
    op.create_index("ix_app_quiz_specs_student_id", "quiz_specs", ["student_id"], schema="app")
    op.create_table(
        "rubrics",
        _id(),
        _student(),
        sa.Column(
            "spec_id", UUID(as_uuid=True), sa.ForeignKey("app.quiz_specs.id"), nullable=False
        ),
        sa.Column("question", sa.String(4000), nullable=False),
        sa.Column("points", JSONB(), nullable=False),
        sa.Column("fingerprint", sa.String(16), nullable=False),
        _created_at(),
        schema="app",
    )
    op.create_index("ix_app_rubrics_student_id", "rubrics", ["student_id"], schema="app")
    op.create_table(
        "quiz_attempts",
        _id(),
        _student(),
        sa.Column("rubric_id", UUID(as_uuid=True), sa.ForeignKey("app.rubrics.id"), nullable=False),
        sa.Column("answer_hash", sa.String(64), nullable=False),
        _created_at(),
        sa.UniqueConstraint("rubric_id", "answer_hash"),
        schema="app",
    )
    op.create_index(
        "ix_app_quiz_attempts_student_id", "quiz_attempts", ["student_id"], schema="app"
    )
    op.create_table(
        "grades",
        sa.Column(
            "attempt_id",
            UUID(as_uuid=True),
            sa.ForeignKey("app.quiz_attempts.id"),
            primary_key=True,
        ),
        sa.Column("result", JSONB(), nullable=False),
        schema="app",
    )
    op.create_table(
        "study_plans",
        _id(),
        _student(),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("items", JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        _created_at(),
        schema="app",
    )
    op.create_index("ix_app_study_plans_student_id", "study_plans", ["student_id"], schema="app")


def downgrade() -> None:
    for table in (
        "study_plans",
        "grades",
        "quiz_attempts",
        "rubrics",
        "quiz_specs",
        "review_schedule",
    ):
        op.drop_table(table, schema="app")
