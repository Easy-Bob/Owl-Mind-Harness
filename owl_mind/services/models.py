"""Tables in the app schema. Other tables are added by the plan that first uses them."""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def sqlalchemy_url(database_url: str) -> str:
    """Select the psycopg 3 driver, which is the only Postgres driver installed."""
    return database_url.replace("postgresql://", "postgresql+psycopg://", 1)


class Base(DeclarativeBase):
    metadata = MetaData(schema="app")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    notice_accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # The invite code's hash, for turning live agent runs on per cohort (M4-G).
    invite_hash: Mapped[str | None] = mapped_column(String(64))


class Thread(Base):
    __tablename__ = "threads"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# A run waiting for the student's approval holds its slot, like one that is running.
_BUSY = "status IN ('queued', 'running', 'waiting_approval')"


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index(
            "ix_app_runs_one_active_per_thread",
            "thread_id",
            unique=True,
            postgresql_where=text(_BUSY),
        ),
        Index(
            "ix_app_runs_client_message",
            "student_id",
            "client_message_id",
            unique=True,
            postgresql_where=text("client_message_id IS NOT NULL"),
        ),
        # One at a time for a student too, so a second chat can't double their spending.
        Index(
            "ix_app_runs_one_active_per_student",
            "student_id",
            unique=True,
            postgresql_where=text(_BUSY),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("threads.id"), index=True)
    # The thread's owner, copied here so the index above can see it.
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"))
    # Sent by the web with each message, so a repeat returns this run (M4-C).
    client_message_id: Mapped[uuid.UUID | None] = mapped_column()
    # The task kind and tutors, so a worker taking the run over can resume it.
    setup: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))
    route: Mapped[str | None] = mapped_column(String(20))
    task_kind: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewSchedule(Base):
    __tablename__ = "review_schedule"

    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), primary_key=True)
    topic: Mapped[str] = mapped_column(String(200), primary_key=True)
    ease: Mapped[float] = mapped_column(Float)
    interval_days: Mapped[int] = mapped_column(Integer)
    repetitions: Mapped[int] = mapped_column(Integer)
    due_date: Mapped[date] = mapped_column(Date, index=True)


class QuizSpec(Base):
    __tablename__ = "quiz_specs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), index=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Rubric(Base):
    __tablename__ = "rubrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), index=True)
    spec_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("quiz_specs.id"))
    question: Mapped[str] = mapped_column(String(4000))
    points: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    fingerprint: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    __table_args__ = (UniqueConstraint("rubric_id", "answer_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), index=True)
    rubric_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rubrics.id"))
    answer_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Grade(Base):
    __tablename__ = "grades"

    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("quiz_attempts.id"), primary_key=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)


class StudyPlan(Base):
    __tablename__ = "study_plans"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RunEvent(Base):
    __tablename__ = "run_events"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Handoff(Base):
    """A forum draft. Its text reaches GitHub only after the student approves it."""

    __tablename__ = "handoffs"
    __table_args__ = (Index("ix_handoffs_student_thread", "student_id", "thread_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"))
    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("threads.id"))
    status: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    graded: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    category: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    posted_as: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"))
    improvements: Mapped[int] = mapped_column(Integer, server_default=text("0"))


class ForumPost(Base):
    """A posted discussion, and when it is next checked for replies."""

    __tablename__ = "forum_posts"
    __table_args__ = (
        Index("ix_forum_posts_due", "next_poll_at", postgresql_where=text("state = 'open'")),
    )

    handoff_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("handoffs.id"), primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    node_id: Mapped[str] = mapped_column(Text)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    state: Mapped[str] = mapped_column(Text, server_default=text("'open'"))
    next_poll_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failures: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ForumReply(Base):
    """Someone else's text from the forum. Shown to the student, never given to an agent."""

    __tablename__ = "forum_replies"
    __table_args__ = (Index("ix_forum_replies_handoff", "handoff_id"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    handoff_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("forum_posts.handoff_id", ondelete="CASCADE")
    )
    author: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_answer: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    seen: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ForumAccount(Base):
    """A connected GitHub account. Both tokens are encrypted by the app."""

    __tablename__ = "forum_accounts"

    student_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), primary_key=True
    )
    github_login: Mapped[str] = mapped_column(Text)
    access_token: Mapped[bytes] = mapped_column(BYTEA)
    refresh_token: Mapped[bytes | None] = mapped_column(BYTEA)
    access_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    refresh_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    needs_reconnect: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Approval(Base):
    """A change to a student's plan or progress, waiting for their answer on a card."""

    __tablename__ = "approvals"
    __table_args__ = (
        Index(
            "ix_app_approvals_one_pending_per_student",
            "student_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"))
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"))
    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("threads.id"))
    kind: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    setup: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(Text)
    resume_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryWrite(Base):
    """When a chat's long-term memory was last written, and from what."""

    __tablename__ = "memory_writes"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True
    )
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"))
    input_hash: Mapped[str | None] = mapped_column(Text)
    written_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QuizPost(Base):
    """A posted quiz, kept so a resumed run still shows it."""

    __tablename__ = "quiz_posts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id"), index=True)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"))
    topic: Mapped[str] = mapped_column(Text)
    quiz: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
