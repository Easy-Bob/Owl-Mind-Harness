"""Study records in the app schema. Every method is scoped to one student."""

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.models import (
    Grade,
    QuizAttempt,
    QuizSpec,
    ReviewSchedule,
    Rubric,
    StudyPlan,
)


@dataclass(frozen=True)
class ReviewState:
    ease: float
    interval_days: int
    repetitions: int
    due_date: date


@dataclass(frozen=True)
class PinnedRubric:
    id: uuid.UUID
    question: str
    points: list[dict[str, Any]]
    fingerprint: str


@runtime_checkable
class StudyStore(Protocol):
    async def get_review(self, student_id: uuid.UUID, topic: str) -> ReviewState | None: ...

    async def save_review(self, student_id: uuid.UUID, topic: str, state: ReviewState) -> None: ...

    async def due_topics(self, student_id: uuid.UUID, today: date) -> list[str]: ...

    async def save_spec(self, student_id: uuid.UUID, spec: dict[str, Any]) -> uuid.UUID: ...

    async def pin_rubric(
        self,
        student_id: uuid.UUID,
        spec_id: uuid.UUID,
        question: str,
        points: list[dict[str, Any]],
        fingerprint: str,
    ) -> uuid.UUID | None:
        """Return None if the spec is not this student's."""

    async def get_rubric(self, student_id: uuid.UUID, rubric_id: uuid.UUID) -> PinnedRubric | None:
        """Return None if the rubric is not this student's."""

    async def record_grade(
        self,
        student_id: uuid.UUID,
        rubric_id: uuid.UUID,
        answer_hash: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep one grade per attempt, and return the grade that was kept."""

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID: ...


class InMemoryStudyStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.reviews: dict[tuple[uuid.UUID, str], ReviewState] = {}
        self.specs: dict[uuid.UUID, uuid.UUID] = {}
        self.rubrics: dict[uuid.UUID, tuple[uuid.UUID, PinnedRubric]] = {}
        self.grades: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
        self.plans: dict[uuid.UUID, tuple[uuid.UUID, str, list[dict[str, Any]]]] = {}

    async def get_review(self, student_id: uuid.UUID, topic: str) -> ReviewState | None:
        return self.reviews.get((student_id, topic))

    async def save_review(self, student_id: uuid.UUID, topic: str, state: ReviewState) -> None:
        self.reviews[(student_id, topic)] = state

    async def due_topics(self, student_id: uuid.UUID, today: date) -> list[str]:
        due = [
            (state.due_date, topic)
            for (owner, topic), state in self.reviews.items()
            if owner == student_id and state.due_date <= today
        ]
        return [topic for _, topic in sorted(due)]

    async def save_spec(self, student_id: uuid.UUID, spec: dict[str, Any]) -> uuid.UUID:
        spec_id = uuid.uuid4()
        self.specs[spec_id] = student_id
        return spec_id

    async def pin_rubric(
        self,
        student_id: uuid.UUID,
        spec_id: uuid.UUID,
        question: str,
        points: list[dict[str, Any]],
        fingerprint: str,
    ) -> uuid.UUID | None:
        if self.specs.get(spec_id) != student_id:
            return None
        rubric_id = uuid.uuid4()
        self.rubrics[rubric_id] = (
            student_id,
            PinnedRubric(rubric_id, question, points, fingerprint),
        )
        return rubric_id

    async def get_rubric(self, student_id: uuid.UUID, rubric_id: uuid.UUID) -> PinnedRubric | None:
        owner, rubric = self.rubrics.get(rubric_id, (None, None))
        return rubric if owner == student_id else None

    async def record_grade(
        self,
        student_id: uuid.UUID,
        rubric_id: uuid.UUID,
        answer_hash: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return self.grades.setdefault((rubric_id, answer_hash), result)

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID:
        plan_id = uuid.uuid4()
        self.plans[plan_id] = (student_id, title, items)
        return plan_id


class PostgresStudyStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_review(self, student_id: uuid.UUID, topic: str) -> ReviewState | None:
        async with self._sessions() as session:
            row = await session.get(ReviewSchedule, (student_id, topic))
        if row is None:
            return None
        return ReviewState(row.ease, row.interval_days, row.repetitions, row.due_date)

    async def save_review(self, student_id: uuid.UUID, topic: str, state: ReviewState) -> None:
        values = {
            "ease": state.ease,
            "interval_days": state.interval_days,
            "repetitions": state.repetitions,
            "due_date": state.due_date,
        }
        statement = (
            insert(ReviewSchedule)
            .values(student_id=student_id, topic=topic, **values)
            .on_conflict_do_update(index_elements=["student_id", "topic"], set_=values)
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def due_topics(self, student_id: uuid.UUID, today: date) -> list[str]:
        statement = (
            select(ReviewSchedule.topic)
            .where(ReviewSchedule.student_id == student_id, ReviewSchedule.due_date <= today)
            .order_by(ReviewSchedule.due_date, ReviewSchedule.topic)
        )
        async with self._sessions() as session:
            return list((await session.scalars(statement)).all())

    async def save_spec(self, student_id: uuid.UUID, spec: dict[str, Any]) -> uuid.UUID:
        row = QuizSpec(id=uuid.uuid4(), student_id=student_id, spec=spec)
        async with self._sessions.begin() as session:
            session.add(row)
        return row.id

    async def pin_rubric(
        self,
        student_id: uuid.UUID,
        spec_id: uuid.UUID,
        question: str,
        points: list[dict[str, Any]],
        fingerprint: str,
    ) -> uuid.UUID | None:
        async with self._sessions.begin() as session:
            spec = await session.get(QuizSpec, spec_id)
            if spec is None or spec.student_id != student_id:
                return None
            row = Rubric(
                id=uuid.uuid4(),
                student_id=student_id,
                spec_id=spec_id,
                question=question,
                points=points,
                fingerprint=fingerprint,
            )
            session.add(row)
        return row.id

    async def get_rubric(self, student_id: uuid.UUID, rubric_id: uuid.UUID) -> PinnedRubric | None:
        async with self._sessions() as session:
            row = await session.get(Rubric, rubric_id)
        if row is None or row.student_id != student_id:
            return None
        return PinnedRubric(row.id, row.question, row.points, row.fingerprint)

    async def record_grade(
        self,
        student_id: uuid.UUID,
        rubric_id: uuid.UUID,
        answer_hash: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._sessions.begin() as session:
            attempt_id = await session.scalar(
                insert(QuizAttempt)
                .values(
                    id=uuid.uuid4(),
                    student_id=student_id,
                    rubric_id=rubric_id,
                    answer_hash=answer_hash,
                )
                .on_conflict_do_nothing()
                .returning(QuizAttempt.id)
            )
            if attempt_id is not None:
                session.add(Grade(attempt_id=attempt_id, result=result))
                return result
            kept = await session.scalar(
                select(Grade.result)
                .join(QuizAttempt, QuizAttempt.id == Grade.attempt_id)
                .where(QuizAttempt.rubric_id == rubric_id, QuizAttempt.answer_hash == answer_hash)
            )
            return kept

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID:
        row = StudyPlan(
            id=uuid.uuid4(), student_id=student_id, title=title, items=items, status="proposed"
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return row.id
