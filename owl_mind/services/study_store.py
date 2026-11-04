"""Study records in the app schema. Every method is scoped to one student."""

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import delete, select, update
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
from owl_mind.services.sm2 import DEFAULT_EASE


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
    # The quiz's topic, so a graded answer can record the student's recall of it.
    topic: str = ""


@dataclass(frozen=True)
class SavedPlan:
    id: uuid.UUID
    title: str
    items: list[dict[str, Any]]
    status: str


def new_review(review_date: str) -> ReviewState:
    """A topic's first review state: the starting ease, due on the plan's date."""
    return ReviewState(DEFAULT_EASE, 0, 0, date.fromisoformat(review_date))


@runtime_checkable
class StudyStore(Protocol):
    async def get_review(self, student_id: uuid.UUID, topic: str) -> ReviewState | None: ...

    async def save_review(self, student_id: uuid.UUID, topic: str, state: ReviewState) -> None: ...

    async def due_topics(self, student_id: uuid.UUID, today: date) -> list[str]: ...

    async def review_counts(self, student_id: uuid.UUID, today: date) -> tuple[int, int]:
        """Return how many topics are scheduled, and how many of those are due."""

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
    ) -> tuple[dict[str, Any], bool]:
        """Keep one grade per attempt. Return the grade that was kept, and whether it's new."""

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID: ...

    async def get_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> SavedPlan | None: ...

    async def commit_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> bool:
        """Commit a proposed plan once, adding its new topics to the review schedule."""

    async def reset_progress(self, student_id: uuid.UUID, topic: str | None) -> int:
        """Clear review state for one topic, or all topics with None. Return how many."""


class InMemoryStudyStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.reviews: dict[tuple[uuid.UUID, str], ReviewState] = {}
        self.specs: dict[uuid.UUID, tuple[uuid.UUID, dict[str, Any]]] = {}
        self.rubrics: dict[uuid.UUID, tuple[uuid.UUID, PinnedRubric]] = {}
        self.grades: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
        self.plans: dict[uuid.UUID, tuple[uuid.UUID, SavedPlan]] = {}

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

    async def review_counts(self, student_id: uuid.UUID, today: date) -> tuple[int, int]:
        owned = [s for (owner, _), s in self.reviews.items() if owner == student_id]
        return len(owned), sum(state.due_date <= today for state in owned)

    async def save_spec(self, student_id: uuid.UUID, spec: dict[str, Any]) -> uuid.UUID:
        spec_id = uuid.uuid4()
        self.specs[spec_id] = (student_id, spec)
        return spec_id

    async def pin_rubric(
        self,
        student_id: uuid.UUID,
        spec_id: uuid.UUID,
        question: str,
        points: list[dict[str, Any]],
        fingerprint: str,
    ) -> uuid.UUID | None:
        owner, spec = self.specs.get(spec_id, (None, {}))
        if owner != student_id:
            return None
        rubric_id = uuid.uuid4()
        self.rubrics[rubric_id] = (
            student_id,
            PinnedRubric(rubric_id, question, points, fingerprint, spec.get("topic", "")),
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
    ) -> tuple[dict[str, Any], bool]:
        key = (rubric_id, answer_hash)
        new = key not in self.grades
        return self.grades.setdefault(key, result), new

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID:
        plan_id = uuid.uuid4()
        self.plans[plan_id] = (student_id, SavedPlan(plan_id, title, items, "proposed"))
        return plan_id

    async def get_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> SavedPlan | None:
        owner, plan = self.plans.get(plan_id, (None, None))
        return plan if owner == student_id else None

    async def commit_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> bool:
        plan = await self.get_plan(student_id, plan_id)
        if plan is None or plan.status != "proposed":
            return False
        self.plans[plan_id] = (student_id, SavedPlan(plan.id, plan.title, plan.items, "committed"))
        for item in plan.items:
            if (student_id, item["topic"]) not in self.reviews:
                self.reviews[(student_id, item["topic"])] = new_review(item["review_date"])
        return True

    async def reset_progress(self, student_id: uuid.UUID, topic: str | None) -> int:
        keys = [k for k in self.reviews if k[0] == student_id and topic in (None, k[1])]
        for key in keys:
            del self.reviews[key]
        return len(keys)


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

    async def review_counts(self, student_id: uuid.UUID, today: date) -> tuple[int, int]:
        statement = select(ReviewSchedule.due_date).where(ReviewSchedule.student_id == student_id)
        async with self._sessions() as session:
            dates = (await session.scalars(statement)).all()
        return len(dates), sum(due <= today for due in dates)

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
            spec = await session.get(QuizSpec, row.spec_id)
        topic = (spec.spec or {}).get("topic", "") if spec else ""
        return PinnedRubric(row.id, row.question, row.points, row.fingerprint, topic)

    async def record_grade(
        self,
        student_id: uuid.UUID,
        rubric_id: uuid.UUID,
        answer_hash: str,
        result: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
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
                return result, True
            kept = await session.scalar(
                select(Grade.result)
                .join(QuizAttempt, QuizAttempt.id == Grade.attempt_id)
                .where(QuizAttempt.rubric_id == rubric_id, QuizAttempt.answer_hash == answer_hash)
            )
            return kept, False

    async def save_plan(
        self, student_id: uuid.UUID, title: str, items: list[dict[str, Any]]
    ) -> uuid.UUID:
        row = StudyPlan(
            id=uuid.uuid4(), student_id=student_id, title=title, items=items, status="proposed"
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return row.id

    async def get_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> SavedPlan | None:
        async with self._sessions() as session:
            row = await session.get(StudyPlan, plan_id)
        if row is None or row.student_id != student_id:
            return None
        return SavedPlan(row.id, row.title, row.items, row.status)

    async def commit_plan(self, student_id: uuid.UUID, plan_id: uuid.UUID) -> bool:
        async with self._sessions.begin() as session:
            # Only a proposed plan changes, so committing twice applies it once.
            items = await session.scalar(
                update(StudyPlan)
                .where(
                    StudyPlan.id == plan_id,
                    StudyPlan.student_id == student_id,
                    StudyPlan.status == "proposed",
                )
                .values(status="committed")
                .returning(StudyPlan.items)
            )
            if items is None:
                return False
            for item in items:
                state = new_review(item["review_date"])
                await session.execute(
                    insert(ReviewSchedule)
                    .values(
                        student_id=student_id,
                        topic=item["topic"],
                        ease=state.ease,
                        interval_days=state.interval_days,
                        repetitions=state.repetitions,
                        due_date=state.due_date,
                    )
                    .on_conflict_do_nothing()
                )
        return True

    async def reset_progress(self, student_id: uuid.UUID, topic: str | None) -> int:
        statement = delete(ReviewSchedule).where(ReviewSchedule.student_id == student_id)
        if topic is not None:
            statement = statement.where(ReviewSchedule.topic == topic)
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
        return result.rowcount
