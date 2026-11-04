"""Needs the compose services: docker compose up -d --wait

The same checks as the in-memory store, so unit tests can trust it stands in for Postgres.
"""

import asyncio
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.study_store import InMemoryStudyStore, PostgresStudyStore, ReviewState

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 11, 2)
POINTS = [{"id": "a", "weight": 2.0, "description": "d"}]


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


def new_student() -> uuid.UUID:
    student_id = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as connection:
        connection.execute(
            "INSERT INTO app.students (id, email) VALUES (%s, %s)",
            (student_id, f"{student_id}@test.invalid"),
        )
    return student_id


def scenario(make_store):
    """Run body(store) in one event loop, because an async engine belongs to its loop."""

    def run(body):
        async def main():
            store, cleanup = make_store()
            try:
                return await body(store)
            finally:
                await cleanup()

        return asyncio.run(main())

    return run


def postgres_store():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    return PostgresStudyStore(async_sessionmaker(engine, expire_on_commit=False)), engine.dispose


def memory_store():
    async def nothing():
        return None

    return InMemoryStudyStore(), nothing


@pytest.fixture(params=[memory_store, postgres_store], ids=["memory", "postgres"])
def run(request):
    return scenario(request.param)


def test_review_round_trip_and_update(run):
    alice = new_student()

    async def body(store):
        assert await store.get_review(alice, "bfs") is None
        await store.save_review(alice, "bfs", ReviewState(2.5, 1, 1, TODAY))
        await store.save_review(alice, "bfs", ReviewState(2.6, 6, 2, TODAY))
        return await store.get_review(alice, "bfs")

    assert run(body) == ReviewState(2.6, 6, 2, TODAY)


def test_due_topics_are_ordered_and_scoped_to_one_student(run):
    alice, bob = new_student(), new_student()

    async def body(store):
        await store.save_review(alice, "later", ReviewState(2.5, 1, 1, date(2026, 11, 1)))
        await store.save_review(alice, "sooner", ReviewState(2.5, 1, 1, date(2026, 10, 1)))
        await store.save_review(alice, "future", ReviewState(2.5, 1, 1, date(2027, 1, 1)))
        await store.save_review(bob, "other", ReviewState(2.5, 1, 1, date(2026, 10, 1)))
        return await store.due_topics(alice, TODAY)

    assert run(body) == ["sooner", "later"]


def test_rubric_is_scoped_to_one_student(run):
    alice, bob = new_student(), new_student()

    async def body(store):
        spec = await store.save_spec(alice, {"topic": "bfs"})
        assert await store.pin_rubric(bob, spec, "Q?", POINTS, "fp") is None
        rubric = await store.pin_rubric(alice, spec, "Q?", POINTS, "fp")
        pinned = await store.get_rubric(alice, rubric)
        return (
            pinned,
            await store.get_rubric(bob, rubric),
            await store.get_rubric(alice, uuid.uuid4()),
        )

    pinned, other, unknown = run(body)
    assert (pinned.question, pinned.points, pinned.fingerprint) == ("Q?", POINTS, "fp")
    assert (other, unknown) == (None, None)


def test_one_grade_per_attempt(run):
    alice = new_student()

    async def body(store):
        spec = await store.save_spec(alice, {"topic": "bfs"})
        rubric = await store.pin_rubric(alice, spec, "Q?", POINTS, "fp")
        first = await store.record_grade(alice, rubric, "hash-1", {"score": 1})
        again = await store.record_grade(alice, rubric, "hash-1", {"score": 2})
        other = await store.record_grade(alice, rubric, "hash-2", {"score": 3})
        return first, again, other

    assert run(body) == (({"score": 1}, True), ({"score": 1}, False), ({"score": 3}, True))


def test_a_rubric_knows_its_quizs_topic(run):
    alice = new_student()

    async def body(store):
        spec = await store.save_spec(alice, {"topic": "bfs"})
        rubric = await store.pin_rubric(alice, spec, "Q?", POINTS, "fp")
        return (await store.get_rubric(alice, rubric)).topic

    assert run(body) == "bfs"


ITEMS = [
    {"topic": "bfs", "note": "", "review_date": "2026-11-09"},
    {"topic": "heaps", "note": "", "review_date": "2026-11-03"},
]


def test_committing_a_plan_adds_only_its_new_topics_once(run):
    alice, bob = new_student(), new_student()
    kept = ReviewState(2.5, 6, 2, date(2026, 11, 20))

    async def body(store):
        await store.save_review(alice, "bfs", kept)
        plan = await store.save_plan(alice, "Week 1", ITEMS)
        stranger = await store.commit_plan(bob, plan)
        first = await store.commit_plan(alice, plan)
        again = await store.commit_plan(alice, plan)
        return (
            stranger,
            first,
            again,
            (await store.get_plan(alice, plan)).status,
            await store.get_review(alice, "bfs"),
            await store.get_review(alice, "heaps"),
            await store.get_plan(bob, plan),
        )

    stranger, first, again, status, bfs, heaps, hidden = run(body)
    assert (stranger, first, again, status) == (False, True, False, "committed")
    assert bfs == kept
    assert heaps.due_date == date(2026, 11, 3)
    assert hidden is None


def test_resetting_progress_clears_one_topic_or_all_for_one_student(run):
    alice, bob = new_student(), new_student()
    state = ReviewState(2.5, 1, 1, date(2026, 11, 5))

    async def body(store):
        for student in (alice, bob):
            for topic in ("bfs", "heaps", "tries"):
                await store.save_review(student, topic, state)
        one = await store.reset_progress(alice, "heaps")
        left = await store.review_counts(alice, date(2026, 11, 1))
        everything = await store.reset_progress(alice, None)
        return one, left, everything, await store.review_counts(bob, date(2026, 11, 1))

    assert run(body) == (1, (2, 0), 2, (3, 0))


def test_plan_is_saved_as_proposed():
    alice = new_student()

    async def body(store):
        return await store.save_plan(alice, "Week 1", [{"topic": "bfs"}])

    plan_id = scenario(postgres_store)(body)

    with psycopg.connect(get_settings().database_url) as connection:
        row = connection.execute(
            "SELECT status, student_id FROM app.study_plans WHERE id = %s", (plan_id,)
        ).fetchone()
    assert row == ("proposed", alice)
