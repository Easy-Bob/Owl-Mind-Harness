"""Integration: M4-C's records on real Postgres and Redis. A message ID is kept once per student,
a run's setup and quiz survive for the worker that takes it over, a run's spend is counted
apart, and an answered approval without its resume run is found."""

import asyncio
import uuid
from datetime import timedelta

import pytest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.budget import RedisBudgetLedger
from owl_mind.core.config import get_settings
from owl_mind.services.approvals import PostgresApprovalStore
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_store import ActiveRunExists, PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore
from tests.test_runtime_integration import migrated, new_student  # noqa: F401  Fixture.

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


async def with_stores(body):
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        return await body(
            PostgresRunStore(sessions),
            PostgresStudyStore(sessions),
            PostgresApprovalStore(sessions),
        )
    finally:
        await engine.dispose()


def test_a_message_id_starts_one_run_and_finds_it_again():
    alice, bob = new_student(), new_student()
    sent = uuid.uuid4()

    async def body(runs, study, approvals):
        thread = await runs.create_thread(alice, "t")
        run_id = await runs.create_run(thread.id, sent)
        await runs.set_run_status(run_id, "completed")
        with pytest.raises(ActiveRunExists):
            await runs.create_run(thread.id, sent)
        # Another student may happen to send the same ID.
        other = await runs.create_thread(bob, "t")
        await runs.create_run(other.id, sent)
        return run_id, await runs.find_by_message(alice, sent), thread.id

    run_id, found, thread_id = asyncio.run(with_stores(body))
    assert found.id == run_id and found.thread_id == thread_id


def test_a_taken_over_run_has_its_setup_quiz_and_active_state():
    alice = new_student()
    setup = {"task_kind": "assess", "subagents": ["practice_coach"], "left_out": []}

    async def body(runs, study, approvals):
        thread = await runs.create_thread(alice, "t")
        run_id = await runs.create_run(thread.id)
        await runs.set_run_status(run_id, "running")
        await runs.set_run_setup(run_id, "orchestrated", "assess", setup)
        await study.save_quiz_post(alice, run_id, "heaps", "1. Q?\nA. a\nB. b")
        active = await runs.active_run(alice, thread.id)
        stranger = await runs.active_run(uuid.uuid4(), thread.id)
        return run_id, await runs.takeover(run_id), await study.quiz_posts(run_id), active, stranger

    run_id, found, quizzes, active, stranger = asyncio.run(with_stores(body))
    assert found.setup == setup and found.route == "orchestrated"
    assert quizzes == ["1. Q?\nA. a\nB. b"]
    assert active.id == run_id and stranger is None


def test_an_answered_approval_without_a_resume_run_is_found_once_it_has_waited():
    alice = new_student()

    async def body(runs, study, approvals):
        thread = await runs.create_thread(alice, "t")
        run_id = await runs.create_run(thread.id)
        asked = {"kind": "commit_plan", "summary": "s", "items": []}
        approval = await approvals.create(run_id, alice, thread.id, asked, {"task_kind": "plan"})
        await approvals.decide(approval.id, "approved")
        fresh = await approvals.unresumed(timedelta(minutes=2))
        due = await approvals.unresumed(timedelta(0))
        await runs.set_run_status(run_id, "completed")
        resume = await runs.create_run(thread.id)
        await approvals.set_resume_run(approval.id, resume)
        after = await approvals.unresumed(timedelta(0))
        return approval.id, fresh, due, after

    approval_id, fresh, due, after = asyncio.run(with_stores(body))
    assert approval_id not in {a.id for a, _ in fresh}
    assert [(a.id, student) for a, student in due if a.id == approval_id] == [(approval_id, alice)]
    assert approval_id not in {a.id for a, _ in after}


def test_a_runs_spend_is_counted_apart_from_the_day():
    run_id = uuid.uuid4()

    async def body():
        client = Redis.from_url(get_settings().redis_url)
        try:
            ledger = RedisBudgetLedger(client)
            await ledger.charge_run(run_id, 120)
            await ledger.charge_run(run_id, 30)
            ttl = await client.ttl(f"budget:run:{run_id}")
            return await ledger.run_spent(run_id), await ledger.run_spent(uuid.uuid4()), ttl
        finally:
            await client.delete(f"budget:run:{run_id}")
            await client.aclose()

    spent, unknown, ttl = asyncio.run(body())
    assert (spent, unknown) == (150, 0)
    assert 0 < ttl <= 24 * 3600
