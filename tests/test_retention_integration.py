"""Integration: retention and account deletion (M4-G) on real Postgres, with the real saver and
store. Old times are made by moving rows' timestamps back."""

import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import _record_turn, read_history
from owl_mind.services.memory import Memory, namespace
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.retention import Retention, _month
from tests.test_runtime_integration import migrated, new_student  # noqa: F401  Fixture.

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


def sql(statement: str, *params):
    with psycopg.connect(get_settings().database_url) as db:
        cursor = db.execute(statement, params)
        return cursor.fetchall() if cursor.description else None


def chat(student: uuid.UUID, days_ago: float, status: str = "completed") -> tuple[uuid.UUID, ...]:
    """A thread with one run made days_ago, as the service makes them."""
    thread, run = uuid.uuid4(), uuid.uuid4()
    sql("INSERT INTO app.threads (id, student_id, title) VALUES (%s, %s, 't')", thread, student)
    sql(
        "INSERT INTO app.runs (id, thread_id, student_id, status, route, task_kind)"
        " VALUES (%s, %s, %s, %s, 'orchestrated', 'learn')",
        run, thread, student, status,
    )  # fmt: skip
    when = datetime.now(UTC) - timedelta(days=days_ago)
    sql("UPDATE app.threads SET created_at = %s WHERE id = %s", when, thread)
    sql("UPDATE app.runs SET created_at = %s WHERE id = %s", when, run)
    return thread, run


async def with_retention(body):
    settings = get_settings()
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with open_persistence(settings.database_url) as persistence:
            retention = Retention(sessions, persistence.checkpointer, persistence.store)
            return await body(retention, persistence)
    finally:
        await engine.dispose()


def checkpoints(thread: uuid.UUID) -> int:
    return sql("SELECT count(*) FROM lg_checkpoint.checkpoints WHERE thread_id = %s", str(thread))[
        0
    ][0]


def test_an_idle_chat_keeps_only_its_latest_checkpoint_and_all_its_history():
    student = new_student()
    idle, _ = chat(student, days_ago=2)
    recent, _ = chat(student, days_ago=0)
    busy, _ = chat(student, days_ago=2, status="waiting_approval")

    async def body(retention, persistence):
        for thread in (idle, recent, busy):
            for n in range(3):
                await _record_turn(persistence.checkpointer, thread, f"q{n}", f"a{n}")
        before = {t: checkpoints(t) for t in (idle, recent, busy)}
        await retention.prune_checkpoints(datetime.now(UTC))
        history = await read_history(persistence.checkpointer, idle)
        return before, history

    before, history = asyncio.run(with_retention(body))
    assert before[idle] >= 3
    assert checkpoints(idle) == 1
    assert checkpoints(recent) == before[recent] and checkpoints(busy) == before[busy]
    assert [m["content"] for m in history] == ["q0", "a0", "q1", "a1", "q2", "a2"]


def test_a_chat_idle_past_the_period_goes_with_its_checkpoints_runs_and_episode():
    student = new_student()
    old, old_run = chat(student, days_ago=181)
    kept, _ = chat(student, days_ago=10)
    call = uuid.uuid4()
    sql(
        "INSERT INTO ops.model_calls (call_id, called_at, run_id, component, provider, model,"
        " usage_complete, outcome, latency_ms) VALUES (%s, now(), %s, 'orchestrator', 'deepseek',"
        " 'deepseek-flash', false, 'success', 5)",
        call, old_run,
    )  # fmt: skip
    sql(
        "INSERT INTO app.run_events (run_id, key, kind, payload) VALUES (%s, 'k', 'done', %s)",
        old_run, Jsonb({"status": "completed"}),
    )  # fmt: skip

    async def body(retention, persistence):
        memory = Memory(persistence.store)
        for thread in (old, kept):
            await _record_turn(persistence.checkpointer, thread, "q", "a")
            await memory.save(student, thread, {"strengths": []}, f"episode of {thread}")
        deleted = await retention.delete_old_chats(datetime.now(UTC))
        episodes = [e.summary for e in await memory.episodes(student)]
        return deleted, episodes

    deleted, episodes = asyncio.run(with_retention(body))
    assert deleted >= 1
    assert sql("SELECT count(*) FROM app.threads WHERE id = %s", old)[0][0] == 0
    assert sql("SELECT count(*) FROM app.runs WHERE id = %s", old_run)[0][0] == 0
    assert checkpoints(old) == 0 and checkpoints(kept) >= 1
    assert episodes == [f"episode of {kept}"]
    # The call stays, for cost totals, without its link to the deleted run.
    assert sql("SELECT run_id FROM ops.model_calls WHERE call_id = %s", call) == [(None,)]


def test_old_access_log_rows_and_model_calls_are_trimmed():
    student, old_call, new_call = new_student(), uuid.uuid4(), uuid.uuid4()
    long_ago = datetime.now(UTC) - timedelta(days=401)
    for call, when in ((old_call, long_ago), (new_call, datetime.now(UTC))):
        sql(
            "INSERT INTO ops.model_calls (call_id, called_at, component, provider, model,"
            " usage_complete, outcome, latency_ms) VALUES (%s, %s, 'intent', 'deepseek',"
            " 'deepseek-flash', false, 'success', 5)",
            call, when,
        )  # fmt: skip
    view = uuid.uuid4()
    sql(
        "INSERT INTO ops.access_log (id, viewer_id, student_id, reason, viewed_at)"
        " VALUES (%s, %s, %s, 'test', %s)",
        view, uuid.uuid4(), student, long_ago,
    )  # fmt: skip

    asyncio.run(with_retention(lambda retention, _: retention.trim_logs(datetime.now(UTC))))

    calls = {r[0] for r in sql("SELECT call_id FROM ops.model_calls WHERE call_id = ANY(%s)",
                                [old_call, new_call])}  # fmt: skip
    assert calls == {new_call}
    assert sql("SELECT count(*) FROM ops.access_log WHERE id = %s", view)[0][0] == 0


def test_partitions_are_made_ahead_and_making_them_again_is_safe():
    today = date.today()

    async def body(retention, _):
        await retention.ensure_partitions(today)
        return await retention.ensure_partitions(today)

    again = asyncio.run(with_retention(body))
    assert again == []
    for ahead in range(4):
        month = _month(today, ahead)
        name = f"ops.model_calls_{month.year}_{month.month:02d}"
        assert sql("SELECT to_regclass(%s)", name) != [(None,)]


def test_deleting_an_account_leaves_nothing_that_names_the_student():
    student = new_student()
    email = sql("SELECT email FROM app.students WHERE id = %s", student)[0][0]
    thread, run = chat(student, days_ago=0)
    sql(
        "INSERT INTO ops.model_calls (call_id, called_at, run_id, component, provider, model,"
        " input_tokens, output_tokens, usage_complete, cost_usd, outcome, latency_ms)"
        " VALUES (%s, now(), %s,"
        " 'orchestrator', 'deepseek', 'deepseek-flash', 100, 20, true, 0.001, 'success', 5)",
        uuid.uuid4(), run,
    )  # fmt: skip
    sql(
        "INSERT INTO app.run_events (run_id, key, kind, payload) VALUES (%s, 'k', 'done', %s)",
        run, Jsonb({"status": "completed"}),
    )  # fmt: skip
    sql(
        "INSERT INTO app.quiz_posts (id, run_id, student_id, topic, quiz)"
        " VALUES (%s, %s, %s, 'heaps', 'quiz')",
        uuid.uuid4(), run, student,
    )  # fmt: skip
    sql(
        "INSERT INTO app.review_schedule (student_id, topic, ease, interval_days, repetitions,"
        " due_date) VALUES (%s, 'heaps', 2.5, 1, 0, current_date)",
        student,
    )  # fmt: skip
    sql(
        "INSERT INTO ops.activity_events (event_id, student_id, thread_id, event_type,"
        " occurred_at) VALUES (%s, %s, %s, 'message_sent', now())",
        uuid.uuid4(), student, thread,
    )  # fmt: skip
    sql(
        "INSERT INTO ops.access_log (id, viewer_id, student_id, reason, viewed_at)"
        " VALUES (%s, %s, %s, 'test', now())",
        uuid.uuid4(), uuid.uuid4(), student,
    )  # fmt: skip
    before = sql("SELECT coalesce(sum(runs), 0) FROM ops.deleted_usage")[0][0]

    async def body(retention, persistence):
        await _record_turn(persistence.checkpointer, thread, "my secret question", "answer")
        await Memory(persistence.store).save(student, thread, {"strengths": ["heaps"]}, "ep")
        done = await retention.delete_account(student)
        left = await persistence.store.asearch(namespace(student), limit=10)
        return done, left

    done, left = asyncio.run(with_retention(body))
    assert done is True and left == []
    # Every row in every table, read as text: none holds the student's ID or email.
    tables = sql(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables"
        " WHERE table_schema IN ('app', 'ops', 'lg_checkpoint', 'lg_store')"
        " AND table_type = 'BASE TABLE'"
    )
    found = []
    for (table,) in tables:
        for needle in (str(student), email, str(thread)):
            hits = sql(f"SELECT count(*) FROM {table} t WHERE t::text LIKE %s", f"%{needle}%")
            if hits[0][0]:
                found.append((table, needle))
    assert found == []
    assert sql("SELECT coalesce(sum(runs), 0) FROM ops.deleted_usage")[0][0] == before + 1


def test_deleting_an_unknown_account_does_nothing():
    done = asyncio.run(with_retention(lambda r, _: r.delete_account(uuid.uuid4())))
    assert done is False
