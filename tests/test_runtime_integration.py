"""Needs the compose services: docker compose up -d --wait

The same run store and event bus checks as the in-memory versions, plus whole turns on real
Postgres and Redis.
"""

import asyncio
import json
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
import redis
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.api.app import create_app
from owl_mind.api.deps import current_student
from owl_mind.core import telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.events import InMemoryEventBus, RedisEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import get_chat_model
from owl_mind.core.persistence import open_persistence
from owl_mind.core.runtime import Services
from owl_mind.services.accounts import PostgresAccountStore, RedisLoginLimiter
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_store import ActiveRunExists, InMemoryRunStore, PostgresRunStore
from owl_mind.services.study_store import PostgresStudyStore
from owl_mind.worker.telemetry import ensure_group, process_once
from tests.fake_deepseek import FakeDeepSeek
from tests.test_filesystem import call, say, task
from tests.test_runtime import intent, kinds, parse_sse, text_of

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def new_student() -> uuid.UUID:
    student_id = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as connection:
        connection.execute(
            "INSERT INTO app.students (id, email) VALUES (%s, %s)",
            (student_id, f"{student_id}@test.invalid"),
        )
    return student_id


def in_loop(body):
    return asyncio.run(body())


@asynccontextmanager
async def postgres_runs():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    yield PostgresRunStore(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


@asynccontextmanager
async def memory_runs():
    yield InMemoryRunStore()


@asynccontextmanager
async def redis_bus():
    client = Redis.from_url(get_settings().redis_url)
    yield RedisEventBus(client)
    await client.aclose()


@asynccontextmanager
async def memory_bus():
    yield InMemoryEventBus()


# -- both run stores behave the same -----------------------------------------


@pytest.fixture(params=[memory_runs, postgres_runs], ids=["memory", "postgres"])
def runs(request):
    return request.param


def test_thread_is_scoped_to_one_student(runs):
    alice, bob = new_student(), new_student()

    async def body():
        async with runs() as store:
            thread = await store.create_thread(alice, "title")
            return (
                await store.get_thread(alice, thread.id),
                await store.get_thread(bob, thread.id),
                await store.list_threads(bob),
            )

    own, other, listed = in_loop(body)
    assert own.title == "title"
    assert (other, listed) == (None, [])


def test_one_unfinished_run_per_thread(runs):
    alice = new_student()

    async def body():
        async with runs() as store:
            thread = await store.create_thread(alice, "t")
            first = await store.create_run(thread.id)
            with pytest.raises(ActiveRunExists):
                await store.create_run(thread.id)
            await store.set_run_status(first, "completed")
            return await store.create_run(thread.id)

    assert in_loop(body)


def test_run_is_scoped_to_one_student(runs):
    alice, bob = new_student(), new_student()

    async def body():
        async with runs() as store:
            thread = await store.create_thread(alice, "t")
            run_id = await store.create_run(thread.id)
            return await store.get_run(alice, run_id), await store.get_run(bob, run_id)

    own, other = in_loop(body)
    assert own.status == "queued"
    assert other is None


def test_unfinished_runs_are_failed_and_finished_runs_are_kept(runs):
    alice = new_student()

    async def body():
        async with runs() as store:
            done = await store.create_thread(alice, "a")
            done_run = await store.create_run(done.id)
            await store.set_run_status(done_run, "completed")
            active = await store.create_thread(alice, "b")
            active_run = await store.create_run(active.id)
            await store.fail_unfinished_runs()
            return (
                (await store.get_run(alice, done_run)).status,
                (await store.get_run(alice, active_run)).status,
            )

    assert in_loop(body) == ("completed", "failed")


def test_postgres_keeps_one_event_per_key():
    alice = new_student()

    async def body():
        async with postgres_runs() as store:
            thread = await store.create_thread(alice, "t")
            run_id = await store.create_run(thread.id)
            await store.record_event(run_id, "k", "usage", {"n": 1})
            await store.record_event(run_id, "k", "usage", {"n": 2})
            return run_id

    run_id = in_loop(body)
    with psycopg.connect(get_settings().database_url) as connection:
        rows = connection.execute(
            "SELECT kind, payload FROM app.run_events WHERE run_id = %s", (run_id,)
        ).fetchall()
    assert rows == [("usage", {"n": 2})]


# -- both event buses behave the same ----------------------------------------


@pytest.fixture(params=[memory_bus, redis_bus], ids=["memory", "redis"])
def bus(request):
    return request.param


def test_events_can_be_read_from_any_point(bus):
    run_id = uuid.uuid4()

    async def body():
        async with bus() as events:
            await events.publish(run_id, "queued", {})
            await events.publish(run_id, "text_delta", {"text": "hi"})
            await events.publish(run_id, "done", {"status": "completed"})
            everything = await events.read(run_id, "0", 100)
            rest = await events.read(run_id, everything[0][0], 100)
            return everything, rest

    everything, rest = in_loop(body)
    assert [kind for _, kind, _ in everything] == ["queued", "text_delta", "done"]
    assert rest == everything[1:]


def test_reading_with_nothing_new_returns_empty_after_waiting(bus):
    async def body():
        async with bus() as events:
            return await events.read(uuid.uuid4(), "0", 50)

    assert in_loop(body) == []


def test_waiting_reader_gets_a_later_event(bus):
    run_id = uuid.uuid4()

    async def body():
        async with bus() as events:
            reader = asyncio.create_task(events.read(run_id, "0", 3000))
            await asyncio.sleep(0.05)
            await events.publish(run_id, "queued", {})
            return await reader

    assert [kind for _, kind, _ in in_loop(body)] == ["queued"]


def test_redis_stream_expires_within_the_hour():
    run_id = uuid.uuid4()

    async def body():
        async with redis_bus() as events:
            await events.publish(run_id, "queued", {})

    in_loop(body)
    ttl = redis.Redis.from_url(get_settings().redis_url).ttl(f"stream:run:{run_id}")
    assert 0 < ttl <= 3600


# -- whole turns on real services --------------------------------------------


class RealWorld:
    def __init__(self, student: uuid.UUID, *replies: dict) -> None:
        self.fake = FakeDeepSeek(list(replies))
        settings = get_settings()

        def get_model(tier, *, component, max_tokens):
            model = get_chat_model(tier, component=component, max_tokens=max_tokens)
            self.fake.attach(model)
            return model

        @asynccontextmanager
        async def factory():
            engine = create_async_engine(sqlalchemy_url(settings.database_url))
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            client = Redis.from_url(settings.redis_url, socket_timeout=15)
            async with open_persistence(settings.database_url) as persistence:
                yield Services(
                    accounts=PostgresAccountStore(sessions),
                    limiter=RedisLoginLimiter(client),
                    runs=PostgresRunStore(sessions),
                    study=PostgresStudyStore(sessions),
                    events=RedisEventBus(client),
                    store=persistence.store,
                    checkpointer=persistence.checkpointer,
                    intent=IntentStep(
                        lambda tier, c, m: get_model(tier, component=c, max_tokens=m)
                    ),
                    get_model=get_model,
                )
            await client.aclose()
            await engine.dispose()

        self.app = create_app(factory)
        self.app.dependency_overrides[current_student] = lambda: student


def test_full_turn_on_real_services_survives_a_restart():
    student = new_student()
    world = RealWorld(
        student,
        intent(problem_help=0.9),
        task("practice_coach"),
        call("build_hint", level=1),
        say("report"),
        say("the final answer"),
    )
    with TestClient(world.app) as client:
        started = client.post("/threads", json={"message": "help"}).json()
        response = client.get(f"/runs/{started['run_id']}/stream")

    events = parse_sse(response.text)
    assert text_of(events) == "the final answer"
    assert kinds(events)[-1] == "done"

    # A new app, as after a restart.
    with TestClient(RealWorld(student).app) as client:
        history = client.get(f"/threads/{started['thread_id']}/messages").json()
        listed = client.get("/threads").json()

    assert [m["content"] for m in history] == ["help", "the final answer"]
    assert listed[0]["id"] == started["thread_id"]

    with psycopg.connect(get_settings().database_url) as connection:
        status = connection.execute(
            "SELECT status FROM app.runs WHERE id = %s", (started["run_id"],)
        ).fetchone()
        recorded = connection.execute(
            "SELECT kind FROM app.run_events WHERE run_id = %s", (started["run_id"],)
        ).fetchall()
    assert status == ("completed",)
    assert ("subagent_start",) in recorded
    assert ("text_delta",) not in recorded


def test_reconnect_resumes_from_a_real_redis_event_id():
    student = new_student()
    world = RealWorld(student, intent(concept_explain=0.9), say("answer"))
    with TestClient(world.app) as client:
        run_id = client.post("/threads", json={"message": "x"}).json()["run_id"]
        first = client.get(f"/runs/{run_id}/stream").text
        event_id = first.split("\n", 1)[0].removeprefix("id: ")
        rest = client.get(f"/runs/{run_id}/stream", headers={"Last-Event-ID": event_id}).text

    assert parse_sse(rest) == parse_sse(first)[1:]


def test_restart_during_a_run_marks_it_failed_and_keeps_the_thread():
    student = new_student()

    async def seed():
        async with postgres_runs() as store:
            thread = await store.create_thread(student, "left running")
            return await store.create_run(thread.id)

    run_id = in_loop(seed)

    with TestClient(RealWorld(student).app) as client:
        assert client.get("/threads").json()[0]["title"] == "left running"
        events = parse_sse(client.get(f"/runs/{run_id}/stream").text)

    assert events[-1] == ("done", {"status": "failed"})


# -- activity through the real consumer ---------------------------------------


def drain(client, db) -> None:
    ensure_group(client)
    while process_once(client, db):
        pass


def test_activity_reaches_the_tables_once_with_daily_counts():
    student, thread = uuid.uuid4(), uuid.uuid4()
    settings = get_settings()
    telemetry.record_activity("session_started", student, thread)
    telemetry.record_activity("message_sent", student, thread)
    telemetry.record_activity("message_sent", student, thread)
    client = redis.Redis.from_url(settings.redis_url)
    # A redelivered event must not count twice.
    last = client.xrevrange("stream:telemetry", count=1)[0][1][b"event"]
    client.xadd("stream:telemetry", {"event": last})

    with psycopg.connect(settings.database_url) as db:
        drain(client, db)
        events = db.execute(
            "SELECT event_type, count(*) FROM ops.activity_events WHERE student_id = %s"
            " GROUP BY event_type ORDER BY event_type",
            (student,),
        ).fetchall()
        daily = db.execute(
            "SELECT day, sessions_started, messages_sent FROM ops.activity_daily"
            " WHERE student_id = %s",
            (student,),
        ).fetchall()

    assert events == [("message_sent", 2), ("session_started", 1)]
    assert daily == [(datetime.now(UTC).date(), 1, 2)]


def test_database_refuses_any_other_activity_type():
    student, thread = uuid.uuid4(), uuid.uuid4()
    settings = get_settings()
    bad = {
        "kind": "activity",
        "event_id": str(uuid.uuid4()),
        "event_type": "page_viewed",
        "student_id": str(student),
        "thread_id": str(thread),
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    client = redis.Redis.from_url(settings.redis_url)
    client.xadd("stream:telemetry", {"event": json.dumps(bad)})

    with psycopg.connect(settings.database_url) as db:
        drain(client, db)
        row = db.execute(
            "SELECT count(*) FROM ops.activity_events WHERE student_id = %s", (student,)
        ).fetchone()

    assert row == (0,)
