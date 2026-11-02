"""Needs the compose services: docker compose up -d --wait

Admin SQL against real tables, checked against seeded rows.
"""

import asyncio
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.services.admin_store import PostgresAdminStore
from owl_mind.services.models import sqlalchemy_url
from tests.guard_rails import check_access_logged

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET_TITLE = "my private first message"
SECRET_TODO = "my private todo text"


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


def call_row(run_id, **overrides) -> dict:
    return {
        "call_id": uuid.uuid4(),
        "called_at": datetime.now(UTC),
        "run_id": run_id,
        "component": "orchestrator",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_hit_tokens": 75,
        "cache_miss_tokens": 25,
        "reasoning_tokens": None,
        "usage_complete": True,
        "cost_usd": Decimal("0.5"),
        "latency_ms": 5,
        "outcome": "success",
        "prompt_version": "m1",
        **overrides,
    }


class Seed:
    """One student and one run with two model calls, one of them with unknown usage."""

    def __init__(self) -> None:
        self.student, self.thread, self.run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        self.email = f"seed-{self.student.hex}@example.com"
        self.component = f"component-{self.student.hex[:8]}"
        calls = [
            call_row(self.run, component=self.component),
            call_row(
                self.run,
                component=self.component,
                input_tokens=None,
                output_tokens=None,
                cache_hit_tokens=None,
                cache_miss_tokens=None,
                usage_complete=False,
                cost_usd=None,
                outcome="error",
            ),
        ]
        with psycopg.connect(get_settings().database_url) as db:
            db.execute(
                "INSERT INTO app.students (id, email) VALUES (%s, %s)", (self.student, self.email)
            )
            db.execute(
                "INSERT INTO app.threads (id, student_id, title) VALUES (%s, %s, %s)",
                (self.thread, self.student, SECRET_TITLE),
            )
            db.execute(
                "INSERT INTO app.runs (id, thread_id, status, route, task_kind)"
                " VALUES (%s, %s, 'completed', 'orchestrated', 'learn')",
                (self.run, self.thread),
            )
            events = [
                ("000001:tool_start", "tool_start", {"name": "build_hint", "args": "secret"}),
                ("000002:todo_delta", "todo_delta", {"todos": [{"content": SECRET_TODO}]}),
                ("000003:done", "done", {"status": "completed"}),
            ]
            for key, kind, payload in events:
                db.execute(
                    "INSERT INTO app.run_events (run_id, key, kind, payload)"
                    " VALUES (%s, %s, %s, %s)",
                    (self.run, key, kind, Jsonb(payload)),
                )
            for call in calls:
                db.execute(
                    """
                    INSERT INTO ops.model_calls (
                        call_id, called_at, run_id, component, provider, model, input_tokens,
                        output_tokens, cache_hit_tokens, cache_miss_tokens, reasoning_tokens,
                        usage_complete, cost_usd, latency_ms, outcome, prompt_version
                    ) VALUES (
                        %(call_id)s, %(called_at)s, %(run_id)s, %(component)s, %(provider)s,
                        %(model)s, %(input_tokens)s, %(output_tokens)s, %(cache_hit_tokens)s,
                        %(cache_miss_tokens)s, %(reasoning_tokens)s, %(usage_complete)s,
                        %(cost_usd)s, %(latency_ms)s, %(outcome)s, %(prompt_version)s
                    )
                    """,
                    call,
                )


def with_store(body):
    async def main():
        engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
        try:
            return await body(PostgresAdminStore(async_sessionmaker(engine)))
        finally:
            await engine.dispose()

    return asyncio.run(main())


def log_rows(run_id) -> list[tuple]:
    with psycopg.connect(get_settings().database_url) as db:
        return db.execute(
            "SELECT viewer_id, student_id, reason FROM ops.access_log WHERE run_id = %s", (run_id,)
        ).fetchall()


def test_list_finds_a_run_by_email_or_id_and_writes_no_log():
    seed = Seed()

    async def body(store):
        return await store.list_requests(seed.email, 1, 25), await store.list_requests(
            str(seed.run), 1, 25
        )

    by_email, by_id = with_store(body)

    assert by_email["total"] == by_id["total"] == 1
    row = by_email["items"][0]
    assert row["id"] == str(seed.run)
    assert (row["route"], row["task_kind"], row["status"]) == ("orchestrated", "learn", "completed")
    assert row["model_calls"] == 2
    assert row["tokens"] == 110
    assert row["cost_usd"] == 0.5
    assert row["unknown_usage"] == 1
    assert log_rows(seed.run) == []


def test_list_holds_no_message_text():
    seed = Seed()

    result = with_store(lambda store: store.list_requests(seed.email, 1, 25))

    assert SECRET_TITLE not in str(result)


def test_search_cannot_inject_sql():
    result = with_store(lambda store: store.list_requests("' OR 1=1 --", 1, 25))

    assert result["total"] == 0


def test_list_pages_newest_first():
    seed = Seed()
    newer = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as db:
        db.execute("UPDATE app.runs SET status = 'failed' WHERE id = %s", (seed.run,))
        db.execute(
            "INSERT INTO app.runs (id, thread_id, status, created_at)"
            " VALUES (%s, %s, 'completed', %s)",
            (newer, seed.thread, datetime.now(UTC) + timedelta(minutes=1)),
        )

    async def body(store):
        return await store.list_requests(seed.email, 1, 1), await store.list_requests(
            seed.email, 2, 1
        )

    first, second = with_store(body)

    assert first["total"] == second["total"] == 2
    assert [first["items"][0]["id"], second["items"][0]["id"]] == [str(newer), str(seed.run)]


def test_run_with_no_model_call_still_appears_with_no_cost():
    seed = Seed()
    quiet = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as db:
        db.execute("UPDATE app.runs SET status = 'failed' WHERE id = %s", (seed.run,))
        db.execute(
            "INSERT INTO app.runs (id, thread_id, status, route)"
            " VALUES (%s, %s, 'completed', 'rule')",
            (quiet, seed.thread),
        )

    result = with_store(lambda store: store.list_requests(str(quiet), 1, 25))

    row = result["items"][0]
    assert (row["route"], row["model_calls"], row["tokens"], row["cost_usd"]) == (
        "rule",
        0,
        None,
        None,
    )


def test_detail_matches_model_calls_and_writes_exactly_one_log_row():
    seed = Seed()
    viewer = uuid.uuid4()
    before = len(log_rows(seed.run))

    detail = with_store(lambda store: store.get_request(seed.run, viewer))

    assert check_access_logged(before, len(log_rows(seed.run))) == []
    assert log_rows(seed.run) == [(viewer, seed.student, "request detail")]
    assert detail["email"] == seed.email
    assert [call["input_tokens"] for call in detail["calls"]] == [100, None]
    assert [call["cost_usd"] for call in detail["calls"]] == [0.5, None]
    assert detail["calls"][1]["reasoning_tokens"] is None


def test_detail_holds_no_message_text_or_tool_arguments():
    seed = Seed()

    detail = with_store(lambda store: store.get_request(seed.run, uuid.uuid4()))

    assert SECRET_TITLE not in str(detail)
    assert SECRET_TODO not in str(detail)
    assert "secret" not in str(detail)
    assert [event["kind"] for event in detail["events"]] == ["tool_start", "todo_delta", "done"]
    assert detail["events"][1]["data"] == {"count": 1}


def test_unknown_run_returns_nothing_and_logs_nothing():
    run_id = uuid.uuid4()

    assert with_store(lambda store: store.get_request(run_id, uuid.uuid4())) is None
    assert log_rows(run_id) == []


def test_summary_groups_match_seeded_calls_and_mark_unknowns():
    seed = Seed()

    summary = with_store(lambda store: store.model_call_summary(30))

    row = next(r for r in summary["by_component"] if r["name"] == seed.component)
    assert (row["calls"], row["input_tokens"], row["output_tokens"], row["cost_usd"]) == (
        2,
        100,
        10,
        0.5,
    )
    assert row["unknown_usage"] == 1
    assert {"success", "error"} <= {outcome["name"] for outcome in summary["outcomes"]}
    assert 0 <= summary["cache_hit_rate"] <= 1
    assert "learn" in {group["name"] for group in summary["by_task_kind"]}
    assert summary["totals"]["unknown_usage"] >= 1
    assert all("run_id" in run for run in summary["top_runs"])


def test_summary_window_leaves_out_older_calls():
    seed = Seed()
    with psycopg.connect(get_settings().database_url) as db:
        db.execute(
            "UPDATE ops.model_calls SET called_at = now() - interval '40 days'"
            " WHERE component = %s",
            (seed.component,),
        )

    recent, older = with_store(lambda store: _both(store))

    assert seed.component not in {row["name"] for row in recent["by_component"]}
    assert seed.component in {row["name"] for row in older["by_component"]}


async def _both(store):
    return await store.model_call_summary(30), await store.model_call_summary(60)
