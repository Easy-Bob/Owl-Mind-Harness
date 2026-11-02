"""Needs the compose services: docker compose up -d --wait

Run with: uv run pytest -m integration
"""

import base64
import json
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psycopg
import pytest
import redis
import uvicorn
from langchain_core.messages import HumanMessage

from owl_mind.api.app import app
from owl_mind.core.config import get_settings
from owl_mind.core.model import ModelTier, get_chat_model
from owl_mind.core.telemetry import RUN_ID_KEY, TELEMETRY_STREAM
from owl_mind.worker import telemetry as worker
from owl_mind.worker.telemetry import ensure_group, process_once
from tests.fake_deepseek import USAGE, FakeDeepSeek

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMETHEUS = "http://localhost:9091"
GRAFANA = "http://localhost:3002"
ROW_COLUMNS = {
    "call_id",
    "called_at",
    "run_id",
    "component",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
    "usage_complete",
    "cost_usd",
    "latency_ms",
    "outcome",
    "prompt_version",
}


@pytest.fixture(scope="module", autouse=True)
def migrated():
    settings = get_settings()
    database, cache = urlsplit(settings.database_url), urlsplit(settings.redis_url)
    assert (database.hostname, database.port) == ("localhost", 5433), "not the compose Postgres"
    assert (cache.hostname, cache.port) == ("localhost", 6380), "not the compose Redis"
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def client():
    return redis.Redis.from_url(get_settings().redis_url)


@pytest.fixture
def db():
    with psycopg.connect(get_settings().database_url) as connection:
        yield connection


def drain(client, db) -> None:
    ensure_group(client)
    while process_once(client, db):
        pass


def event(**overrides) -> dict:
    return {
        "call_id": str(uuid.uuid4()),
        "called_at": "2026-11-02T02:00:00+00:00",  # Monday, peak
        "run_id": None,
        "component": "test",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "latency_ms": 10,
        "outcome": "success",
        "prompt_version": None,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_hit_tokens": 1_000_000,
        "cache_miss_tokens": 0,
        "reasoning_tokens": None,
        **overrides,
    }


def send(client, payload: dict) -> None:
    client.xadd(TELEMETRY_STREAM, {"event": json.dumps(payload)})


def row(db, call_id: str) -> tuple | None:
    return db.execute(
        "SELECT cost_usd, usage_complete, reasoning_tokens, input_tokens"
        " FROM ops.model_calls WHERE call_id = %s",
        (call_id,),
    ).fetchone()


def test_prices_are_seeded_for_both_models_and_bands(db):
    rows = db.execute("SELECT model, band FROM ops.model_prices").fetchall()

    assert {("deepseek-flash", "peak"), ("deepseek-flash", "off_peak")} <= set(rows)
    assert {("deepseek-v4-pro", "peak"), ("deepseek-v4-pro", "off_peak")} <= set(rows)


def test_model_calls_is_partitioned_with_a_default(db):
    rows = db.execute(
        "SELECT child.relname FROM pg_inherits"
        " JOIN pg_class child ON child.oid = inhrelid"
        " JOIN pg_class parent ON parent.oid = inhparent"
        " WHERE parent.relname = 'model_calls'"
    ).fetchall()

    assert {"model_calls_default", "model_calls_2026_11"} <= {name for (name,) in rows}


def test_real_model_call_becomes_one_row(client, db):
    drain(client, db)
    run_id = str(uuid.uuid4())
    model = get_chat_model(ModelTier.SMALL, component="integration")
    FakeDeepSeek([{"role": "assistant", "content": "hi"}]).attach(model)

    model.invoke([HumanMessage("hello")], config={"metadata": {RUN_ID_KEY: run_id}})
    drain(client, db)

    rows = db.execute(
        "SELECT component, model, input_tokens, output_tokens, outcome FROM ops.model_calls"
        " WHERE run_id = %s",
        (run_id,),
    ).fetchall()
    assert rows == [
        (
            "integration",
            "deepseek-flash",
            USAGE["prompt_tokens"],
            USAGE["completion_tokens"],
            "success",
        )
    ]


def test_rows_have_no_column_for_message_text(db):
    rows = db.execute(
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_schema = 'ops' AND table_name = 'model_calls'"
    ).fetchall()

    assert {name for (name,) in rows} == ROW_COLUMNS


def test_stored_cost_uses_the_peak_band_price(client, db):
    payload = event(cache_hit_tokens=500_000, cache_miss_tokens=500_000, output_tokens=1_000_000)
    send(client, payload)
    drain(client, db)

    # Flash at peak: 0.5 x 0.006 + 0.5 x 0.30 + 1 x 1.20
    assert row(db, payload["call_id"])[0] == Decimal("1.353")


def test_stored_cost_uses_the_off_peak_band_price(client, db):
    payload = event(called_at="2026-11-07T02:00:00+00:00")  # Saturday
    send(client, payload)
    drain(client, db)

    # Flash off-peak: 1 x 0.003 + 1 x 0.60
    assert row(db, payload["call_id"])[0] == Decimal("0.603")


def test_old_cost_is_unchanged_by_a_later_price(client, db):
    model = f"test-model-{uuid.uuid4()}"
    insert_price = "INSERT INTO ops.model_prices VALUES (%s, 'peak', %s, 1, 1, 1)"
    db.execute(insert_price, (model, datetime(2026, 11, 1, tzinfo=UTC)))
    db.commit()
    first = event(model=model, cache_hit_tokens=0, cache_miss_tokens=0, output_tokens=1_000_000)
    send(client, first)
    drain(client, db)

    db.execute(insert_price, (model, datetime(2026, 11, 3, tzinfo=UTC)))
    db.execute(
        "UPDATE ops.model_prices SET output_usd_per_mtok = 9 WHERE model = %s"
        " AND effective_from = %s",
        (model, datetime(2026, 11, 3, tzinfo=UTC)),
    )
    db.commit()
    later = dict(first, call_id=str(uuid.uuid4()), called_at="2026-11-04T02:00:00+00:00")
    send(client, later)
    drain(client, db)

    assert row(db, first["call_id"])[0] == Decimal(1)
    assert row(db, later["call_id"])[0] == Decimal(9)


def test_missing_usage_writes_a_row_and_marks_the_gap(client, db):
    payload = event(
        input_tokens=None, output_tokens=None, cache_hit_tokens=None, cache_miss_tokens=None
    )
    send(client, payload)
    drain(client, db)

    assert row(db, payload["call_id"]) == (None, False, None, None)


def test_unknown_price_writes_a_row_with_no_cost(client, db):
    payload = event(model="no-such-model")
    send(client, payload)
    drain(client, db)

    assert row(db, payload["call_id"])[0] is None


def test_redelivered_event_does_not_make_a_second_row(client, db):
    payload = event()
    send(client, payload)
    send(client, payload)
    drain(client, db)

    count = db.execute(
        "SELECT count(*) FROM ops.model_calls WHERE call_id = %s", (payload["call_id"],)
    ).fetchone()
    assert count == (1,)


def test_event_waits_in_the_stream_while_the_consumer_is_down(client, db):
    drain(client, db)
    payload = event()
    send(client, payload)
    assert row(db, payload["call_id"]) is None

    drain(client, db)

    assert row(db, payload["call_id"]) is not None


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"call_id": "missing fields"}),
        json.dumps(event(call_id="not-a-uuid")),
        json.dumps(event(outcome="not-an-outcome")),
    ],
    ids=["not-json", "missing-fields", "bad-uuid", "bad-outcome"],
)
def test_malformed_event_is_dropped_and_keeps_good_rows_around_it(client, db, bad):
    drain(client, db)
    before, after = event(), event()
    send(client, before)
    client.xadd(TELEMETRY_STREAM, {"event": bad})
    send(client, after)

    drain(client, db)

    assert row(db, before["call_id"]) is not None
    assert row(db, after["call_id"]) is not None
    assert client.xpending(TELEMETRY_STREAM, "telemetry")["pending"] == 0


def test_event_left_unacknowledged_by_a_crash_is_recovered(client, db, monkeypatch):
    drain(client, db)
    payload = event()
    send(client, payload)
    # A consumer reads the event, then dies before writing it.
    client.xreadgroup("telemetry", "crashed-consumer", {TELEMETRY_STREAM: ">"}, count=1)
    assert client.xpending(TELEMETRY_STREAM, "telemetry")["pending"] == 1
    monkeypatch.setattr(worker, "_STALE_MS", 0)

    drain(client, db)

    assert row(db, payload["call_id"]) is not None
    assert client.xpending(TELEMETRY_STREAM, "telemetry")["pending"] == 0


def wait_for(check, seconds: float = 40):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        last = check()
        if last:
            return last
        time.sleep(1)
    raise AssertionError(f"timed out, last result: {last}")


@pytest.fixture
def running_app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    get_settings.cache_clear()
    # Prometheus runs in a container, so the app must listen beyond localhost.
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(lambda: server.started, seconds=10)
    yield
    server.should_exit = True
    thread.join(timeout=5)


def test_prometheus_and_grafana_see_the_app_metrics(running_app):
    component = f"prometheus-{uuid.uuid4().hex[:8]}"
    model = get_chat_model(ModelTier.SMALL, component=component)
    FakeDeepSeek([{"role": "assistant", "content": "hi"}]).attach(model)
    model.invoke([HumanMessage("hello")])

    def target_is_up():
        data = httpx.get(f"{PROMETHEUS}/api/v1/targets").json()
        return [t for t in data["data"]["activeTargets"] if t["health"] == "up"]

    wait_for(target_is_up)

    query = {"query": f'llm_calls_total{{component="{component}"}}'}

    def prometheus_has_metric():
        return httpx.get(f"{PROMETHEUS}/api/v1/query", params=query).json()["data"]["result"]

    assert wait_for(prometheus_has_metric)[0]["value"][1] == "1"

    token = base64.b64encode(b"admin:owl_dev").decode()
    response = httpx.get(
        f"{GRAFANA}/api/datasources/proxy/uid/prometheus/api/v1/query",
        params=query,
        headers={"Authorization": f"Basic {token}"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["result"][0]["value"][1] == "1"
