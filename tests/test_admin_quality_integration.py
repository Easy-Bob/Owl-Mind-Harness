"""Integration: the Quality and Reliability pages (M4-F) against seeded rows in real Postgres
and Redis. The local database holds other rows too, so figures are checked by what the seed
adds. Seeded text that must never be shown is planted in titles, events and the dead letters."""

import asyncio
import json
import uuid

import psycopg
import pytest
from redis.asyncio import Redis

from owl_mind.core.config import get_settings
from owl_mind.core.runtime import RunRequest
from owl_mind.worker.runs import DEAD, HEARTBEATS, JOB_STATUS, RedisRunQueue, _encode, worker_status
from tests.test_admin_integration import (
    SECRET_TITLE,
    SECRET_TODO,
    Seed,
    migrated,  # noqa: F401  Applied to all.
    with_store,
)

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

SECRET_MESSAGE = "zqx-secret-dead-letter-message"


def quality_seed() -> Seed:
    return Seed(
        extra_events=[
            ("000004:integrity", "integrity", {"result": "blocked", "reason": SECRET_TODO}),
            ("000005:answer_stats", "answer_stats", {"citations": 2}),
        ]
    )


def by_version(view, version: str) -> dict:
    return next((r for r in view["live"] if r["prompt_version"] == version), {})


def test_quality_counts_integrity_and_citations_by_prompt_version():
    version = "m1"  # The seed's model calls carry it.
    before = with_store(lambda store: store.quality(30))
    quality_seed()
    after = with_store(lambda store: store.quality(30))

    old, new = by_version(before, version), by_version(after, version)
    assert new["runs"] - old.get("runs", 0) == 1
    assert new["integrity_blocked"] - old.get("integrity_blocked", 0) == 1
    assert new["learn_cited"] - old.get("learn_cited", 0) == 1
    shown = json.dumps(after)
    assert SECRET_TITLE not in shown and SECRET_TODO not in shown


def test_reliability_lists_stopped_runs_by_day_and_reason():
    before = with_store(lambda store: store.reliability(14))
    seed = Seed(extra_events=[("000004:done", "done", {"status": "stopped", "reason": "x_limit"})])
    with psycopg.connect(get_settings().database_url) as db:
        db.execute("UPDATE app.runs SET status = 'stopped' WHERE id = %s", (seed.run,))
    after = with_store(lambda store: store.reliability(14))

    def stopped(view) -> int:
        return sum(r["runs"] for r in view["trouble"] if r["reason"] == "x_limit")

    assert stopped(after) - stopped(before) == 1
    assert SECRET_TITLE not in json.dumps(after)


def test_worker_status_reads_the_dead_letters_without_their_message():
    async def body():
        client = Redis.from_url(get_settings().redis_url)
        queue = RedisRunQueue(client)
        await queue.ensure_group()
        name = f"test-{uuid.uuid4().hex[:6]}"
        run_id = uuid.uuid4()
        raw = _encode(RunRequest(run_id, uuid.uuid4(), uuid.uuid4(), SECRET_MESSAGE))
        entry = await client.xadd(DEAD, {"run": raw, "deliveries": "4"})
        await client.hset(HEARTBEATS, name, "1")
        await client.hset(JOB_STATUS, f"{name}-job:success", "2")
        try:
            return run_id, name, await worker_status(queue)
        finally:
            await client.xdel(DEAD, entry)
            await client.hdel(HEARTBEATS, name)
            await client.hdel(JOB_STATUS, f"{name}-job:success")
            await client.aclose()

    run_id, name, status = asyncio.run(body())
    assert {"run_id": str(run_id), "deliveries": 4} in [
        {k: d[k] for k in ("run_id", "deliveries")} for d in status["dead_letters"]
    ]
    assert any(w["worker"] == name for w in status["workers"])
    assert {"job": f"{name}-job", "success": 2.0, "failure": None} in status["jobs"]
    assert SECRET_MESSAGE not in json.dumps(status)
