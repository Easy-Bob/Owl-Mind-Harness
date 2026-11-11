"""The run queue and the worker that runs it (M4-B).

The API saves a run and adds it to stream:runs. Workers read the stream as one consumer group,
so each run goes to one worker. A message a dead worker left unacknowledged is taken over after
the claim lapses. After 3 deliveries a run goes to stream:runs:dead and is failed. Background
jobs run here too, each under a Redis lease so one worker runs it at a time.

Usage: uv run python -m owl_mind.worker.runs
"""

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from owl_mind.core import metrics
from owl_mind.core.budget import RedisBudgetLedger
from owl_mind.core.config import get_settings
from owl_mind.core.runtime import RunRequest, Services, execute_run
from owl_mind.services.run_store import FINISHED, WAITING

logger = logging.getLogger(__name__)

STREAM = "stream:runs"
DEAD = "stream:runs:dead"
GROUP = "runners"
# Longer than any run can take (the longest time limit is 200 seconds), so a slow run is never
# taken over while its worker is still on it.
CLAIM_MS = 10 * 60 * 1000
MAX_DELIVERIES = 3
INTERRUPTED = "This answer was interrupted. Please ask again."
# Read by the Admin Reliability page (M4-F): when each worker last reported, and each job's last
# success and failure. Hashes keyed by worker name and by job.
HEARTBEATS = "workers:heartbeat"
JOB_STATUS = "jobs:status"
HEARTBEAT_SECONDS = 15
# Old entries are trimmed so the stream doesn't grow forever; far more than can be waiting.
_STREAM_MAX = 100_000
# A run taken over later than this is failed rather than resumed.
RESUME_WITHIN = timedelta(hours=1)


def _encode(request: RunRequest) -> str:
    data = asdict(request)
    for key in ("run_id", "thread_id", "student_id"):
        data[key] = str(data[key])
    return json.dumps(data)


def _decode(raw: str) -> RunRequest:
    data = json.loads(raw)
    for key in ("run_id", "thread_id", "student_id"):
        data[key] = uuid.UUID(data[key])
    return RunRequest(**data)


class RedisRunQueue:
    def __init__(self, client: Redis) -> None:
        self.client = client

    async def ensure_group(self) -> None:
        try:
            await self.client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue(self, request: RunRequest) -> None:
        await self.client.xadd(STREAM, {"run": _encode(request)}, maxlen=_STREAM_MAX)

    async def next(self, consumer: str, block_ms: int, claim_ms: int = CLAIM_MS):
        """The next message for this worker: an abandoned one first, then a new one."""
        _, stale, _ = await self.client.xautoclaim(
            STREAM, GROUP, consumer, claim_ms, start_id="0", count=1
        )
        if stale:
            return stale[0]
        found = await self.client.xreadgroup(
            GROUP, consumer, {STREAM: ">"}, count=1, block=block_ms or None
        )
        return found[0][1][0] if found else None

    async def deliveries(self, message_id: bytes | str) -> int:
        pending = await self.client.xpending_range(
            STREAM, GROUP, min=message_id, max=message_id, count=1
        )
        return pending[0]["times_delivered"] if pending else 1

    async def ack(self, message_id) -> None:
        await self.client.xack(STREAM, GROUP, message_id)

    async def dead_letter(self, message_id, raw: str, deliveries: int) -> None:
        await self.client.xadd(DEAD, {"run": raw, "deliveries": str(deliveries)}, maxlen=10_000)
        await self.ack(message_id)

    async def depth(self) -> tuple[int, int]:
        """Runs not yet read by any worker, and runs read but not yet acknowledged."""
        for group in await self.client.xinfo_groups(STREAM):
            name = group["name"].decode() if isinstance(group["name"], bytes) else group["name"]
            if name == GROUP:
                return int(group.get("lag") or 0), int(group.get("pending") or 0)
        return 0, 0


async def fail_run(services: Services, run_id: uuid.UUID, message: str) -> None:
    await services.runs.set_run_status(run_id, "failed")
    try:
        await services.events.publish(run_id, "error", {"message": message})
    except Exception as exc:
        logger.warning("Live event lost: %s", type(exc).__name__)


async def handle(
    services: Services,
    queue: RedisRunQueue,
    message: tuple,
    run: Callable[[Services, RunRequest], Awaitable[None]] = execute_run,
) -> str:
    """Run one queued message. Returns what happened, for tests and logs."""
    message_id, fields = message
    raw = fields[b"run"].decode() if isinstance(fields.get(b"run"), bytes) else fields["run"]
    try:
        request = _decode(raw)
    except ValueError, TypeError, KeyError:
        # A malformed entry can never succeed, so it goes straight to the dead letters.
        await queue.dead_letter(message_id, raw, 0)
        metrics.DEAD_LETTERS.labels("malformed").inc()
        return "malformed"
    deliveries = await queue.deliveries(message_id)
    if deliveries > MAX_DELIVERIES:
        await queue.dead_letter(message_id, raw, deliveries)
        await fail_run(services, request.run_id, INTERRUPTED)
        metrics.DEAD_LETTERS.labels("deliveries").inc()
        return "dead"
    status = await services.runs.status_of(request.run_id)
    if status == "queued" and request.queued_at is not None:
        metrics.QUEUE_WAIT.observe(max(0.0, time.time() - request.queued_at))
    if status in (*FINISHED, WAITING):
        await queue.ack(message_id)
        return "already done"
    step = "ran"
    if status == "running":
        # Taken over from a worker that stopped mid-run (M4-C).
        found = await services.runs.takeover(request.run_id)
        if found is None or datetime.now(UTC) - found.created_at > RESUME_WITHIN:
            # Too old to be worth finishing: the student has long since moved on.
            await fail_run(services, request.run_id, INTERRUPTED)
            await queue.ack(message_id)
            return "interrupted"
        metrics.TAKEOVERS.labels("resumed" if found.setup is not None else "restarted").inc()
        if found.setup is not None:
            # The intent step had finished: go on from the last checkpoint, with the same setup,
            # so the question isn't added twice and finished tutor calls aren't paid for again.
            request = replace(request, setup=found.setup, from_checkpoint=True)
            step = "resumed"
        else:
            # Stopped before the intent step chose a setup, so nothing reached the thread yet.
            step = "restarted"
    await run(services, request)
    await queue.ack(message_id)
    return step


async def worker_status(queue: RedisRunQueue, dead_limit: int = 50) -> dict[str, Any]:
    """For the Admin Reliability page (M4-F): workers, jobs, the queue and the dead letters.

    A dead letter holds the queued request, which includes the student's message, so only its
    run ID, attempt count and time are read out of it.
    """
    now = time.time()
    beats = await queue.client.hgetall(HEARTBEATS)
    workers = sorted(
        ({"worker": k.decode(), "seconds_ago": round(now - float(v), 1)} for k, v in beats.items()),
        key=lambda w: w["seconds_ago"],
    )
    jobs: dict[str, dict[str, float | None]] = {}
    for key, value in (await queue.client.hgetall(JOB_STATUS)).items():
        name, _, outcome = key.decode().rpartition(":")
        jobs.setdefault(name, {"success": None, "failure": None})[outcome] = float(value)
    dead = []
    for entry_id, fields in await queue.client.xrevrange(DEAD, count=dead_limit):
        try:
            run_id = json.loads(fields[b"run"].decode())["run_id"]
        except ValueError, KeyError, TypeError:
            run_id = None
        dead.append(
            {
                "run_id": run_id,
                "deliveries": int(fields.get(b"deliveries", b"0")),
                "at": int(entry_id.decode().split("-")[0]) / 1000,
            }
        )
    lag, pending = await queue.depth()
    return {
        "workers": workers,
        "jobs": [{"job": name, **times} for name, times in sorted(jobs.items())],
        "dead_letters": dead,
        "dead_letter_total": await queue.client.xlen(DEAD),
        "queue": {"lag": lag, "pending": pending},
    }


class JobLease:
    """One holder at a time for a background job, across every worker."""

    def __init__(self, client: Redis, name: str, owner: str, ttl_ms: int) -> None:
        self.client, self.key, self.owner, self.ttl_ms = client, f"lease:{name}", owner, ttl_ms

    async def hold(self) -> bool:
        if await self.client.set(self.key, self.owner, nx=True, px=self.ttl_ms):
            return True
        current = await self.client.get(self.key)
        if current is not None and current.decode() == self.owner:
            await self.client.pexpire(self.key, self.ttl_ms)
            return True
        return False


async def _job_status(client: Redis | None, name: str, outcome: str) -> None:
    if client is not None:
        try:
            await client.hset(JOB_STATUS, f"{name}:{outcome}", str(time.time()))
        except Exception as exc:
            logger.warning("Job status not kept: %s", type(exc).__name__)


async def job_loop(
    lease: JobLease, interval_s: float, job: Callable[[], Awaitable[Any]], name: str
) -> None:
    while True:
        try:
            if await lease.hold():
                await job()
                metrics.JOB_RUNS.labels(name, "success").inc()
                metrics.JOB_LAST_SUCCESS.labels(name).set_to_current_time()
                await _job_status(lease.client, name, "success")
        except Exception as exc:
            logger.warning("%s job failed: %s", name, type(exc).__name__)
            metrics.JOB_RUNS.labels(name, "failure").inc()
            await _job_status(lease.client, name, "failure")
        await asyncio.sleep(interval_s)


async def heartbeat(queue: RedisRunQueue, name: str) -> None:
    """Report this worker as alive, and sample the queue's depth for the dashboards."""
    while True:
        metrics.WORKER_HEARTBEAT.set_to_current_time()
        try:
            await queue.client.hset(HEARTBEATS, name, str(time.time()))
            lag, pending = await queue.depth()
            metrics.QUEUE_LAG.set(lag)
            metrics.QUEUE_PENDING.set(pending)
            spend = await RedisBudgetLedger(queue.client).spent(
                uuid.UUID(int=0), datetime.now(UTC).date()
            )
            limit = get_settings().global_daily_usd * 1_000_000
            metrics.BUDGET_GLOBAL_SHARE.set(spend.global_micro_usd / limit if limit else 0)
        except Exception as exc:
            logger.warning("Heartbeat not sent: %s", type(exc).__name__)
        await asyncio.sleep(HEARTBEAT_SECONDS)


async def consume(services: Services, queue: RedisRunQueue, consumer: str) -> None:
    while True:
        try:
            message = await queue.next(consumer, block_ms=5000)
            if message is not None:
                await handle(services, queue, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Run worker step failed: %s", type(exc).__name__)
            await asyncio.sleep(1)


async def serve() -> None:
    from prometheus_client import start_http_server

    from owl_mind.api.app import _production_services
    from owl_mind.worker.jobs import jobs

    settings = get_settings()
    name = f"{socket.gethostname()}-{os.getpid()}"
    if settings.worker_metrics_port:
        # Each worker on a host needs its own port: set WORKER_METRICS_PORT per worker.
        start_http_server(settings.worker_metrics_port)
    async with _production_services() as services:
        queue = services.queue
        await queue.ensure_group()
        tasks = [
            asyncio.create_task(consume(services, queue, f"{name}-{n}"))
            for n in range(settings.worker_concurrency)
        ]
        tasks.append(asyncio.create_task(heartbeat(queue, name)))
        for job_name, interval, job in jobs(services):
            lease = JobLease(queue.client, job_name, name, int(interval * 3 * 1000))
            tasks.append(asyncio.create_task(job_loop(lease, interval, job, job_name)))
        await asyncio.gather(*tasks)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
