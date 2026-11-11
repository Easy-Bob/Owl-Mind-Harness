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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

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
        await self.client.xadd(STREAM, {"run": _encode(request)})

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
        return "malformed"
    deliveries = await queue.deliveries(message_id)
    if deliveries > MAX_DELIVERIES:
        await queue.dead_letter(message_id, raw, deliveries)
        await fail_run(services, request.run_id, INTERRUPTED)
        return "dead"
    status = await services.runs.status_of(request.run_id)
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


async def job_loop(
    lease: JobLease, interval_s: float, job: Callable[[], Awaitable[Any]], name: str
) -> None:
    while True:
        try:
            if await lease.hold():
                await job()
        except Exception as exc:
            logger.warning("%s job failed: %s", name, type(exc).__name__)
        await asyncio.sleep(interval_s)


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


async def serve(concurrency: int = 4) -> None:
    from owl_mind.api.app import _production_services
    from owl_mind.worker.jobs import jobs

    name = f"{socket.gethostname()}-{os.getpid()}"
    async with _production_services() as services:
        queue = services.queue
        await queue.ensure_group()
        tasks = [
            asyncio.create_task(consume(services, queue, f"{name}-{n}")) for n in range(concurrency)
        ]
        for job_name, interval, job in jobs(services):
            lease = JobLease(queue.client, job_name, name, int(interval * 3 * 1000))
            tasks.append(asyncio.create_task(job_loop(lease, interval, job, job_name)))
        await asyncio.gather(*tasks)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
