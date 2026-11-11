"""Integration: the run queue and job leases on real Redis, with two workers and a crash."""

import asyncio
import uuid

import pytest
from redis.asyncio import Redis

from owl_mind.core.config import get_settings
from owl_mind.core.runtime import RunRequest
from owl_mind.worker import runs as worker
from owl_mind.worker.runs import DEAD, GROUP, STREAM, JobLease, RedisRunQueue, handle
from tests.test_workers import queued_world

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

CLAIM_MS = 300  # Short, so a stopped worker's run is taken over within the test.


async def with_queue(body):
    client = Redis.from_url(get_settings().redis_url)
    await client.delete(STREAM, DEAD)
    queue = RedisRunQueue(client)
    await queue.ensure_group()
    try:
        return await body(queue, client)
    finally:
        await client.delete(STREAM, DEAD)
        await client.aclose()


def test_two_workers_finish_every_run_once_even_when_one_stops_mid_run():
    world, _ = queued_world()
    finished: list[uuid.UUID] = []
    started: set[uuid.UUID] = set()

    async def run(services, request):
        started.add(request.run_id)
        await asyncio.sleep(0.01)
        await services.runs.set_run_status(request.run_id, "completed")
        finished.append(request.run_id)

    async def body(queue, client):
        requests = []
        for _ in range(50):
            thread = await world.runs.create_thread(uuid.uuid4(), "t")
            run_id = await world.runs.create_run(thread.id)
            requests.append(RunRequest(run_id, thread.id, uuid.uuid4(), "hi"))
            await queue.enqueue(requests[-1])
        # A worker that takes one run and stops before finishing it, as if it crashed.
        crashed = await queue.next("crashed", block_ms=100, claim_ms=CLAIM_MS)
        lost = worker._decode(crashed[1][b"run"].decode())
        await asyncio.sleep(CLAIM_MS / 1000 + 0.1)

        async def drain(consumer):
            while len({*finished}) < 50:
                message = await queue.next(consumer, block_ms=50, claim_ms=CLAIM_MS)
                if message is not None:
                    await handle(world.services, queue, message, run=run)

        await asyncio.wait_for(asyncio.gather(drain("w1"), drain("w2")), timeout=30)
        pending = await client.xpending(STREAM, GROUP)
        return requests, lost, pending

    requests, lost, pending = asyncio.run(with_queue(body))
    assert sorted(finished) == sorted(r.run_id for r in requests)  # Each once.
    assert lost.run_id in finished  # The stopped worker's run was taken over.
    assert pending["pending"] == 0
    assert all(world.runs.runs[r.run_id].status == "completed" for r in requests)


def test_a_message_delivered_too_often_is_dead_lettered_and_its_run_failed():
    world, _ = queued_world()

    async def body(queue, client):
        thread = await world.runs.create_thread(uuid.uuid4(), "t")
        run_id = await world.runs.create_run(thread.id)
        await queue.enqueue(RunRequest(run_id, thread.id, uuid.uuid4(), "hi"))
        message = None
        for n in range(worker.MAX_DELIVERIES + 1):  # Each worker stops without acknowledging.
            await asyncio.sleep(CLAIM_MS / 1000 + 0.05)
            message = await queue.next(f"w{n}", block_ms=100, claim_ms=CLAIM_MS)
        step = await handle(world.services, queue, message)
        return run_id, step, await client.xlen(DEAD), await client.xpending(STREAM, GROUP)

    run_id, step, dead, pending = asyncio.run(with_queue(body))
    assert step == "dead"
    assert dead == 1 and pending["pending"] == 0
    assert world.runs.runs[run_id].status == "failed"


def test_a_job_lease_has_one_holder_until_it_lapses():
    async def body(queue, client):
        name = f"test-{uuid.uuid4().hex[:6]}"
        first, second = JobLease(client, name, "a", 200), JobLease(client, name, "b", 200)
        holds = [await first.hold(), await second.hold(), await first.hold()]
        await asyncio.sleep(0.3)
        holds.append(await second.hold())
        await client.delete(f"lease:{name}")
        return holds

    assert asyncio.run(with_queue(body)) == [True, False, True, True]


def test_a_worker_reports_its_heartbeat_the_queue_depth_and_each_jobs_outcome():
    from owl_mind.core import metrics
    from owl_mind.worker.runs import HEARTBEATS, JOB_STATUS, heartbeat, job_loop

    world, _ = queued_world()
    name = f"test-{uuid.uuid4().hex[:6]}"

    async def body(queue, client):
        for _ in range(3):
            thread = await world.runs.create_thread(uuid.uuid4(), "t")
            await queue.enqueue(RunRequest(await world.runs.create_run(thread.id), thread.id,
                                           uuid.uuid4(), "hi"))  # fmt: skip
        await queue.next("w1", block_ms=100)  # One read and not yet acknowledged.
        beat = asyncio.create_task(heartbeat(queue, name))

        async def fails():
            raise RuntimeError("job broke")

        lease = JobLease(client, f"{name}-job", name, 5000)
        jobs = [
            asyncio.create_task(job_loop(lease, 60, lambda: asyncio.sleep(0), f"{name}-ok")),
            asyncio.create_task(
                job_loop(JobLease(client, f"{name}-bad", name, 5000), 60, fails, f"{name}-bad")
            ),  # fmt: skip
        ]
        await asyncio.sleep(0.3)
        for task in (beat, *jobs):
            task.cancel()
        beats = await client.hget(HEARTBEATS, name)
        status = await client.hgetall(JOB_STATUS)
        await client.hdel(HEARTBEATS, name)
        await client.delete(f"lease:{name}-job", f"lease:{name}-bad")
        await client.hdel(JOB_STATUS, f"{name}-ok:success", f"{name}-bad:failure")
        return beats, {k.decode() for k in status}

    beats, status = asyncio.run(with_queue(body))
    assert beats is not None
    assert (metrics.QUEUE_LAG._value.get(), metrics.QUEUE_PENDING._value.get()) == (2, 1)
    assert {f"{name}-ok:success", f"{name}-bad:failure"} <= status
