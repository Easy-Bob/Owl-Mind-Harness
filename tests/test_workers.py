"""Offline: with a queue, the API hands runs to workers; a worker handles each message once."""

import asyncio
import uuid

from owl_mind.core.runtime import RunRequest
from owl_mind.worker import runs as worker
from owl_mind.worker.runs import INTERRUPTED, _decode, _encode, handle
from tests.test_filesystem import ALICE
from tests.test_runtime import World, intent, say, start
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.


class FakeQueue:
    """Stands in for Redis: records what's queued, acknowledged and dead-lettered."""

    def __init__(self, deliveries: int = 1) -> None:
        self.queued: list[RunRequest] = []
        self.acked: list = []
        self.dead: list = []
        self._deliveries = deliveries

    async def enqueue(self, request: RunRequest) -> None:
        self.queued.append(request)

    async def deliveries(self, message_id) -> int:
        return self._deliveries

    async def ack(self, message_id) -> None:
        self.acked.append(message_id)

    async def dead_letter(self, message_id, raw: str, deliveries: int) -> None:
        self.dead.append((message_id, deliveries))
        self.acked.append(message_id)


def queued_world() -> tuple[World, FakeQueue]:
    world, queue = World(intent(concept_explain=0.9), say("answer")), FakeQueue()
    world.services = world.services.__class__(**{**vars(world.services), "queue": queue})
    return world, queue


def test_with_a_queue_the_api_queues_the_run_and_runs_nothing_itself():
    world, queue = queued_world()
    with world.client() as client:
        started = start(client, "explain heaps")

    [request] = queue.queued
    assert str(request.run_id) == started["run_id"] and request.message == "explain heaps"
    assert world.fake.requests == []
    assert world.runs.runs[request.run_id].status == "queued"


def test_a_queued_run_survives_encoding():
    request = RunRequest(uuid.uuid4(), uuid.uuid4(), ALICE, "hi", resume={"approved": True})

    assert _decode(_encode(request)) == request


def message(request: RunRequest) -> tuple:
    return (b"1-0", {b"run": _encode(request).encode()})


def queued_run(world: World) -> RunRequest:
    async def make():
        thread = await world.runs.create_thread(ALICE, "t")
        return RunRequest(await world.runs.create_run(thread.id), thread.id, ALICE, "explain heaps")

    return asyncio.run(make())


def test_a_worker_runs_a_queued_run_once_and_acknowledges_it():
    world, queue = queued_world()
    request = queued_run(world)

    step = asyncio.run(handle(world.services, queue, message(request)))

    assert step == "ran"
    assert world.runs.runs[request.run_id].status == "completed"
    assert queue.acked == [b"1-0"]
    again = asyncio.run(handle(world.services, queue, message(request)))
    assert again == "already done"
    assert len(world.fake.requests) == 2  # Intent and orchestrator, once.


def test_a_run_another_worker_left_mid_way_is_failed_not_left_running():
    world, queue = queued_world()
    request = queued_run(world)
    asyncio.run(world.runs.set_run_status(request.run_id, "running"))

    step = asyncio.run(handle(world.services, queue, message(request)))

    assert step == "interrupted"
    assert world.runs.runs[request.run_id].status == "failed"
    [(_, kind, data)] = world.bus.events[request.run_id]
    assert (kind, data) == ("error", {"message": INTERRUPTED})


def test_a_run_delivered_more_than_three_times_goes_to_the_dead_letters():
    world, _ = queued_world()
    queue = FakeQueue(deliveries=4)
    request = queued_run(world)

    step = asyncio.run(handle(world.services, queue, message(request)))

    assert step == "dead"
    assert queue.dead == [(b"1-0", 4)]
    assert world.runs.runs[request.run_id].status == "failed"
    assert world.fake.requests == []


def test_a_malformed_entry_goes_to_the_dead_letters_at_once():
    world, queue = queued_world()

    step = asyncio.run(handle(world.services, queue, (b"9-0", {b"run": b"not json"})))

    assert step == "malformed"
    assert queue.dead == [(b"9-0", 0)]


def test_every_background_job_is_listed_for_the_workers():
    from owl_mind.worker.jobs import jobs

    world, _ = queued_world()
    names = [name for name, _, _ in jobs(world.services)]

    assert names == ["approvals"]  # Memory and the forum aren't set up in this test world.
    assert worker.MAX_DELIVERIES == 3
