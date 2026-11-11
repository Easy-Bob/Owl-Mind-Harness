"""Offline: a resent message gets the first copy's run, and a run a worker left mid-way is
resumed from its checkpoint, with nothing asked or paid for twice (M4-C)."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from owl_mind.core.middleware import FanOutMiddleware
from owl_mind.core.runtime import RunRequest, execute_run, read_history
from owl_mind.services.run_store import ActiveRunExists
from owl_mind.worker import jobs as worker_jobs
from owl_mind.worker.runs import INTERRUPTED, handle
from tests.fake_deepseek import _is_review
from tests.test_filesystem import ALICE, say, task
from tests.test_runtime import World, intent, start, stream, text_of
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.
from tests.test_workers import FakeQueue, message


class WorkerStopped(Exception):
    pass


def crash_on_agent_call(world: World, number: int) -> None:
    """The nth agent request fails (reviews don't count), leaving the thread's checkpoint just
    before it, where a worker that died at that moment would leave it."""
    seen = 0
    original = world.fake.handler

    def handler(request):
        nonlocal seen
        if not _is_review(json.loads(request.content)):
            seen += 1
            if seen == number:
                world.fake.handler = original
                raise WorkerStopped
        return original(request)

    world.fake.handler = handler


def a_queued_run(world: World, text: str) -> RunRequest:
    async def make():
        thread = await world.runs.create_thread(ALICE, "t")
        return RunRequest(await world.runs.create_run(thread.id), thread.id, ALICE, text)

    return asyncio.run(make())


def humans(world: World, thread_id: uuid.UUID) -> list[str]:
    history = asyncio.run(read_history(world.saver, thread_id))
    return [m["content"] for m in history if m["role"] == "user"]


def test_a_run_left_after_a_tutor_finished_resumes_without_asking_the_tutor_again():
    world = World(
        intent(concept_explain=0.9),
        task("concept_tutor"),
        say("tutor report text"),
        say("Heaps keep the smallest item on top."),
    )
    world.services = world.services.__class__(**{**vars(world.services), "queue": FakeQueue()})
    request = a_queued_run(world, "explain heaps")
    # Intent step, orchestrator, tutor: the worker stops as the orchestrator is asked again.
    crash_on_agent_call(world, 4)
    asyncio.run(execute_run(world.services, request))
    # A dead worker leaves the run as running, with its checkpoint where the call failed.
    asyncio.run(world.runs.set_run_status(request.run_id, "running"))
    world.bus.events[request.run_id].clear()
    asked_before = len(world.fake.agent_requests)
    spent_before = world.budgets.runs[request.run_id]

    step = asyncio.run(handle(world.services, FakeQueue(), message(request)))

    assert step == "resumed"
    assert world.runs.runs[request.run_id].status == "completed"
    # Only the orchestrator's last call is made again; no intent step and no second tutor call.
    assert len(world.fake.agent_requests) == asked_before + 1
    assert humans(world, request.thread_id) == ["explain heaps"]
    events = world.bus.events[request.run_id]
    assert "Heaps keep the smallest item on top." in text_of((k, d) for _, k, d in events)
    # The spend before the stop still counts toward the run's budget.
    assert world.budgets.runs[request.run_id] > spent_before


def test_a_run_left_before_the_intent_step_finished_starts_again():
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("r"), say("Answer."))
    request = a_queued_run(world, "explain heaps")
    asyncio.run(world.runs.set_run_status(request.run_id, "running"))

    step = asyncio.run(handle(world.services, FakeQueue(), message(request)))

    assert step == "restarted"
    assert world.runs.runs[request.run_id].status == "completed"
    assert humans(world, request.thread_id) == ["explain heaps"]


def test_a_run_left_more_than_an_hour_ago_is_failed():
    world = World()
    request = a_queued_run(world, "explain heaps")
    asyncio.run(world.runs.set_run_status(request.run_id, "running"))
    world.runs.created[request.run_id] = datetime.now(UTC) - timedelta(hours=2)

    step = asyncio.run(handle(world.services, FakeQueue(), message(request)))

    assert step == "interrupted"
    assert world.runs.runs[request.run_id].status == "failed"
    [(_, kind, data)] = world.bus.events[request.run_id]
    assert (kind, data) == ("error", {"message": INTERRUPTED})


def test_the_same_message_sent_twice_gets_one_run():
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("r"), say("Answer."))
    body = {"message": "explain heaps", "client_message_id": str(uuid.uuid4())}
    with world.client() as client:
        first = client.post("/threads", json=body)
        stream(client, first.json()["run_id"])
        again = client.post("/threads", json=body)
        follow_up = {"message": "more", "client_message_id": str(uuid.uuid4())}
        thread_id = first.json()["thread_id"]
        world.fake.replies += [say("x"), say("y"), say("z")]
        sent = client.post(f"/threads/{thread_id}/messages", json=follow_up)
        resent = client.post(f"/threads/{thread_id}/messages", json=follow_up)

    assert again.status_code == 201 and again.json() == first.json()
    assert resent.json() == sent.json()
    assert len(world.runs.runs) == 2
    assert len(world.runs.threads) == 1


def test_a_message_without_an_id_still_works_as_before():
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("r"), say("Answer."))
    with world.client() as client:
        started = start(client, "explain heaps")
        events = stream(client, started["run_id"])

    assert text_of(events) == "Answer."


def test_the_message_id_is_kept_once_per_student():
    world = World()

    async def body():
        thread = await world.runs.create_thread(ALICE, "t")
        sent = uuid.uuid4()
        run_id = await world.runs.create_run(thread.id, sent)
        await world.runs.set_run_status(run_id, "completed")
        with pytest.raises(ActiveRunExists):
            await world.runs.create_run(thread.id, sent)
        return run_id, await world.runs.find_by_message(ALICE, sent)

    run_id, found = asyncio.run(body())
    assert found.id == run_id


def test_a_reopened_chat_finds_the_answer_still_being_written():
    world = World()
    with world.client() as client:
        # Made after startup, which fails unfinished runs when there are no workers.
        request = a_queued_run(world, "explain heaps")
        busy = client.get(f"/threads/{request.thread_id}/active")
        asyncio.run(world.runs.set_run_status(request.run_id, "completed"))
        idle = client.get(f"/threads/{request.thread_id}/active")
        world.student = uuid.uuid4()
        other = client.get(f"/threads/{request.thread_id}/active")

    assert busy.json()["run_id"] == str(request.run_id)
    assert idle.status_code == 404 and other.status_code == 404


def test_fan_out_counts_the_calls_a_resumed_turn_already_finished():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    fan_out = FanOutMiddleware(frozenset({"concept_tutor"}))
    calls = [{"name": "task", "args": {"subagent_type": "concept_tutor"}, "id": i} for i in "abc"]
    fan_out._seed(
        [
            HumanMessage("earlier"),
            AIMessage("", tool_calls=[dict(calls[0])]),
            ToolMessage("old", tool_call_id="a"),
            HumanMessage("explain heaps"),
            AIMessage("", tool_calls=[dict(calls[1]), dict(calls[2])]),
            ToolMessage("done", tool_call_id="b"),
            ToolMessage("refused", tool_call_id="c", status="error"),
        ]
    )

    assert fan_out.calls == 1 and fan_out.called == {"concept_tutor": 1}


def test_an_answered_approval_that_never_got_its_resume_run_gets_one():
    world = World()
    queue = FakeQueue()
    world.services = world.services.__class__(**{**vars(world.services), "queue": queue})

    async def body():
        thread = await world.runs.create_thread(ALICE, "t")
        paused = await world.runs.create_run(thread.id)
        await world.runs.set_run_status(paused, "waiting_approval")
        asked = {"kind": "commit_plan", "summary": "s", "items": []}
        setup = {"task_kind": "plan", "subagents": ["planner"], "paused_run": str(paused)}
        approval = await world.approvals.create(paused, ALICE, thread.id, asked, setup)
        await world.approvals.decide(approval.id, "approved")
        # The API stopped here, before it made the resume run.
        soon = await worker_jobs.resume_answered(world.services)
        world.approvals.decided_at[approval.id] -= worker_jobs.RESUME_GRACE
        later = await worker_jobs.resume_answered(world.services)
        again = await worker_jobs.resume_answered(world.services)
        return paused, approval, soon, later, again

    paused, approval, soon, later, again = asyncio.run(body())
    assert (soon, later, again) == (0, 1, 0)
    [request] = queue.queued
    assert request.resume == {"approved": True} and request.setup["paused_run"] == str(paused)
    assert world.approvals.resumed[approval.id] == request.run_id
    assert world.runs.runs[paused].status == "completed"
