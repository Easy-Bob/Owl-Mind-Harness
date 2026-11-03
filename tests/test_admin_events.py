"""Offline: retrieval and handoff steps recorded as metadata-only run events for Admin."""

import asyncio
import dataclasses
import json
import uuid

import pytest

from owl_mind.forum.improve import Improver
from owl_mind.forum.replies import ReplyPoller
from owl_mind.services.admin_store import safe_event
from owl_mind.services.run_events import RunEventLog
from owl_mind.services.run_store import InMemoryRunStore
from tests.test_auth import ALICE_EMAIL, logged_in
from tests.test_auth import settings as auth_settings  # noqa: F401  Applied to every test.
from tests.test_forum_replies import reply
from tests.test_handoff import FakeForum, approve, ask_for_tutor, draft_of, handoff_world
from tests.test_improve import rewrite

SECRET = "zqx-secret-marker"


def timeline(world, kind: str = "handoff") -> list[dict]:
    """What Admin would show, in key order. Later steps' keys also order across runs."""
    events = sorted(
        (
            (key, stored_kind, payload)
            for (run_id, key), (stored_kind, payload) in world.runs.events.items()
        ),
        key=lambda event: event[0],
    )
    return [safe_event(k, payload) for _, k, payload in events if k == kind]


def stored_text(world) -> str:
    return json.dumps([safe_event(kind, payload) for kind, payload in world.runs.events.values()])


# -- safe_event -------------------------------------------------------------


def test_retrieval_keeps_only_its_named_fields():
    payload = {
        "result_count": 2,
        "top_score": 0.03,
        "corpus_version": 4,
        "cache_hit": False,
        "latency_ms": 7,
        "query": SECRET,
        "results": [SECRET],
    }

    assert safe_event("retrieval", payload) == {
        "result_count": 2,
        "top_score": 0.03,
        "corpus_version": 4,
        "cache_hit": False,
        "latency_ms": 7,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"step": SECRET},
        {"step": "drafted", "body": SECRET, "title": SECRET},
        {"step": "drafted", "removed": {SECRET: 1, "email": SECRET}},
        {"step": "posted", "url": f"https://evil.example/{SECRET}"},
        {"step": "approved", "url": "https://github.com/o/f/discussions/1"},
        {"step": "post_failed", "status": SECRET, "attempts": SECRET},
        {"step": "fallback", "reason": SECRET},
        {"step": "posted", "posted_as": SECRET},
        {"step": "improved", "remaining": True},
    ],
)
def test_handoff_drops_any_value_that_could_be_text(payload):
    safe = safe_event("handoff", payload)

    assert SECRET not in json.dumps(safe)
    assert set(safe) <= {"step", "removed"} or "remaining" not in safe


def test_handoff_keeps_counts_words_and_a_forum_url():
    payload = {
        "step": "posted",
        "url": "https://github.com/owner/forum/discussions/42",
        "posted_as": "bot",
        "removed": {"email": 2, "phone": 1},
    }

    assert safe_event("handoff", payload) == payload


# -- the handoff path -------------------------------------------------------


def test_a_handoff_records_each_step_in_order_with_no_text():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client, f"talk to a TA, mail {ALICE_EMAIL} {SECRET}")
        draft = draft_of(client, thread_id)
        revised = approve(client, draft, body=f"{SECRET} password = hunter22").json()
        approve(client, revised)
        poller = ReplyPoller(world.handoff_store, forum, world.clock, world.events)
        forum.replies["D_1"] = [reply(1), reply(2)]
        world.clock.advance(minutes=5)
        asyncio.run(poller.poll_due())

    steps = timeline(world)
    assert [step["step"] for step in steps] == [
        "drafted",
        "similar_found",
        "needs_review",
        "approved",
        "posted",
        "replies_arrived",
    ]
    # The email was in both the title and the body.
    assert steps[0]["removed"] == {"known detail": 2}
    assert steps[1] == {"step": "similar_found", "count": 1}
    assert steps[2]["removed"] == {"secret": 1}
    assert steps[4] == {
        "step": "posted",
        "url": "https://github.com/owner/forum/discussions/1",
        "posted_as": "bot",
    }
    assert steps[5] == {"step": "replies_arrived", "count": 2}
    for secret in (SECRET, ALICE_EMAIL, "hunter22", "reply 1"):
        assert secret not in stored_text(world)


def test_a_failed_post_shows_the_failure_then_the_fallback():
    forum = FakeForum()
    forum.failures = 10
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        approve(client, draft_of(client, thread_id))
        for _ in range(2):
            asyncio.run(world.services.handoffs.retry_pending())

    steps = [s for s in timeline(world) if s["step"] in ("post_failed", "fallback")]
    assert steps == [
        {"step": "post_failed", "attempts": 1, "status": "retrying"},
        {"step": "post_failed", "attempts": 2, "status": "retrying"},
        {"step": "post_failed", "attempts": 3, "status": "fallback"},
        {"step": "fallback", "reason": "retries_used"},
    ]


def test_rate_limits_and_cancels_are_recorded():
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        first, _ = ask_for_tutor(client)
        approve(client, draft_of(client, first))
        second, _ = ask_for_tutor(client)
        draft = draft_of(client, second)
        approve(client, draft)
        client.post(f"/handoffs/{draft['id']}/cancel")

    steps = [s["step"] for s in timeline(world)]
    assert "rate_limited" in steps
    assert steps[-1] == "cancelled"


def test_a_rewrite_records_its_counts_and_its_failures():
    world = handoff_world(FakeForum())
    world.fake.replies.extend([rewrite(body=f"mail {SECRET}@x.org"), rewrite(body="")])
    improver = Improver(world.handoff_store, world.services.get_model, world.events)
    world.services = dataclasses.replace(world.services, improver=improver)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        fields = {"title": draft["title"], "body": draft["body"]}
        client.post(f"/handoffs/{draft['id']}/improve", json=fields)
        client.post(f"/handoffs/{draft['id']}/improve", json=fields)

    steps = [s for s in timeline(world) if s["step"].startswith("improve")]
    assert steps == [
        {"step": "improved", "removed": {"email": 1}, "remaining": 2},
        {"step": "improve_failed"},
    ]
    assert SECRET not in stored_text(world)


def test_polling_records_failures_and_when_it_stops():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        approve(client, draft_of(client, thread_id))
    poller = ReplyPoller(world.handoff_store, forum, world.clock, world.events)
    forum.down = True
    world.clock.advance(minutes=5)
    asyncio.run(poller.poll_due())
    forum.down = False
    forum.gone.add("D_1")
    world.clock.advance(hours=1)
    asyncio.run(poller.poll_due())

    steps = timeline(world)[-2:]
    assert steps == [
        {"step": "poll_failed", "failures": 1},
        {"step": "polling_stopped", "reason": "discussion_gone"},
    ]


class BrokenRuns(InMemoryRunStore):
    async def record_event(self, run_id, key, kind, payload) -> None:
        if kind == "handoff":
            raise RuntimeError("database down")
        await super().record_event(run_id, key, kind, payload)


def test_a_failing_recorder_never_fails_a_handoff():
    forum = FakeForum()
    world = handoff_world(forum)
    world.services.handoffs.events = RunEventLog(BrokenRuns(), world.clock)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        posted = approve(client, draft_of(client, thread_id))

    assert posted.json()["status"] == "posted"
    assert len(forum.posts) == 1


def test_later_steps_sort_after_the_runs_own_events():
    runs = InMemoryRunStore()
    run_id = uuid.uuid4()
    asyncio.run(runs.record_event(run_id, "000009:done", "done", {"status": "completed"}))
    asyncio.run(RunEventLog(runs).record(run_id, "handoff", {"step": "posted"}))

    keys = sorted(key for (_, key) in runs.events)

    assert keys[0] == "000009:done"
    assert keys[1].startswith("t")


def test_an_event_with_no_run_is_skipped():
    runs = InMemoryRunStore()

    asyncio.run(RunEventLog(runs).record(None, "handoff", {"step": "posted"}))

    assert runs.events == {}


def test_admin_route_filter_refuses_unknown_routes():
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        student = client.get("/admin/requests", params={"route": "handoff"})

    assert student.status_code == 403
