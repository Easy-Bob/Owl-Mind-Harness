"""Offline: changes to a student's plan or progress wait for their approval on a card."""

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

from owl_mind.api.app import expire_approvals
from owl_mind.services.admin_store import safe_event
from owl_mind.services.study_store import ReviewState
from tests.test_filesystem import ALICE, BOB, call, say, task
from tests.test_runtime import (
    World,
    intent,
    kinds,
    start,
    stream,
    telemetry_events,  # noqa: F401  Applied to all.
    text_of,
)
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.

PLAN_ITEMS = [
    {"topic": "bfs", "note": "", "review_date": "2026-11-09"},
    {"topic": "heaps", "note": "", "review_date": "2026-11-03"},
]


def plan_world(*after: dict) -> tuple[World, uuid.UUID]:
    """A student with a proposed plan, and a planner that asks to commit it."""
    world = World()
    plan_id = asyncio.run(world.study.save_plan(ALICE, "Midterm plan", PLAN_ITEMS))
    world.study.reviews[(ALICE, "bfs")] = ReviewState(2.5, 6, 2, date(2026, 11, 9))
    world.fake.replies = [
        intent(study_plan=0.9),
        task("study_planner"),
        call("commit_study_plan", plan_id=str(plan_id)),
        *after,
    ]
    return world, plan_id


def plan_status(world: World, plan_id: uuid.UUID) -> str:
    return world.study.plans[plan_id][1].status


def paused(client, world):
    started = start(client, "save my midterm plan")
    return started, stream(client, started["run_id"])


# -- pausing -------------------------------------------------------------------


def test_a_commit_pauses_the_run_and_changes_nothing_until_the_student_answers():
    world, plan_id = plan_world()
    before = dict(world.study.reviews)
    with world.client() as client:
        started, events = paused(client, world)
        card = client.get(f"/threads/{started['thread_id']}/approval")

    assert events[-1] == ("done", {"status": "waiting_approval"})
    [request] = [data for kind, data in events if kind == "approval_request"]
    assert request["summary"] == "Commit 'Midterm plan', 2 items. 1 new topics join your reviews."
    assert request["items"] == [
        {"topic": "bfs", "review_date": "2026-11-09"},
        {"topic": "heaps", "review_date": "2026-11-03"},
    ]
    assert plan_status(world, plan_id) == "proposed"
    assert world.study.reviews == before
    assert world.runs.runs[uuid.UUID(started["run_id"])].status == "waiting_approval"
    assert card.status_code == 200
    assert card.json()["summary"] == request["summary"]


def test_a_waiting_student_can_send_nothing_new_until_they_answer():
    world, _ = plan_world()
    with world.client() as client:
        started, _ = paused(client, world)
        same = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "hi"})
        other = client.post("/threads", json={"message": "hi"})

    assert (same.status_code, other.status_code) == (409, 409)


# -- answering -------------------------------------------------------------------


def test_approving_resumes_the_run_and_commits_the_plan():
    world, plan_id = plan_world(say("committed it"), say("Your plan is saved."))
    with world.client() as client:
        started, _ = paused(client, world)
        resumed = client.post(f"/runs/{started['run_id']}/approve")
        events = stream(client, resumed.json()["run_id"])
        card = client.get(f"/threads/{started['thread_id']}/approval")

    assert resumed.status_code == 200
    assert text_of(events) == "Your plan is saved."
    assert events[-1] == ("done", {"status": "completed"})
    assert plan_status(world, plan_id) == "committed"
    assert world.study.reviews[(ALICE, "heaps")].due_date == date(2026, 11, 3)
    assert world.study.reviews[(ALICE, "bfs")].interval_days == 6  # Existing topics keep theirs.
    assert world.runs.runs[uuid.UUID(started["run_id"])].status == "completed"
    assert card.status_code == 404
    # The resumed run asks the intent step nothing: only the tutor and the orchestrator run.
    assert len(world.fake.requests) == 5


def test_rejecting_resumes_the_run_and_changes_nothing():
    world, plan_id = plan_world(say("not committed"), say("OK, I left your plan as it was."))
    before = dict(world.study.reviews)
    with world.client() as client:
        started, _ = paused(client, world)
        resumed = client.post(f"/runs/{started['run_id']}/reject")
        events = stream(client, resumed.json()["run_id"])

    assert text_of(events) == "OK, I left your plan as it was."
    assert plan_status(world, plan_id) == "proposed"
    assert world.study.reviews == before
    tool_results = [
        m["content"] for m in world.fake.requests[-2]["messages"] if m["role"] == "tool"
    ]
    assert any("The student said no" in text for text in tool_results)


def test_a_double_click_on_approve_applies_the_change_once():
    world, plan_id = plan_world(say("committed it"), say("Saved."))
    with world.client() as client:
        started, _ = paused(client, world)
        first = client.post(f"/runs/{started['run_id']}/approve")
        second = client.post(f"/runs/{started['run_id']}/approve")
        stream(client, first.json()["run_id"])

    assert first.status_code == 200
    assert second.status_code in (404, 409)
    assert plan_status(world, plan_id) == "committed"


def test_another_student_cannot_answer_or_see_the_card():
    world, plan_id = plan_world()
    with world.client() as client:
        started, _ = paused(client, world)
        world.student = BOB
        answers = [client.post(f"/runs/{started['run_id']}/{a}") for a in ("approve", "reject")]
        card = client.get(f"/threads/{started['thread_id']}/approval")

    assert [r.status_code for r in answers] == [404, 404]
    assert card.status_code == 404
    assert plan_status(world, plan_id) == "proposed"


# -- expiry ---------------------------------------------------------------------


def test_an_unanswered_approval_expires_and_the_chat_can_go_on():
    world, plan_id = plan_world(intent(concept_explain=0.9), say("Heaps keep order."))
    with world.client() as client:
        started, _ = paused(client, world)
        for approval_id, (owner, info) in list(world.approvals.rows.items()):
            past = datetime.now(UTC) - timedelta(minutes=1)
            world.approvals.rows[approval_id] = (owner, replace(info, expires_at=past))
        expired = client.portal.call(expire_approvals, world.services)
        late = client.post(f"/runs/{started['run_id']}/approve")
        next_run = client.post(
            f"/threads/{started['thread_id']}/messages", json={"message": "explain heaps"}
        )
        events = stream(client, next_run.json()["run_id"])

    assert expired == 1
    assert late.status_code == 404
    assert world.runs.runs[uuid.UUID(started["run_id"])].status == "stopped"
    assert plan_status(world, plan_id) == "proposed"
    assert text_of(events) == "Heaps keep order."


# -- resetting progress ------------------------------------------------------------


def reset_world(*arguments: dict, after=()) -> World:
    world = World()
    for topic in ("bfs", "heaps", "tries"):
        world.study.reviews[(ALICE, topic)] = ReviewState(2.5, 1, 1, date(2026, 11, 5))
    world.fake.replies = [
        intent(study_plan=0.9),
        task("study_planner"),
        call("reset_progress", **arguments[0]),
        *after,
    ]
    return world


def test_resetting_all_progress_names_how_many_topics_and_waits():
    world = reset_world({"all_topics": True}, after=(say("reset"), say("Done, fresh start.")))
    with world.client() as client:
        started, events = paused(client, world)
        [request] = [data for kind, data in events if kind == "approval_request"]
        assert len(world.study.reviews) == 3
        resumed = client.post(f"/runs/{started['run_id']}/approve")
        stream(client, resumed.json()["run_id"])

    assert request["summary"] == "Reset progress for all 3 topics."
    assert world.study.reviews == {}


def test_resetting_one_topic_clears_only_that_topic():
    world = reset_world({"topic": "Heaps"}, after=(say("reset"), say("Done.")))
    with world.client() as client:
        started, events = paused(client, world)
        resumed = client.post(f"/runs/{started['run_id']}/approve")
        stream(client, resumed.json()["run_id"])

    [request] = [data for kind, data in events if kind == "approval_request"]
    assert request["summary"] == "Reset progress for 'heaps'."
    assert {topic for _, topic in world.study.reviews} == {"bfs", "tries"}


def test_a_reset_of_a_topic_with_no_progress_asks_nothing():
    world = reset_world(
        {"topic": "graphs"}, after=(say("no progress there"), say("Nothing to reset."))
    )
    with world.client() as client:
        _, events = paused(client, world)

    assert "approval_request" not in kinds(events)
    assert events[-1] == ("done", {"status": "completed"})


def test_a_plan_that_does_not_exist_asks_nothing():
    world = World()
    world.fake.replies = [
        intent(study_plan=0.9),
        task("study_planner"),
        call("commit_study_plan", plan_id=str(uuid.uuid4())),
        say("no such plan"),
        say("I couldn't find that plan."),
    ]
    with world.client() as client:
        _, events = paused(client, world)

    assert "approval_request" not in kinds(events)
    assert text_of(events) == "I couldn't find that plan."


# -- what Admin sees -------------------------------------------------------------------


def test_admin_sees_the_kind_and_step_but_not_the_summary_or_topics():
    request = {"id": "x", "kind": "commit_plan", "summary": "Commit 'secret'", "items": [{}]}

    assert safe_event("approval_request", request) == {"kind": "commit_plan"}
    assert safe_event("approval", {"step": "approved", "kind": "reset_progress"}) == {
        "step": "approved",
        "kind": "reset_progress",
    }
