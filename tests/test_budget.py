"""Offline: every budget stops spending once used up, through the real routes and agent loop."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from owl_mind.core import replies, runtime
from owl_mind.core.budget import InMemoryBudgetLedger, RunMeter, budget_step, micro_usd
from owl_mind.core.config import get_settings
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.limits import RUN_LIMITS, RunLimit
from tests.fake_deepseek import USAGE
from tests.test_factory import Run, tool_results
from tests.test_filesystem import ALICE, call, say, task
from tests.test_runtime import (
    HINT,
    World,
    intent,
    kinds,
    start,
    stream,
    telemetry_events,  # noqa: F401  Applied to all.
    text_of,
)
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.

PER_CALL = USAGE["prompt_tokens"] + USAGE["completion_tokens"]  # 15 tokens a call.


def today():
    return datetime.now(UTC).date()


def tool_names(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools", [])}


def recorded(world: World, run_id: str, kind: str) -> list[dict]:
    run = uuid.UUID(run_id)
    events = sorted((key, value) for (rid, key), value in world.runs.events.items() if rid == run)
    return [data for _, (event_kind, data) in events if event_kind == kind]


# -- the run budget and its ladder -------------------------------------------


def test_the_ladder_steps_down_at_70_and_90_percent_and_no_call_starts_past_100(monkeypatch):
    # 100 tokens at 15 a call: the 6th call is at 75%, the 7th at 90%, and an 8th would be 105%.
    monkeypatch.setitem(RUN_LIMITS, TaskKind.PRACTICE, RunLimit(20, 120, 100))
    world = World(
        intent(problem_help=0.9),
        task("practice_coach"),
        call("build_hint", level=1),
        say("tutor report"),
        call("get_due_topics"),
        task("practice_coach"),  # Offered no more at 75%, and refused when called anyway.
        call("get_due_topics"),  # Asked for no tools at 90%, but called one anyway.
        say("never sent"),
    )
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    requests = world.fake.requests
    assert len(requests) == 7
    assert "task" in tool_names(requests[4])
    assert "task" not in tool_names(requests[5])
    assert "used most of its budget" in requests[5]["messages"][0]["content"]
    assert requests[6].get("tool_choice") == "none"
    assert "No tools now" in requests[6]["messages"][0]["content"]
    refusals = [m["content"] for m in requests[6]["messages"] if m["role"] == "tool"]
    assert any("budget is nearly used" in text for text in refusals)

    steps = [data for kind, data in events if kind == "budget_step"]
    assert steps == [
        {"step": "wind_down", "percent": 75},
        {"step": "final", "percent": 90},
        {"step": "stopped", "percent": 105},
    ]
    assert recorded(world, run_id, "budget_step") == steps
    assert events[-1] == ("done", {"status": "stopped", "reason": "token_limit"})
    assert replies.RUN_STOPPED in text_of(events)
    assert world.budgets.students[(ALICE, today())] == 7 * PER_CALL


def test_a_stop_inside_a_tutor_stops_the_run_instead_of_becoming_a_tool_error(monkeypatch):
    monkeypatch.setitem(RUN_LIMITS, TaskKind.PRACTICE, RunLimit(20, 120, 50))
    world = World(
        intent(problem_help=0.9),
        task("practice_coach"),
        call("build_hint", level=1),
        call("build_hint", level=2),  # The tutor's call at 90% ignores the no-tools request.
        say("never sent"),
        say("never sent"),
    )
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    assert len(world.fake.requests) == 4
    assert events[-1] == ("done", {"status": "stopped", "reason": "token_limit"})
    assert "subagent_end" not in kinds(events)


def test_a_turn_well_within_budget_takes_no_step(monkeypatch):
    world = World(
        intent(problem_help=0.9), task("practice_coach"), say("report"), say("answer"), HINT
    )
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    assert "budget_step" not in kinds(events)
    assert events[-1] == ("done", {"status": "completed"})
    # The integrity reviewer's call counts too.
    assert world.budgets.students[(ALICE, today())] == 5 * PER_CALL


@pytest.mark.parametrize(
    ("share", "step"),
    [(0.0, None), (0.69, None), (0.7, "wind_down"), (0.89, "wind_down"), (0.9, "final")],
)
def test_ladder_steps(share, step):
    assert budget_step(share) == step


# -- daily budgets -------------------------------------------------------------


def test_a_student_at_their_daily_budget_gets_the_message_and_no_model_call():
    world = World(intent(concept_explain=0.9), say("never sent"))
    world.budgets.students[(ALICE, today())] = get_settings().student_daily_tokens
    with world.client() as client:
        run_id = start(client, "Can you explain how a binary heap keeps its order?")["run_id"]
        events = stream(client, run_id)

    assert world.fake.requests == []
    assert text_of(events) == replies.STUDENT_BUDGET
    assert events[-1] == ("done", {"status": "stopped", "reason": "student_budget"})


def test_a_student_at_their_daily_budget_still_gets_fixed_replies():
    world = World()
    world.budgets.students[(ALICE, today())] = get_settings().student_daily_tokens
    with world.client() as client:
        events = stream(client, start(client, "hello")["run_id"])

    assert text_of(events) == replies.TEXT[runtime.Reply.GREETING]
    assert events[-1] == ("done", {"status": "completed"})


def test_another_students_spending_does_not_count_against_this_one():
    world = World(intent(concept_explain=0.9), say("answer"))
    world.budgets.students[(uuid.UUID(int=99), today())] = get_settings().student_daily_tokens
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    assert text_of(events) == "answer"


def test_the_global_budget_stops_every_agent_run():
    world = World(intent(concept_explain=0.9), say("never sent"))
    world.budgets.totals[today()] = round(get_settings().global_daily_usd * 1_000_000)
    with world.client() as client:
        run_id = start(client, "Can you explain how a binary heap keeps its order?")["run_id"]
        events = stream(client, run_id)

    assert world.fake.requests == []
    assert text_of(events) == replies.AGENTS_PAUSED
    assert events[-1] == ("done", {"status": "stopped", "reason": "global_budget"})


@pytest.mark.parametrize("missing", ["down", "none"])
def test_no_agent_run_starts_without_the_counters(missing):
    world = World(intent(concept_explain=0.9), say("never sent"))
    if missing == "down":
        world.budgets.down = True
    else:
        world.services = world.services.__class__(**{**vars(world.services), "budgets": None})
    with world.client() as client:
        study = stream(client, start(client, "Explain how a binary heap keeps its order")["run_id"])
        greeting = stream(client, start(client, "hello")["run_id"])

    assert world.fake.requests == []
    assert text_of(study) == replies.AGENTS_PAUSED
    assert study[-1] == ("done", {"status": "stopped", "reason": "budget_unavailable"})
    assert text_of(greeting) == replies.TEXT[runtime.Reply.GREETING]


def test_counters_lost_mid_run_stop_it_at_the_next_call():
    world = World(intent(problem_help=0.9), task("practice_coach"), say("never sent"))
    original = world.budgets.charge
    charges = 0

    async def charge_then_fail(*args):
        nonlocal charges
        charges += 1
        if charges > 1:
            raise ConnectionError("lost")
        await original(*args)

    world.budgets.charge = charge_then_fail
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    assert len(world.fake.requests) == 2
    assert events[-1] == ("done", {"status": "stopped", "reason": "budget_unavailable"})


# -- counting ------------------------------------------------------------------


def test_a_reply_with_no_usage_counts_as_the_calls_cap_not_zero():
    world = World(intent(concept_explain=0.9), say("answer"))
    world.fake.usage = None
    with world.client() as client:
        stream(client, start(client)["run_id"])

    # Both calls count at least their max_tokens: 200 for the intent step, 1,200 for the answer.
    assert world.budgets.students[(ALICE, today())] > 1400
    assert world.budgets.totals[today()] > 0


def test_two_runs_at_once_from_one_student_both_count():
    async def scenario():
        ledger = InMemoryBudgetLedger()
        first, second = RunMeter(ledger, ALICE, 1000), RunMeter(ledger, ALICE, 1000)
        await asyncio.gather(first._charge(300, 5), second._charge(400, 7))
        return ledger

    ledger = asyncio.run(scenario())

    assert ledger.students[(ALICE, today())] == 700
    assert ledger.totals[today()] == 12


def test_the_cost_estimate_rounds_up_and_never_undercounts_an_unknown_model():
    assert micro_usd("deepseek-flash", 1_000_000, 0) == 300_000
    assert micro_usd("deepseek-flash", 1, 0) == 1
    assert micro_usd("some-new-model", 1000, 1000) >= micro_usd("deepseek-v4-pro", 1000, 1000)


# -- result caps ---------------------------------------------------------------


def test_a_long_tutor_report_is_cut_before_the_orchestrator_reads_it():
    run = Run(task("concept_tutor"), say("x" * 20_000), say("final"))

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    [report] = tool_results(result)
    assert report.startswith("x" * 8000)
    assert "[Cut to 8000 of 20000 characters.]" in report
    assert len(report) < 8100


# -- summarization ---------------------------------------------------------------


def test_a_long_thread_is_summarized_once_and_the_summary_counts(telemetry_events):  # noqa: F811
    world = World(intent(concept_explain=0.9), say("short summary"), say("answer"))
    with world.client() as client:
        started = start(client, "hello")
        stream(client, started["run_id"])
        thread = uuid.UUID(started["thread_id"])
        for _ in range(30):  # About 30,000 tokens of earlier chat.
            asyncio.run(runtime._record_turn(world.saver, thread, "q" * 2000, "a" * 2000))
        before = world.budgets.students.get((ALICE, today()), 0)
        next_run = client.post(f"/threads/{thread}/messages", json={"message": "explain heaps"})
        events = stream(client, next_run.json()["run_id"])

    components = [event.get("component") for event in telemetry_events]
    assert components.count("summarization") == 1
    assert text_of(events) == "answer"
    assert world.budgets.students[(ALICE, today())] - before == 3 * PER_CALL
    answer_request = world.fake.requests[-1]
    assert sum(len(str(m.get("content") or "")) for m in answer_request["messages"]) < 60_000
