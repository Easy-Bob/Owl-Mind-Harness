"""Offline: the checks around every agent step, in the real agent loop with scripted replies."""

import json

import pytest

from owl_mind.core import factory
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.middleware import (
    ARGUMENT_CHARS,
    MAX_TASK_CALLS,
    ContextMiddleware,
    FanOutMiddleware,
    ToolArgumentMiddleware,
    ToolErrorMiddleware,
)
from tests.guard_rails import check_agent_middleware
from tests.test_factory import Run, tool_results
from tests.test_filesystem import call, say, task
from tests.test_filesystem import settings as filesystem_settings  # noqa: F401  Applied to all.
from tests.test_search import FakeSearch, SearchRun, result

SECRET = "zqx-secret-from-an-exception"


def tool_messages(run: Run, request: int) -> list[str]:
    """Tool results as the model saw them in one request, including a tutor's own tools."""
    return [m["content"] for m in run.fake.requests[request]["messages"] if m["role"] == "tool"]


# -- fan-out ----------------------------------------------------------------


def test_task_calls_beyond_the_cap_are_refused_and_the_turn_finishes():
    replies = []
    for number in range(MAX_TASK_CALLS):
        replies += [task("concept_tutor"), say(f"tutor answer {number}")]
    replies += [task("concept_tutor"), task("concept_tutor"), say("final answer")]
    run = Run(*replies)

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    results = tool_results(result)
    assert results[:MAX_TASK_CALLS] == [f"tutor answer {n}" for n in range(MAX_TASK_CALLS)]
    assert all("used its 3 tutor calls" in text for text in results[MAX_TASK_CALLS:])
    assert len(results) == MAX_TASK_CALLS + 2
    assert result["messages"][-1].content == "final answer"


def test_retries_on_one_tutor_cannot_use_up_the_other_tutors_call():
    run = Run(
        task("concept_tutor"),
        say("weak answer"),
        task("concept_tutor"),
        say("better answer"),
        task("concept_tutor"),
        task("quiz_master"),
        say("quiz"),
        say("final"),
    )

    result = run.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    results = tool_results(result)
    assert results[:2] == ["weak answer", "better answer"]
    assert "Keep a call for quiz_master" in results[2]
    assert results[3] == "quiz"


def test_once_both_tutors_ran_the_spare_call_can_retry_either():
    run = Run(
        task("concept_tutor"),
        say("explanation"),
        task("quiz_master"),
        say("weak quiz"),
        task("quiz_master"),
        say("better quiz"),
        say("final"),
    )

    result = run.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    assert tool_results(result) == ["explanation", "weak quiz", "better quiz"]


def test_a_one_tutor_turn_can_use_all_three_calls():
    run = Run(
        task("concept_tutor"), say("part 1"),
        task("concept_tutor"), say("part 2"),
        task("concept_tutor"), say("part 3"),
        say("final"),
    )  # fmt: skip

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert tool_results(result) == ["part 1", "part 2", "part 3"]


def test_a_tutor_the_intent_step_chose_is_allowed():
    run = Run(task("practice_coach"), say("hint"), say("final"))

    result = run.invoke(TaskKind.PRACTICE, [Subagent.PRACTICE_COACH])

    assert tool_results(result) == ["hint"]


# -- tool arguments ---------------------------------------------------------


def test_a_too_long_argument_is_refused_before_the_tool_runs():
    search = FakeSearch([result()])
    long_query = "bloom " * 200
    run = SearchRun(
        search,
        task("concept_tutor"),
        call("materials_search", query=long_query),
        say("tutor done"),
        say("final"),
    )

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    [refusal] = tool_messages(run, 2)
    limit = ARGUMENT_CHARS[("materials_search", "query")]
    assert f"at most {limit} characters" in refusal
    assert search.calls == []


def test_control_characters_are_removed_before_the_tool_runs():
    search = FakeSearch([result()])
    run = SearchRun(
        search,
        task("concept_tutor"),
        call("materials_search", query="bloom\x00 fil\x1bters"),
        say("tutor done"),
        say("final"),
    )

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert search.calls == [("bloom filters", 5)]


# -- tool errors ------------------------------------------------------------


class FailingSearch(FakeSearch):
    async def search_with_stats(self, query: str, limit: int = 5):
        raise RuntimeError(f"database error near {SECRET}")


def test_a_failing_tool_becomes_a_fixed_message_and_the_run_continues(caplog):
    run = SearchRun(
        FailingSearch(),
        task("concept_tutor"),
        call("materials_search", query="bloom filters"),
        say("answered without the notes"),
        say("final"),
    )

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    [message] = tool_messages(run, 2)
    assert message.startswith("materials_search failed (RuntimeError)")
    assert tool_results(result) == ["answered without the notes"]
    everything = json.dumps(run.fake.requests) + caplog.text + str(result)
    assert SECRET not in everything


# -- context block ----------------------------------------------------------


def test_every_agent_gets_the_code_written_context_block_after_its_prompt():
    run = Run(task("concept_tutor"), say("tutor answer"), say("final"))

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR], message="zqx-student-text")

    for request in run.fake.requests:
        system = request["messages"][0]["content"]
        system = system if isinstance(system, str) else json.dumps(system)
        assert "## This request" in system
        assert "- Today: 2026-11-02" in system
        assert "- Kind of request: learn" in system
        assert "- Tutors available: concept_tutor" in system
        assert "zqx-student-text" not in system
        assert system.index("## This request") > 0


# -- attached to every agent ------------------------------------------------


def built_agents(monkeypatch, task_kind, subagents) -> dict:
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    factory.build_owl_agent(task_kind, subagents, store=None, get_model=Run().get_model)
    return captured


@pytest.mark.parametrize(
    ("task_kind", "subagents"),
    [
        (TaskKind.LEARN, [Subagent.CONCEPT_TUTOR]),
        (TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER]),
        (TaskKind.PLAN, [Subagent.STUDY_PLANNER]),
    ],
)
def test_every_agent_the_factory_builds_has_the_checks_in_order(monkeypatch, task_kind, subagents):
    captured = built_agents(monkeypatch, task_kind, subagents)

    assert check_agent_middleware(captured) == []
    orchestrator = [type(m) for m in captured["middleware"]][:4]
    assert orchestrator == [
        ToolErrorMiddleware,
        ToolArgumentMiddleware,
        FanOutMiddleware,
        ContextMiddleware,
    ]


def test_the_guard_reports_an_agent_with_a_missing_check(monkeypatch):
    captured = built_agents(monkeypatch, TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])
    captured["subagents"][0]["middleware"] = [
        m for m in captured["subagents"][0]["middleware"] if not isinstance(m, ToolErrorMiddleware)
    ]
    captured["middleware"] = [
        m for m in captured["middleware"] if not isinstance(m, FanOutMiddleware)
    ]

    violations = check_agent_middleware(captured)

    assert {v.path for v in violations} == {"concept_tutor", "orchestrator"}


def test_every_model_call_in_a_run_has_a_component(monkeypatch):
    events = []
    monkeypatch.setattr("owl_mind.core.telemetry.publish", events.append)
    run = Run(task("concept_tutor"), say("tutor answer"), say("final"))

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert len(events) == 3
    assert {event["component"] for event in events} == {"orchestrator", "agent:concept_tutor"}
