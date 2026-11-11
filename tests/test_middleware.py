"""Offline: the checks around every agent step, in the real agent loop with scripted replies."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from owl_mind.core import factory
from owl_mind.core.config import get_settings
from owl_mind.core.factory import BudgetedSummarization
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.middleware import (
    ARGUMENT_CHARS,
    MAX_TASK_CALLS,
    BudgetMiddleware,
    ContextMiddleware,
    FanOutMiddleware,
    JoinAnswerMiddleware,
    ResultCapMiddleware,
    ToolArgumentMiddleware,
    ToolErrorMiddleware,
)
from tests.budget_helpers import meter
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


def test_task_calls_beyond_the_cap_are_refused_and_the_turn_finishes(monkeypatch):
    # Rewrite mode, so the orchestrator keeps going after both tutors have replied.
    monkeypatch.setenv("COMPOSITE_ANSWER", "rewrite")
    get_settings.cache_clear()
    run = Run(
        task("concept_tutor"), say("one"),
        task("quiz_master"), say("two"),
        task("concept_tutor"), say("three"),
        task("quiz_master"),
        say("final answer"),
    )  # fmt: skip

    result = run.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    results = tool_results(result)
    assert results[:MAX_TASK_CALLS] == ["one", "two", "three"]
    assert "used its 3 tutor calls" in results[MAX_TASK_CALLS]
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
    assert "already asked twice" in results[2]
    assert results[3] == "quiz"


def test_once_both_tutors_ran_the_spare_call_can_retry_either(monkeypatch):
    # A joined answer is made as soon as both have replied, so retries need the rewrite mode.
    monkeypatch.setenv("COMPOSITE_ANSWER", "rewrite")
    get_settings.cache_clear()
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


def test_a_tutor_gets_one_retry_and_then_the_orchestrator_answers_with_what_it_has():
    run = Run(
        task("concept_tutor"), say("failed: no notes"),
        task("concept_tutor"), say("failed again"),
        task("concept_tutor"),
        say("Here's what I can say; the notes didn't cover the rest."),
    )  # fmt: skip

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    results = tool_results(result)
    assert results[:2] == ["failed: no notes", "failed again"]
    assert "already asked twice" in results[2] and "what's missing" in results[2]
    assert result["messages"][-1].content.startswith("Here's what I can say")


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


def text_of(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else json.dumps(content)


def test_every_agent_gets_the_code_written_context_block_as_its_last_message():
    run = Run(task("concept_tutor"), say("tutor answer"), say("final"))

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR], message="zqx-student-text")

    for request in run.fake.agent_requests:
        last = request["messages"][-1]
        assert last["role"] == "system"
        block = text_of(last)
        assert block.startswith("## This request")
        assert "- Today: 2026-11-02" in block
        assert "- Kind of request: learn" in block
        assert "- Tutors available: concept_tutor" in block
        assert "zqx-student-text" not in block
        assert "## This request" not in text_of(request["messages"][0])
    # Only in the request: the chat's history never holds the block.
    assert all("## This request" not in str(m.content) for m in result["messages"])


def test_the_prompt_beginning_is_identical_for_different_students_and_days():
    import uuid
    from datetime import date

    from langchain_core.messages import HumanMessage
    from langgraph.store.memory import InMemoryStore

    from owl_mind.tools.context import RunContext

    def first_requests(student, today) -> list[dict]:
        run = Run(task("concept_tutor"), say("tutor answer"), say("final"))
        agent = factory.build_owl_agent(
            TaskKind.LEARN,
            [Subagent.CONCEPT_TUTOR],
            store=InMemoryStore(),
            meter=meter(student),
            get_model=run.get_model,
        )
        context = RunContext(student_id=student, store=run.store, today=today)
        asyncio.run(agent.ainvoke({"messages": [HumanMessage("help me")]}, context=context))
        return run.fake.agent_requests

    one = first_requests(uuid.UUID(int=1), date(2026, 11, 2))
    two = first_requests(uuid.UUID(int=2), date(2026, 12, 25))

    # A first request holds no chat yet, so all of it but the block must match.
    assert one[0]["messages"][:-1] == two[0]["messages"][:-1]
    for a, b in zip(one, two, strict=True):
        # The system prompt (with the skills list) and the tools are the cacheable beginning.
        assert a["messages"][0] == b["messages"][0]
        assert a.get("tools") == b.get("tools")
        assert a["messages"][-1] != b["messages"][-1]


# -- attached to every agent ------------------------------------------------


def built_agents(monkeypatch, task_kind, subagents) -> dict:
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    factory.build_owl_agent(
        task_kind, subagents, store=None, meter=meter(), get_model=Run().get_model
    )
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
    built = [type(m) for m in captured["middleware"]]
    if task_kind is TaskKind.COMPOSITE:
        assert built[0] is JoinAnswerMiddleware
    orchestrator = [m for m in built if m is not JoinAnswerMiddleware][:7]
    assert orchestrator == [
        BudgetMiddleware,
        ToolErrorMiddleware,
        ResultCapMiddleware,
        ToolArgumentMiddleware,
        FanOutMiddleware,
        ContextMiddleware,
        BudgetedSummarization,
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

    assert len(events) == 4
    assert {event["component"] for event in events} == {
        "orchestrator",
        "agent:concept_tutor",
        "integrity_reviewer",
    }


# -- joined two-tutor answers ------------------------------------------------

COMPOSITE = [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER]
EXPLANATION = "A Bloom filter sets k bits per item [cite:a1b2]."
QUIZ = "1. What does a set bit tell you?"


def test_a_two_tutor_answer_is_the_replies_joined_with_no_rewrite_call():
    run = Run(task("concept_tutor"), say(EXPLANATION), task("quiz_master"), say(QUIZ))

    result = run.invoke(TaskKind.COMPOSITE, COMPOSITE)

    expected = f"{EXPLANATION}\n\nNow let's check your understanding.\n\n{QUIZ}"
    assert result["messages"][-1].content == expected
    assert len(run.fake.requests) == 4  # Two orchestrator calls to delegate, one per tutor.


def test_tutors_replies_join_in_the_intent_steps_order_whatever_order_they_ran():
    run = Run(task("quiz_master"), say(QUIZ), task("concept_tutor"), say(EXPLANATION))

    result = run.invoke(TaskKind.COMPOSITE, COMPOSITE)

    assert result["messages"][-1].content.startswith(EXPLANATION)


def test_the_rewrite_mode_has_the_orchestrator_write_the_answer(monkeypatch):
    monkeypatch.setenv("COMPOSITE_ANSWER", "rewrite")
    get_settings.cache_clear()
    run = Run(
        task("concept_tutor"), say(EXPLANATION), task("quiz_master"), say(QUIZ), say("rewritten")
    )

    result = run.invoke(TaskKind.COMPOSITE, COMPOSITE)

    assert result["messages"][-1].content == "rewritten"
    assert len(run.fake.requests) == 5


def ask(call_id: str, tutor: str) -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"id": call_id, "name": "task", "args": {"subagent_type": tutor}}]
    )


def reply(call_id: str, text: str, status: str = "success") -> ToolMessage:
    return ToolMessage(text, tool_call_id=call_id, name="task", status=status)


def test_a_failed_or_missing_tutor_reply_leaves_the_answer_to_the_orchestrator():
    join = JoinAnswerMiddleware(("concept_tutor", "quiz_master"))
    asked = [HumanMessage("help"), ask("1", "concept_tutor"), ask("2", "quiz_master")]

    failed = [*asked, reply("1", EXPLANATION), reply("2", "quiz_master failed", "error")]
    missing = [*asked, reply("1", EXPLANATION)]
    retried = [*failed, ask("3", "quiz_master"), reply("3", QUIZ)]

    assert join.joined(failed) is None
    assert join.joined(missing) is None
    assert join.joined(retried).endswith(QUIZ)


def test_only_this_turns_replies_are_joined():
    join = JoinAnswerMiddleware(("concept_tutor", "quiz_master"))
    earlier = [HumanMessage("first"), ask("1", "concept_tutor"), reply("1", "old explanation")]
    now = [HumanMessage("second"), ask("2", "quiz_master"), reply("2", QUIZ)]

    assert join.joined([*earlier, *now]) is None
