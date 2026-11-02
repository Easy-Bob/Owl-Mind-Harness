"""Offline: the scope proof, through the real agent loop with only the network replaced."""

import asyncio
import json
import uuid
from datetime import date

import pytest
from deepagents import HarnessProfile, register_harness_profile
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from owl_mind.core import factory, telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.factory import LiveAgentRunsDisabled, build_owl_agent
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.model import get_chat_model
from owl_mind.core.subagents import TOOL_SCOPES, TUTORS
from owl_mind.services.study_store import InMemoryStudyStore
from owl_mind.tools.context import RunContext
from tests.fake_deepseek import FakeDeepSeek
from tests.guard_rails import check_models_from_door, check_tool_scopes

ALICE = uuid.UUID(int=1)
TODAY = date(2026, 11, 2)
# What DeepAgents gives every agent. Permissions restrict what they can reach.
FILE_TOOLS = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete"}


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    monkeypatch.setattr(telemetry, "publish", lambda event: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def call(name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def task(subagent: str) -> dict:
    return call("task", description="Help the student", subagent_type=subagent)


def say(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


class Run:
    def __init__(self, *replies: dict, status: int = 200) -> None:
        self.fake = FakeDeepSeek(list(replies), status=status)
        self.models: list = []
        self.store = InMemoryStudyStore()

    def get_model(self, tier, *, component, max_tokens):
        model = get_chat_model(tier, component=component, max_tokens=max_tokens)
        self.fake.attach(model)
        self.models.append((component, tier, max_tokens, model))
        return model

    def invoke(self, task_kind, subagents, message="help me"):
        agent = build_owl_agent(
            task_kind, subagents, store=InMemoryStore(), get_model=self.get_model
        )
        context = RunContext(student_id=ALICE, store=self.store, today=TODAY)
        return asyncio.run(agent.ainvoke({"messages": [HumanMessage(message)]}, context=context))

    def tool_names(self, request: int) -> set[str]:
        tools = self.fake.requests[request].get("tools", [])
        return {tool["function"]["name"] for tool in tools}

    def task_description(self, request: int = 0) -> str:
        for tool in self.fake.requests[request]["tools"]:
            if tool["function"]["name"] == "task":
                return tool["function"]["description"]
        raise AssertionError("no task tool")


def tool_results(result) -> list[str]:
    return [m.content for m in result["messages"] if m.type == "tool"]


# -- the live gate ----------------------------------------------------------


def test_factory_refuses_while_live_agent_runs_are_off(monkeypatch):
    monkeypatch.delenv("LIVE_AGENT_RUNS_ENABLED")
    get_settings.cache_clear()
    run = Run(say())

    with pytest.raises(LiveAgentRunsDisabled):
        run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert run.models == []
    assert run.fake.requests == []


# -- the scope proof --------------------------------------------------------


def test_task_tool_lists_only_the_selected_subagents():
    run = Run(say())

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    description = run.task_description()
    assert "- concept_tutor:" in description
    for other in ("general-purpose:", "practice_coach:", "study_planner:", "quiz_master:"):
        assert f"- {other}" not in description


def test_execute_is_never_offered():
    run = Run(task("concept_tutor"), say(), say())

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert all("execute" not in run.tool_names(i) for i in range(len(run.fake.requests)))


@pytest.mark.parametrize("target", ["general-purpose", "practice_coach"])
def test_task_call_to_an_unavailable_subagent_is_refused_and_the_run_continues(target):
    run = Run(task(target), say("carried on"))

    result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    assert "does not exist" in tool_results(result)[0]
    # One request for the task call and one to carry on, with none for a subagent.
    assert len(run.fake.requests) == 2
    assert result["messages"][-1].content == "carried on"


def test_removing_the_profile_makes_the_scope_proof_fail(monkeypatch):
    """Proves the checks above detect the problem they exist for."""
    register_harness_profile(
        "deepseek",
        HarnessProfile(
            general_purpose_subagent=factory.GeneralPurposeSubagentProfile(enabled=True)
        ),
    )
    try:
        run = Run(task("general-purpose"), say("subagent ran"), say("carried on"))

        result = run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

        assert len(run.fake.requests) == 3
        assert tool_results(result) == ["subagent ran"]
        assert "- general-purpose:" in run.task_description()
    finally:
        register_harness_profile(
            "deepseek",
            HarnessProfile(
                general_purpose_subagent=factory.GeneralPurposeSubagentProfile(enabled=False),
                excluded_tools=frozenset({"execute"}),
            ),
        )


@pytest.mark.parametrize("subagent", list(Subagent))
def test_each_tutor_is_offered_exactly_its_scope(subagent):
    run = Run(task(subagent.value), say(), say())

    run.invoke(TaskKind.LEARN, [subagent])

    assert run.tool_names(1) - FILE_TOOLS == set(TOOL_SCOPES[subagent])


def test_tool_outside_a_tutors_scope_does_not_run():
    run = Run(
        task("concept_tutor"),
        call("build_hint", level=3, topic="bfs"),
        say("tutor answer"),
        say("final"),
    )

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    tutor_turn = json.dumps(run.fake.requests[2]["messages"])
    assert "build_hint is not a valid tool" in tutor_turn
    assert "withholds_solution" not in tutor_turn


def test_no_tutor_selected_means_no_task_tool():
    run = Run(say("You can do this."))

    result = run.invoke(TaskKind.SUPPORT, [])

    assert "task" not in run.tool_names(0)
    assert result["messages"][-1].content == "You can do this."


# -- how the turn is set up -------------------------------------------------


def test_steps_are_chosen_by_the_model_in_either_order():
    code = "for a in x:\n    pass\n"
    orders = [
        [call("build_hint", level=1), call("analyze_complexity", code=code)],
        [call("analyze_complexity", code=code), call("build_hint", level=1)],
        [call("build_hint", level=2)],
    ]
    for tool_calls in orders:
        run = Run(task("practice_coach"), *tool_calls, say("tutor"), say("final"))

        result = run.invoke(TaskKind.PRACTICE, [Subagent.PRACTICE_COACH])

        assert result["messages"][-1].content == "final"
        assert len(run.fake.requests) == len(tool_calls) + 3


def test_tools_in_a_subagent_read_the_student_from_context():
    run = Run(
        task("study_planner"),
        call("schedule_review", topic="bfs", quality=5),
        say("scheduled"),
        say("final"),
    )

    run.invoke(TaskKind.PLAN, [Subagent.STUDY_PLANNER])

    assert (ALICE, "bfs") in run.store.reviews


def test_student_cannot_be_set_through_a_tool_argument():
    other = uuid.UUID(int=2)
    run = Run(
        task("study_planner"),
        call("schedule_review", topic="bfs", quality=5, student_id=str(other), today="2020-01-01"),
        say("scheduled"),
        say("final"),
    )

    run.invoke(TaskKind.PLAN, [Subagent.STUDY_PLANNER])

    assert (other, "bfs") not in run.store.reviews
    assert run.store.reviews[(ALICE, "bfs")].due_date.year == TODAY.year


def test_components_and_token_caps_are_set_per_agent():
    run = Run(say())

    run.invoke(TaskKind.PRACTICE, [Subagent.PRACTICE_COACH])

    labels = {component: max_tokens for component, _, max_tokens, _ in run.models}
    assert labels == {
        "orchestrator": 1200,
        "agent:practice_coach": TUTORS[Subagent.PRACTICE_COACH].max_tokens,
    }


def test_write_todos_is_only_offered_on_composite_turns():
    single, composite = Run(say()), Run(say())

    single.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])
    composite.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    assert "write_todos" not in single.tool_names(0)
    assert "write_todos" in composite.tool_names(0)


def test_composite_turn_can_reach_both_selected_tutors():
    run = Run(
        task("concept_tutor"), say("explained"), task("quiz_master"), say("quizzed"), say("end")
    )

    result = run.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    assert tool_results(result) == ["explained", "quizzed"]


def test_every_prompt_requires_english():
    run = Run(task("concept_tutor"), say(), say())

    run.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR], message="\u00bfQu\u00e9 es BFS?")

    for request in run.fake.requests:
        assert "Always reply in English" in json.dumps(request["messages"][0]["content"])


def test_prompt_prefix_and_tools_are_the_same_on_every_build():
    first, second = Run(say()), Run(say())

    first.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])
    second.invoke(TaskKind.LEARN, [Subagent.CONCEPT_TUTOR], message="something else")

    assert first.fake.requests[0]["messages"][0] == second.fake.requests[0]["messages"][0]
    assert first.fake.requests[0]["tools"] == second.fake.requests[0]["tools"]


# -- guard rails on the real setup and on deliberate violations -------------


def test_every_model_given_to_an_agent_comes_from_the_door():
    run = Run(say())

    run.invoke(TaskKind.COMPOSITE, [Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER])

    assert check_models_from_door([model for *_, model in run.models]) == []


def test_guard_reports_a_model_built_outside_the_door():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    assert len(check_models_from_door([GenericFakeChatModel(messages=iter([]))])) == 1


def test_declared_scopes_match_what_each_tutor_is_built_with():
    built = {subagent: [tool.name for tool in TUTORS[subagent].tools] for subagent in Subagent}

    assert check_tool_scopes(TOOL_SCOPES, built) == []


def test_guard_reports_a_tutor_built_with_an_undeclared_tool():
    built = {Subagent.CONCEPT_TUTOR: ["get_prerequisites", "build_hint"]}
    declared = {Subagent.CONCEPT_TUTOR: ("get_prerequisites",)}

    assert len(check_tool_scopes(declared, built)) == 1
