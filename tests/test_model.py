"""Offline: the real client and agent loop run, with only the network layer replaced."""

import json

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.model import ModelTier, get_chat_model
from tests.fake_deepseek import USAGE, FakeDeepSeek, tool_call


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@tool
def lookup_code(name: str) -> str:
    """Return the code for a name."""
    return f"code-{name}"


# A real component of each tier, since the door checks every component against its table.
ON = {ModelTier.SMALL: "intent", ModelTier.LARGE: "intent:layer3"}


def test_tiers_map_to_deepseek_model_ids():
    assert get_chat_model(ModelTier.SMALL, component="intent").model_name == "deepseek-flash"
    # LARGE stays Flash until the owner approves another model (M3-F).
    assert get_chat_model("large", component="intent:layer3").model_name == "deepseek-flash"


def test_the_large_model_comes_from_its_setting(monkeypatch):
    monkeypatch.setenv("LARGE_MODEL", "deepseek-v4-pro")
    get_settings.cache_clear()

    assert get_chat_model("large", component="orchestrator:composite").model_name == (
        "deepseek-v4-pro"
    )
    assert get_chat_model("small", component="orchestrator").model_name == "deepseek-flash"


def test_a_component_calling_another_tier_is_refused():
    with pytest.raises(ValueError, match="orchestrator uses the small tier, not large"):
        get_chat_model("large", component="orchestrator")
    with pytest.raises(ValueError, match="No tier is set"):
        get_chat_model("small", component="made-up")


def test_every_tutor_is_small_and_the_table_names_every_component_in_use():
    from owl_mind.core.model import COMPONENT_TIERS, tier_for

    assert tier_for("agent:concept_tutor") is ModelTier.SMALL
    assert set(COMPONENT_TIERS) >= {
        "intent", "orchestrator", "orchestrator:composite", "summarization",
        "integrity_reviewer", "memory:write", "materials:topics", "handoff:improve",
    }  # fmt: skip


def test_every_tier_connects_to_the_configured_base_url():
    for tier in ModelTier:
        assert get_chat_model(tier, component=ON[tier]).api_base == "https://api.deepseek.com"


def test_unknown_tier_is_named_in_the_error():
    with pytest.raises(ValueError, match="'huge'.*small, large"):
        get_chat_model("huge", component="intent")


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    get_settings.cache_clear()

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        get_chat_model(ModelTier.SMALL, component="intent")


def test_api_key_is_hidden_when_printed():
    assert "not-a-real-key" not in repr(get_chat_model(ModelTier.SMALL, component="intent"))


@pytest.mark.parametrize("tier", list(ModelTier))
def test_request_turns_thinking_off(tier):
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}])
    model = get_chat_model(tier, component=ON[tier], max_tokens=50)
    fake.attach(model)

    model.invoke([HumanMessage("hello")])

    body = fake.requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 50
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


def test_request_keeps_thinking_off_with_tools_bound():
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}])
    model = get_chat_model(ModelTier.SMALL, component="intent")
    fake.attach(model)

    model.bind_tools([lookup_code]).invoke([HumanMessage("hello")])

    assert fake.requests[0]["thinking"] == {"type": "disabled"}
    assert fake.requests[0]["tools"]


def test_reasoning_tokens_can_be_read_from_usage():
    usage = dict(USAGE, completion_tokens_details={"reasoning_tokens": 3})
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}], usage=usage)
    model = get_chat_model(ModelTier.SMALL, component="intent")
    fake.attach(model)

    message = model.invoke([HumanMessage("hello")])

    assert message.usage_metadata["output_token_details"]["reasoning"] == 3


def test_deep_agent_tool_loop_with_two_tool_calls():
    fake = FakeDeepSeek(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call_1", "alpha")],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call_2", "beta")],
            },
            {"role": "assistant", "content": "code-alpha and code-beta"},
        ]
    )
    model = get_chat_model(ModelTier.SMALL, component="intent")
    fake.attach(model)
    agent = create_deep_agent(model=model, tools=[lookup_code])

    result = agent.invoke({"messages": [HumanMessage("Find the codes for alpha and beta")]})

    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_results == ["code-alpha", "code-beta"]
    assert result["messages"][-1].content == "code-alpha and code-beta"
    assert len(fake.requests) == 3
    assert all(body["thinking"] == {"type": "disabled"} for body in fake.requests)
    assert "code-alpha" in json.dumps(fake.requests[1]["messages"])
