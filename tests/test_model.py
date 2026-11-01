"""Offline: the real client and agent loop run, with only the network layer replaced."""

import json

import httpx
import openai
import pytest
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.model import ModelTier, get_chat_model

USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 10,
    "completion_tokens_details": {"reasoning_tokens": 0},
}


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def completion(message: dict) -> dict:
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": USAGE,
    }


def tool_call(call_id: str, code_name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup_code", "arguments": json.dumps({"name": code_name})},
    }


class FakeDeepSeek:
    """Replays set replies and records every request body the client would send."""

    def __init__(self, replies: list[dict], usage: dict = USAGE) -> None:
        self.replies = list(replies)
        self.usage = usage
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        body = dict(completion(self.replies.pop(0)), usage=self.usage)
        return httpx.Response(200, json=body)

    def attach(self, model) -> None:
        client = openai.OpenAI(
            api_key="not-a-real-key",
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(transport=httpx.MockTransport(self)),
        )
        model.root_client = client
        model.client = client.chat.completions


@tool
def lookup_code(name: str) -> str:
    """Return the code for a name."""
    return f"code-{name}"


def test_tiers_map_to_deepseek_model_ids():
    assert get_chat_model(ModelTier.SMALL).model_name == "deepseek-flash"
    # LARGE uses Flash until a better model is chosen. See the TODO in core/model.py.
    assert get_chat_model("large").model_name == "deepseek-flash"


def test_every_tier_connects_to_the_configured_base_url():
    for tier in ModelTier:
        assert get_chat_model(tier).api_base == "https://api.deepseek.com"


def test_unknown_tier_is_named_in_the_error():
    with pytest.raises(ValueError, match="'huge'.*small, large"):
        get_chat_model("huge")


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    get_settings.cache_clear()

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        get_chat_model(ModelTier.SMALL)


def test_api_key_is_hidden_when_printed():
    assert "not-a-real-key" not in repr(get_chat_model(ModelTier.SMALL))


@pytest.mark.parametrize("tier", list(ModelTier))
def test_request_turns_thinking_off(tier):
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}])
    model = get_chat_model(tier, max_tokens=50)
    fake.attach(model)

    model.invoke([HumanMessage("hello")])

    body = fake.requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 50
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


def test_request_keeps_thinking_off_with_tools_bound():
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}])
    model = get_chat_model(ModelTier.SMALL)
    fake.attach(model)

    model.bind_tools([lookup_code]).invoke([HumanMessage("hello")])

    assert fake.requests[0]["thinking"] == {"type": "disabled"}
    assert fake.requests[0]["tools"]


def test_reasoning_tokens_can_be_read_from_usage():
    usage = dict(USAGE, completion_tokens_details={"reasoning_tokens": 3})
    fake = FakeDeepSeek([{"role": "assistant", "content": "hi"}], usage=usage)
    model = get_chat_model(ModelTier.SMALL)
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
    model = get_chat_model(ModelTier.SMALL)
    fake.attach(model)
    agent = create_deep_agent(model=model, tools=[lookup_code])

    result = agent.invoke({"messages": [HumanMessage("Find the codes for alpha and beta")]})

    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_results == ["code-alpha", "code-beta"]
    assert result["messages"][-1].content == "code-alpha and code-beta"
    assert len(fake.requests) == 3
    assert all(body["thinking"] == {"type": "disabled"} for body in fake.requests)
    assert "code-alpha" in json.dumps(fake.requests[1]["messages"])
