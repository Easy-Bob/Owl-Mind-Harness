"""Calls DeepSeek and costs a small amount. Run only on request: uv run pytest -m live"""

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.model import ModelTier, get_chat_model

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]

CODES = {"alpha": "7291", "beta": "3846"}


@tool
def lookup_code(name: str) -> str:
    """Return the code for a name. Call this once for each name."""
    return CODES[name]


@pytest.fixture
def small_model():
    try:
        if get_settings().deepseek_api_key is None:
            pytest.skip("DEEPSEEK_API_KEY is not set")
    except ConfigError as exc:
        pytest.skip(str(exc))
    return get_chat_model(ModelTier.SMALL, max_tokens=300)


def test_tool_loop_completes_with_no_reasoning_tokens(small_model):
    agent = create_deep_agent(
        model=small_model,
        tools=[lookup_code],
        system_prompt="Answer using lookup_code. Reply with only the codes.",
    )

    result = agent.invoke({"messages": [HumanMessage("What are the codes for alpha and beta?")]})

    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert sorted(tool_results) == sorted(CODES.values())

    answer = result["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert all(code in answer.content for code in CODES.values())

    model_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    for message in model_messages:
        details = message.usage_metadata.get("output_token_details", {})
        assert details.get("reasoning", 0) == 0
        assert not message.additional_kwargs.get("reasoning_content")
