"""Offline: accounting events, with only the network and the publisher replaced."""

from datetime import UTC, datetime

import pytest
from deepagents import create_deep_agent
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from prometheus_client import REGISTRY

from owl_mind.api.app import app
from owl_mind.core import telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.model import ModelTier, get_chat_model
from owl_mind.worker.telemetry import time_band
from tests.fake_deepseek import USAGE, FakeDeepSeek
from tests.guard_rails import check_every_call_recorded, check_telemetry_failure_is_isolated

SECRET_TEXT = "private student message"
EVENT_FIELDS = {
    "call_id",
    "called_at",
    "run_id",
    "component",
    "provider",
    "model",
    "latency_ms",
    "outcome",
    "prompt_version",
    "input_tokens",
    "output_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
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


@pytest.fixture
def events(monkeypatch):
    published = []
    monkeypatch.setattr(telemetry, "publish", published.append)
    return published


def model_with(fake: FakeDeepSeek, component: str = "test"):
    model = get_chat_model(ModelTier.SMALL, component=component)
    fake.attach(model)
    return model


def answer(content: str = "hi") -> dict:
    return {"role": "assistant", "content": content}


def test_one_call_makes_one_event_with_metadata_only(events):
    model = model_with(FakeDeepSeek([answer()]), component="intent")

    model.invoke(
        [HumanMessage(SECRET_TEXT)],
        config={
            "metadata": {
                telemetry.RUN_ID_KEY: "11111111-1111-1111-1111-111111111111",
                telemetry.PROMPT_VERSION_KEY: "v1",
            }
        },
    )

    assert len(events) == 1
    event = events[0]
    assert set(event) == EVENT_FIELDS
    assert SECRET_TEXT not in str(event)
    assert event["component"] == "intent"
    assert event["model"] == "deepseek-flash"
    assert event["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert event["prompt_version"] == "v1"
    assert event["outcome"] == "success"
    assert event["input_tokens"] == USAGE["prompt_tokens"]
    assert event["output_tokens"] == USAGE["completion_tokens"]


def test_cache_tokens_are_split(events):
    usage = dict(USAGE, prompt_tokens_details={"cached_tokens": 4})
    model = model_with(FakeDeepSeek([answer()], usage=usage))

    model.invoke([HumanMessage("hello")])

    assert events[0]["cache_hit_tokens"] == 4
    assert events[0]["cache_miss_tokens"] == USAGE["prompt_tokens"] - 4


def test_missing_reasoning_field_stays_unknown(events):
    usage = {key: value for key, value in USAGE.items() if key != "completion_tokens_details"}
    model = model_with(FakeDeepSeek([answer()], usage=usage))

    model.invoke([HumanMessage("hello")])

    assert events[0]["reasoning_tokens"] is None


def test_response_with_no_usage_still_makes_an_event(events):
    model = model_with(FakeDeepSeek([answer()], usage=None))

    model.invoke([HumanMessage("hello")])

    assert events[0]["outcome"] == "success"
    assert events[0]["input_tokens"] is None
    assert events[0]["output_tokens"] is None


def test_truncated_response_is_marked(events):
    model = model_with(FakeDeepSeek([answer()], finish_reason="length"))

    model.invoke([HumanMessage("hello")])

    assert events[0]["outcome"] == "truncated"


def test_failed_call_is_recorded_as_error(events):
    model = model_with(FakeDeepSeek([answer()], status=400))

    with pytest.raises(Exception, match="400"):
        model.invoke([HumanMessage("hello")])

    assert [event["outcome"] for event in events] == ["error"]


def test_subagent_calls_are_recorded_under_their_own_component(events):
    fake = FakeDeepSeek(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": ('{"description": "explain", "subagent_type": "tutor"}'),
                        },
                    }
                ],
            },
            answer("from the tutor"),
            answer("final"),
        ]
    )
    main = model_with(fake, component="orchestrator")
    tutor = model_with(fake, component="agent:tutor")
    agent = create_deep_agent(
        model=main,
        subagents=[
            {"name": "tutor", "description": "Explains", "system_prompt": "Teach.", "model": tutor}
        ],
    )

    agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"metadata": {telemetry.RUN_ID_KEY: "22222222-2222-2222-2222-222222222222"}},
    )

    assert [event["component"] for event in events] == [
        "orchestrator",
        "agent:tutor",
        "orchestrator",
    ]
    assert {event["run_id"] for event in events} == {"22222222-2222-2222-2222-222222222222"}
    assert check_every_call_recorded(len(fake.requests), len(events)) == []


def lost_events() -> float:
    return REGISTRY.get_sample_value("telemetry_events_lost_total") or 0


def test_failed_publish_does_not_fail_the_call_and_counts_the_loss(monkeypatch):
    def broken_publish(event):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(telemetry, "publish", broken_publish)
    model = model_with(FakeDeepSeek([answer("still works")]))
    before = lost_events()

    result = model.invoke([HumanMessage("hello")])

    assert result.content == "still works"
    assert lost_events() == before + 1


def test_unreachable_redis_does_not_fail_the_call():
    # The default test run blocks sockets, so the real publisher cannot connect.
    model = model_with(FakeDeepSeek([answer("still works")]))
    before = lost_events()

    assert model.invoke([HumanMessage("hello")]).content == "still works"
    assert lost_events() == before + 1


def test_guard_rail_accepts_isolated_telemetry(monkeypatch):
    def broken_publish(event):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(telemetry, "publish", broken_publish)
    model = model_with(FakeDeepSeek([answer()]))

    assert check_telemetry_failure_is_isolated(lambda: model.invoke([HumanMessage("hi")])) == []


def test_guard_rail_reports_a_call_that_telemetry_breaks():
    def call_broken_by_telemetry():
        raise ConnectionError("redis is down")

    assert len(check_telemetry_failure_is_isolated(call_broken_by_telemetry)) == 1


@pytest.mark.parametrize(("calls", "recorded"), [(3, 2), (2, 3), (1, 0)])
def test_guard_rail_reports_calls_that_do_not_match_events(calls, recorded):
    assert len(check_every_call_recorded(calls, recorded)) == 1


def test_guard_rail_accepts_matching_counts():
    assert check_every_call_recorded(3, 3) == []


def test_component_is_required():
    with pytest.raises(TypeError, match="component"):
        get_chat_model(ModelTier.SMALL)


@pytest.mark.parametrize(
    ("called_at", "band"),
    [
        (datetime(2026, 11, 2, 1, 0, tzinfo=UTC), "peak"),  # Monday, start of first window
        (datetime(2026, 11, 2, 3, 59, tzinfo=UTC), "peak"),
        (datetime(2026, 11, 2, 4, 0, tzinfo=UTC), "off_peak"),  # between the windows
        (datetime(2026, 11, 2, 6, 0, tzinfo=UTC), "peak"),
        (datetime(2026, 11, 2, 9, 59, tzinfo=UTC), "peak"),
        (datetime(2026, 11, 2, 10, 0, tzinfo=UTC), "off_peak"),
        (datetime(2026, 11, 2, 0, 59, tzinfo=UTC), "off_peak"),
        (datetime(2026, 11, 7, 2, 0, tzinfo=UTC), "off_peak"),  # Saturday
        (datetime(2026, 11, 8, 8, 0, tzinfo=UTC), "off_peak"),  # Sunday
    ],
)
def test_time_band(called_at, band):
    assert time_band(called_at) == band


def test_time_band_converts_other_timezones_to_utc():
    called_at = datetime.fromisoformat("2026-11-01T20:00:00-05:00")  # Monday 01:00 UTC

    assert time_band(called_at) == "peak"


def test_metrics_endpoint_serves_llm_metrics(events):
    model_with(FakeDeepSeek([answer()]), component="metrics-test").invoke([HumanMessage("hi")])

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert 'llm_calls_total{component="metrics-test"' in response.text
    assert 'llm_tokens_total{component="metrics-test"' in response.text
    assert 'llm_latency_ms_bucket{component="metrics-test"' in response.text
    assert "telemetry_events_lost_total" in response.text
