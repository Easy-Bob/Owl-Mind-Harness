"""Token accounting for every model call. A telemetry failure never fails a run."""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import redis
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Histogram

from owl_mind.core.config import get_settings

logger = logging.getLogger(__name__)

TELEMETRY_STREAM = "stream:telemetry"
_STREAM_MAX_LENGTH = 100_000

# Callers set these in a run's config metadata. LangChain passes them down to subagent calls.
RUN_ID_KEY = "owl_run_id"
PROMPT_VERSION_KEY = "owl_prompt_version"

# Opening a tab, viewing history or reconnecting records nothing.
ACTIVITY_EVENT_TYPES = frozenset({"session_started", "message_sent"})

LLM_CALLS = Counter("llm_calls_total", "Model calls", ["component", "model", "outcome"])
LLM_TOKENS = Counter("llm_tokens_total", "Model tokens", ["component", "model", "kind"])
LLM_LATENCY = Histogram(
    "llm_latency_ms",
    "Model call latency in milliseconds",
    ["component", "model"],
    buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000),
)
EVENTS_LOST = Counter("telemetry_events_lost_total", "Telemetry events that could not be sent")


@lru_cache
def _client(url: str) -> redis.Redis:
    # Short timeouts: a slow Redis must not hold up a student's run.
    return redis.Redis.from_url(url, socket_connect_timeout=0.2, socket_timeout=0.2)


def publish(event: dict[str, Any]) -> None:
    client = _client(get_settings().redis_url)
    client.xadd(
        TELEMETRY_STREAM,
        {"event": json.dumps(event)},
        maxlen=_STREAM_MAX_LENGTH,
        approximate=True,
    )


def record_activity(event_type: str, student_id: uuid.UUID, thread_id: uuid.UUID) -> None:
    """Record user activity. Metadata only, and a failure never reaches the student."""
    if event_type not in ACTIVITY_EVENT_TYPES:
        raise ValueError(f"Not an activity event type: {event_type}")
    try:
        publish(
            {
                "kind": "activity",
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "student_id": str(student_id),
                "thread_id": str(thread_id),
                "occurred_at": datetime.now(UTC).isoformat(),
            }
        )
    except Exception as exc:
        EVENTS_LOST.inc()
        logger.warning("Activity event lost: %s", type(exc).__name__)


def _usage_fields(message: Any) -> dict[str, Any]:
    """Read tokens from a response. Anything the provider left out stays None, meaning unknown."""
    usage = getattr(message, "usage_metadata", None) or {}
    hit = usage.get("input_token_details", {}).get("cache_read")
    input_tokens = usage.get("input_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": usage.get("output_tokens"),
        "cache_hit_tokens": hit,
        "cache_miss_tokens": None if hit is None or input_tokens is None else input_tokens - hit,
        "reasoning_tokens": usage.get("output_token_details", {}).get("reasoning"),
    }


class TokenAccounting(BaseCallbackHandler):
    """Attached to every model in core/model.py, so subagent calls are counted too."""

    def __init__(self, component: str, model: str) -> None:
        self.component = component
        self.model = model
        self._started: dict[uuid.UUID, tuple[datetime, float, dict[str, Any]]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._started[run_id] = (datetime.now(UTC), time.monotonic(), metadata or {})

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        self._record(run_id, response=response)

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        self._record(run_id, response=None)

    def _record(self, call_id: uuid.UUID, response: LLMResult | None) -> None:
        try:
            event = self._event(call_id, response)
            self._count(event)
            publish(event)
        except Exception as exc:
            EVENTS_LOST.inc()
            logger.warning("Telemetry event lost: %s", type(exc).__name__)

    def _event(self, call_id: uuid.UUID, response: LLMResult | None) -> dict[str, Any]:
        called_at, started, metadata = self._started.pop(call_id)
        outcome = "error"
        usage = _usage_fields(None)
        if response is not None:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            usage = _usage_fields(message)
            finish = getattr(message, "response_metadata", {}).get("finish_reason")
            outcome = "truncated" if finish == "length" else "success"
        return {
            "call_id": str(call_id),
            "called_at": called_at.isoformat(),
            "run_id": metadata.get(RUN_ID_KEY),
            "component": self.component,
            "provider": "deepseek",
            "model": self.model,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "outcome": outcome,
            "prompt_version": metadata.get(PROMPT_VERSION_KEY),
            **usage,
        }

    def _count(self, event: dict[str, Any]) -> None:
        labels = {"component": event["component"], "model": event["model"]}
        LLM_CALLS.labels(**labels, outcome=event["outcome"]).inc()
        LLM_LATENCY.labels(**labels).observe(event["latency_ms"])
        for kind in ("input", "output"):
            if (tokens := event[f"{kind}_tokens"]) is not None:
                LLM_TOKENS.labels(**labels, kind=kind).inc(tokens)
