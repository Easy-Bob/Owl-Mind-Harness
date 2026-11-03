"""Read-only Admin data. Message text never leaves here, and that includes thread titles."""

import re
import uuid
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Event payloads may hold model-written text, so only these fields are ever shown.
_SAFE_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "tool_start": ("name", "agent"),
    "tool_end": ("name", "status"),
    "subagent_start": ("name",),
    "subagent_end": ("status",),
    "usage": ("input_tokens", "output_tokens", "model_calls"),
    "done": ("status", "reason"),
    "retrieval": ("result_count", "top_score", "corpus_version", "cache_hit", "latency_ms"),
}

# Handoff events are checked value by value, because any free text could be a student's.
_HANDOFF_STEPS = {
    "drafted",
    "similar_found",
    "improved",
    "improve_failed",
    "improve_limit",
    "needs_review",
    "approved",
    "rate_limited",
    "reconnect_needed",
    "cancelled",
    "posted",
    "post_failed",
    "fallback",
    "replies_arrived",
    "poll_failed",
    "polling_stopped",
}
_REDACTION_KINDS = {"email", "secret", "student id", "netid", "phone", "name", "known detail"}
_HANDOFF_WORDS = {
    "status": {"retrying", "fallback"},
    "reason": {
        "no_forum",
        "retries_used",
        "reconnect_needed",
        "discussion_gone",
        "window_ended",
    },
    "posted_as": {"bot", "student"},
}
_HANDOFF_NUMBERS = ("count", "attempts", "remaining", "failures")
_FORUM_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/discussions/\d+")


def _safe_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    step = payload.get("step")
    if step not in _HANDOFF_STEPS:
        return {}
    safe: dict[str, Any] = {"step": step}
    removed = payload.get("removed")
    if isinstance(removed, dict):
        # Counts by kind only, never what was removed.
        safe["removed"] = {
            kind: count
            for kind, count in removed.items()
            if kind in _REDACTION_KINDS and isinstance(count, int) and not isinstance(count, bool)
        }
    for name in _HANDOFF_NUMBERS:
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            safe[name] = value
    for name, allowed in _HANDOFF_WORDS.items():
        if payload.get(name) in allowed:
            safe[name] = payload[name]
    if isinstance(payload.get("graded"), bool):
        safe["graded"] = payload["graded"]
    url = payload.get("url")
    if step == "posted" and isinstance(url, str) and _FORUM_URL.fullmatch(url):
        safe["url"] = url
    return safe


_RUNS = """
FROM app.runs r
JOIN app.threads t ON t.id = r.thread_id
JOIN app.students s ON s.id = t.student_id
WHERE (:query = '' OR s.email ILIKE '%' || :query || '%' OR r.id::text = :query)
    AND (:route = '' OR r.route = :route)
"""

_LIST = f"""
SELECT r.id, s.email, r.status, r.route, r.task_kind, r.created_at,
    (SELECT count(*) FROM ops.model_calls m WHERE m.run_id = r.id) AS model_calls,
    (SELECT sum(m.input_tokens + m.output_tokens) FROM ops.model_calls m WHERE m.run_id = r.id)
        AS tokens,
    (SELECT sum(m.cost_usd) FROM ops.model_calls m WHERE m.run_id = r.id) AS cost_usd,
    (SELECT count(*) FROM ops.model_calls m WHERE m.run_id = r.id AND NOT m.usage_complete)
        AS unknown_usage,
    EXISTS (SELECT 1 FROM app.run_events e WHERE e.run_id = r.id AND e.kind = 'retrieval')
        AS searched,
    (SELECT e.payload ->> 'step' FROM app.run_events e
        WHERE e.run_id = r.id AND e.kind = 'handoff' ORDER BY e.key DESC LIMIT 1) AS handoff_step
{_RUNS}
ORDER BY r.created_at DESC, r.id
LIMIT :limit OFFSET :offset
"""

_DETAIL = """
SELECT r.id, s.id AS student_id, s.email, r.status, r.route, r.task_kind, r.created_at
FROM app.runs r
JOIN app.threads t ON t.id = r.thread_id
JOIN app.students s ON s.id = t.student_id
WHERE r.id = :run_id
"""

_EVENTS = """
SELECT key, kind, payload, created_at FROM app.run_events WHERE run_id = :run_id ORDER BY key
"""

_CALLS = """
SELECT called_at, component, model, input_tokens, output_tokens, cache_hit_tokens,
    cache_miss_tokens, reasoning_tokens, cost_usd, latency_ms, outcome, usage_complete
FROM ops.model_calls WHERE run_id = :run_id ORDER BY called_at
"""

_LOG_VIEW = """
INSERT INTO ops.access_log (id, viewer_id, student_id, run_id, reason)
VALUES (:id, :viewer_id, :student_id, :run_id, 'request detail')
"""

_WINDOW = "m.called_at >= now() - make_interval(days => :days)"

_TOTALS = f"""
SELECT count(*) AS calls, sum(m.input_tokens) AS input_tokens,
    sum(m.output_tokens) AS output_tokens,
    sum(m.cost_usd) AS cost_usd, count(*) FILTER (WHERE NOT m.usage_complete) AS unknown_usage,
    sum(m.cache_hit_tokens) AS cache_hit, sum(m.cache_miss_tokens) AS cache_miss
FROM ops.model_calls m WHERE {_WINDOW}
"""


def _grouped(column: str, join: str = "") -> str:
    return f"""
SELECT {column} AS name, count(*) AS calls, sum(m.input_tokens) AS input_tokens,
    sum(m.output_tokens) AS output_tokens, sum(m.cost_usd) AS cost_usd,
    count(*) FILTER (WHERE NOT m.usage_complete) AS unknown_usage
FROM ops.model_calls m {join} WHERE {_WINDOW}
GROUP BY 1 ORDER BY sum(m.cost_usd) DESC NULLS LAST, 1
"""


_BY_COMPONENT = _grouped("m.component")
_BY_MODEL = _grouped("m.model")
_BY_TASK_KIND = _grouped("r.task_kind", "LEFT JOIN app.runs r ON r.id = m.run_id")
_OUTCOMES = f"""
SELECT m.outcome AS name, count(*) AS calls FROM ops.model_calls m WHERE {_WINDOW}
GROUP BY 1 ORDER BY 1
"""
_TOP_RUNS = f"""
SELECT m.run_id, sum(m.cost_usd) AS cost_usd, count(*) AS calls
FROM ops.model_calls m WHERE {_WINDOW} AND m.run_id IS NOT NULL AND m.cost_usd IS NOT NULL
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10
"""


def safe_event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "handoff":
        return _safe_handoff(payload)
    if kind == "todo_delta":
        return {"count": len(payload.get("todos", []))}
    if kind == "usage":
        # This event covers the agent only. The intent call appears in the model call table.
        return {
            f"agent_{name}": payload[name] for name in _SAFE_EVENT_FIELDS[kind] if name in payload
        }
    return {field: payload[field] for field in _SAFE_EVENT_FIELDS.get(kind, ()) if field in payload}


def _clean(row: Any) -> dict[str, Any]:
    """JSON-ready values. NULL stays None, which the page shows as unknown and never as 0."""
    values = dict(row._mapping)
    for key, value in values.items():
        if isinstance(value, uuid.UUID):
            values[key] = str(value)
        elif hasattr(value, "isoformat"):
            values[key] = value.isoformat()
        elif type(value).__name__ == "Decimal":
            values[key] = float(value)
    return values


@runtime_checkable
class AdminStore(Protocol):
    async def list_requests(
        self, query: str, page: int, size: int, route: str = ""
    ) -> dict[str, Any]: ...

    async def get_request(self, run_id: uuid.UUID, viewer_id: uuid.UUID) -> dict[str, Any] | None:
        """Return one run's timeline, and record that this viewer opened it."""

    async def model_call_summary(self, days: int) -> dict[str, Any]: ...


class PostgresAdminStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def list_requests(
        self, query: str, page: int, size: int, route: str = ""
    ) -> dict[str, Any]:
        filters = {"query": query.strip(), "route": route}
        params = {**filters, "limit": size, "offset": (page - 1) * size}
        async with self._sessions() as session:
            rows = (await session.execute(text(_LIST), params)).all()
            total = await session.scalar(text(f"SELECT count(*) {_RUNS}"), filters)
        items = []
        for row in rows:
            item = _clean(row)
            # The step comes from stored data, so only known step names are passed on.
            if item["handoff_step"] not in _HANDOFF_STEPS:
                item["handoff_step"] = None
            items.append(item)
        return {"items": items, "total": total, "page": page, "size": size, "route": route}

    async def get_request(self, run_id: uuid.UUID, viewer_id: uuid.UUID) -> dict[str, Any] | None:
        params = {"run_id": run_id}
        # One transaction: if the view can't be logged, the detail isn't returned.
        async with self._sessions.begin() as session:
            run = (await session.execute(text(_DETAIL), params)).first()
            if run is None:
                return None
            await session.execute(
                text(_LOG_VIEW),
                {
                    "id": uuid.uuid4(),
                    "viewer_id": viewer_id,
                    "student_id": run.student_id,
                    "run_id": run_id,
                },
            )
            events = (await session.execute(text(_EVENTS), params)).all()
            calls = (await session.execute(text(_CALLS), params)).all()
        detail = _clean(run)
        detail.pop("student_id")
        detail["events"] = [
            {
                "kind": event.kind,
                "at": event.created_at.isoformat(),
                "data": safe_event(event.kind, event.payload),
            }
            for event in events
        ]
        detail["calls"] = [_clean(call) for call in calls]
        return detail

    async def model_call_summary(self, days: int) -> dict[str, Any]:
        params = {"days": days}
        async with self._sessions() as session:

            async def rows(sql: str) -> list[dict[str, Any]]:
                return [_clean(row) for row in (await session.execute(text(sql), params)).all()]

            totals = (await rows(_TOTALS))[0]
            summary = {
                "by_component": await rows(_BY_COMPONENT),
                "by_model": await rows(_BY_MODEL),
                "by_task_kind": await rows(_BY_TASK_KIND),
                "outcomes": await rows(_OUTCOMES),
                "top_runs": await rows(_TOP_RUNS),
            }
        hit, miss = totals.pop("cache_hit"), totals.pop("cache_miss")
        known = (hit or 0) + (miss or 0)
        return {
            "days": days,
            "totals": totals,
            "cache_hit_rate": None if not known else (hit or 0) / known,
            **summary,
        }
