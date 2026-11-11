"""Admin's Overview, Users and Savings figures. Counts, tokens, costs and statuses only.

Every query reads metadata: runs, run events and ops.model_calls. None reads a message, a
thread title or memory. A missing figure stays None, which the page shows as unknown, never 0.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# DeepSeek's peak hours, as worker/telemetry.py's time_band: weekdays 01:00-04:00 and
# 06:00-10:00 UTC. Kept in SQL so a saving uses the price of the band the call was made in.
_BAND = """CASE WHEN extract(isodow FROM m.called_at AT TIME ZONE 'UTC') <= 5
    AND (extract(hour FROM m.called_at AT TIME ZONE 'UTC') BETWEEN 1 AND 3
         OR extract(hour FROM m.called_at AT TIME ZONE 'UTC') BETWEEN 6 AND 9)
    THEN 'peak' ELSE 'off_peak' END"""

_RUNS = """
SELECT coalesce(route, 'none') AS route, status, count(*) AS runs
FROM app.runs WHERE created_at >= :since GROUP BY 1, 2 ORDER BY 1, 2
"""
_STOPPED = """
SELECT coalesce(e.payload->>'reason', 'unknown') AS reason, count(*) AS runs
FROM app.run_events e JOIN app.runs r ON r.id = e.run_id
WHERE r.created_at >= :since AND e.kind = 'done' AND e.payload->>'status' = 'stopped'
GROUP BY 1 ORDER BY 2 DESC, 1
"""
_CALLS = """
SELECT count(*) AS calls, sum(cost_usd) AS cost_usd,
    count(*) FILTER (WHERE cost_usd IS NULL) AS unknown_cost,
    count(*) FILTER (WHERE outcome = 'error') AS errors,
    count(*) FILTER (WHERE outcome = 'truncated') AS truncated
FROM ops.model_calls WHERE called_at >= :since
"""
_USERS = """
SELECT s.id AS student_id, s.email,
    count(DISTINCT r.id) FILTER (WHERE r.created_at >= :today) AS runs_today,
    count(DISTINCT r.id) AS runs_30d,
    sum(m.input_tokens + m.output_tokens) FILTER (WHERE m.called_at >= :today) AS tokens_today,
    sum(m.input_tokens + m.output_tokens) AS tokens_30d,
    sum(m.cost_usd) FILTER (WHERE m.called_at >= :today) AS cost_today,
    sum(m.cost_usd) AS cost_30d,
    count(m.call_id) FILTER (WHERE NOT m.usage_complete) AS unknown_usage
FROM app.students s
LEFT JOIN app.runs r ON r.student_id = s.id AND r.created_at >= :since
LEFT JOIN ops.model_calls m ON m.run_id = r.id
WHERE (:query = '' OR s.email ILIKE '%' || :query || '%')
GROUP BY s.id, s.email
ORDER BY sum(m.cost_usd) DESC NULLS LAST, s.email
LIMIT :limit
"""
_USER_DAYS = """
SELECT (r.created_at AT TIME ZONE 'UTC')::date AS day, count(DISTINCT r.id) AS runs,
    sum(m.input_tokens + m.output_tokens) AS tokens, sum(m.cost_usd) AS cost_usd,
    count(m.call_id) FILTER (WHERE NOT m.usage_complete) AS unknown_usage
FROM app.runs r LEFT JOIN ops.model_calls m ON m.run_id = r.id
WHERE r.student_id = :student AND r.created_at >= :since
GROUP BY 1 ORDER BY 1 DESC
"""
_USER_RUNS = """
SELECT id AS run_id, created_at, route, task_kind, status FROM app.runs
WHERE student_id = :student ORDER BY created_at DESC LIMIT 20
"""
_LOG_USER_VIEW = """
INSERT INTO ops.access_log (id, viewer_id, student_id, run_id, reason)
VALUES (:id, :viewer, :student, NULL, 'user usage')
"""
_CACHE = f"""
SELECT sum(m.cache_hit_tokens) AS cache_hit_tokens,
    sum(m.cache_miss_tokens) AS cache_miss_tokens,
    sum(m.cache_hit_tokens * (p.cache_miss_usd_per_mtok - p.cache_hit_usd_per_mtok) / 1000000)
        AS saved_usd
FROM ops.model_calls m
LEFT JOIN LATERAL (
    SELECT cache_hit_usd_per_mtok, cache_miss_usd_per_mtok FROM ops.model_prices p
    WHERE p.model = m.model AND p.band = {_BAND} AND p.effective_from <= m.called_at
    ORDER BY p.effective_from DESC LIMIT 1
) p ON true
WHERE m.called_at >= :since
"""
_SUMMARIES = """
SELECT count(*) AS calls, sum(input_tokens - output_tokens) AS tokens_removed,
    sum(cost_usd) AS cost_usd
FROM ops.model_calls WHERE component = 'summarization' AND called_at >= :since
"""
_EVENTS = """
SELECT e.kind, coalesce(e.payload->>'step', e.payload->>'result', e.payload->>'mode') AS name,
    count(*) AS count
FROM app.run_events e JOIN app.runs r ON r.id = e.run_id
WHERE r.created_at >= :since AND e.kind IN ('budget_step', 'integrity', 'answer')
GROUP BY 1, 2 ORDER BY 1, 2
"""
_NO_AGENT = """
SELECT count(*) FILTER (WHERE route IN ('rule', 'template', 'handoff')) AS without_agent,
    count(*) FILTER (WHERE route = 'orchestrated') AS with_agent
FROM app.runs WHERE created_at >= :since
"""

# Event values are checked against what code writes, as Admin does everywhere: anything else
# could be text, so it's counted as "other" and never shown.
_KNOWN = {
    "budget_step": {"wind_down", "final", "stopped"},
    "integrity": {"passed", "retried", "blocked", "withheld"},
    "answer": {"joined"},
}

HOW = {
    "cache": "Cache-hit tokens x (normal input price - cache price), at each call's own price.",
    "summaries": "Each summary replaced about (its input - its output) tokens of history in later "
    "calls. Its own cost is shown beside it.",
    "ladder": "Runs that stepped down instead of spending past their budget.",
    "integrity": "Answers checked before showing: passed, rewritten once, or replaced by a hint.",
    "joined": "Two-tutor answers joined in code, each saving the orchestrator's rewrite call.",
    "no_agent": "Runs answered by a rule, template or handoff, with no agent.",
}


def _row(row: Any) -> dict[str, Any]:
    values = dict(row._mapping)
    for key, value in values.items():
        if isinstance(value, uuid.UUID):
            values[key] = str(value)
        elif hasattr(value, "isoformat"):
            values[key] = value.isoformat()
        elif type(value).__name__ == "Decimal":
            values[key] = float(value)
    return values


async def _rows(session: AsyncSession, sql: str, params: dict) -> list[dict[str, Any]]:
    return [_row(row) for row in (await session.execute(text(sql), params)).all()]


def today_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def overview(session: AsyncSession) -> dict[str, Any]:
    windows = {"today": today_start(), "last_7_days": datetime.now(UTC) - timedelta(days=7)}
    result: dict[str, Any] = {}
    for name, since in windows.items():
        params = {"since": since}
        result[name] = {
            "runs": await _rows(session, _RUNS, params),
            "stopped": await _rows(session, _STOPPED, params),
            "calls": (await _rows(session, _CALLS, params))[0],
        }
    return result


async def users(session: AsyncSession, query: str, limit: int) -> list[dict[str, Any]]:
    params = {
        "since": datetime.now(UTC) - timedelta(days=30),
        "today": today_start(),
        "query": query.strip(),
        "limit": limit,
    }
    return await _rows(session, _USERS, params)


async def user_usage(
    session: AsyncSession, student_id: uuid.UUID, viewer_id: uuid.UUID
) -> dict[str, Any] | None:
    """One student's days and recent runs. The view is logged in the same transaction."""
    email = await session.scalar(
        text("SELECT email FROM app.students WHERE id = :id"), {"id": student_id}
    )
    if email is None:
        return None
    await session.execute(
        text(_LOG_USER_VIEW), {"id": uuid.uuid4(), "viewer": viewer_id, "student": student_id}
    )
    params = {"student": student_id, "since": datetime.now(UTC) - timedelta(days=30)}
    return {
        "student_id": str(student_id),
        "email": email,
        "days": await _rows(session, _USER_DAYS, params),
        "runs": await _rows(session, _USER_RUNS, {"student": student_id}),
    }


async def savings(session: AsyncSession, days: int) -> dict[str, Any]:
    params = {"since": datetime.now(UTC) - timedelta(days=days)}
    cache = (await _rows(session, _CACHE, params))[0]
    hit, miss = cache["cache_hit_tokens"], cache["cache_miss_tokens"]
    known = (hit or 0) + (miss or 0)
    cache["cache_hit_rate"] = None if not known else (hit or 0) / known
    counts: dict[str, dict[str, int]] = {}
    for row in await _rows(session, _EVENTS, params):
        name = row["name"] if row["name"] in _KNOWN[row["kind"]] else "other"
        group = counts.setdefault(row["kind"], {})
        group[name] = group.get(name, 0) + row["count"]
    return {
        "days": days,
        "how": HOW,
        "cache": cache,
        "summaries": (await _rows(session, _SUMMARIES, params))[0],
        "ladder": counts.get("budget_step", {}),
        "integrity": counts.get("integrity", {}),
        "joined": counts.get("answer", {}).get("joined", 0),
        "runs": (await _rows(session, _NO_AGENT, params))[0],
    }
