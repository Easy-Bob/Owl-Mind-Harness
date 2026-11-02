"""Reads stream:telemetry, calculates cost and writes ops.model_calls."""

import json
import logging
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import psycopg
import redis

from owl_mind.core.config import get_settings
from owl_mind.core.telemetry import TELEMETRY_STREAM

logger = logging.getLogger(__name__)

GROUP = "telemetry"
CONSUMER = "telemetry-1"
_STALE_MS = 60_000
_BATCH = 100

_INSERT = """
INSERT INTO ops.model_calls (
    call_id, called_at, run_id, component, provider, model, input_tokens, output_tokens,
    cache_hit_tokens, cache_miss_tokens, reasoning_tokens, usage_complete, cost_usd, latency_ms,
    outcome, prompt_version
) VALUES (
    %(call_id)s, %(called_at)s, %(run_id)s, %(component)s, %(provider)s, %(model)s,
    %(input_tokens)s, %(output_tokens)s, %(cache_hit_tokens)s, %(cache_miss_tokens)s,
    %(reasoning_tokens)s, %(usage_complete)s, %(cost_usd)s, %(latency_ms)s, %(outcome)s,
    %(prompt_version)s
)
ON CONFLICT (call_id, called_at) DO NOTHING
"""

_PRICE = """
SELECT cache_hit_usd_per_mtok, cache_miss_usd_per_mtok, output_usd_per_mtok
FROM ops.model_prices
WHERE model = %s AND band = %s AND effective_from <= %s
ORDER BY effective_from DESC
LIMIT 1
"""


_INSERT_ACTIVITY = """
INSERT INTO ops.activity_events (event_id, student_id, thread_id, event_type, occurred_at)
VALUES (%(event_id)s, %(student_id)s, %(thread_id)s, %(event_type)s, %(occurred_at)s)
ON CONFLICT (event_id) DO NOTHING
"""

_COUNT_ACTIVITY = """
INSERT INTO ops.activity_daily (student_id, day, sessions_started, messages_sent)
VALUES (%(student_id)s, %(day)s, %(sessions)s, %(messages)s)
ON CONFLICT (student_id, day) DO UPDATE SET
    sessions_started = ops.activity_daily.sessions_started + EXCLUDED.sessions_started,
    messages_sent = ops.activity_daily.messages_sent + EXCLUDED.messages_sent
"""


def _write_activity(event: dict[str, Any], db: psycopg.Connection) -> None:
    # Count only a newly written event, so a redelivered one can't count twice.
    if db.execute(_INSERT_ACTIVITY, event).rowcount == 0:
        return
    day = datetime.fromisoformat(event["occurred_at"]).astimezone(UTC).date()
    db.execute(
        _COUNT_ACTIVITY,
        {
            "student_id": event["student_id"],
            "day": day,
            "sessions": int(event["event_type"] == "session_started"),
            "messages": int(event["event_type"] == "message_sent"),
        },
    )


def time_band(called_at: datetime) -> str:
    """DeepSeek's peak hours in UTC. Chinese public holidays are off-peak but not modeled yet."""
    utc = called_at.astimezone(UTC)
    is_peak_hour = 1 <= utc.hour < 4 or 6 <= utc.hour < 10
    return "peak" if utc.weekday() < 5 and is_peak_hour else "off_peak"


def calculate_cost(event: dict[str, Any], db: psycopg.Connection) -> Decimal | None:
    """Return None when tokens or the price are unknown. A guessed cost would look real."""
    tokens = (event["cache_hit_tokens"], event["cache_miss_tokens"], event["output_tokens"])
    if None in tokens:
        return None
    called_at = datetime.fromisoformat(event["called_at"])
    price = db.execute(_PRICE, (event["model"], time_band(called_at), called_at)).fetchone()
    if price is None:
        logger.warning("No price for model %s", event["model"])
        return None
    return sum(count * rate for count, rate in zip(tokens, price, strict=True)) / Decimal(1_000_000)


def _row(event: dict[str, Any], db: psycopg.Connection) -> dict[str, Any]:
    usage_complete = None not in (
        event["input_tokens"],
        event["output_tokens"],
        event["cache_hit_tokens"],
    )
    return {**event, "usage_complete": usage_complete, "cost_usd": calculate_cost(event, db)}


def ensure_group(client: redis.Redis) -> None:
    try:
        client.xgroup_create(TELEMETRY_STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def _read(client: redis.Redis, block_ms: int) -> list[tuple[bytes, dict[bytes, bytes]]]:
    # Messages a crashed consumer left unacknowledged come first.
    _, stale, _ = client.xautoclaim(
        TELEMETRY_STREAM, GROUP, CONSUMER, _STALE_MS, start_id="0", count=_BATCH
    )
    if stale:
        return stale
    streams = client.xreadgroup(
        GROUP, CONSUMER, {TELEMETRY_STREAM: ">"}, count=_BATCH, block=block_ms or None
    )
    return streams[0][1] if streams else []


def process_once(client: redis.Redis, db: psycopg.Connection, block_ms: int = 0) -> int:
    """Handle one batch and return its size. Rows commit before ack, so a crash only repeats."""
    messages = _read(client, block_ms)
    for message_id, fields in messages:
        try:
            # A savepoint per event, so dropping a bad one keeps the good rows in this batch.
            with db.transaction():
                event = json.loads(fields[b"event"])
                if event.get("kind") == "activity":
                    _write_activity(event, db)
                else:
                    db.execute(_INSERT, _row(event, db))
        except KeyError, ValueError, TypeError, psycopg.DataError, psycopg.IntegrityError:
            # A malformed event can never succeed, so retrying would block the stream.
            logger.warning("Dropped malformed telemetry event %s", message_id)
    db.commit()
    if messages:
        client.xack(TELEMETRY_STREAM, GROUP, *[message_id for message_id, _ in messages])
    return len(messages)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url)
    while True:
        # Unacknowledged events are read again, so reconnecting loses nothing.
        try:
            with psycopg.connect(settings.database_url) as db:
                while True:
                    ensure_group(client)
                    process_once(client, db, block_ms=5000)
        except (redis.RedisError, psycopg.OperationalError) as exc:
            logger.warning("Service unavailable, retrying: %s", type(exc).__name__)
            time.sleep(1)


if __name__ == "__main__":
    main()
