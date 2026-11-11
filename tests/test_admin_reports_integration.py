"""Integration: Overview, Users and Savings figures against seeded rows in real Postgres.

The local database holds other rows too, so totals are checked by what the seed adds.
"""

import json
import uuid
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.types.json import Jsonb

from owl_mind.core.config import get_settings
from owl_mind.worker.telemetry import time_band
from tests.guard_rails import check_access_logged
from tests.test_admin_integration import (
    SECRET_TITLE,
    SECRET_TODO,
    Seed,
    migrated,  # noqa: F401  Applied to all.
    with_store,
)

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

# Seed's first call: 100 in, 10 out, 75 cache hits, $0.50. Its second: unknown usage, an error.
PRICES = {"peak": (0.006, 0.30), "off_peak": (0.003, 0.15)}


def stopped_seed() -> Seed:
    return Seed(
        extra_events=[
            ("000004:done", "done", {"status": "stopped", "reason": "token_limit"}),
            ("000005:budget_step", "budget_step", {"step": "wind_down", "percent": 72}),
            ("000006:integrity", "integrity", {"result": "retried", "reason": "full_code"}),
            ("000007:answer", "answer", {"mode": "joined", "parts": 2}),
        ]
    )


def summary_call(run_id) -> None:
    with psycopg.connect(get_settings().database_url) as db:
        db.execute(
            "INSERT INTO ops.model_calls (call_id, called_at, run_id, component, provider, model,"
            " input_tokens, output_tokens, cache_hit_tokens, cache_miss_tokens, usage_complete,"
            " cost_usd, latency_ms, outcome) VALUES (%s, now(), %s, 'summarization', 'deepseek',"
            " 'deepseek-flash', 3000, 400, 0, 3000, true, 0.001, 9, 'success')",
            (uuid.uuid4(), run_id),
        )


def test_overview_counts_what_the_seed_adds_today():
    before = with_store(lambda store: store.overview())
    stopped_seed()
    after = with_store(lambda store: store.overview())

    def calls(view):
        return view["today"]["calls"]

    assert calls(after)["calls"] - calls(before)["calls"] == 2
    assert calls(after)["errors"] - calls(before)["errors"] == 1
    assert calls(after)["unknown_cost"] - calls(before)["unknown_cost"] == 1
    assert round(calls(after)["cost_usd"] - (calls(before)["cost_usd"] or 0), 6) == 0.5

    def stopped(view, reason):
        return next((r["runs"] for r in view["today"]["stopped"] if r["reason"] == reason), 0)

    assert stopped(after, "token_limit") - stopped(before, "token_limit") == 1

    def runs(view):
        return sum(r["runs"] for r in view["last_7_days"]["runs"] if r["route"] == "orchestrated")

    assert runs(after) - runs(before) == 1


def test_users_show_a_students_usage_and_unknowns():
    seed = Seed()

    [row] = with_store(lambda store: store.users(seed.email))

    assert row["email"] == seed.email
    assert (row["runs_today"], row["runs_30d"]) == (1, 1)
    # The call with unknown usage adds nothing, and is counted as unknown instead.
    assert (row["tokens_today"], row["tokens_30d"]) == (110, 110)
    assert (row["cost_today"], row["cost_30d"]) == (0.5, 0.5)
    assert row["unknown_usage"] == 1


def test_a_student_with_no_runs_reads_unknown_not_zero():
    student = uuid.uuid4()
    email = f"quiet-{student.hex}@example.com"
    with psycopg.connect(get_settings().database_url) as db:
        db.execute("INSERT INTO app.students (id, email) VALUES (%s, %s)", (student, email))

    [row] = with_store(lambda store: store.users(email))

    assert (row["runs_today"], row["runs_30d"]) == (0, 0)
    assert row["tokens_30d"] is None and row["cost_30d"] is None


def test_opening_a_students_usage_writes_one_access_log_row():
    seed = Seed()
    viewer = uuid.uuid4()

    def rows() -> int:
        with psycopg.connect(get_settings().database_url) as db:
            return db.execute(
                "SELECT count(*) FROM ops.access_log WHERE student_id = %s AND viewer_id = %s",
                (seed.student, viewer),
            ).fetchone()[0]

    before = rows()
    usage = with_store(lambda store: store.user_usage(seed.student, viewer))
    after = rows()

    assert check_access_logged(before, after) == []
    [day] = usage["days"]
    assert (day["runs"], day["tokens"], day["cost_usd"], day["unknown_usage"]) == (1, 110, 0.5, 1)
    assert [run["run_id"] for run in usage["runs"]] == [str(seed.run)]
    with psycopg.connect(get_settings().database_url) as db:
        reason = db.execute(
            "SELECT reason, run_id FROM ops.access_log WHERE viewer_id = %s", (viewer,)
        ).fetchone()
    assert reason == ("user usage", None)


def test_an_unknown_student_logs_nothing():
    assert with_store(lambda store: store.user_usage(uuid.uuid4(), uuid.uuid4())) is None


def test_savings_count_cache_summaries_ladder_integrity_and_joined_answers():
    before = with_store(lambda store: store.savings(1))
    seed = stopped_seed()
    summary_call(seed.run)
    after = with_store(lambda store: store.savings(1))

    hit_price, miss_price = PRICES[time_band(datetime.now(UTC))]
    expected = 75 * (miss_price - hit_price) / 1_000_000
    assert round((after["cache"]["saved_usd"] or 0) - (before["cache"]["saved_usd"] or 0), 9) == (
        round(expected, 9)
    )
    gained = after["summaries"]["tokens_removed"] - (before["summaries"]["tokens_removed"] or 0)
    assert gained == 2600
    assert after["ladder"].get("wind_down", 0) - before["ladder"].get("wind_down", 0) == 1
    assert after["integrity"].get("retried", 0) - before["integrity"].get("retried", 0) == 1
    assert after["joined"] - before["joined"] == 1
    assert after["runs"]["with_agent"] - before["runs"]["with_agent"] == 1
    assert all(after["how"][key] for key in ("cache", "summaries", "ladder", "joined"))


def test_no_page_holds_message_text_or_titles():
    seed = stopped_seed()
    with psycopg.connect(get_settings().database_url) as db:
        db.execute(
            "INSERT INTO app.run_events (run_id, key, kind, payload) VALUES (%s, %s, %s, %s)",
            (seed.run, "000009:integrity", "integrity", Jsonb({"result": SECRET_TODO})),
        )

    async def body(store):
        return [
            await store.overview(),
            await store.users(seed.email),
            await store.user_usage(seed.student, uuid.uuid4()),
            await store.savings(1),
        ]

    shown = json.dumps(with_store(body), default=str)
    assert SECRET_TITLE not in shown and SECRET_TODO not in shown
    assert "build_hint" not in shown and "args" not in shown
