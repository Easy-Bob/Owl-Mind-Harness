"""Data retention and account deletion (M4-G).

A nightly job:
- keeps only each idle chat's latest checkpoint, which is all its history needs;
- deletes chats idle past the chat period (default 180 days) with their checkpoints, run
  records and memory episodes;
- deletes access-log rows and model calls past the log period (default 400 days);
- creates the next months' partitions of ops.model_calls ahead of time.

Deleting an account removes everything that belongs to the student. What they used stays only
as counts per day in ops.deleted_usage, and their model calls lose the link to any run.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.memory import EPISODES, Memory, namespace

logger = logging.getLogger(__name__)

# A chat with an unfinished run is never pruned or deleted.
_BUSY = "('queued', 'running', 'waiting_approval')"
_IDLE_THREADS = f"""
SELECT t.id FROM app.threads t
WHERE NOT EXISTS (
    SELECT 1 FROM app.runs r
    WHERE r.thread_id = t.id AND (r.created_at >= :before OR r.status IN {_BUSY})
) AND t.created_at < :before
"""
_PRUNE_CHECKPOINTS = """
WITH latest AS (
    SELECT DISTINCT ON (thread_id, checkpoint_ns) thread_id, checkpoint_ns, checkpoint_id
    FROM lg_checkpoint.checkpoints WHERE thread_id = ANY(:threads)
    ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
)
DELETE FROM lg_checkpoint.checkpoints c
WHERE c.thread_id = ANY(:threads) AND NOT EXISTS (
    SELECT 1 FROM latest l WHERE l.thread_id = c.thread_id
        AND l.checkpoint_ns = c.checkpoint_ns AND l.checkpoint_id = c.checkpoint_id
)
"""
_PRUNE_WRITES = """
DELETE FROM lg_checkpoint.checkpoint_writes w
WHERE w.thread_id = ANY(:threads) AND NOT EXISTS (
    SELECT 1 FROM lg_checkpoint.checkpoints c WHERE c.thread_id = w.thread_id
        AND c.checkpoint_ns = w.checkpoint_ns AND c.checkpoint_id = w.checkpoint_id
)
"""
# A blob stays while the kept checkpoint still names its version of the channel.
_PRUNE_BLOBS = """
DELETE FROM lg_checkpoint.checkpoint_blobs b
WHERE b.thread_id = ANY(:threads) AND NOT EXISTS (
    SELECT 1 FROM lg_checkpoint.checkpoints c WHERE c.thread_id = b.thread_id
        AND c.checkpoint_ns = b.checkpoint_ns
        AND c.checkpoint->'channel_versions'->>b.channel = b.version
)
"""
# Rows that hang off a chat, deepest first, so no foreign key blocks a delete.
_THREAD_ROWS = [
    "DELETE FROM app.forum_replies WHERE handoff_id IN"
    " (SELECT id FROM app.handoffs WHERE thread_id = ANY(:ids))",
    "DELETE FROM app.forum_posts WHERE handoff_id IN"
    " (SELECT id FROM app.handoffs WHERE thread_id = ANY(:ids))",
    "DELETE FROM app.handoffs WHERE thread_id = ANY(:ids)",
    "DELETE FROM app.approvals WHERE thread_id = ANY(:ids)",
    "DELETE FROM app.quiz_posts WHERE run_id IN"
    " (SELECT id FROM app.runs WHERE thread_id = ANY(:ids))",
    "DELETE FROM app.run_events WHERE run_id IN"
    " (SELECT id FROM app.runs WHERE thread_id = ANY(:ids))",
    "DELETE FROM app.runs WHERE thread_id = ANY(:ids)",
    "DELETE FROM app.memory_writes WHERE thread_id = ANY(:ids)",
    "DELETE FROM ops.activity_events WHERE thread_id = ANY(:ids)",
    "DELETE FROM app.threads WHERE id = ANY(:ids)",
]
# Everything else of one student's, once their chats are gone.
_STUDENT_ROWS = [
    "DELETE FROM app.grades WHERE attempt_id IN"
    " (SELECT id FROM app.quiz_attempts WHERE student_id = :student)",
    "DELETE FROM app.quiz_attempts WHERE student_id = :student",
    "DELETE FROM app.rubrics WHERE student_id = :student",
    "DELETE FROM app.quiz_specs WHERE student_id = :student",
    "DELETE FROM app.review_schedule WHERE student_id = :student",
    "DELETE FROM app.study_plans WHERE student_id = :student",
    "DELETE FROM app.forum_accounts WHERE student_id = :student",
    "DELETE FROM ops.activity_events WHERE student_id = :student",
    "DELETE FROM ops.activity_daily WHERE student_id = :student",
    # Admin's record of who looked at this student keeps the view, not the student.
    f"UPDATE ops.access_log SET student_id = '{uuid.UUID(int=0)}' WHERE student_id = :student",
    "DELETE FROM app.students WHERE id = :student",
]
_USAGE = """
SELECT (r.created_at AT TIME ZONE 'UTC')::date AS day, count(DISTINCT r.id) AS runs,
    count(m.call_id) AS calls,
    coalesce(sum(coalesce(m.input_tokens, 0) + coalesce(m.output_tokens, 0)), 0) AS tokens,
    coalesce(sum(m.cost_usd), 0) AS cost_usd
FROM app.runs r LEFT JOIN ops.model_calls m ON m.run_id = r.id
WHERE r.student_id = :student GROUP BY 1
"""
_KEEP_USAGE = """
INSERT INTO ops.deleted_usage (day, accounts, runs, model_calls, tokens, cost_usd)
VALUES (:day, :accounts, :runs, :calls, :tokens, :cost_usd)
ON CONFLICT (day) DO UPDATE SET accounts = ops.deleted_usage.accounts + EXCLUDED.accounts,
    runs = ops.deleted_usage.runs + EXCLUDED.runs,
    model_calls = ops.deleted_usage.model_calls + EXCLUDED.model_calls,
    tokens = ops.deleted_usage.tokens + EXCLUDED.tokens,
    cost_usd = ops.deleted_usage.cost_usd + EXCLUDED.cost_usd
"""


@dataclass(frozen=True)
class Periods:
    chat_days: int = 180
    log_days: int = 400
    checkpoint_idle_days: int = 1
    partitions_ahead: int = 3


def _month(day: date, ahead: int) -> date:
    index = day.year * 12 + day.month - 1 + ahead
    return date(index // 12, index % 12 + 1, 1)


class Retention:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        checkpointer: BaseCheckpointSaver,
        store: BaseStore,
        periods: Periods | None = None,
    ) -> None:
        self._sessions = sessions
        self._checkpointer = checkpointer
        self._store = store
        self.periods = periods or Periods()

    async def _threads(self, session: AsyncSession, before: datetime) -> list[uuid.UUID]:
        return list((await session.scalars(text(_IDLE_THREADS), {"before": before})).all())

    async def prune_checkpoints(self, now: datetime) -> int:
        """Keep only the latest checkpoint of each chat idle for a day. History reads only it."""
        before = now - timedelta(days=self.periods.checkpoint_idle_days)
        async with self._sessions.begin() as session:
            threads = [str(t) for t in await self._threads(session, before)]
            if not threads:
                return 0
            removed = await session.execute(text(_PRUNE_CHECKPOINTS), {"threads": threads})
            await session.execute(text(_PRUNE_WRITES), {"threads": threads})
            await session.execute(text(_PRUNE_BLOBS), {"threads": threads})
            return removed.rowcount or 0

    async def _delete_threads(
        self, session: AsyncSession, threads: list[tuple[uuid.UUID, uuid.UUID]]
    ) -> None:
        """Delete chats' rows in the session; their checkpoints and memory after it commits."""
        ids = [thread for thread, _ in threads]
        # Model calls stay for cost totals, without the link to a deleted run.
        await session.execute(
            text(
                "UPDATE ops.model_calls SET run_id = NULL WHERE run_id IN"
                " (SELECT id FROM app.runs WHERE thread_id = ANY(:ids))"
            ),
            {"ids": ids},
        )
        for statement in _THREAD_ROWS:
            await session.execute(text(statement), {"ids": ids})

    async def _forget(self, threads: list[tuple[uuid.UUID, uuid.UUID]]) -> None:
        for thread, student in threads:
            await self._checkpointer.adelete_thread(str(thread))
            space = namespace(student)
            for item in await self._store.asearch(space, limit=1000):
                if item.key.startswith(EPISODES) and item.key.endswith(f"-{str(thread)[:8]}.md"):
                    await self._store.adelete(space, item.key)

    async def delete_old_chats(self, now: datetime) -> int:
        before = now - timedelta(days=self.periods.chat_days)
        async with self._sessions.begin() as session:
            ids = await self._threads(session, before)
            rows = await session.execute(
                text("SELECT id, student_id FROM app.threads WHERE id = ANY(:ids)"), {"ids": ids}
            )
            threads = [(row.id, row.student_id) for row in rows]
            await self._delete_threads(session, threads)
        await self._forget(threads)
        return len(threads)

    async def trim_logs(self, now: datetime) -> dict[str, int]:
        before = now - timedelta(days=self.periods.log_days)
        async with self._sessions.begin() as session:
            views = await session.execute(
                text("DELETE FROM ops.access_log WHERE viewed_at < :before"), {"before": before}
            )
            calls = await session.execute(
                text("DELETE FROM ops.model_calls WHERE called_at < :before"), {"before": before}
            )
        return {"access_log": views.rowcount or 0, "model_calls": calls.rowcount or 0}

    async def ensure_partitions(self, today: date) -> list[str]:
        """This month's and the next months' partitions, so no call lands in the default one."""
        made = []
        async with self._sessions.begin() as session:
            for ahead in range(self.periods.partitions_ahead + 1):
                start, end = _month(today, ahead), _month(today, ahead + 1)
                name = f"model_calls_{start.year}_{start.month:02d}"
                exists = await session.scalar(
                    text("SELECT to_regclass(:name)"), {"name": f"ops.{name}"}
                )
                if exists is None:
                    await session.execute(
                        text(
                            f"CREATE TABLE ops.{name} PARTITION OF ops.model_calls "
                            f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
                        )
                    )
                    made.append(name)
        return made

    async def run(self, now: datetime | None = None) -> dict[str, Any]:
        """One nightly pass. Each step runs even if an earlier one failed."""
        now = now or datetime.now(UTC)
        result: dict[str, Any] = {}
        steps = {
            "partitions": lambda: self.ensure_partitions(now.date()),
            "checkpoints": lambda: self.prune_checkpoints(now),
            "old_chats": lambda: self.delete_old_chats(now),
            "logs": lambda: self.trim_logs(now),
        }
        for name, step in steps.items():
            try:
                result[name] = await step()
            except Exception as exc:
                logger.warning("Retention step %s failed: %s", name, type(exc).__name__)
                result[name] = "failed"
        if "failed" in result.values():
            raise RuntimeError(f"Retention steps failed: {result}")
        return result

    async def delete_account(self, student_id: uuid.UUID) -> bool:
        """Remove everything of one student's. False if there's no such account."""
        async with self._sessions.begin() as session:
            exists = await session.scalar(
                text("SELECT 1 FROM app.students WHERE id = :student"), {"student": student_id}
            )
            if exists is None:
                return False
            usage = (await session.execute(text(_USAGE), {"student": student_id})).all()
            for n, row in enumerate(usage):
                await session.execute(
                    text(_KEEP_USAGE),
                    {
                        "day": row.day,
                        "accounts": 1 if n == 0 else 0,
                        "runs": row.runs,
                        "calls": row.calls,
                        "tokens": row.tokens,
                        "cost_usd": row.cost_usd,
                    },
                )
            rows = await session.execute(
                text("SELECT id, student_id FROM app.threads WHERE student_id = :student"),
                {"student": student_id},
            )
            threads = [(row.id, row.student_id) for row in rows]
            await self._delete_threads(session, threads)
            # Rows tied to the student but not to a chat that still exists.
            for statement in (
                "DELETE FROM app.quiz_posts WHERE student_id = :student",
                "DELETE FROM app.approvals WHERE student_id = :student",
                "DELETE FROM app.memory_writes WHERE student_id = :student",
                "DELETE FROM app.forum_replies WHERE handoff_id IN"
                " (SELECT id FROM app.handoffs WHERE student_id = :student)",
                "DELETE FROM app.forum_posts WHERE handoff_id IN"
                " (SELECT id FROM app.handoffs WHERE student_id = :student)",
                "DELETE FROM app.handoffs WHERE student_id = :student",
                *_STUDENT_ROWS,
            ):
                await session.execute(text(statement), {"student": student_id})
        await self._forget(threads)
        await Memory(self._store).delete_all(student_id)
        return True
