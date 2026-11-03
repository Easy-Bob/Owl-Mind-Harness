"""Handoff drafts, forum posts and their saved replies, in memory or in Postgres."""

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.forum.github import Reply

FIRST_POLL = timedelta(minutes=5)


@dataclass(frozen=True)
class Handoff:
    id: uuid.UUID
    student_id: uuid.UUID
    thread_id: uuid.UUID
    status: str
    title: str
    body: str
    graded: bool
    category: str
    url: str | None = None
    attempts: int = 0
    created_at: datetime | None = None
    # A GitHub login, or "bot" for the Owl-Mind app.
    posted_as: str | None = None


@dataclass(frozen=True)
class TrackedPost:
    """A posted discussion that may still get replies."""

    handoff_id: uuid.UUID
    node_id: str
    posted_at: datetime
    failures: int


@dataclass(frozen=True)
class SavedReply:
    id: str
    handoff_id: uuid.UUID
    author: str
    body: str
    created_at: datetime
    is_answer: bool
    seen: bool


@runtime_checkable
class HandoffStore(Protocol):
    async def save(self, handoff: Handoff) -> None: ...

    async def get(self, student_id: uuid.UUID, handoff_id: uuid.UUID) -> Handoff | None: ...

    async def for_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> list[Handoff]: ...

    async def claim(self, handoff_id: uuid.UUID, expected: str, status: str) -> bool:
        """Change the status only if it's still the expected one, so two approvals post once."""

    async def last_posted_at(self, student_id: uuid.UUID) -> datetime | None: ...

    async def record_post(
        self, handoff_id: uuid.UUID, url: str, node_id: str, posted_at: datetime
    ) -> None:
        """Also schedules the first check for replies."""

    async def due_for_retry(self) -> list[Handoff]: ...

    async def fail_unfinished(self) -> None:
        """After a restart, a post that was being sent may or may not exist on the forum.
        Retrying could post it twice, so the student gets the copyable fallback instead."""

    async def lease_due_posts(
        self, now: datetime, limit: int, lease: timedelta
    ) -> list[TrackedPost]:
        """Posts due a check. A leased post isn't handed to another poller until the lease ends."""

    async def save_replies(
        self,
        handoff_id: uuid.UUID,
        replies: list[Reply],
        complete: bool,
        next_poll_at: datetime | None,
    ) -> int:
        """Store a check's result and return how many replies are new. With no next time, or
        with a deleted discussion, the post stops being checked."""

    async def poll_failed(
        self, handoff_id: uuid.UUID, failures: int, next_poll_at: datetime | None
    ) -> None: ...

    async def discussion_gone(self, handoff_id: uuid.UUID) -> None: ...

    async def replies_for(
        self, student_id: uuid.UUID, handoff_ids: list[uuid.UUID]
    ) -> list[SavedReply]: ...

    async def unread_counts(self, student_id: uuid.UUID) -> dict[uuid.UUID, int]:
        """Unseen replies per thread."""

    async def mark_seen(self, student_id: uuid.UUID, handoff_id: uuid.UUID) -> None: ...


@dataclass
class _Post:
    url: str
    node_id: str
    posted_at: datetime
    state: str = "open"
    next_poll_at: datetime | None = None
    failures: int = 0
    lease_until: datetime | None = None


class InMemoryHandoffStore:
    def __init__(self) -> None:
        self.handoffs: dict[uuid.UUID, Handoff] = {}
        self.posts: dict[uuid.UUID, _Post] = {}
        self.replies: dict[str, SavedReply] = {}

    async def save(self, handoff: Handoff) -> None:
        self.handoffs[handoff.id] = handoff

    async def get(self, student_id, handoff_id) -> Handoff | None:
        handoff = self.handoffs.get(handoff_id)
        return handoff if handoff and handoff.student_id == student_id else None

    async def for_thread(self, student_id, thread_id) -> list[Handoff]:
        return [
            h
            for h in self.handoffs.values()
            if h.student_id == student_id and h.thread_id == thread_id
        ]

    async def claim(self, handoff_id, expected, status) -> bool:
        handoff = self.handoffs.get(handoff_id)
        if handoff is None or handoff.status != expected:
            return False
        self.handoffs[handoff_id] = replace(handoff, status=status)
        return True

    async def last_posted_at(self, student_id) -> datetime | None:
        times = [
            post.posted_at
            for handoff_id, post in self.posts.items()
            if self.handoffs[handoff_id].student_id == student_id
        ]
        return max(times, default=None)

    async def record_post(self, handoff_id, url, node_id, posted_at) -> None:
        self.posts.setdefault(
            handoff_id, _Post(url, node_id, posted_at, next_poll_at=posted_at + FIRST_POLL)
        )

    async def due_for_retry(self) -> list[Handoff]:
        return [h for h in self.handoffs.values() if h.status == "retrying"]

    async def fail_unfinished(self) -> None:
        for handoff in list(self.handoffs.values()):
            if handoff.status == "posting":
                self.handoffs[handoff.id] = replace(handoff, status="fallback")

    async def lease_due_posts(self, now, limit, lease) -> list[TrackedPost]:
        due = sorted(
            (
                (handoff_id, post)
                for handoff_id, post in self.posts.items()
                if post.state == "open"
                and post.next_poll_at is not None
                and post.next_poll_at <= now
                and (post.lease_until is None or post.lease_until < now)
            ),
            key=lambda item: item[1].next_poll_at,
        )[:limit]
        for _, post in due:
            post.lease_until = now + lease
        return [TrackedPost(i, p.node_id, p.posted_at, p.failures) for i, p in due]

    async def save_replies(self, handoff_id, replies, complete, next_poll_at) -> int:
        new = 0
        fetched = {reply.id for reply in replies}
        if complete:
            for reply_id in [r.id for r in self.replies.values() if r.handoff_id == handoff_id]:
                if reply_id not in fetched:
                    del self.replies[reply_id]
        for reply in replies:
            saved = self.replies.get(reply.id)
            if saved is None:
                new += 1
                seen = False
            else:
                seen = saved.seen
            self.replies[reply.id] = SavedReply(
                reply.id, handoff_id, reply.author, reply.body, reply.created_at,
                reply.is_answer, seen,
            )  # fmt: skip
        post = self.posts[handoff_id]
        post.failures, post.lease_until, post.next_poll_at = 0, None, next_poll_at
        if next_poll_at is None:
            post.state = "done"
        return new

    async def poll_failed(self, handoff_id, failures, next_poll_at) -> None:
        post = self.posts[handoff_id]
        post.failures, post.lease_until, post.next_poll_at = failures, None, next_poll_at
        if next_poll_at is None:
            post.state = "done"

    async def discussion_gone(self, handoff_id) -> None:
        post = self.posts[handoff_id]
        post.state, post.lease_until, post.next_poll_at = "gone", None, None

    async def replies_for(self, student_id, handoff_ids) -> list[SavedReply]:
        mine = {
            h
            for h in handoff_ids
            if h in self.handoffs and self.handoffs[h].student_id == student_id
        }
        found = [r for r in self.replies.values() if r.handoff_id in mine]
        return sorted(found, key=lambda r: r.created_at)

    async def unread_counts(self, student_id) -> dict[uuid.UUID, int]:
        counts: dict[uuid.UUID, int] = {}
        for reply in self.replies.values():
            handoff = self.handoffs[reply.handoff_id]
            if handoff.student_id == student_id and not reply.seen:
                counts[handoff.thread_id] = counts.get(handoff.thread_id, 0) + 1
        return counts

    async def mark_seen(self, student_id, handoff_id) -> None:
        if await self.get(student_id, handoff_id) is None:
            return
        for reply_id, reply in list(self.replies.items()):
            if reply.handoff_id == handoff_id:
                self.replies[reply_id] = replace(reply, seen=True)


_SAVE = """
INSERT INTO app.handoffs (id, student_id, thread_id, status, title, body, graded, category,
    url, attempts, posted_as)
VALUES (:id, :student_id, :thread_id, :status, :title, :body, :graded, :category, :url, :attempts,
    :posted_as)
ON CONFLICT (id) DO UPDATE SET status = :status, title = :title, body = :body, graded = :graded,
    url = :url, attempts = :attempts, posted_as = :posted_as, updated_at = now()
"""
_COLUMNS = (
    "id, student_id, thread_id, status, title, body, graded, category, url, attempts, created_at,"
    " posted_as"
)
_LEASE = """
UPDATE app.forum_posts SET lease_until = :now + :lease
WHERE handoff_id IN (
    SELECT handoff_id FROM app.forum_posts
    WHERE state = 'open' AND next_poll_at <= :now
        AND (lease_until IS NULL OR lease_until < :now)
    ORDER BY next_poll_at
    LIMIT :limit
    FOR UPDATE SKIP LOCKED
)
RETURNING handoff_id, node_id, posted_at, failures
"""
_UPSERT_REPLY = """
INSERT INTO app.forum_replies (id, handoff_id, author, body, created_at, is_answer)
VALUES (:id, :handoff_id, :author, :body, :created_at, :is_answer)
ON CONFLICT (id) DO UPDATE SET body = :body, is_answer = :is_answer
RETURNING (xmax = 0) AS inserted
"""
_REPLY_COLUMNS = "r.id, r.handoff_id, r.author, r.body, r.created_at, r.is_answer, r.seen"


class PostgresHandoffStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, handoff: Handoff) -> None:
        values = {k: v for k, v in vars(handoff).items() if k != "created_at"}
        async with self._sessions.begin() as session:
            await session.execute(text(_SAVE), values)

    async def _select(self, where: str, params: dict) -> list[Handoff]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(f"SELECT {_COLUMNS} FROM app.handoffs WHERE {where} ORDER BY created_at"),
                params,
            )
        return [Handoff(**row._mapping) for row in rows]

    async def get(self, student_id, handoff_id) -> Handoff | None:
        found = await self._select(
            "id = :id AND student_id = :student", {"id": handoff_id, "student": student_id}
        )
        return found[0] if found else None

    async def for_thread(self, student_id, thread_id) -> list[Handoff]:
        return await self._select(
            "student_id = :student AND thread_id = :thread",
            {"student": student_id, "thread": thread_id},
        )

    async def claim(self, handoff_id, expected, status) -> bool:
        async with self._sessions.begin() as session:
            claimed = await session.scalar(
                text(
                    "UPDATE app.handoffs SET status = :status, updated_at = now()"
                    " WHERE id = :id AND status = :expected RETURNING id"
                ),
                {"id": handoff_id, "expected": expected, "status": status},
            )
        return claimed is not None

    async def last_posted_at(self, student_id) -> datetime | None:
        async with self._sessions() as session:
            return await session.scalar(
                text(
                    "SELECT max(p.posted_at) FROM app.forum_posts p"
                    " JOIN app.handoffs h ON h.id = p.handoff_id WHERE h.student_id = :student"
                ),
                {"student": student_id},
            )

    async def record_post(self, handoff_id, url, node_id, posted_at) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "INSERT INTO app.forum_posts"
                    " (handoff_id, url, node_id, posted_at, next_poll_at)"
                    " VALUES (:id, :url, :node, :posted_at, :next)"
                    " ON CONFLICT (handoff_id) DO NOTHING"
                ),
                {
                    "id": handoff_id,
                    "url": url,
                    "node": node_id,
                    "posted_at": posted_at,
                    "next": posted_at + FIRST_POLL,
                },
            )

    async def due_for_retry(self) -> list[Handoff]:
        return await self._select("status = 'retrying'", {})

    async def fail_unfinished(self) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text("UPDATE app.handoffs SET status = 'fallback' WHERE status = 'posting'")
            )

    async def lease_due_posts(self, now, limit, lease) -> list[TrackedPost]:
        async with self._sessions.begin() as session:
            rows = await session.execute(text(_LEASE), {"now": now, "lease": lease, "limit": limit})
            return [TrackedPost(**row._mapping) for row in rows]

    async def save_replies(self, handoff_id, replies, complete, next_poll_at) -> int:
        new = 0
        async with self._sessions.begin() as session:
            if complete:
                await session.execute(
                    text(
                        "DELETE FROM app.forum_replies"
                        " WHERE handoff_id = :id AND NOT (id = ANY(:kept))"
                    ),
                    {"id": handoff_id, "kept": [reply.id for reply in replies]},
                )
            for reply in replies:
                inserted = await session.scalar(
                    text(_UPSERT_REPLY),
                    {
                        "id": reply.id,
                        "handoff_id": handoff_id,
                        "author": reply.author,
                        "body": reply.body,
                        "created_at": reply.created_at,
                        "is_answer": reply.is_answer,
                    },
                )
                new += bool(inserted)
            await session.execute(
                text(
                    "UPDATE app.forum_posts SET failures = 0, lease_until = NULL,"
                    " next_poll_at = :next,"
                    " state = CASE WHEN CAST(:next AS timestamptz) IS NULL THEN 'done'"
                    " ELSE state END"
                    " WHERE handoff_id = :id"
                ),
                {"id": handoff_id, "next": next_poll_at},
            )
        return new

    async def poll_failed(self, handoff_id, failures, next_poll_at) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE app.forum_posts SET failures = :failures, lease_until = NULL,"
                    " next_poll_at = :next,"
                    " state = CASE WHEN CAST(:next AS timestamptz) IS NULL THEN 'done'"
                    " ELSE state END"
                    " WHERE handoff_id = :id"
                ),
                {"id": handoff_id, "failures": failures, "next": next_poll_at},
            )

    async def discussion_gone(self, handoff_id) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE app.forum_posts SET state = 'gone', lease_until = NULL,"
                    " next_poll_at = NULL WHERE handoff_id = :id"
                ),
                {"id": handoff_id},
            )

    async def replies_for(self, student_id, handoff_ids) -> list[SavedReply]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(
                    f"SELECT {_REPLY_COLUMNS} FROM app.forum_replies r"
                    " JOIN app.handoffs h ON h.id = r.handoff_id"
                    " WHERE h.student_id = :student AND r.handoff_id = ANY(:ids)"
                    " ORDER BY r.created_at"
                ),
                {"student": student_id, "ids": list(handoff_ids)},
            )
        return [SavedReply(**row._mapping) for row in rows]

    async def unread_counts(self, student_id) -> dict[uuid.UUID, int]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(
                    "SELECT h.thread_id, count(*) FROM app.forum_replies r"
                    " JOIN app.handoffs h ON h.id = r.handoff_id"
                    " WHERE h.student_id = :student AND NOT r.seen GROUP BY h.thread_id"
                ),
                {"student": student_id},
            )
        return {thread_id: count for thread_id, count in rows}

    async def mark_seen(self, student_id, handoff_id) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE app.forum_replies r SET seen = true FROM app.handoffs h"
                    " WHERE h.id = r.handoff_id AND h.student_id = :student"
                    " AND r.handoff_id = :id AND NOT r.seen"
                ),
                {"student": student_id, "id": handoff_id},
            )
