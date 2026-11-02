"""Tutor handoff: a fixed path with no model. Draft, redact, approve, then post.

Nothing reaches the forum without the student approving the exact text that will be posted.
"""

import re
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.forum.github import Forum, ForumUnavailable, SimilarPost
from owl_mind.forum.redact import CODE_LINES, escape_markdown, for_graded_problem, redact, trim_code

POST_INTERVAL = timedelta(minutes=10)
MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 6000
MAX_RETRIES = 3

_ERROR_LINE = re.compile(
    r"^.*(?:\w*(?:Error|Exception)\b|\bTraceback\b|\bfailed\b).*$", re.MULTILINE
)


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


class HandoffError(Exception):
    """The request can't be done. The message is safe to show the student."""


class NeedsReview(Exception):
    """Redaction changed the text, so the student must see and approve the new version."""

    def __init__(self, handoff: Handoff) -> None:
        self.handoff = handoff


@runtime_checkable
class HandoffStore(Protocol):
    async def save(self, handoff: Handoff) -> None: ...

    async def get(self, student_id: uuid.UUID, handoff_id: uuid.UUID) -> Handoff | None: ...

    async def for_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> list[Handoff]: ...

    async def claim(self, handoff_id: uuid.UUID, expected: str, status: str) -> bool:
        """Change the status only if it's still the expected one, so two approvals post once."""

    async def last_posted_at(self, student_id: uuid.UUID) -> datetime | None: ...

    async def record_post(self, handoff_id: uuid.UUID, url: str, node_id: str) -> None: ...

    async def due_for_retry(self) -> list[Handoff]: ...

    async def fail_unfinished(self) -> None:
        """After a restart, a post that was being sent may or may not exist on the forum.
        Retrying could post it twice, so the student gets the copyable fallback instead."""


def known_values(email: str) -> list[str]:
    """The student's own email and the name parts of it are always removed, wherever they appear."""
    local = email.split("@", 1)[0]
    parts = re.split(r"[._+-]", local)
    return [email, local, *(part for part in parts if len(part) >= 3)]


def summarize(message: str, earlier: list[str]) -> tuple[str, str]:
    """Build a draft from a template. Code, not a model, writes it."""
    first_line = next((line.strip() for line in message.splitlines() if line.strip()), "")
    title = (first_line or "Question from a student")[:MAX_TITLE_CHARS]
    errors = [m.group(0) for m in _ERROR_LINE.finditer("\n".join([*earlier, message]))]
    sections = [f"**Question**\n\n{message.strip()}"]
    if earlier:
        context = "\n\n".join(e.strip() for e in earlier)
        sections.append(f"**Earlier in the conversation**\n\n{context}")
    if errors:
        listed = "\n".join(f"- {line.strip()}" for line in errors[:5])
        sections.append(f"**Error messages**\n\n{listed}")
    sections.append("_Posted from Owl-Mind by a student, after reviewing this text._")
    return title, "\n\n".join(sections)


def prepare(title: str, body: str, graded: bool, known_values: list[str]) -> tuple[str, str, dict]:
    """Redact, trim and escape exactly as the text will be posted."""
    body = for_graded_problem(body) if graded else trim_code(body, CODE_LINES)
    clean_title, clean_body = redact(title, known_values), redact(body, known_values)
    removed: dict[str, int] = {}
    for counts in (clean_title.removed, clean_body.removed):
        for kind, count in counts.items():
            removed[kind] = removed.get(kind, 0) + count
    return clean_title.text[:MAX_TITLE_CHARS], clean_body.text[:MAX_BODY_CHARS], removed


class HandoffService:
    def __init__(self, store: HandoffStore, forum: Forum | None, category: str) -> None:
        self.store = store
        self.forum = forum
        self.category = category

    async def draft(
        self,
        student_id: uuid.UUID,
        thread_id: uuid.UUID,
        message: str,
        earlier: list[str],
        known_values: list[str],
    ) -> tuple[Handoff, dict[str, int], list[SimilarPost]]:
        title, body = summarize(message, earlier)
        title, body, removed = prepare(title, body, False, known_values)
        handoff = Handoff(
            uuid.uuid4(),
            student_id,
            thread_id,
            "draft",
            title,
            body,
            False,
            self.category,
            created_at=datetime.now(UTC),
        )
        await self.store.save(handoff)
        similar: list[SimilarPost] = []
        if self.forum is not None:
            try:
                similar = await self.forum.search_similar(title)
            except ForumUnavailable:
                pass  # The duplicate check is a convenience. A draft never depends on it.
        return handoff, removed, similar

    async def approve(
        self,
        student_id: uuid.UUID,
        handoff_id: uuid.UUID,
        title: str,
        body: str,
        graded: bool,
        known_values: list[str],
    ) -> Handoff:
        handoff = await self.store.get(student_id, handoff_id)
        if handoff is None:
            raise LookupError
        if handoff.status != "draft":
            raise HandoffError("This draft was already sent or cancelled.")
        title = " ".join(title.split())
        if not title or not body.strip():
            raise HandoffError("A post needs a title and some text.")
        clean_title, clean_body, _ = prepare(title, body, graded, known_values)
        if (clean_title, clean_body) != (title, body):
            # Redaction changed what the student approved, so they must see the new text first.
            revised = replace(handoff, title=clean_title, body=clean_body, graded=graded)
            await self.store.save(revised)
            raise NeedsReview(revised)
        last = await self.store.last_posted_at(student_id)
        if last is not None and datetime.now(UTC) - last < POST_INTERVAL:
            raise HandoffError("You can post once every 10 minutes. Try again shortly.")
        await self.store.save(replace(handoff, title=title, body=body, graded=graded))
        if not await self.store.claim(handoff_id, "draft", "posting"):
            raise HandoffError("This draft was already sent or cancelled.")
        return await self._post(await self.store.get(student_id, handoff_id))

    async def _post(self, handoff: Handoff) -> Handoff:
        if self.forum is None:
            failed = replace(handoff, status="fallback", attempts=handoff.attempts + 1)
            await self.store.save(failed)
            return failed
        try:
            post = await self.forum.post_question(
                handoff.category, escape_markdown(handoff.title), escape_markdown(handoff.body)
            )
        except ForumUnavailable:
            attempts = handoff.attempts + 1
            status = "retrying" if attempts < MAX_RETRIES else "fallback"
            failed = replace(handoff, status=status, attempts=attempts)
            await self.store.save(failed)
            return failed
        posted = replace(handoff, status="posted", url=post.url, attempts=handoff.attempts + 1)
        await self.store.save(posted)
        await self.store.record_post(handoff.id, post.url, post.node_id)
        return posted

    async def retry_pending(self) -> int:
        """Called by a background task. Only drafts the student already approved are retried."""
        retried = 0
        for handoff in await self.store.due_for_retry():
            if await self.store.claim(handoff.id, "retrying", "posting"):
                await self._post(replace(handoff, status="posting"))
                retried += 1
        return retried

    async def cancel(self, student_id: uuid.UUID, handoff_id: uuid.UUID) -> Handoff:
        handoff = await self.store.get(student_id, handoff_id)
        if handoff is None:
            raise LookupError
        if not await self.store.claim(handoff_id, "draft", "cancelled"):
            raise HandoffError("Only a draft can be cancelled.")
        return replace(handoff, status="cancelled")


def public(handoff: Handoff, home_url: str | None) -> dict[str, Any]:
    return {
        "id": str(handoff.id),
        "status": handoff.status,
        "title": handoff.title,
        "body": handoff.body,
        "graded": handoff.graded,
        "url": handoff.url,
        "forum_url": home_url,
    }


class InMemoryHandoffStore:
    def __init__(self) -> None:
        self.handoffs: dict[uuid.UUID, Handoff] = {}
        self.posts: dict[uuid.UUID, tuple[str, str]] = {}
        self.posted_at: dict[uuid.UUID, datetime] = {}

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
        times = [t for i, t in self.posted_at.items() if self.handoffs[i].student_id == student_id]
        return max(times, default=None)

    async def record_post(self, handoff_id, url, node_id) -> None:
        self.posts[handoff_id] = (url, node_id)
        self.posted_at[handoff_id] = datetime.now(UTC)

    async def due_for_retry(self) -> list[Handoff]:
        return [h for h in self.handoffs.values() if h.status == "retrying"]

    async def fail_unfinished(self) -> None:
        for handoff in list(self.handoffs.values()):
            if handoff.status == "posting":
                self.handoffs[handoff.id] = replace(handoff, status="fallback")


_SAVE = """
INSERT INTO app.handoffs (id, student_id, thread_id, status, title, body, graded, category,
    url, attempts)
VALUES (:id, :student_id, :thread_id, :status, :title, :body, :graded, :category, :url, :attempts)
ON CONFLICT (id) DO UPDATE SET status = :status, title = :title, body = :body, graded = :graded,
    url = :url, attempts = :attempts, updated_at = now()
"""
_COLUMNS = (
    "id, student_id, thread_id, status, title, body, graded, category, url, attempts, created_at"
)


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

    async def record_post(self, handoff_id, url, node_id) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "INSERT INTO app.forum_posts (handoff_id, url, node_id)"
                    " VALUES (:id, :url, :node) ON CONFLICT (handoff_id) DO NOTHING"
                ),
                {"id": handoff_id, "url": url, "node": node_id},
            )

    async def due_for_retry(self) -> list[Handoff]:
        return await self._select("status = 'retrying'", {})

    async def fail_unfinished(self) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text("UPDATE app.handoffs SET status = 'fallback' WHERE status = 'posting'")
            )
