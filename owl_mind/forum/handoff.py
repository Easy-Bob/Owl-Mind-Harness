"""Tutor handoff: a fixed path with no model. Draft, redact, approve, then post.

Nothing reaches the forum without the student approving the exact text that will be posted.
"""

import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from owl_mind.forum.accounts import RECONNECT_MESSAGE, ForumAccounts, ReconnectNeeded
from owl_mind.forum.github import Forum, ForumAuthFailed, ForumUnavailable, SimilarPost
from owl_mind.forum.redact import CODE_LINES, escape_markdown, for_graded_problem, redact, trim_code
from owl_mind.forum.store import Handoff, HandoffStore, SavedReply

POST_INTERVAL = timedelta(minutes=10)
MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 6000
MAX_RETRIES = 3

_ERROR_LINE = re.compile(
    r"^.*(?:\w*(?:Error|Exception)\b|\bTraceback\b|\bfailed\b).*$", re.MULTILINE
)


class HandoffError(Exception):
    """The request can't be done. The message is safe to show the student."""


class NeedsReview(Exception):
    """Redaction changed the text, so the student must see and approve the new version."""

    def __init__(self, handoff: Handoff) -> None:
        self.handoff = handoff


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
    def __init__(
        self,
        store: HandoffStore,
        forum: Forum | None,
        category: str,
        accounts: ForumAccounts | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self.forum = forum
        self.category = category
        self.accounts = accounts
        self._clock = clock

    async def _author(self, student_id: uuid.UUID) -> tuple[str, str] | None:
        """The student's own token and login, or None for the bot. Never silently the bot when
        a connected account has stopped working."""
        if self.accounts is None:
            return None
        try:
            return await self.accounts.token_for(student_id)
        except ReconnectNeeded as exc:
            raise HandoffError(RECONNECT_MESSAGE) from exc

    async def post_as(self, student_id: uuid.UUID) -> str:
        if self.accounts is None:
            return "bot"
        status = await self.accounts.status(student_id)
        return status["login"] if status["connected"] else "bot"

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
            created_at=self._clock(),
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
        if last is not None and self._clock() - last < POST_INTERVAL:
            raise HandoffError("You can post once every 10 minutes. Try again shortly.")
        author = await self._author(student_id)
        await self.store.save(replace(handoff, title=title, body=body, graded=graded))
        if not await self.store.claim(handoff_id, "draft", "posting"):
            raise HandoffError("This draft was already sent or cancelled.")
        return await self._post(await self.store.get(student_id, handoff_id), author)

    async def _post(self, handoff: Handoff, author: tuple[str, str] | None) -> Handoff:
        if self.forum is None:
            failed = replace(handoff, status="fallback", attempts=handoff.attempts + 1)
            await self.store.save(failed)
            return failed
        token, login = author if author else (None, "bot")
        try:
            post = await self.forum.post_question(
                handoff.category,
                escape_markdown(handoff.title),
                escape_markdown(handoff.body),
                user_token=token,
            )
        except ForumAuthFailed:
            if token is None:
                raise
            # The student's token was refused. Give the draft back rather than post as the bot.
            await self.accounts.mark_broken(handoff.student_id)
            await self.store.save(replace(handoff, status="draft"))
            raise HandoffError(RECONNECT_MESSAGE) from None
        except ForumUnavailable:
            attempts = handoff.attempts + 1
            status = "retrying" if attempts < MAX_RETRIES else "fallback"
            failed = replace(handoff, status=status, attempts=attempts)
            await self.store.save(failed)
            return failed
        posted = replace(
            handoff,
            status="posted",
            url=post.url,
            attempts=handoff.attempts + 1,
            posted_as=login,
        )
        await self.store.save(posted)
        await self.store.record_post(handoff.id, post.url, post.node_id, self._clock())
        return posted

    async def retry_pending(self) -> int:
        """Called by a background task. Only drafts the student already approved are retried."""
        retried = 0
        for handoff in await self.store.due_for_retry():
            if await self.store.claim(handoff.id, "retrying", "posting"):
                handoff = replace(handoff, status="posting")
                try:
                    author = await self._author(handoff.student_id)
                    await self._post(handoff, author)
                except HandoffError:
                    # Nobody is waiting to read the message, so give the copyable text instead.
                    await self.store.save(replace(handoff, status="fallback"))
                retried += 1
        return retried

    async def cancel(self, student_id: uuid.UUID, handoff_id: uuid.UUID) -> Handoff:
        handoff = await self.store.get(student_id, handoff_id)
        if handoff is None:
            raise LookupError
        if not await self.store.claim(handoff_id, "draft", "cancelled"):
            raise HandoffError("Only a draft can be cancelled.")
        return replace(handoff, status="cancelled")


def public(
    handoff: Handoff,
    home_url: str | None,
    post_as: str | None = None,
    replies: list[SavedReply] = (),
) -> dict[str, Any]:
    return {
        "id": str(handoff.id),
        "status": handoff.status,
        "title": handoff.title,
        "body": handoff.body,
        "graded": handoff.graded,
        "url": handoff.url,
        "forum_url": home_url,
        # Who a draft will post as, or who a post was made by: a GitHub login, or "bot".
        "post_as": handoff.posted_as or post_as,
        "replies": [
            {
                "id": reply.id,
                "author": reply.author,
                "body": reply.body,
                "created_at": reply.created_at.isoformat(),
                "is_answer": reply.is_answer,
                "new": not reply.seen,
            }
            for reply in replies
        ],
    }
