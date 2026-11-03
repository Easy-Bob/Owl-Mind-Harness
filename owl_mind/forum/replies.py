"""Checks posted discussions for replies, on a schedule that survives a restart.

Replies are other people's text. They're saved and shown to the student as forum replies, and
never added to an agent's messages, so they can't steer an agent and cost no tokens.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from owl_mind.forum.github import DiscussionGone, Forum, ForumUnavailable
from owl_mind.forum.store import HandoffStore, TrackedPost

logger = logging.getLogger(__name__)

POLL_EVERY = timedelta(hours=1)
POLL_FOR = timedelta(days=14)
RETRY_FIRST = timedelta(minutes=5)
RETRY_MAX = timedelta(hours=6)
BATCH = 20
# Longer than one check takes, so a slow check isn't picked up twice.
LEASE = timedelta(minutes=2)


def next_poll(posted_at: datetime, now: datetime, delay: timedelta = POLL_EVERY) -> datetime | None:
    """The next check, or None once the post is older than the polling window."""
    following = now + delay
    return following if following <= posted_at + POLL_FOR else None


def retry_delay(failures: int) -> timedelta:
    return min(RETRY_FIRST * 2 ** (failures - 1), RETRY_MAX)


class ReplyPoller:
    def __init__(
        self,
        store: HandoffStore,
        forum: Forum,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        events: Any = None,
    ) -> None:
        self.store = store
        self.forum = forum
        self._clock = clock
        self.events = events

    async def _record(self, post: TrackedPost, step: str, **data: Any) -> None:
        if self.events is not None:
            await self.events.record(post.run_id, "handoff", {"step": step, **data})

    async def poll_due(self) -> int:
        """Check every post that is due. Returns how many new replies were saved."""
        new = 0
        for post in await self.store.lease_due_posts(self._clock(), BATCH, LEASE):
            new += await self._check(post)
        return new

    async def _check(self, post: TrackedPost) -> int:
        try:
            result = await self.forum.get_replies(post.node_id)
        except DiscussionGone:
            await self.store.discussion_gone(post.handoff_id)
            await self._record(post, "polling_stopped", reason="discussion_gone")
            return 0
        except ForumUnavailable as exc:
            # GitHub's rate limit also lands here, so failures wait longer each time.
            failures = post.failures + 1
            logger.warning("Forum reply check failed: %s", type(exc).__name__)
            when = next_poll(post.posted_at, self._clock(), retry_delay(failures))
            await self.store.poll_failed(post.handoff_id, failures, when)
            await self._record(post, "poll_failed", failures=failures)
            if when is None:
                await self._record(post, "polling_stopped", reason="window_ended")
            return 0
        when = next_poll(post.posted_at, self._clock())
        new = await self.store.save_replies(post.handoff_id, result.replies, result.complete, when)
        if new:
            await self._record(post, "replies_arrived", count=new)
        if when is None:
            await self._record(post, "polling_stopped", reason="window_ended")
        return new
