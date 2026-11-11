"""The background jobs, run by workers under a lease (M4-B), or in the API with no workers."""

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta

from owl_mind.core.runtime import RunRequest, Services
from owl_mind.forum.replies import ReplyPoller
from owl_mind.services.run_store import ActiveRunExists

logger = logging.getLogger(__name__)

HANDOFF_RETRY_SECONDS = 60
APPROVAL_CHECK_SECONDS = 300
RETENTION_SECONDS = 24 * 3600
RESUME_CHECK_SECONDS = 60
# Long enough that the API, which makes the resume run straight after the answer, is done.
RESUME_GRACE = timedelta(minutes=2)


async def expire_approvals(services: Services) -> int:
    """Expired approvals count as rejected: their runs stop and free the student's slot."""
    runs = await services.approvals.expire_due()
    for run_id in runs:
        await services.runs.record_event(
            run_id, "approval:expired", "approval", {"step": "expired"}
        )
        await services.runs.set_run_status(run_id, "stopped")
    return len(runs)


async def resume_answered(services: Services) -> int:
    """Start the resume run for an answered approval that never got one (M4-C)."""
    started = 0
    for approval, student in await services.approvals.unresumed(RESUME_GRACE):
        await services.runs.set_run_status(approval.run_id, "completed")
        try:
            run_id = await services.runs.create_run(approval.thread_id)
        except ActiveRunExists:
            continue  # The student has another answer being written; tried again next pass.
        await services.approvals.set_resume_run(approval.id, run_id)
        resume = {"approved": approval.status == "approved"}
        await services.queue.enqueue(
            RunRequest(
                run_id,
                approval.thread_id,
                student,
                "",
                resume=resume,
                setup=approval.setup,
                queued_at=time.time(),
            )
        )
        started += 1
    if started:
        logger.info("Resumed %d answered approvals", started)
    return started


def jobs(services: Services) -> list[tuple[str, float, Callable[[], Awaitable[object]]]]:
    """Each job this setup has: its name, how often it runs, and one pass of it."""
    found: list[tuple[str, float, Callable[[], Awaitable[object]]]] = []
    if services.approvals is not None:
        found.append(("approvals", APPROVAL_CHECK_SECONDS, lambda: expire_approvals(services)))
        if services.queue is not None:
            found.append(("resume", RESUME_CHECK_SECONDS, lambda: resume_answered(services)))
    if services.retention is not None:
        found.append(("retention", RETENTION_SECONDS, services.retention.run))
    if services.memory is not None:
        found.append(("memory", APPROVAL_CHECK_SECONDS, services.memory.run_due))
    if services.handoffs is not None:
        handoffs = services.handoffs
        poller = (
            ReplyPoller(handoffs.store, handoffs.forum, events=handoffs.events)
            if handoffs.forum is not None
            else None
        )

        async def forum() -> None:
            await handoffs.retry_pending()
            if poller is not None:
                await poller.poll_due()

        found.append(("forum", HANDOFF_RETRY_SECONDS, forum))
    return found
