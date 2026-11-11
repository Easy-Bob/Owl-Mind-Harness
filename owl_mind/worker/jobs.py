"""The background jobs, run by workers under a lease (M4-B), or in the API with no workers."""

from collections.abc import Awaitable, Callable

from owl_mind.core.runtime import Services
from owl_mind.forum.replies import ReplyPoller

HANDOFF_RETRY_SECONDS = 60
APPROVAL_CHECK_SECONDS = 300


async def expire_approvals(services: Services) -> int:
    """Expired approvals count as rejected: their runs stop and free the student's slot."""
    runs = await services.approvals.expire_due()
    for run_id in runs:
        await services.runs.record_event(
            run_id, "approval:expired", "approval", {"step": "expired"}
        )
        await services.runs.set_run_status(run_id, "stopped")
    return len(runs)


def jobs(services: Services) -> list[tuple[str, float, Callable[[], Awaitable[object]]]]:
    """Each job this setup has: its name, how often it runs, and one pass of it."""
    found: list[tuple[str, float, Callable[[], Awaitable[object]]]] = []
    if services.approvals is not None:
        found.append(("approvals", APPROVAL_CHECK_SECONDS, lambda: expire_approvals(services)))
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
