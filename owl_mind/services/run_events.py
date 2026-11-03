"""Records metadata-only events on a run after it has finished, such as a later forum post.

A run's own events have numbered keys. These keys start with "t", a timestamp and a counter,
so they sort after the run's events and in the order they happened, even within one clock tick.
"""

import itertools
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from owl_mind.services.run_store import RunStore

logger = logging.getLogger(__name__)


class RunEventLog:
    def __init__(
        self, runs: RunStore, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._runs = runs
        self._clock = clock
        self._sequence = itertools.count()

    async def record(self, run_id: uuid.UUID | None, kind: str, data: dict[str, Any]) -> None:
        """Never raises: a lost event must not fail a post, a search or a check for replies."""
        if run_id is None:
            return
        key = f"t{self._clock():%Y%m%dT%H%M%S%f}:{next(self._sequence):09d}:{kind}"
        try:
            await self._runs.record_event(run_id, key, kind, data)
        except Exception as exc:
            logger.warning("Run event not recorded: %s", type(exc).__name__)
