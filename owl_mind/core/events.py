"""The live event stream for one run. Redis keeps it for an hour so a client can reconnect."""

import asyncio
import json
import uuid
from typing import Any, Protocol, runtime_checkable

from redis.asyncio import Redis

TERMINAL_KINDS = ("done", "error")
_STREAM_TTL_SECONDS = 3600
_MAX_EVENTS_PER_RUN = 10_000


def _key(run_id: uuid.UUID) -> str:
    return f"stream:run:{run_id}"


@runtime_checkable
class EventBus(Protocol):
    async def publish(self, run_id: uuid.UUID, kind: str, data: dict[str, Any]) -> None: ...

    async def read(
        self, run_id: uuid.UUID, after: str, block_ms: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return (event id, kind, data) after an event id, waiting up to block_ms for any."""


class RedisEventBus:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def publish(self, run_id: uuid.UUID, kind: str, data: dict[str, Any]) -> None:
        key = _key(run_id)
        await self._client.xadd(
            key,
            {"kind": kind, "data": json.dumps(data)},
            maxlen=_MAX_EVENTS_PER_RUN,
            approximate=True,
        )
        await self._client.expire(key, _STREAM_TTL_SECONDS)

    async def read(
        self, run_id: uuid.UUID, after: str, block_ms: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        streams = await self._client.xread({_key(run_id): after}, block=block_ms)
        if not streams:
            return []
        return [
            (
                event_id.decode(),
                fields[b"kind"].decode(),
                json.loads(fields[b"data"]),
            )
            for event_id, fields in streams[0][1]
        ]


class InMemoryEventBus:
    """The same behavior for unit tests, with no Redis."""

    def __init__(self) -> None:
        self.events: dict[uuid.UUID, list[tuple[str, str, dict[str, Any]]]] = {}
        self._changed = asyncio.Event()

    async def publish(self, run_id: uuid.UUID, kind: str, data: dict[str, Any]) -> None:
        events = self.events.setdefault(run_id, [])
        events.append((f"{len(events) + 1}-0", kind, data))
        self._changed.set()

    async def read(
        self, run_id: uuid.UUID, after: str, block_ms: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        start = int(after.split("-")[0])
        found = self.events.get(run_id, [])[start:]
        if found or not block_ms:
            return found
        self._changed.clear()
        try:
            await asyncio.wait_for(self._changed.wait(), block_ms / 1000)
        except TimeoutError:
            return []
        return self.events.get(run_id, [])[start:]
