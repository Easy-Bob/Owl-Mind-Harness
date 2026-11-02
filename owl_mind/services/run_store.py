"""Threads, runs and run events in the app schema. Student-facing reads filter by owner."""

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.models import Run, RunEvent, Thread

UNFINISHED = ("queued", "running")
FINISHED = ("completed", "failed", "stopped")


class ActiveRunExists(Exception):
    """This thread already has a run in progress."""


@dataclass(frozen=True)
class ThreadInfo:
    id: uuid.UUID
    title: str
    created_at: datetime


@dataclass(frozen=True)
class RunInfo:
    id: uuid.UUID
    thread_id: uuid.UUID
    status: str
    route: str | None = None
    task_kind: str | None = None


@runtime_checkable
class RunStore(Protocol):
    async def create_thread(self, student_id: uuid.UUID, title: str) -> ThreadInfo: ...

    async def list_threads(self, student_id: uuid.UUID) -> list[ThreadInfo]: ...

    async def get_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> ThreadInfo | None:
        """Return None if the thread is not this student's."""

    async def create_run(self, thread_id: uuid.UUID) -> uuid.UUID:
        """Raise ActiveRunExists if the thread has an unfinished run."""

    async def get_run(self, student_id: uuid.UUID, run_id: uuid.UUID) -> RunInfo | None:
        """Return None if the run is not this student's."""

    async def set_run_status(self, run_id: uuid.UUID, status: str) -> None: ...

    async def set_run_setup(self, run_id: uuid.UUID, route: str, task_kind: str | None) -> None: ...

    async def fail_unfinished_runs(self) -> int: ...

    async def record_event(
        self, run_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]
    ) -> None:
        """Save one event per key. A repeat overwrites."""


class InMemoryRunStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.threads: dict[uuid.UUID, tuple[uuid.UUID, ThreadInfo]] = {}
        self.runs: dict[uuid.UUID, RunInfo] = {}
        self.events: dict[tuple[uuid.UUID, str], tuple[str, dict[str, Any]]] = {}

    async def create_thread(self, student_id: uuid.UUID, title: str) -> ThreadInfo:
        info = ThreadInfo(uuid.uuid4(), title, datetime.now(UTC))
        self.threads[info.id] = (student_id, info)
        return info

    async def list_threads(self, student_id: uuid.UUID) -> list[ThreadInfo]:
        owned = [info for owner, info in self.threads.values() if owner == student_id]
        return sorted(owned, key=lambda info: info.created_at, reverse=True)

    async def get_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> ThreadInfo | None:
        owner, info = self.threads.get(thread_id, (None, None))
        return info if owner == student_id else None

    async def create_run(self, thread_id: uuid.UUID) -> uuid.UUID:
        if any(r.thread_id == thread_id and r.status in UNFINISHED for r in self.runs.values()):
            raise ActiveRunExists
        run_id = uuid.uuid4()
        self.runs[run_id] = RunInfo(run_id, thread_id, "queued")
        return run_id

    async def get_run(self, student_id: uuid.UUID, run_id: uuid.UUID) -> RunInfo | None:
        run = self.runs.get(run_id)
        if run is None or await self.get_thread(student_id, run.thread_id) is None:
            return None
        return run

    async def set_run_status(self, run_id: uuid.UUID, status: str) -> None:
        self.runs[run_id] = replace(self.runs[run_id], status=status)

    async def set_run_setup(self, run_id: uuid.UUID, route: str, task_kind: str | None) -> None:
        self.runs[run_id] = replace(self.runs[run_id], route=route, task_kind=task_kind)

    async def fail_unfinished_runs(self) -> int:
        unfinished = [r.id for r in self.runs.values() if r.status in UNFINISHED]
        for run_id in unfinished:
            await self.set_run_status(run_id, "failed")
        return len(unfinished)

    async def record_event(
        self, run_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]
    ) -> None:
        self.events[(run_id, key)] = (kind, payload)


class PostgresRunStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_thread(self, student_id: uuid.UUID, title: str) -> ThreadInfo:
        row = Thread(id=uuid.uuid4(), student_id=student_id, title=title)
        async with self._sessions.begin() as session:
            session.add(row)
            await session.flush()
            await session.refresh(row)
        return ThreadInfo(row.id, row.title, row.created_at)

    async def list_threads(self, student_id: uuid.UUID) -> list[ThreadInfo]:
        statement = (
            select(Thread)
            .where(Thread.student_id == student_id)
            .order_by(Thread.created_at.desc(), Thread.id)
        )
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
        return [ThreadInfo(row.id, row.title, row.created_at) for row in rows]

    async def get_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> ThreadInfo | None:
        async with self._sessions() as session:
            row = await session.get(Thread, thread_id)
        if row is None or row.student_id != student_id:
            return None
        return ThreadInfo(row.id, row.title, row.created_at)

    async def create_run(self, thread_id: uuid.UUID) -> uuid.UUID:
        row = Run(id=uuid.uuid4(), thread_id=thread_id, status="queued")
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            # A partial unique index allows one unfinished run per thread, even across processes.
            raise ActiveRunExists from exc
        return row.id

    async def get_run(self, student_id: uuid.UUID, run_id: uuid.UUID) -> RunInfo | None:
        statement = (
            select(Run)
            .join(Thread, Thread.id == Run.thread_id)
            .where(Run.id == run_id, Thread.student_id == student_id)
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
        if row is None:
            return None
        return RunInfo(row.id, row.thread_id, row.status, row.route, row.task_kind)

    async def set_run_status(self, run_id: uuid.UUID, status: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(update(Run).where(Run.id == run_id).values(status=status))

    async def set_run_setup(self, run_id: uuid.UUID, route: str, task_kind: str | None) -> None:
        values = {"route": route, "task_kind": task_kind}
        async with self._sessions.begin() as session:
            await session.execute(update(Run).where(Run.id == run_id).values(**values))

    async def fail_unfinished_runs(self) -> int:
        statement = update(Run).where(Run.status.in_(UNFINISHED)).values(status="failed")
        async with self._sessions.begin() as session:
            return (await session.execute(statement)).rowcount

    async def record_event(
        self, run_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]
    ) -> None:
        values = {"kind": kind, "payload": payload}
        statement = (
            insert(RunEvent)
            .values(run_id=run_id, key=key, **values)
            .on_conflict_do_update(index_elements=["run_id", "key"], set_=values)
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)
