"""Threads, runs and run events in the app schema. Student-facing reads filter by owner."""

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.models import Run, RunEvent, Thread

# Left unfinished by a restart, so failed at startup.
UNFINISHED = ("queued", "running")
# Waiting for the student's approval. It survives a restart and holds the student's slot.
WAITING = "waiting_approval"
BUSY = (*UNFINISHED, WAITING)
FINISHED = ("completed", "failed", "stopped")


class ActiveRunExists(Exception):
    """This student already has a run in progress, in this thread or another."""


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


@dataclass(frozen=True)
class Takeover:
    """What a worker needs to resume a run another worker left."""

    created_at: datetime
    route: str | None
    setup: dict[str, Any] | None


@runtime_checkable
class RunStore(Protocol):
    async def create_thread(self, student_id: uuid.UUID, title: str) -> ThreadInfo: ...

    async def list_threads(self, student_id: uuid.UUID) -> list[ThreadInfo]: ...

    async def get_thread(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> ThreadInfo | None:
        """Return None if the thread is not this student's."""

    async def create_run(
        self, thread_id: uuid.UUID, client_message_id: uuid.UUID | None = None
    ) -> uuid.UUID:
        """Raise ActiveRunExists if the thread's student has an unfinished run anywhere."""

    async def find_by_message(
        self, student_id: uuid.UUID, client_message_id: uuid.UUID
    ) -> RunInfo | None:
        """The run this message ID already started, so a repeat returns it (M4-C)."""

    async def takeover(self, run_id: uuid.UUID) -> Takeover | None: ...

    async def active_run(self, student_id: uuid.UUID, thread_id: uuid.UUID) -> RunInfo | None:
        """The chat's answer still being written, so a reopened chat can follow it."""

    async def delete_empty_thread(self, thread_id: uuid.UUID) -> None:
        """Remove a thread that never got a run, such as one refused for a run elsewhere."""

    async def get_run(self, student_id: uuid.UUID, run_id: uuid.UUID) -> RunInfo | None:
        """Return None if the run is not this student's."""

    async def set_run_status(self, run_id: uuid.UUID, status: str) -> None: ...

    async def set_run_setup(
        self, run_id: uuid.UUID, route: str, task_kind: str | None, setup: dict | None = None
    ) -> None: ...

    async def fail_unfinished_runs(self) -> int: ...

    async def status_of(self, run_id: uuid.UUID) -> str | None:
        """For workers, which act for no particular student."""

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
        self.messages: dict[tuple[uuid.UUID, uuid.UUID], uuid.UUID] = {}
        self.setups: dict[uuid.UUID, dict] = {}
        self.created: dict[uuid.UUID, datetime] = {}

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

    async def create_run(self, thread_id, client_message_id=None) -> uuid.UUID:
        student = self.threads[thread_id][0]
        busy = {self.threads[r.thread_id][0] for r in self.runs.values() if r.status in BUSY}
        if student in busy or (student, client_message_id) in self.messages:
            raise ActiveRunExists
        run_id = uuid.uuid4()
        self.runs[run_id] = RunInfo(run_id, thread_id, "queued")
        self.created[run_id] = datetime.now(UTC)
        if client_message_id is not None:
            self.messages[(student, client_message_id)] = run_id
        return run_id

    async def find_by_message(self, student_id, client_message_id) -> RunInfo | None:
        run_id = self.messages.get((student_id, client_message_id))
        return self.runs.get(run_id) if run_id else None

    async def takeover(self, run_id: uuid.UUID) -> Takeover | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        return Takeover(self.created[run_id], run.route, self.setups.get(run_id))

    async def active_run(self, student_id, thread_id) -> RunInfo | None:
        if await self.get_thread(student_id, thread_id) is None:
            return None
        return next(
            (r for r in self.runs.values() if r.thread_id == thread_id and r.status in UNFINISHED),
            None,
        )

    async def delete_empty_thread(self, thread_id: uuid.UUID) -> None:
        if not any(r.thread_id == thread_id for r in self.runs.values()):
            self.threads.pop(thread_id, None)

    async def get_run(self, student_id: uuid.UUID, run_id: uuid.UUID) -> RunInfo | None:
        run = self.runs.get(run_id)
        if run is None or await self.get_thread(student_id, run.thread_id) is None:
            return None
        return run

    async def set_run_status(self, run_id: uuid.UUID, status: str) -> None:
        self.runs[run_id] = replace(self.runs[run_id], status=status)

    async def set_run_setup(self, run_id, route, task_kind, setup=None) -> None:
        self.runs[run_id] = replace(self.runs[run_id], route=route, task_kind=task_kind)
        if setup is not None:
            self.setups[run_id] = setup

    async def status_of(self, run_id: uuid.UUID) -> str | None:
        run = self.runs.get(run_id)
        return run.status if run else None

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

    async def create_run(self, thread_id, client_message_id=None) -> uuid.UUID:
        try:
            async with self._sessions.begin() as session:
                student = await session.scalar(
                    select(Thread.student_id).where(Thread.id == thread_id)
                )
                row = Run(
                    id=uuid.uuid4(),
                    thread_id=thread_id,
                    student_id=student,
                    status="queued",
                    client_message_id=client_message_id,
                )
                session.add(row)
        except IntegrityError as exc:
            # Partial unique indexes allow one unfinished run per thread and per student, even
            # across processes.
            raise ActiveRunExists from exc
        return row.id

    async def delete_empty_thread(self, thread_id: uuid.UUID) -> None:
        async with self._sessions.begin() as session:
            has_runs = await session.scalar(select(Run.id).where(Run.thread_id == thread_id))
            if has_runs is None:
                await session.execute(delete(Thread).where(Thread.id == thread_id))

    async def status_of(self, run_id: uuid.UUID) -> str | None:
        async with self._sessions() as session:
            return await session.scalar(select(Run.status).where(Run.id == run_id))

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

    async def find_by_message(self, student_id, client_message_id) -> RunInfo | None:
        statement = select(Run).where(
            Run.student_id == student_id, Run.client_message_id == client_message_id
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
        if row is None:
            return None
        return RunInfo(row.id, row.thread_id, row.status, row.route, row.task_kind)

    async def active_run(self, student_id, thread_id) -> RunInfo | None:
        statement = select(Run).where(
            Run.thread_id == thread_id,
            Run.student_id == student_id,
            Run.status.in_(UNFINISHED),
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
        if row is None:
            return None
        return RunInfo(row.id, row.thread_id, row.status, row.route, row.task_kind)

    async def takeover(self, run_id: uuid.UUID) -> Takeover | None:
        async with self._sessions() as session:
            row = await session.get(Run, run_id)
        return Takeover(row.created_at, row.route, row.setup) if row else None

    async def set_run_setup(self, run_id, route, task_kind, setup=None) -> None:
        values: dict[str, Any] = {"route": route, "task_kind": task_kind}
        if setup is not None:
            values["setup"] = setup
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
