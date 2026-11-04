"""Approvals waiting for a student's answer. Every read is scoped to one student.

A decision is claimed with one conditional update, so a double click applies it once, and an
expired approval can't be approved.
"""

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.models import Approval

EXPIRES_AFTER = timedelta(hours=24)


@dataclass(frozen=True)
class ApprovalInfo:
    id: uuid.UUID
    run_id: uuid.UUID
    thread_id: uuid.UUID
    kind: str
    summary: str
    items: list[dict[str, Any]]
    setup: dict[str, Any]
    status: str
    expires_at: datetime


def _info(row: Approval) -> ApprovalInfo:
    return ApprovalInfo(
        row.id,
        row.run_id,
        row.thread_id,
        row.kind,
        row.summary,
        row.items,
        row.setup,
        row.status,
        row.expires_at,
    )


@runtime_checkable
class ApprovalStore(Protocol):
    async def create(
        self,
        run_id: uuid.UUID,
        student_id: uuid.UUID,
        thread_id: uuid.UUID,
        request: dict[str, Any],
        setup: dict[str, Any],
    ) -> ApprovalInfo: ...

    async def pending_for_run(
        self, student_id: uuid.UUID, run_id: uuid.UUID
    ) -> ApprovalInfo | None:
        """The run's approval while it can still be answered. None for another student's."""

    async def pending_for_thread(
        self, student_id: uuid.UUID, thread_id: uuid.UUID
    ) -> ApprovalInfo | None: ...

    async def decide(self, approval_id: uuid.UUID, status: str) -> bool:
        """Claim the decision. False if it was already answered or has expired."""

    async def set_resume_run(self, approval_id: uuid.UUID, run_id: uuid.UUID) -> None: ...

    async def expire_due(self) -> list[uuid.UUID]:
        """Expire every approval past its time, and return their runs."""


class InMemoryApprovalStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, tuple[uuid.UUID, ApprovalInfo]] = {}
        self.resumed: dict[uuid.UUID, uuid.UUID] = {}

    async def create(self, run_id, student_id, thread_id, request, setup) -> ApprovalInfo:
        if any(o == student_id and a.status == "pending" for o, a in self.rows.values()):
            raise ValueError("This student already has a pending approval.")
        info = ApprovalInfo(
            uuid.uuid4(),
            run_id,
            thread_id,
            request["kind"],
            request["summary"],
            request.get("items", []),
            setup,
            "pending",
            datetime.now(UTC) + EXPIRES_AFTER,
        )
        self.rows[info.id] = (student_id, info)
        return info

    def _pending(self, student_id, match) -> ApprovalInfo | None:
        now = datetime.now(UTC)
        for owner, info in self.rows.values():
            if owner == student_id and info.status == "pending" and info.expires_at > now:
                if match(info):
                    return info
        return None

    async def pending_for_run(self, student_id, run_id) -> ApprovalInfo | None:
        return self._pending(student_id, lambda info: info.run_id == run_id)

    async def pending_for_thread(self, student_id, thread_id) -> ApprovalInfo | None:
        return self._pending(student_id, lambda info: info.thread_id == thread_id)

    async def decide(self, approval_id: uuid.UUID, status: str) -> bool:
        owner, info = self.rows[approval_id]
        if info.status != "pending" or info.expires_at <= datetime.now(UTC):
            return False
        self.rows[approval_id] = (owner, replace(info, status=status))
        return True

    async def set_resume_run(self, approval_id: uuid.UUID, run_id: uuid.UUID) -> None:
        self.resumed[approval_id] = run_id

    async def expire_due(self) -> list[uuid.UUID]:
        now = datetime.now(UTC)
        expired = []
        for approval_id, (owner, info) in list(self.rows.items()):
            if info.status == "pending" and info.expires_at <= now:
                self.rows[approval_id] = (owner, replace(info, status="expired"))
                expired.append(info.run_id)
        return expired


class PostgresApprovalStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, run_id, student_id, thread_id, request, setup) -> ApprovalInfo:
        row = Approval(
            id=uuid.uuid4(),
            run_id=run_id,
            student_id=student_id,
            thread_id=thread_id,
            kind=request["kind"],
            summary=request["summary"],
            items=request.get("items", []),
            setup=setup,
            status="pending",
            expires_at=datetime.now(UTC) + EXPIRES_AFTER,
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return _info(row)

    async def _pending(self, student_id: uuid.UUID, *where) -> ApprovalInfo | None:
        statement = select(Approval).where(
            Approval.student_id == student_id,
            Approval.status == "pending",
            Approval.expires_at > datetime.now(UTC),
            *where,
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
        return _info(row) if row else None

    async def pending_for_run(self, student_id, run_id) -> ApprovalInfo | None:
        return await self._pending(student_id, Approval.run_id == run_id)

    async def pending_for_thread(self, student_id, thread_id) -> ApprovalInfo | None:
        return await self._pending(student_id, Approval.thread_id == thread_id)

    async def decide(self, approval_id: uuid.UUID, status: str) -> bool:
        now = datetime.now(UTC)
        statement = (
            update(Approval)
            .where(
                Approval.id == approval_id,
                Approval.status == "pending",
                Approval.expires_at > now,
            )
            .values(status=status, decided_at=now)
            .returning(Approval.id)
        )
        async with self._sessions.begin() as session:
            return await session.scalar(statement) is not None

    async def set_resume_run(self, approval_id: uuid.UUID, run_id: uuid.UUID) -> None:
        statement = update(Approval).where(Approval.id == approval_id).values(resume_run_id=run_id)
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def expire_due(self) -> list[uuid.UUID]:
        now = datetime.now(UTC)
        statement = (
            update(Approval)
            .where(Approval.status == "pending", Approval.expires_at <= now)
            .values(status="expired", decided_at=now)
            .returning(Approval.run_id)
        )
        async with self._sessions.begin() as session:
            return list((await session.scalars(statement)).all())
