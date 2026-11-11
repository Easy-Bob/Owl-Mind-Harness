"""Accounts and failed-login counting. The admin role is never stored here."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.services.models import Student

_WINDOW_SECONDS = 15 * 60


class EmailTaken(Exception):
    """An account already uses this email."""


@dataclass(frozen=True)
class Account:
    id: uuid.UUID
    email: str
    password_hash: str | None
    # The hash of the invite code the account signed up with, for cohorts (M4-G).
    invite_hash: str | None = None


@runtime_checkable
class AccountStore(Protocol):
    async def create(
        self, email: str, password_hash: str, invite_hash: str | None = None
    ) -> Account:
        """Raise EmailTaken if the email is in use. Emails are compared in lower case."""

    async def get_by_email(self, email: str) -> Account | None: ...

    async def get_by_id(self, account_id: uuid.UUID) -> Account | None: ...


@runtime_checkable
class LoginLimiter(Protocol):
    async def failures(self, key: str) -> int: ...

    async def record_failure(self, key: str) -> None: ...

    async def clear(self, key: str) -> None: ...


class InMemoryAccountStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.accounts: dict[uuid.UUID, Account] = {}

    async def create(self, email: str, password_hash: str, invite_hash=None) -> Account:
        if await self.get_by_email(email):
            raise EmailTaken
        account = Account(uuid.uuid4(), email.lower(), password_hash, invite_hash)
        self.accounts[account.id] = account
        return account

    async def get_by_email(self, email: str) -> Account | None:
        lowered = email.lower()
        return next((a for a in self.accounts.values() if a.email == lowered), None)

    async def get_by_id(self, account_id: uuid.UUID) -> Account | None:
        return self.accounts.get(account_id)


class PostgresAccountStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, email: str, password_hash: str, invite_hash=None) -> Account:
        row = Student(
            id=uuid.uuid4(),
            email=email.lower(),
            password_hash=password_hash,
            notice_accepted_at=datetime.now(UTC),
            invite_hash=invite_hash,
        )
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            raise EmailTaken from exc
        return Account(row.id, row.email, row.password_hash, row.invite_hash)

    async def get_by_email(self, email: str) -> Account | None:
        statement = select(Student).where(Student.email == email.lower())
        async with self._sessions() as session:
            row = await session.scalar(statement)
        return (
            None if row is None else Account(row.id, row.email, row.password_hash, row.invite_hash)
        )

    async def get_by_id(self, account_id: uuid.UUID) -> Account | None:
        async with self._sessions() as session:
            row = await session.get(Student, account_id)
        return (
            None if row is None else Account(row.id, row.email, row.password_hash, row.invite_hash)
        )


class InMemoryLoginLimiter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def failures(self, key: str) -> int:
        return self.counts.get(key, 0)

    async def record_failure(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    async def clear(self, key: str) -> None:
        self.counts.pop(key, None)


class RedisLoginLimiter:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def failures(self, key: str) -> int:
        return int(await self._client.get(key) or 0)

    async def record_failure(self, key: str) -> None:
        # The window starts at the first failure, and later failures don't extend it.
        if await self._client.incr(key) == 1:
            await self._client.expire(key, _WINDOW_SECONDS)

    async def clear(self, key: str) -> None:
        await self._client.delete(key)
