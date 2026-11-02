"""Needs the compose services: docker compose up -d --wait

The same account and limiter checks as the in-memory versions, plus login on real services.
"""

import asyncio
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
import pytest
import redis
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.api.app import create_app
from owl_mind.core.config import get_settings
from owl_mind.services.accounts import (
    EmailTaken,
    InMemoryAccountStore,
    InMemoryLoginLimiter,
    PostgresAccountStore,
    RedisLoginLimiter,
)
from owl_mind.services.models import sqlalchemy_url

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "correct horse battery"


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "s" * 32)
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("INVITE_CODES", "invite-1")
    monkeypatch.setenv("CONFIGURED_ADMINS", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def email() -> str:
    return f"user-{uuid.uuid4().hex}@example.com"


@asynccontextmanager
async def postgres_accounts():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    yield PostgresAccountStore(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


@asynccontextmanager
async def memory_accounts():
    yield InMemoryAccountStore()


@asynccontextmanager
async def redis_limiter():
    client = Redis.from_url(get_settings().redis_url)
    yield RedisLoginLimiter(client)
    await client.aclose()


@asynccontextmanager
async def memory_limiter():
    yield InMemoryLoginLimiter()


@pytest.fixture(params=[memory_accounts, postgres_accounts], ids=["memory", "postgres"])
def accounts(request):
    return request.param


@pytest.fixture(params=[memory_limiter, redis_limiter], ids=["memory", "redis"])
def limiter(request):
    return request.param


def test_account_round_trip_ignores_letter_case(accounts):
    address = email()

    async def body():
        async with accounts() as store:
            created = await store.create(address.upper(), "hash")
            return (
                created,
                await store.get_by_email(address),
                await store.get_by_id(created.id),
                await store.get_by_email("nobody@example.com"),
                await store.get_by_id(uuid.uuid4()),
            )

    created, by_email, by_id, no_email, no_id = asyncio.run(body())
    assert created.email == address
    assert by_email == by_id == created
    assert (no_email, no_id) == (None, None)


def test_same_email_twice_is_refused_in_any_case(accounts):
    address = email()

    async def body():
        async with accounts() as store:
            await store.create(address, "hash")
            with pytest.raises(EmailTaken):
                await store.create(address.upper(), "hash")

    asyncio.run(body())


def test_failure_counts_are_kept_per_key_and_can_be_cleared(limiter):
    key, other = f"rl:test:{uuid.uuid4()}", f"rl:test:{uuid.uuid4()}"

    async def body():
        async with limiter() as counts:
            assert await counts.failures(key) == 0
            await counts.record_failure(key)
            await counts.record_failure(key)
            two = await counts.failures(key)
            none = await counts.failures(other)
            await counts.clear(key)
            return two, none, await counts.failures(key)

    assert asyncio.run(body()) == (2, 0, 0)


def test_redis_failure_count_expires_and_later_failures_do_not_extend_it():
    key = f"rl:test:{uuid.uuid4()}"
    client = redis.Redis.from_url(get_settings().redis_url)

    async def body():
        async with redis_limiter() as counts:
            await counts.record_failure(key)
            client.expire(key, 100)
            await counts.record_failure(key)

    asyncio.run(body())
    assert 0 < client.ttl(key) <= 100


def test_production_app_signs_up_logs_in_and_stores_only_a_hash():
    address = email()
    # The real services, as the deployed app uses them.
    with TestClient(create_app()) as client:
        signed_up = client.post(
            "/auth/signup",
            json={
                "email": address,
                "password": PASSWORD,
                "invite_code": "invite-1",
                "accepted_notice": True,
            },
        )
        logged_in = client.post("/auth/login", json={"email": address, "password": PASSWORD})
        me = client.get("/auth/me")
        threads = client.get("/threads")
        admin = client.get("/admin/me")

    assert signed_up.status_code == 201
    assert logged_in.status_code == 200
    assert me.json() == {"email": address, "role": "student"}
    assert threads.json() == []
    assert admin.status_code == 403

    with psycopg.connect(get_settings().database_url) as connection:
        row = connection.execute(
            "SELECT password_hash, notice_accepted_at FROM app.students WHERE email = %s",
            (address,),
        ).fetchone()
    assert row[0].startswith("$argon2")
    assert PASSWORD not in row[0]
    assert row[1] is not None


def test_production_app_limits_failed_logins_with_real_redis():
    address = email()
    with TestClient(create_app()) as client:
        codes = [
            client.post(
                "/auth/login", json={"email": address, "password": "wrong password"}
            ).status_code
            for _ in range(6)
        ]

    assert codes == [401] * 5 + [429]
