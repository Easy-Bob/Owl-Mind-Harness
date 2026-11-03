"""Integration: the budget counters in real Redis."""

import asyncio
import uuid
from datetime import date

import pytest
from redis.asyncio import Redis

from owl_mind.core.budget import RedisBudgetLedger, Spend
from owl_mind.core.config import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

# A day no real run uses, so these counters never mix with a developer's own.
DAY = date(2001, 1, 1)


async def with_ledger(scenario):
    client = Redis.from_url(get_settings().redis_url)
    try:
        await client.delete(f"budget:global:{DAY.isoformat()}")
        return await scenario(RedisBudgetLedger(client), client)
    finally:
        keys = [key async for key in client.scan_iter(f"budget:*{DAY.isoformat()}*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


def test_charges_add_up_per_student_and_for_everyone():
    alice, bob = uuid.uuid4(), uuid.uuid4()

    async def scenario(ledger, client):
        await ledger.charge(alice, DAY, 300, 5)
        await ledger.charge(bob, DAY, 100, 2)
        return await ledger.spent(alice, DAY), await ledger.spent(bob, DAY)

    assert asyncio.run(with_ledger(scenario)) == (Spend(300, 7), Spend(100, 7))


def test_charges_at_the_same_moment_are_never_lost():
    alice = uuid.uuid4()

    async def scenario(ledger, client):
        await asyncio.gather(*[ledger.charge(alice, DAY, 10, 1) for _ in range(50)])
        return await ledger.spent(alice, DAY)

    assert asyncio.run(with_ledger(scenario)) == Spend(500, 50)


def test_counters_expire_so_old_days_do_not_pile_up():
    alice = uuid.uuid4()

    async def scenario(ledger, client):
        await ledger.charge(alice, DAY, 10, 1)
        return await client.ttl(f"budget:student:{DAY.isoformat()}:{alice}")

    assert 0 < asyncio.run(with_ledger(scenario)) <= 2 * 24 * 3600


def test_nothing_spent_reads_as_zero():
    async def scenario(ledger, client):
        return await ledger.spent(uuid.uuid4(), DAY)

    assert asyncio.run(with_ledger(scenario)) == Spend(0, 0)
