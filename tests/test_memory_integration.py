"""Integration: which chats are due for memory, claims, and memory search in real Postgres."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.core.persistence import EMBEDDING_DIMS, open_persistence
from owl_mind.services.memory import Memory, PostgresMemoryLog
from owl_mind.services.models import sqlalchemy_url
from tests.test_memory import toy_embed
from tests.test_runtime_integration import new_student

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

AGO = datetime.now(UTC) - timedelta(hours=2)


def padded_embed(texts: list[str]) -> list[list[float]]:
    """The toy embedder, padded to the store's real vector size."""
    return [v + [0.0] * (EMBEDDING_DIMS - len(v)) for v in toy_embed(texts)]


def chat(student, *runs: tuple[str, str]) -> uuid.UUID:
    """A thread with runs of (route, status), all two hours old."""
    thread = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as db:
        db.execute(
            "INSERT INTO app.threads (id, student_id, title, created_at) VALUES (%s, %s, 't', %s)",
            (thread, student, AGO),
        )
        for route, status in runs:
            db.execute(
                "INSERT INTO app.runs (id, thread_id, student_id, status, route, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (uuid.uuid4(), thread, student, status, route, AGO),
            )
    return thread


async def with_log(body):
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    try:
        return await body(PostgresMemoryLog(async_sessionmaker(engine, expire_on_commit=False)))
    finally:
        await engine.dispose()


def test_only_idle_chats_with_a_tutor_turn_and_new_turns_are_due():
    student = new_student()
    tutored = chat(student, ("template", "completed"), ("orchestrated", "completed"))
    greetings = chat(student, ("template", "completed"))
    # A waiting run holds the student's slot, so a second chat may stay busy only by being older.
    busy = chat(new_student(), ("orchestrated", "waiting_approval"))

    async def body(log):
        due = {c.thread_id for c in await log.due(datetime.now(UTC) - timedelta(minutes=30), 500)}
        early = {c.thread_id for c in await log.due(AGO - timedelta(minutes=1), 500)}
        claimed = await log.claim(tutored, student, "h1")
        await log.written(tutored, "h1")
        after = {c.thread_id for c in await log.due(datetime.now(UTC), 500)}
        return due, early, claimed, after

    due, early, claimed, after = asyncio.run(with_log(body))
    assert tutored in due
    assert greetings not in due and busy not in due
    assert tutored not in early  # Not idle long enough yet.
    assert claimed
    assert tutored not in after  # Written after its last turn.


def test_two_claims_at_once_give_one_writer_and_an_unchanged_hash_none():
    student = new_student()
    thread = chat(student, ("orchestrated", "completed"))

    async def body(log):
        first = await asyncio.gather(*[log.claim(thread, student, "h1") for _ in range(5)])
        await log.written(thread, "h1")
        same = await log.claim(thread, student, "h1")
        changed = await log.claim(thread, student, "h2")
        return first, same, changed

    first, same, changed = asyncio.run(with_log(body))
    assert sum(first) == 1
    assert (same, changed) == (False, True)


def test_memory_is_searched_by_meaning_and_kept_apart_in_the_real_store():
    alice, bob = uuid.uuid4(), uuid.uuid4()

    async def body():
        # The real store with a vector index, as the app opens it.
        async with open_persistence(get_settings().database_url, embed=padded_embed) as persistence:
            memory = Memory(persistence.store)
            profile = {"strengths": [], "weak_topics": ["Heaps"], "style": ""}
            for text in ("Markov chain transitions.", "Bloom filter hash positives.", "Heaps."):
                await memory.save(alice, uuid.uuid4(), profile, text)
            found = await memory.search(alice, "bloom filter hash")
            other = await memory.search(bob, "bloom filter hash")
            deleted = await memory.delete_all(alice)
            left = await memory.episodes(alice)
            return found, other, deleted, left

    found, other, deleted, left = asyncio.run(body())
    assert found[0].summary == "Bloom filter hash positives."
    assert other == []
    assert (deleted, left) == (4, [])
