"""Needs the compose services: docker compose up -d --wait

The same checks as the in-memory store, so unit tests can trust it stands in for Postgres.
"""

import asyncio
import uuid
from dataclasses import replace

import psycopg
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.forum.handoff import (
    Handoff,
    HandoffService,
    InMemoryHandoffStore,
    PostgresHandoffStore,
)
from owl_mind.services.models import sqlalchemy_url
from tests.test_handoff import FakeForum
from tests.test_study_store_integration import (
    migrated,  # noqa: F401  Applied to every test.
    new_student,
    scenario,
)

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


def new_thread(student_id: uuid.UUID) -> uuid.UUID:
    thread_id = uuid.uuid4()
    with psycopg.connect(get_settings().database_url) as connection:
        connection.execute(
            "INSERT INTO app.threads (id, student_id, title) VALUES (%s, %s, 't')",
            (thread_id, student_id),
        )
    return thread_id


def postgres_store():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    return PostgresHandoffStore(async_sessionmaker(engine, expire_on_commit=False)), engine.dispose


def memory_store():
    async def nothing():
        return None

    return InMemoryHandoffStore(), nothing


@pytest.fixture(params=[memory_store, postgres_store], ids=["memory", "postgres"])
def run(request):
    return scenario(request.param)


def handoff(student_id: uuid.UUID, thread_id: uuid.UUID, **changes) -> Handoff:
    fields = dict(status="draft", title="T", body="B", graded=False, category="Q&A")
    return Handoff(uuid.uuid4(), student_id, thread_id, **{**fields, **changes})


def test_saved_handoff_reads_back_only_for_its_student(run):
    alice, bob = new_student(), new_student()
    thread = new_thread(alice)
    draft = handoff(alice, thread)

    async def body(store):
        await store.save(draft)
        await store.save(replace(draft, title="Edited", graded=True))
        return (
            await store.get(alice, draft.id),
            await store.get(bob, draft.id),
            await store.for_thread(alice, thread),
            await store.for_thread(bob, thread),
        )

    mine, theirs, listed, other = run(body)

    assert (mine.title, mine.graded, mine.body) == ("Edited", True, "B")
    assert theirs is None
    assert [h.id for h in listed] == [draft.id]
    assert other == []


def test_claim_changes_the_status_only_once(run):
    alice = new_student()
    draft = handoff(alice, new_thread(alice))

    async def body(store):
        await store.save(draft)
        results = await asyncio.gather(
            *(store.claim(draft.id, "draft", "posting") for _ in range(5))
        )
        return results, (await store.get(alice, draft.id)).status

    results, status = run(body)

    assert sorted(results) == [False, False, False, False, True]
    assert status == "posting"


def test_last_post_time_and_retry_list(run):
    alice = new_student()
    thread = new_thread(alice)
    posted, waiting = handoff(alice, thread, status="posted"), handoff(alice, thread)

    async def body(store):
        before = await store.last_posted_at(alice)
        await store.save(posted)
        await store.record_post(posted.id, "https://github.com/o/f/discussions/1", "D_1")
        await store.record_post(posted.id, "https://github.com/o/f/discussions/1", "D_1")
        await store.save(replace(waiting, status="retrying"))
        due = await store.due_for_retry()
        return before, await store.last_posted_at(alice), due

    before, after, due = run(body)

    assert before is None
    assert after is not None
    assert waiting.id in {h.id for h in due}


def test_restart_turns_an_unfinished_post_into_a_fallback(run):
    alice = new_student()
    draft = handoff(alice, new_thread(alice), status="posting")

    async def body(store):
        await store.save(draft)
        await store.fail_unfinished()
        return (await store.get(alice, draft.id)).status

    assert run(body) == "fallback"


def test_status_outside_the_list_is_refused():
    alice = new_student()
    bad = handoff(alice, new_thread(alice), status="published")

    async def body(store):
        await store.save(bad)

    with pytest.raises(Exception, match="check"):
        scenario(postgres_store)(body)


def test_whole_service_posts_through_postgres():
    alice = new_student()
    thread = new_thread(alice)
    forum = FakeForum()

    async def body(store):
        service = HandoffService(store, forum, "Q&A")
        draft, _, _ = await service.draft(alice, thread, "Why is BFS slow?", [], [])
        return await service.approve(alice, draft.id, draft.title, draft.body, False, [])

    posted = scenario(postgres_store)(body)

    assert posted.status == "posted"
    assert len(forum.posts) == 1
