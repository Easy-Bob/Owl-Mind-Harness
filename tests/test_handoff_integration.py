"""Needs the compose services: docker compose up -d --wait

The same checks as the in-memory store, so unit tests can trust it stands in for Postgres.
"""

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.forum.accounts import (
    ForumAccount,
    InMemoryAccountStore,
    PostgresAccountStore,
    TokenCipher,
)
from owl_mind.forum.github import Reply
from owl_mind.forum.handoff import HandoffService
from owl_mind.forum.store import Handoff, InMemoryHandoffStore, PostgresHandoffStore
from owl_mind.services.models import sqlalchemy_url
from tests.test_handoff import FakeForum
from tests.test_study_store_integration import (
    migrated,  # noqa: F401  Applied to every test.
    new_student,
    scenario,
)

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)
LEASE = timedelta(minutes=2)


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
        await store.record_post(posted.id, "https://github.com/o/f/discussions/1", "D_1", NOW)
        await store.record_post(posted.id, "https://github.com/o/f/discussions/1", "D_1", NOW)
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


# -- replies ----------------------------------------------------------------


def posted_handoff(store_student=None):
    student = store_student or new_student()
    return handoff(student, new_thread(student), status="posted")


def reply(number: int, *, body: str = "", answer: bool = False) -> Reply:
    created = NOW + timedelta(minutes=number)
    return Reply(f"C{uuid.uuid4().hex[:6]}{number}", "tutor", body or f"r{number}", created, answer)


def test_a_post_is_due_five_minutes_after_posting(run):
    post = posted_handoff()

    async def body(store):
        await store.save(post)
        await store.record_post(post.id, "https://example.invalid/1", "D_1", NOW)
        early = await store.lease_due_posts(NOW + timedelta(minutes=4), 50, LEASE)
        due = await store.lease_due_posts(NOW + timedelta(minutes=5), 50, LEASE)
        return [p.handoff_id for p in early], [p.handoff_id for p in due]

    early, due = run(body)

    assert post.id not in early
    assert post.id in due


def test_concurrent_leases_hand_a_post_to_one_poller(run):
    post = posted_handoff()

    async def body(store):
        await store.save(post)
        await store.record_post(post.id, "https://example.invalid/1", "D_1", NOW)
        later = NOW + timedelta(minutes=5)
        found = await asyncio.gather(*(store.lease_due_posts(later, 50, LEASE) for _ in range(5)))
        return [post.id in {p.handoff_id for p in batch} for batch in found]

    assert sorted(run(body)) == [False, False, False, False, True]


def test_replies_save_once_update_and_delete_like_github(run):
    post = posted_handoff()
    first, second = reply(1), reply(2)

    async def body(store):
        await store.save(post)
        await store.record_post(post.id, "https://example.invalid/1", "D_1", NOW)
        counts = [await store.save_replies(post.id, [first, second], True, NOW + LEASE)]
        edited = replace(first, body="edited", is_answer=True)
        counts.append(await store.save_replies(post.id, [edited, second], True, NOW + LEASE))
        counts.append(await store.save_replies(post.id, [edited], False, NOW + LEASE))
        kept = await store.replies_for(post.student_id, [post.id])
        counts.append(await store.save_replies(post.id, [edited], True, None))
        return counts, kept, await store.replies_for(post.student_id, [post.id])

    counts, kept, final = run(body)

    assert counts == [2, 0, 0, 0]
    assert [(r.body, r.is_answer) for r in kept] == [("edited", True), ("r2", False)]
    assert [r.id for r in final] == [first.id]


def test_unread_counts_and_seen_belong_to_the_student(run):
    post = posted_handoff()
    bob = new_student()

    async def body(store):
        await store.save(post)
        await store.record_post(post.id, "https://example.invalid/1", "D_1", NOW)
        await store.save_replies(post.id, [reply(1), reply(2)], True, NOW + LEASE)
        before = await store.unread_counts(post.student_id)
        await store.mark_seen(bob, post.id)
        bob_marked = await store.unread_counts(post.student_id)
        hidden = await store.replies_for(bob, [post.id])
        await store.mark_seen(post.student_id, post.id)
        return before, bob_marked, hidden, await store.unread_counts(post.student_id)

    before, bob_marked, hidden, after = run(body)

    assert before == {post.thread_id: 2} == bob_marked
    assert hidden == []
    assert after == {}


def test_failed_checks_and_deleted_discussions_change_the_schedule(run):
    failing, deleted = posted_handoff(), posted_handoff()

    async def body(store):
        for post in (failing, deleted):
            await store.save(post)
            await store.record_post(post.id, "https://example.invalid/1", "D_1", NOW)
        due = NOW + timedelta(minutes=5)
        await store.lease_due_posts(due, 50, LEASE)
        await store.poll_failed(failing.id, 1, due + timedelta(minutes=5))
        await store.discussion_gone(deleted.id)
        soon = await store.lease_due_posts(due + timedelta(minutes=4), 50, LEASE)
        later = await store.lease_due_posts(due + timedelta(minutes=5), 50, LEASE)
        return {p.handoff_id: p.failures for p in soon}, {p.handoff_id: p.failures for p in later}

    soon, later = run(body)

    assert failing.id not in soon and deleted.id not in soon
    assert later.get(failing.id) == 1
    assert deleted.id not in later


# -- accounts ---------------------------------------------------------------


def postgres_accounts():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    return PostgresAccountStore(async_sessionmaker(engine, expire_on_commit=False)), engine.dispose


def memory_accounts():
    async def nothing():
        return None

    return InMemoryAccountStore(), nothing


@pytest.fixture(params=[memory_accounts, postgres_accounts], ids=["memory", "postgres"])
def run_accounts(request):
    return scenario(request.param)


def test_account_round_trip_update_and_delete(run_accounts):
    student = new_student()
    account = ForumAccount(student, "gh-user", b"encrypted-a", b"encrypted-r", NOW, None)

    async def body(store):
        await store.save(account)
        saved = await store.get(student)
        await store.save(replace(account, needs_reconnect=True, access_token=b"other"))
        updated = await store.get(student)
        listed = student in {a.student_id for a in await store.all()}
        await store.delete(student)
        return saved, updated, listed, await store.get(student)

    saved, updated, listed, gone = run_accounts(body)

    assert saved == account
    assert (updated.needs_reconnect, updated.access_token) == (True, b"other")
    assert listed
    assert gone is None


def test_the_database_row_holds_no_plaintext_token():
    student = new_student()
    cipher = TokenCipher([Fernet.generate_key().decode()])
    account = ForumAccount(student, "gh-user", cipher.encrypt("ghu_plain_token"), None, None, None)

    async def body(store):
        await store.save(account)

    scenario(postgres_accounts)(body)
    with psycopg.connect(get_settings().database_url) as connection:
        stored = connection.execute(
            "SELECT access_token FROM app.forum_accounts WHERE student_id = %s", (student,)
        ).fetchone()[0]
        # Rows under a throwaway key would look like a lost key to anything run later.
        connection.execute("DELETE FROM app.forum_accounts WHERE student_id = %s", (student,))

    assert b"ghu_plain_token" not in bytes(stored)
    assert cipher.decrypt(bytes(stored)) == "ghu_plain_token"


# -- rewrites ---------------------------------------------------------------


def test_rewrites_are_capped_once_even_when_clicked_together(run):
    alice = new_student()
    draft = handoff(alice, new_thread(alice))

    async def body(store):
        await store.save(draft)
        results = await asyncio.gather(*(store.claim_improvement(draft.id, 3) for _ in range(5)))
        # A later save from an older copy mustn't reset the count.
        await store.save(replace(draft, title="Edited"))
        saved = await store.get(alice, draft.id)
        return results, saved.improvements, saved.title

    results, used, title = run(body)

    assert sorted(r for r in results if r is not None) == [1, 2, 3]
    assert results.count(None) == 2
    assert (used, title) == (3, "Edited")


def test_only_a_draft_can_be_rewritten(run):
    alice = new_student()
    posted = handoff(alice, new_thread(alice), status="posted")

    async def body(store):
        await store.save(posted)
        return await store.claim_improvement(posted.id, 3)

    assert run(body) is None
