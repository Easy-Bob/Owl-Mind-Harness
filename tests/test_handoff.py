"""Offline: the tutor handoff, from the chat message to the forum post, with GitHub replaced."""

import asyncio
import dataclasses
import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from owl_mind.core import replies
from owl_mind.core.config import get_settings
from owl_mind.forum.github import ForumPost, ForumUnavailable, GitHubForum, SimilarPost
from owl_mind.forum.handoff import (
    MAX_RETRIES,
    HandoffService,
    InMemoryHandoffStore,
    known_values,
    summarize,
)
from owl_mind.forum.redact import escape_markdown, for_graded_problem, redact, trim_code
from tests.test_auth import ALICE_EMAIL, logged_in
from tests.test_auth import settings as auth_settings  # noqa: F401  Applied to every test.
from tests.test_runtime import World, parse_sse

ZW = "\u200b"
REPO = "owner/forum"


class FakeForum:
    """Records posts. Set `failures` to make the next posts fail."""

    home_url = "https://github.com/owner/forum/discussions"

    def __init__(self) -> None:
        self.posts: list[tuple[str, str, str]] = []
        self.searches: list[str] = []
        self.failures = 0

    async def search_similar(self, text: str, limit: int = 3) -> list[SimilarPost]:
        self.searches.append(text)
        return [SimilarPost("Dijkstra with negative edges", "https://github.com/owner/forum/1")]

    async def post_question(self, category: str, title: str, body: str) -> ForumPost:
        if self.failures:
            self.failures -= 1
            raise ForumUnavailable("GitHub answered 502")
        self.posts.append((category, title, body))
        return ForumPost(f"https://github.com/owner/forum/discussions/{len(self.posts)}", "D_1")

    async def get_replies(self, node_id: str) -> list:
        return []


def handoff_world(forum: FakeForum | None) -> World:
    world = World()
    world.app.dependency_overrides.clear()
    world.handoff_store = InMemoryHandoffStore()
    world.services = dataclasses.replace(
        world.services, handoffs=HandoffService(world.handoff_store, forum, "Q&A")
    )
    return world


def ask_for_tutor(client, message: str = "I want to talk to a TA") -> tuple[str, list]:
    started = client.post("/threads", json={"message": message}).json()
    events = parse_sse(client.get(f"/runs/{started['run_id']}/stream").text)
    return started["thread_id"], events


def draft_of(client, thread_id: str) -> dict:
    [draft] = client.get("/handoffs", params={"thread_id": thread_id}).json()
    return draft


def approve(client, draft: dict, **changes):
    fields = {"title": draft["title"], "body": draft["body"], "graded": draft["graded"]}
    return client.post(f"/handoffs/{draft['id']}/approve", json={**fields, **changes})


# -- redaction --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "gone"),
    [
        ("mail me at jane.doe+580@rice.edu please", "email", "jane.doe"),
        ("my key is sk-abcdefghijklmnop123", "secret", "sk-abc"),
        ("token ghp_" + "a" * 36, "secret", "ghp_a"),
        ("password = hunter22", "secret", "hunter22"),
        ("API_KEY: zzzyyyxxx", "secret", "zzzyyyxxx"),
        ("my student id is S01234567", "student id", "S01234567"),
        ("Notes by Ann Lee (al42)", "netid", "al42"),
        ("call 713-555-0100", "phone", "555-0100"),
        ("Hi, my name is Jane Doe and I'm stuck", "name", "Jane"),
    ],
)
def test_private_details_are_removed_and_counted(text, kind, gone):
    result = redact(text)

    assert gone not in result.text
    assert result.removed.get(kind, 0) >= 1


@pytest.mark.parametrize(
    "text",
    [
        "I am stuck on Dijkstra with negative edges",
        "This is Python 3, and the heap has 12 items",
        "Why is my loop O(n^2)?",
    ],
)
def test_ordinary_questions_are_left_alone(text):
    assert redact(text).text == text


@pytest.mark.parametrize(
    "text",
    ["password = hunter22 and jane@rice.edu", "my name is Jane Doe, id S01234567, (al42)"],
)
def test_redacting_twice_changes_nothing_more(text):
    once = redact(text).text

    assert redact(once).text == once


def test_the_students_own_email_and_name_parts_are_removed_anywhere():
    values = known_values("jane.doe@rice.edu")
    result = redact("Thanks, JANE. (Doe here, jane.doe on campus)", values)

    assert "jane" not in result.text.lower()
    assert "doe" not in result.text.lower()


def test_short_known_values_are_ignored_so_words_survive():
    assert redact("an ox ate", ["ox", "an"]).text == "an ox ate"


def test_long_code_is_trimmed():
    code = "```python\n" + "\n".join(f"line{i} = {i}" for i in range(30)) + "\n```"

    trimmed = trim_code(code, 10)

    assert "line9 " in trimmed and "line10 " not in trimmed
    assert "20 more lines not shown" in trimmed


def test_graded_problem_keeps_only_a_short_excerpt():
    text = "word " * 400 + "\n```\n" + "\n".join(f"x{i}" for i in range(20)) + "\n```"

    shortened = for_graded_problem(text)

    assert len(shortened) < 500
    assert shortened.endswith("(shortened: graded problem)")


@pytest.mark.parametrize(
    ("text", "escaped"),
    [
        ("ping @prof", f"ping @{ZW}prof"),
        ("see #12", f"see #{ZW}12"),
        ("![x](http://evil/t.png)", f"!{ZW}[x](http://evil/t.png)"),
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("a & b", "a &amp; b"),
        ("# heading and C# code", "# heading and C# code"),
    ],
)
def test_github_cannot_turn_student_text_into_mentions_links_images_or_html(text, escaped):
    assert escape_markdown(text) == escaped


def test_summary_comes_from_a_template_with_errors_listed():
    title, body = summarize(
        "Why does my BFS fail?\nIt prints the wrong order.",
        ["Traceback (most recent call last):\nKeyError: 3"],
    )

    assert title == "Why does my BFS fail?"
    assert "**Question**" in body and "**Earlier in the conversation**" in body
    assert "- KeyError: 3" in body


# -- the chat turn ----------------------------------------------------------


def test_asking_for_a_tutor_drafts_a_post_with_no_model_call():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, events = ask_for_tutor(client, "I want to talk to a TA about BFS")
        draft = draft_of(client, thread_id)
        history = client.get(f"/threads/{thread_id}/messages").json()

    [card] = [data for kind, data in events if kind == "handoff_draft"]
    assert card["id"] == draft["id"]
    assert card["similar"] == [
        {"title": "Dijkstra with negative edges", "url": "https://github.com/owner/forum/1"}
    ]
    text = "".join(data["text"] for kind, data in events if kind == "text_delta")
    assert text == replies.HANDOFF_DRAFTED
    assert history[-1]["content"] == replies.HANDOFF_DRAFTED
    assert world.fake.requests == []
    assert forum.posts == []
    assert draft["status"] == "draft"
    assert draft["title"] == "I want to talk to a TA about BFS"
    assert draft["forum_url"] == FakeForum.home_url


def test_draft_already_has_the_students_email_removed():
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        thread_id, events = ask_for_tutor(client, f"talk to a TA, reach me at {ALICE_EMAIL}")
        draft = draft_of(client, thread_id)

    assert ALICE_EMAIL not in draft["body"]
    [card] = [data for kind, data in events if kind == "handoff_draft"]
    assert card["removed"]


def test_without_a_forum_the_old_reply_stays():
    world = World()
    world.app.dependency_overrides.clear()
    with world.client() as client:
        logged_in(client)
        _, events = ask_for_tutor(client)
        routes = client.get("/handoffs", params={"thread_id": str(uuid.uuid4())})

    assert "".join(d["text"] for k, d in events if k == "text_delta") == (
        replies.HANDOFF_UNAVAILABLE
    )
    assert routes.status_code == 503


def test_the_run_record_keeps_no_draft_text():
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        ask_for_tutor(client, "talk to a TA about my secret plan")

    recorded = json.dumps([event for run in world.runs.events.values() for event in run])
    assert "secret plan" not in recorded


# -- approval ---------------------------------------------------------------


def test_approval_posts_exactly_the_approved_text_once_escaped():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        posted = approve(client, draft, body="Ping @prof about #3, please")
        again = approve(client, draft, body="Ping @prof about #3, please")

    assert posted.status_code == 200, posted.text
    assert posted.json()["status"] == "posted"
    assert posted.json()["url"] == "https://github.com/owner/forum/discussions/1"
    assert forum.posts == [("Q&A", draft["title"], f"Ping @{ZW}prof about #{ZW}3, please")]
    assert again.status_code == 400


def test_two_approvals_at_once_post_once():
    forum = FakeForum()
    store = InMemoryHandoffStore()
    service = HandoffService(store, forum, "Q&A")
    student, thread = uuid.uuid4(), uuid.uuid4()

    async def scenario():
        draft, _, _ = await service.draft(student, thread, "help with BFS", [], [])
        return await asyncio.gather(
            service.approve(student, draft.id, draft.title, draft.body, False, []),
            service.approve(student, draft.id, draft.title, draft.body, False, []),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())

    assert len(forum.posts) == 1
    assert sum(isinstance(r, Exception) for r in results) == 1


def test_text_that_redaction_changes_must_be_approved_again():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        first = approve(client, draft, body="my password = hunter22 fails")
        second = approve(client, first.json())

    assert first.status_code == 409
    assert first.json()["body"] == "my password = [secret removed] fails"
    assert forum.posts == [("Q&A", draft["title"], "my password = [secret removed] fails")]
    assert second.json()["status"] == "posted"


def test_graded_flag_shortens_the_post_before_approval():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        response = approve(client, draft, body="word " * 300, graded=True)

    assert response.status_code == 409
    assert response.json()["body"].endswith("(shortened: graded problem)")
    assert response.json()["graded"] is True
    assert forum.posts == []


def test_another_student_cannot_see_approve_or_cancel_a_draft():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        client.cookies.clear()
        logged_in(client, email="bob@example.com")
        listed = client.get("/handoffs", params={"thread_id": thread_id}).json()
        approved = approve(client, draft)
        cancelled = client.post(f"/handoffs/{draft['id']}/cancel")

    assert listed == []
    assert approved.status_code == cancelled.status_code == 404
    assert forum.posts == []


def test_cancelled_draft_cannot_be_posted():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        cancelled = client.post(f"/handoffs/{draft['id']}/cancel")
        approved = approve(client, draft)

    assert cancelled.json()["status"] == "cancelled"
    assert approved.status_code == 400
    assert forum.posts == []


@pytest.mark.parametrize(("title", "body"), [("   ", "text"), ("Title", "  \n ")])
def test_empty_title_or_body_is_refused(title, body):
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        response = approve(client, draft_of(client, thread_id), title=title, body=body)

    assert response.status_code == 400


def test_one_post_per_ten_minutes():
    forum = FakeForum()
    world = handoff_world(forum)
    with world.client() as client:
        logged_in(client)
        first, _ = ask_for_tutor(client)
        assert approve(client, draft_of(client, first)).status_code == 200
        second, _ = ask_for_tutor(client)
        draft = draft_of(client, second)
        blocked = approve(client, draft)
        for handoff_id in world.handoff_store.posted_at:
            world.handoff_store.posted_at[handoff_id] -= timedelta(minutes=11)
        allowed = approve(client, draft)

    assert blocked.status_code == 400
    assert "10 minutes" in blocked.json()["detail"]
    assert allowed.status_code == 200
    assert len(forum.posts) == 2


# -- failures ---------------------------------------------------------------


def test_github_failure_is_retried_in_the_background_then_given_up():
    forum = FakeForum()
    store = InMemoryHandoffStore()
    service = HandoffService(store, forum, "Q&A")
    student = uuid.uuid4()
    forum.failures = MAX_RETRIES

    async def scenario():
        draft, _, _ = await service.draft(student, uuid.uuid4(), "help", [], [])
        first = await service.approve(student, draft.id, draft.title, draft.body, False, [])
        statuses = [first.status]
        for _ in range(MAX_RETRIES):
            await service.retry_pending()
            statuses.append((await store.get(student, draft.id)).status)
        return statuses

    assert asyncio.run(scenario()) == ["retrying", "retrying", "fallback", "fallback"]
    assert forum.posts == []


def test_retry_posts_once_github_is_back():
    forum = FakeForum()
    store = InMemoryHandoffStore()
    service = HandoffService(store, forum, "Q&A")
    student = uuid.uuid4()
    forum.failures = 1

    async def scenario():
        draft, _, _ = await service.draft(student, uuid.uuid4(), "help", [], [])
        await service.approve(student, draft.id, draft.title, draft.body, False, [])
        await service.retry_pending()
        await service.retry_pending()
        return await store.get(student, draft.id)

    handoff = asyncio.run(scenario())

    assert handoff.status == "posted"
    assert len(forum.posts) == 1


def test_without_github_the_student_gets_a_copyable_fallback():
    world = handoff_world(None)
    with world.client() as client:
        logged_in(client)
        thread_id, events = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        response = approve(client, draft)

    assert [kind for kind, _ in events].count("handoff_draft") == 1
    assert response.json()["status"] == "fallback"
    assert response.json()["body"] == draft["body"]


def test_a_post_interrupted_by_a_restart_is_not_sent_twice():
    store = InMemoryHandoffStore()
    service = HandoffService(store, FakeForum(), "Q&A")
    student = uuid.uuid4()

    async def scenario():
        draft, _, _ = await service.draft(student, uuid.uuid4(), "help", [], [])
        await store.claim(draft.id, "draft", "posting")
        await store.fail_unfinished()
        await service.retry_pending()
        return await store.get(student, draft.id)

    assert asyncio.run(scenario()).status == "fallback"


def test_search_failure_still_gives_a_draft():
    forum = FakeForum()

    async def broken(text, limit=3):
        raise ForumUnavailable("down")

    forum.search_similar = broken
    service = HandoffService(InMemoryHandoffStore(), forum, "Q&A")

    _, _, similar = asyncio.run(service.draft(uuid.uuid4(), uuid.uuid4(), "help", [], []))

    assert similar == []


# -- the GitHub client ------------------------------------------------------


@pytest.fixture
def app_key(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "app.pem"
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return path, key.public_key()


class FakeGitHub:
    def __init__(self, public_key, graphql_errors: bool = False, status: int = 200) -> None:
        self.public_key = public_key
        self.graphql_errors = graphql_errors
        self.status = status
        self.requests: list[httpx.Request] = []
        self.token_grants = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        auth = request.headers["Authorization"]
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            claims = jwt.decode(auth.removeprefix("Bearer "), self.public_key, algorithms=["RS256"])
            assert claims["iss"] == "123"
            return httpx.Response(200, json={"id": 77})
        if path == "/app/installations/77/access_tokens":
            self.token_grants += 1
            return httpx.Response(201, json={"token": "ghs_installation"})
        assert path == "/graphql" and auth == "token ghs_installation"
        if self.status != 200:
            return httpx.Response(self.status, json={})
        if self.graphql_errors:
            return httpx.Response(200, json={"errors": [{"message": "no"}]})
        query = json.loads(request.content)["query"]
        if "createDiscussion" in query:
            discussion = {"id": "D_9", "url": "https://github.com/owner/forum/discussions/9"}
            return httpx.Response(
                200, json={"data": {"createDiscussion": {"discussion": discussion}}}
            )
        if "discussionCategories" in query:
            categories = [{"id": "C_qa", "name": "Q&A"}, {"id": "C_gen", "name": "General"}]
            repository = {"id": "R_1", "discussionCategories": {"nodes": categories}}
            return httpx.Response(200, json={"data": {"repository": repository}})
        if "search(" in query:
            nodes = [
                {"title": "Old question", "url": "https://github.com/owner/forum/discussions/2"}
            ]
            return httpx.Response(200, json={"data": {"search": {"nodes": nodes}}})
        comments = [{"id": "C1", "body": "Use a heap.", "createdAt": "2026-11-02T00:00:00Z"}]
        node = {"answer": {"id": "C1"}, "comments": {"nodes": comments}}
        return httpx.Response(200, json={"data": {"node": node}})

    def variables(self, index: int) -> dict:
        return json.loads(self.requests[index].content)["variables"]


def github(app_key, **options) -> tuple[GitHubForum, FakeGitHub]:
    path, public_key = app_key
    fake = FakeGitHub(public_key, **options)
    return GitHubForum(REPO, "123", path, httpx.MockTransport(fake.handler)), fake


def test_client_posts_to_the_named_category_with_an_installation_token(app_key):
    forum, fake = github(app_key)

    async def scenario():
        post = await forum.post_question("Q&A", "Title", "Body")
        await forum.post_question("Q&A", "Second", "Body")
        return post

    post = asyncio.run(scenario())

    assert post == ForumPost("https://github.com/owner/forum/discussions/9", "D_9")
    assert fake.token_grants == 1
    created = fake.variables(3)
    assert created == {"repository": "R_1", "category": "C_qa", "title": "Title", "body": "Body"}


def test_client_searches_only_the_forum_repository(app_key):
    forum, fake = github(app_key)

    similar = asyncio.run(forum.search_similar("why does   BFS fail"))

    assert similar == [SimilarPost("Old question", "https://github.com/owner/forum/discussions/2")]
    assert fake.variables(2)["query"] == f"repo:{REPO} why does BFS fail"


def test_client_reads_replies_and_the_accepted_answer(app_key):
    forum, _ = github(app_key)

    [reply] = asyncio.run(forum.get_replies("D_9"))

    assert reply.body == "Use a heap." and reply.is_answer


def test_client_refreshes_a_token_close_to_expiry(app_key):
    forum, fake = github(app_key)

    async def scenario():
        await forum.search_similar("a")
        token, _ = forum._token
        forum._token = (token, datetime.now(UTC).timestamp() + 60)
        await forum.search_similar("b")

    asyncio.run(scenario())

    assert fake.token_grants == 2


@pytest.mark.parametrize("options", [{"graphql_errors": True}, {"status": 502}])
def test_client_failures_become_forum_unavailable(app_key, options):
    forum, _ = github(app_key, **options)

    with pytest.raises(ForumUnavailable):
        asyncio.run(forum.post_question("Q&A", "Title", "Body"))


def test_unknown_category_is_refused(app_key):
    forum, _ = github(app_key)

    with pytest.raises(ForumUnavailable):
        asyncio.run(forum.post_question("Announcements", "Title", "Body"))


def test_forum_settings_are_optional(monkeypatch):
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.forum_category == "Q&A"
    assert settings.forum_app_id is None
