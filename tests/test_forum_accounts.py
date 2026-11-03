"""Offline: connecting a student's own GitHub account, with GitHub's login service replaced."""

import asyncio
import json
import logging
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from cryptography.fernet import Fernet, InvalidToken

from owl_mind.api.app import create_app
from owl_mind.forum.accounts import (
    RECONNECT_MESSAGE,
    ForumAccounts,
    GitHubOAuth,
    InMemoryAccountStore,
    ReconnectNeeded,
    TokenCipher,
    rotate_keys,
)
from tests.test_auth import logged_in
from tests.test_auth import settings as auth_settings  # noqa: F401  Applied to every test.
from tests.test_handoff import Clock, FakeForum, approve, ask_for_tutor, draft_of, handoff_world

LOGIN = "student-gh"
EIGHT_HOURS = 8 * 3600


class FakeGitHubLogin:
    """GitHub's token endpoint, user endpoint and token revocation."""

    def __init__(self) -> None:
        self.issued = 0
        self.refresh_works = True
        self.revoked: list[str] = []
        self.requests: list[httpx.Request] = []

    def tokens(self) -> dict:
        self.issued += 1
        return {
            "access_token": f"ghu_access{self.issued}",
            "refresh_token": f"ghr_refresh{self.issued}",
            "expires_in": EIGHT_HOURS,
            "refresh_token_expires_in": 180 * 24 * 3600,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/login/oauth/access_token":
            form = parse_qs(request.content.decode())
            assert form["client_secret"] == ["client-secret"]
            if form.get("code") == ["good-code"]:
                return httpx.Response(200, json=self.tokens())
            if form.get("grant_type") == ["refresh_token"] and self.refresh_works:
                return httpx.Response(200, json=self.tokens())
            return httpx.Response(200, json={"error": "bad_verification_code"})
        if request.url.path == "/user":
            assert request.headers["Authorization"].startswith("Bearer ghu_")
            return httpx.Response(200, json={"login": LOGIN})
        if request.method == "DELETE":
            self.revoked.append(json.loads(request.content)["access_token"])
            return httpx.Response(204)
        return httpx.Response(404)


def make_accounts(clock: Clock, keys: list[str] | None = None):
    github = FakeGitHubLogin()
    oauth = GitHubOAuth(
        "client-id",
        "client-secret",
        "http://localhost:5173/forum/account/callback",
        httpx.MockTransport(github.handler),
        clock,
    )
    store = InMemoryAccountStore()
    cipher = TokenCipher(keys or [Fernet.generate_key().decode()])
    return ForumAccounts(store, oauth, cipher, clock), github


def connected_world(forum: FakeForum | None = None):
    clock = Clock()
    accounts, github = make_accounts(clock)
    forum = forum or FakeForum()
    world = handoff_world(forum, clock, accounts)
    world.accounts_store, world.github, world.forum = accounts.store, github, forum
    return world


def connect(client) -> httpx.Response:
    start = client.get("/forum/account/connect", follow_redirects=False)
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    return client.get(
        "/forum/account/callback",
        params={"code": "good-code", "state": state},
        follow_redirects=False,
    )


# -- encryption -------------------------------------------------------------


def test_stored_tokens_are_not_plaintext():
    cipher = TokenCipher([Fernet.generate_key().decode()])

    stored = cipher.encrypt("ghu_secret_value")

    assert b"ghu_secret_value" not in stored
    assert cipher.decrypt(stored) == "ghu_secret_value"


def test_key_rotation_keeps_every_token_working_under_the_new_key():
    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    clock = Clock()
    accounts, _ = make_accounts(clock, [old])
    student = uuid.uuid4()
    asyncio.run(accounts.connect(student, "good-code"))

    rotated = asyncio.run(rotate_keys(accounts.store, TokenCipher([new, old])))
    account = asyncio.run(accounts.store.get(student))

    assert (rotated.rotated, rotated.unreadable) == (1, 0)
    assert TokenCipher([new]).decrypt(account.access_token) == "ghu_access1"
    assert TokenCipher([new]).decrypt(account.refresh_token) == "ghr_refresh1"
    with pytest.raises(InvalidToken):
        TokenCipher([old]).decrypt(account.access_token)


def test_a_bad_key_setting_is_refused():
    with pytest.raises(ValueError):
        TokenCipher.from_setting("not-a-key")
    with pytest.raises(ValueError):
        TokenCipher.from_setting(" , ")


# -- connecting -------------------------------------------------------------


def test_connecting_stores_an_encrypted_token_and_shows_the_login():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        before = client.get("/forum/account").json()
        finished = connect(client)
        after = client.get("/forum/account").json()

    assert before == {
        "available": True,
        "connected": False,
        "login": None,
        "needs_reconnect": False,
    }
    assert finished.status_code == 303
    assert finished.headers["location"] == "/?forum=connected"
    assert after == {"available": True, "connected": True, "login": LOGIN, "needs_reconnect": False}
    [account] = world.accounts_store.accounts.values()
    assert b"ghu_access1" not in account.access_token


def test_connect_sends_the_student_to_github_with_a_signed_state():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        start = client.get("/forum/account/connect", follow_redirects=False)

    location = urlparse(start.headers["location"])
    query = parse_qs(location.query)
    assert start.status_code == 303
    assert location.netloc == "github.com" and location.path == "/login/oauth/authorize"
    assert query["client_id"] == ["client-id"]
    assert query["redirect_uri"] == ["http://localhost:5173/forum/account/callback"]


@pytest.mark.parametrize("state", ["", "forged", "other student"])
def test_a_callback_without_this_students_state_connects_nothing(state):
    world = connected_world()
    with world.client() as client:
        logged_in(client, email="mallory@example.com")
        mallory_state = parse_qs(
            urlparse(
                client.get("/forum/account/connect", follow_redirects=False).headers["location"]
            ).query
        )["state"][0]
        client.cookies.clear()
        logged_in(client)
        used = mallory_state if state == "other student" else state
        finished = client.get(
            "/forum/account/callback",
            params={"code": "good-code", "state": used},
            follow_redirects=False,
        )

    assert finished.headers["location"] == "/?forum=failed"
    assert world.accounts_store.accounts == {}


def test_a_refused_code_connects_nothing():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        start = client.get("/forum/account/connect", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        finished = client.get(
            "/forum/account/callback",
            params={"code": "bad-code", "state": state},
            follow_redirects=False,
        )

    assert finished.headers["location"] == "/?forum=failed"
    assert world.accounts_store.accounts == {}


def test_without_settings_accounts_are_unavailable():
    world = handoff_world(FakeForum())
    with world.client() as client:
        logged_in(client)
        status = client.get("/forum/account").json()
        start = client.get("/forum/account/connect", follow_redirects=False)

    assert status["available"] is False
    assert start.status_code == 503


def test_account_routes_need_a_login():
    world = connected_world()
    with world.client() as client:
        responses = [
            client.get("/forum/account"),
            client.get("/forum/account/connect", follow_redirects=False),
            client.get("/forum/account/callback", follow_redirects=False),
            client.post("/forum/account/disconnect"),
        ]

    assert {response.status_code for response in responses} == {401}


# -- posting ----------------------------------------------------------------


def test_a_connected_student_posts_as_themself():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        posted = approve(client, draft)

    assert draft["post_as"] == LOGIN
    assert posted.json()["post_as"] == LOGIN
    assert world.forum.tokens == ["ghu_access1"]


def test_without_an_account_the_bot_posts():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        posted = approve(client, draft)

    assert draft["post_as"] == "bot"
    assert posted.json()["post_as"] == "bot"
    assert world.forum.tokens == [None]


def test_an_expiring_token_is_refreshed_before_posting():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        world.clock.advance(hours=8)
        thread_id, _ = ask_for_tutor(client)
        approve(client, draft_of(client, thread_id))

    assert world.forum.tokens == ["ghu_access2"]
    [account] = world.accounts_store.accounts.values()
    assert world.services.forum_accounts.cipher.decrypt(account.refresh_token) == "ghr_refresh2"


def test_a_failed_refresh_asks_to_reconnect_and_never_posts_as_the_bot():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        world.github.refresh_works = False
        world.clock.advance(hours=8)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        refused = approve(client, draft)
        status = client.get("/forum/account").json()
        [still] = client.get("/handoffs", params={"thread_id": thread_id}).json()

    assert refused.status_code == 400
    assert refused.json()["detail"] == RECONNECT_MESSAGE
    assert status["needs_reconnect"] is True
    assert still["status"] == "draft"
    assert world.forum.posts == []


def test_a_revoked_token_asks_to_reconnect_and_keeps_the_draft():
    world = connected_world()
    world.forum.refused_tokens.add("ghu_access1")
    with world.client() as client:
        logged_in(client)
        connect(client)
        thread_id, _ = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        refused = approve(client, draft)
        again = approve(client, draft)
        [still] = client.get("/handoffs", params={"thread_id": thread_id}).json()

    assert refused.status_code == again.status_code == 400
    assert refused.json()["detail"] == again.json()["detail"] == RECONNECT_MESSAGE
    assert still["status"] == "draft"
    assert world.forum.posts == []


def test_reconnecting_clears_the_problem():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        asyncio.run(
            world.services.forum_accounts.mark_broken(next(iter(world.accounts_store.accounts)))
        )
        connect(client)
        status = client.get("/forum/account").json()

    assert status["needs_reconnect"] is False


def test_disconnecting_deletes_the_token_and_revokes_it():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        gone = client.post("/forum/account/disconnect")
        status = client.get("/forum/account").json()
        thread_id, _ = ask_for_tutor(client)
        approve(client, draft_of(client, thread_id))

    assert gone.status_code == 204
    assert world.accounts_store.accounts == {}
    assert world.github.revoked == ["ghu_access1"]
    assert status["connected"] is False
    assert world.forum.tokens == [None]


def test_each_student_sees_only_their_own_account():
    world = connected_world()
    with world.client() as client:
        logged_in(client)
        connect(client)
        client.cookies.clear()
        logged_in(client, email="bob@example.com")
        bobs = client.get("/forum/account").json()
        client.post("/forum/account/disconnect")

    assert bobs["connected"] is False
    assert len(world.accounts_store.accounts) == 1


def test_tokens_never_reach_responses_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    world = connected_world()
    world.forum.refused_tokens.add("ghu_access2")
    seen: list[str] = []
    with world.client() as client:
        logged_in(client)
        seen += [connect(client).text, client.get("/forum/account").text]
        world.clock.advance(hours=8)
        thread_id, events = ask_for_tutor(client)
        draft = draft_of(client, thread_id)
        seen += [approve(client, draft).text, json.dumps(events), json.dumps(draft)]
        seen.append(client.post("/forum/account/disconnect").text)

    everything = "\n".join([*seen, caplog.text])
    for secret in ("ghu_access", "ghr_refresh", "client-secret"):
        assert secret not in everything


def test_rotation_flags_tokens_under_a_lost_key_instead_of_failing():
    clock = Clock()
    lost, kept = make_accounts(clock)[0], make_accounts(clock)[0]
    kept.store = lost.store
    lost_student, kept_student = uuid.uuid4(), uuid.uuid4()
    asyncio.run(lost.connect(lost_student, "good-code"))
    asyncio.run(kept.connect(kept_student, "good-code"))

    result = asyncio.run(rotate_keys(lost.store, kept.cipher))

    assert (result.rotated, result.unreadable) == (1, 1)
    assert lost.store.accounts[lost_student].needs_reconnect is True
    assert lost.store.accounts[kept_student].needs_reconnect is False


def test_a_token_under_a_lost_key_asks_to_reconnect():
    clock = Clock()
    accounts, _ = make_accounts(clock)
    student = uuid.uuid4()
    asyncio.run(accounts.connect(student, "good-code"))
    accounts.cipher = TokenCipher([Fernet.generate_key().decode()])

    with pytest.raises(ReconnectNeeded):
        asyncio.run(accounts.token_for(student))
    assert accounts.store.accounts[student].needs_reconnect is True


def test_the_request_log_never_records_the_callback_code():
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", "/forum/account/callback?code=secret-code&state=s", "1.1", 303),
        None,
    )  # fmt: skip
    other = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", "/threads?x=1", "1.1", 200),
        None,
    )  # fmt: skip

    create_app()
    for handler_filter in logging.getLogger("uvicorn.access").filters:
        handler_filter.filter(record)
        handler_filter.filter(other)

    assert "secret-code" not in record.getMessage()
    assert "/forum/account/callback" in record.getMessage()
    assert "/threads?x=1" in other.getMessage()
