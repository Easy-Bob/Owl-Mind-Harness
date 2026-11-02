"""Offline: signup, login and roles through the real routes and cookie handling."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from owl_mind.api import auth
from owl_mind.api.app import create_app
from owl_mind.api.auth import (
    MAX_ACCOUNT_FAILURES,
    MAX_ADDRESS_FAILURES,
    NOTICE,
    check_jwt_secret,
)
from owl_mind.api.deps import SESSION_COOKIE
from owl_mind.core.config import ConfigError, Settings, get_settings
from owl_mind.core.intent.taxonomy import Reply
from owl_mind.core.replies import TEXT
from tests.guard_rails import _schema_routes, check_admin_routes, check_login_required
from tests.test_runtime import World, parse_sse

SECRET = "s" * 32
ALICE_EMAIL = "alice@example.com"
ADMIN_EMAIL = "owner@example.com"
PASSWORD = "correct horse battery"
INVITE = "invite-code-1"
PUBLIC_ROUTES = {
    "/auth/notice",
    "/auth/signup",
    "/auth/login",
    "/auth/logout",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
}


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CONFIGURED_ADMINS", f" {ADMIN_EMAIL.upper()} , other@example.com")
    monkeypatch.setenv("INVITE_CODES", f"{INVITE}, invite-code-2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def world() -> World:
    world = World()
    world.app.dependency_overrides.clear()
    return world


def signup(client, email=ALICE_EMAIL, password=PASSWORD, invite=INVITE, accepted=True, **extra):
    return client.post(
        "/auth/signup",
        json={
            "email": email,
            "password": password,
            "invite_code": invite,
            "accepted_notice": accepted,
            **extra,
        },
    )


def login(client, email=ALICE_EMAIL, password=PASSWORD):
    return client.post("/auth/login", json={"email": email, "password": password})


def logged_in(client, email=ALICE_EMAIL):
    assert signup(client, email=email).status_code == 201
    assert login(client, email=email).status_code == 200


# -- signup -----------------------------------------------------------------


def test_signup_creates_an_account_with_a_hashed_password(world):
    with world.client() as client:
        response = signup(client)

    assert response.status_code == 201
    assert response.json() == {"email": ALICE_EMAIL, "role": "student"}
    account = next(iter(world.accounts.accounts.values()))
    assert account.password_hash.startswith("$argon2")
    assert PASSWORD not in account.password_hash


@pytest.mark.parametrize("invite", ["wrong", "", "invite-code-"])
def test_bad_invite_code_is_refused(world, invite):
    with world.client() as client:
        assert signup(client, invite=invite).status_code == 403

    assert world.accounts.accounts == {}


def test_second_invite_code_works(world):
    with world.client() as client:
        assert signup(client, invite="invite-code-2").status_code == 201


def test_with_no_invite_codes_nobody_can_sign_up(world, monkeypatch):
    monkeypatch.delenv("INVITE_CODES")
    get_settings.cache_clear()
    with world.client() as client:
        assert signup(client, invite="").status_code == 403
        assert signup(client, invite=INVITE).status_code == 403


def test_unaccepted_notice_is_refused(world):
    with world.client() as client:
        assert signup(client, accepted=False).status_code == 400

    assert world.accounts.accounts == {}


def test_notice_says_who_sees_messages_and_where_they_go():
    assert "DeepSeek" in NOTICE
    assert "China" in NOTICE
    assert "admin" in NOTICE


def test_notice_can_be_read_with_no_login(world):
    with world.client() as client:
        assert client.get("/auth/notice").json() == {"notice": NOTICE}


def test_signup_can_never_grant_admin(world):
    with world.client() as client:
        response = signup(client, role="admin", is_admin=True)
        login(client)
        me = client.get("/auth/me").json()
        admin = client.get("/admin/me")

    assert response.json()["role"] == "student"
    assert me["role"] == "student"
    assert admin.status_code == 403


@pytest.mark.parametrize("email", [ALICE_EMAIL, ALICE_EMAIL.upper(), "Alice@Example.com"])
def test_same_email_cannot_sign_up_twice_in_any_case(world, email):
    with world.client() as client:
        assert signup(client).status_code == 201
        assert signup(client, email=email).status_code == 409


@pytest.mark.parametrize(
    "fields",
    [
        {"email": ""},
        {"email": "not-an-email"},
        {"password": ""},
        {"password": "short"},
        {"password": "x" * 129},
    ],
)
def test_bad_signup_fields_are_refused(world, fields):
    with world.client() as client:
        assert signup(client, **fields).status_code == 422


# -- login, cookies and tokens ----------------------------------------------


def test_login_sets_a_protected_cookie_and_logout_clears_it(world):
    with world.client() as client:
        signup(client)
        response = login(client)
        cookie = response.headers["set-cookie"]
        assert client.get("/auth/me").json() == {"email": ALICE_EMAIL, "role": "student"}
        assert client.post("/auth/logout").status_code == 204
        after = client.get("/auth/me")

    assert f"{SESSION_COOKIE}=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Max-Age=14400" in cookie
    assert "Secure" not in cookie
    assert after.status_code == 401


def test_secure_flag_is_on_by_default(world, monkeypatch):
    monkeypatch.delenv("COOKIE_SECURE")
    get_settings.cache_clear()
    with world.client() as client:
        signup(client)
        cookie = login(client).headers["set-cookie"]

    assert "Secure" in cookie


def test_token_holds_no_password_or_email(world):
    with world.client() as client:
        signup(client)
        token = login(client).cookies[SESSION_COOKIE]

    claims = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert set(claims) == {"sub", "iat", "exp"}


def test_wrong_password_and_unknown_email_get_the_same_answer(world):
    with world.client() as client:
        signup(client)
        wrong = login(client, password="wrong password")
        unknown = login(client, email="nobody@example.com")

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_unknown_email_still_checks_a_password_hash(world, monkeypatch):
    calls = []

    class SpyHasher:
        def __init__(self, hasher) -> None:
            self.hasher = hasher

        def hash(self, password):
            return self.hasher.hash(password)

        def verify(self, *args):
            calls.append(args)
            return self.hasher.verify(*args)

    monkeypatch.setattr(auth, "_hasher", SpyHasher(auth._hasher))
    with world.client() as client:
        login(client, email="nobody@example.com")

    assert len(calls) == 1


def test_email_login_ignores_letter_case(world):
    with world.client() as client:
        signup(client)
        assert login(client, email=ALICE_EMAIL.upper()).status_code == 200


def tokens(account_id: str) -> dict[str, str]:
    now = datetime.now(UTC)
    good = {"sub": account_id, "iat": now, "exp": now + timedelta(hours=1)}
    return {
        "expired": jwt.encode(
            {**good, "exp": now - timedelta(seconds=1)}, SECRET, algorithm="HS256"
        ),
        "wrong-secret": jwt.encode(good, "w" * 32, algorithm="HS256"),
        "no-expiry": jwt.encode({"sub": account_id}, SECRET, algorithm="HS256"),
        "no-subject": jwt.encode({"exp": good["exp"]}, SECRET, algorithm="HS256"),
        "bad-subject": jwt.encode({**good, "sub": "not-a-uuid"}, SECRET, algorithm="HS256"),
        "unknown-account": jwt.encode(
            {**good, "sub": str(uuid.uuid4())}, SECRET, algorithm="HS256"
        ),
        "alg-none": jwt.encode(good, "", algorithm="none"),
        "garbage": "not.a.token",
    }


@pytest.mark.parametrize(
    "kind",
    [
        "expired",
        "wrong-secret",
        "no-expiry",
        "no-subject",
        "bad-subject",
        "unknown-account",
        "alg-none",
        "garbage",
    ],
)
def test_bad_token_gets_401(world, kind):
    with world.client() as client:
        signup(client)
        account_id = str(next(iter(world.accounts.accounts)))
        client.cookies.set(SESSION_COOKIE, tokens(account_id)[kind])
        assert client.get("/auth/me").status_code == 401


def test_altered_token_gets_401(world):
    with world.client() as client:
        signup(client)
        token = login(client).cookies[SESSION_COOKIE]
        client.cookies.clear()
        client.cookies.set(SESSION_COOKIE, token[:-2] + ("aa" if token[-2:] != "aa" else "bb"))
        assert client.get("/auth/me").status_code == 401


def test_deleted_account_cannot_use_its_token(world):
    with world.client() as client:
        logged_in(client)
        world.accounts.accounts.clear()
        assert client.get("/auth/me").status_code == 401


# -- roles ------------------------------------------------------------------


def test_configured_email_is_admin_in_any_case_and_with_spaces(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get("/auth/me").json()["role"] == "admin"
        assert client.get("/admin/me").status_code == 200


def test_removing_an_email_takes_effect_at_once(world, monkeypatch):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get("/admin/me").status_code == 200
        monkeypatch.setenv("CONFIGURED_ADMINS", "")
        get_settings.cache_clear()
        assert client.get("/admin/me").status_code == 403


def test_student_gets_403_and_no_login_gets_401_on_admin_routes(world):
    with world.client() as client:
        assert client.get("/admin/me").status_code == 401
        logged_in(client)
        assert client.get("/admin/me").status_code == 403


def test_every_admin_route_refuses_students_on_the_real_app(world):
    with world.client() as client:
        logged_in(client)
        paths = [path for _, path in _schema_routes(client) if path.startswith("/admin")]
        assert paths
        assert check_admin_routes(client) == []


def rogue_app() -> FastAPI:
    app = FastAPI()
    rogue = APIRouter(prefix="/admin")

    @rogue.get("/secrets")
    def secrets():
        return {}

    app.include_router(rogue)
    return app


def test_guard_reports_an_admin_route_with_no_check():
    with TestClient(rogue_app()) as client:
        assert len(check_admin_routes(client)) == 1


def test_every_other_route_needs_a_login_on_the_real_app(world):
    with world.client() as client:
        routes = _schema_routes(client)
        assert len(routes) >= 8
        assert check_login_required(client, PUBLIC_ROUTES) == []


def test_public_list_is_exactly_what_answers_with_no_login(world):
    with world.client() as client:
        open_routes = {
            path
            for method, path in _schema_routes(client)
            if client.request(method, path).status_code != 401
        }

    assert open_routes <= PUBLIC_ROUTES
    assert "/auth/me" not in open_routes


def test_guard_reports_a_route_with_no_login():
    with TestClient(rogue_app()) as client:
        assert len(check_login_required(client, PUBLIC_ROUTES)) == 1


# -- a real chat as a real user ---------------------------------------------


def test_real_login_reaches_chat_and_users_stay_apart(world):
    with world.client() as alice:
        logged_in(alice)
        started = alice.post("/threads", json={"message": "hi"}).json()
        events = parse_sse(alice.get(f"/runs/{started['run_id']}/stream").text)
        thread, run = started["thread_id"], started["run_id"]

        alice.cookies.clear()
        logged_in(alice, email="bob@example.com")
        responses = [
            alice.get(f"/threads/{thread}/messages"),
            alice.post(f"/threads/{thread}/messages", json={"message": "hi"}),
            alice.get(f"/runs/{run}/stream"),
        ]
        listed = alice.get("/threads").json()

    assert "".join(d["text"] for k, d in events if k == "text_delta") == TEXT[Reply.GREETING]
    assert {response.status_code for response in responses} == {404}
    assert listed == []


def test_chat_routes_refuse_with_no_login(world):
    some_id = uuid.uuid4()
    with world.client() as client:
        responses = [
            client.post("/threads", json={"message": "hi"}),
            client.get("/threads"),
            client.get(f"/threads/{some_id}/messages"),
            client.get(f"/runs/{some_id}/stream"),
        ]

    assert {response.status_code for response in responses} == {401}


# -- failed-login limits ----------------------------------------------------


def test_too_many_failures_block_the_account_even_with_the_right_password(world):
    with world.client() as client:
        signup(client)
        for _ in range(MAX_ACCOUNT_FAILURES):
            assert login(client, password="wrong password").status_code == 401
        assert login(client).status_code == 429


def test_success_clears_the_account_count(world):
    with world.client() as client:
        signup(client)
        for _ in range(MAX_ACCOUNT_FAILURES - 1):
            login(client, password="wrong password")
        assert login(client).status_code == 200
        assert login(client, password="wrong password").status_code == 401
        assert login(client).status_code == 200


def test_too_many_failures_across_accounts_block_the_address(world):
    with world.client() as client:
        for number in range(MAX_ADDRESS_FAILURES):
            login(client, email=f"guess{number}@example.com")
        assert login(client, email="another@example.com").status_code == 429


def test_guessing_invite_codes_blocks_the_address(world):
    with world.client() as client:
        for _ in range(MAX_ADDRESS_FAILURES):
            assert signup(client, invite="guess").status_code == 403
        assert signup(client).status_code == 429


def test_limiter_keys_hold_no_email_address(world):
    with world.client() as client:
        login(client)

    assert world.limiter.counts
    assert all(ALICE_EMAIL not in key for key in world.limiter.counts)


def test_limiter_outage_does_not_lock_people_out(world):
    async def broken(*args):
        raise ConnectionError("redis is down")

    with world.client() as client:
        signup(client)
        world.limiter.failures = broken
        world.limiter.record_failure = broken
        world.limiter.clear = broken
        assert login(client, password="wrong password").status_code == 401
        assert login(client).status_code == 200


# -- secrets stay out of errors and logs -------------------------------------


def test_validation_errors_do_not_echo_the_input(world):
    with world.client() as client:
        response = signup(client, password="short-pw")
        long_invite = signup(client, invite="i" * 201)

    assert response.status_code == 422
    assert "short-pw" not in response.text
    assert "iiii" not in long_invite.text
    assert response.json()["detail"][0]["field"] == ["body", "password"]


def test_secrets_stay_out_of_logs(world, caplog):
    caplog.set_level(logging.DEBUG)
    with world.client() as client:
        signup(client)
        signup(client, invite="guess-code-value")
        token = login(client).cookies[SESSION_COOKIE]
        login(client, password="wrong-password-value")

    for secret in (PASSWORD, "guess-code-value", "wrong-password-value", INVITE, token):
        assert secret not in caplog.text


def test_settings_do_not_print_their_secrets(monkeypatch):
    assert SECRET not in repr(get_settings())
    assert INVITE not in repr(get_settings())


# -- cross-site posts -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "content_type"),
    [
        ("/threads", "message=hi", "application/x-www-form-urlencoded"),
        ("/threads", '{"message": "hi"}', "text/plain"),
        ("/auth/login", "email=a@example.com&password=x", "application/x-www-form-urlencoded"),
    ],
)
def test_state_changing_routes_refuse_form_posts(world, path, body, content_type):
    with world.client() as client:
        logged_in(client)
        response = client.post(path, content=body, headers={"content-type": content_type})

    assert response.status_code == 422


# -- startup ----------------------------------------------------------------


@pytest.mark.parametrize("secret", [None, "short"])
def test_missing_or_short_jwt_secret_refuses_to_start(secret):
    settings = Settings(database_url="x", redis_url="y", jwt_secret=secret, _env_file=None)

    with pytest.raises(ConfigError, match="JWT_SECRET"):
        check_jwt_secret(settings)


def test_production_app_refuses_to_start_with_no_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET")
    get_settings.cache_clear()

    with pytest.raises(ConfigError, match="JWT_SECRET"), TestClient(create_app()):
        pass


def test_real_secret_is_accepted():
    check_jwt_secret(get_settings())
