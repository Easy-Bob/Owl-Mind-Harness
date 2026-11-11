"""Offline: M4-G's cohorts, account deletion route, alert rules, and the retention job's place."""

import asyncio
import re
import uuid
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

from owl_mind.core import replies
from owl_mind.core.config import get_settings
from owl_mind.services.accounts import Account
from owl_mind.services.cohort import allowed, invite_hash
from tests.test_auth import INVITE, PASSWORD, logged_in, signup
from tests.test_filesystem import say, task
from tests.test_runtime import World, intent, start, stream, text_of

REPO = Path(__file__).resolve().parents[1]
ALERTS = REPO / "ops" / "grafana" / "provisioning" / "alerting" / "owl-alerts.yml"


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "DATABASE_URL": "postgresql://db",
        "REDIS_URL": "redis://cache",
        "DEEPSEEK_API_KEY": "not-a-real-key",
        "JWT_SECRET": "s" * 32,
        "COOKIE_SECURE": "false",
        "INVITE_CODES": f"{INVITE},cohort-b",
        "LIVE_AGENT_RUNS_ENABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# -- cohorts -------------------------------------------------------------------------------------


def account(email="alice@rice.edu", code=None) -> Account:
    return Account(uuid.uuid4(), email, None, invite_hash(code) if code else None)


@pytest.mark.parametrize(
    ("cohort", "who", "expected"),
    [
        ("", account(), True),
        ("alice@rice.edu", account(), True),
        ("ALICE@rice.edu", account(), True),
        ("bob@rice.edu", account(), False),
        ("@rice.edu", account(), True),
        ("@rice.edu", account("alice@example.com"), False),
        ("invite:cohort-b", account(code="cohort-b"), True),
        ("invite:cohort-b", account(code="invite-code-1"), False),
        ("invite:cohort-b", account(), False),
        ("bob@rice.edu, invite:cohort-b", account(code="cohort-b"), True),
        ("@rice.edu", None, False),
    ],
)
def test_who_is_in_a_cohort(cohort, who, expected):
    assert allowed(who, cohort) is expected


def test_the_invite_code_is_kept_only_as_a_hash():
    assert invite_hash("cohort-b") != "cohort-b" and len(invite_hash("cohort-b")) == 64
    assert invite_hash(" cohort-b ") == invite_hash("cohort-b")


def test_signup_keeps_the_invite_codes_hash_on_the_account():
    world = World()
    world.app.dependency_overrides.clear()
    with world.client() as client:
        signup(client, email="carol@example.com", invite="cohort-b")

    [created] = world.accounts.accounts.values()
    assert created.invite_hash == invite_hash("cohort-b")


def test_outside_the_cohort_a_study_question_gets_the_switched_off_reply(monkeypatch):
    monkeypatch.setenv("AGENT_COHORT", "invite:cohort-b")
    get_settings.cache_clear()
    world = World(
        intent(concept_explain=0.9),  # Outside: only the intent step runs.
        intent(concept_explain=0.9),
        task("concept_tutor"),
        say("r"),
        say("Answer."),
    )
    world.accounts.accounts[world.student] = Account(world.student, "a@example.com", None)
    with world.client() as client:
        outside = text_of(stream(client, start(client, "explain heaps")["run_id"]))
        world.accounts.accounts[world.student] = Account(
            world.student, "a@example.com", None, invite_hash("cohort-b")
        )
        inside = text_of(stream(client, start(client, "explain heaps")["run_id"]))

    assert outside == replies.AGENTS_UNAVAILABLE
    assert inside == "Answer."


# -- deleting an account ---------------------------------------------------------------------------


class FakeRetention:
    def __init__(self) -> None:
        self.deleted: list[uuid.UUID] = []

    async def delete_account(self, student_id):
        self.deleted.append(student_id)
        return True


def deletion_world(retention=True) -> tuple[World, FakeRetention]:
    world, fake = World(), FakeRetention()
    if retention:
        world.services = world.services.__class__(**{**vars(world.services), "retention": fake})
    world.app.dependency_overrides.clear()
    return world, fake


def delete(client, password=PASSWORD, confirm="delete my account"):
    return client.post("/auth/account/delete", json={"password": password, "confirm": confirm})


def test_an_account_is_deleted_only_with_its_password_and_the_typed_words():
    world, fake = deletion_world()
    with world.client() as client:
        logged_in(client)
        unconfirmed = delete(client, confirm="yes")
        wrong = delete(client, password="not my password")
        done = delete(client)
        after = client.get("/auth/me")

    assert (unconfirmed.status_code, wrong.status_code, done.status_code) == (400, 401, 204)
    assert len(fake.deleted) == 1
    assert after.status_code == 401  # The session cookie went with it.


def test_an_account_with_an_answer_still_being_written_isnt_deleted():
    world, fake = deletion_world()
    with world.client() as client:
        logged_in(client)
        [created] = world.accounts.accounts.values()
        thread = asyncio.run(world.runs.create_thread(created.id, "t"))
        asyncio.run(world.runs.create_run(thread.id))
        busy = delete(client)

    assert busy.status_code == 409 and fake.deleted == []


def test_without_retention_set_up_deletion_says_it_cant():
    world, _ = deletion_world(retention=False)
    with world.client() as client:
        logged_in(client)
        assert delete(client).status_code == 503


# -- alerts and the job --------------------------------------------------------------------------


def test_every_alert_names_a_runbook_that_exists_and_metrics_that_exist():
    rules = yaml.safe_load(ALERTS.read_text())["groups"][0]["rules"]
    known = {m.name for m in REGISTRY.collect()}

    assert len(rules) == 7
    for rule in rules:
        book = rule["annotations"]["runbook_url"].rsplit("/ops/runbooks/", 1)[1]
        assert (REPO / "ops" / "runbooks" / book).exists(), rule["uid"]
        expr = rule["data"][0]["model"]["expr"]
        for name in re.findall(r"\bowl_[a-z_]+", expr):
            assert re.sub(r"_(bucket|sum|count|total)$", "", name) in known, (rule["uid"], name)
        assert "job=" not in expr and "job!=" not in expr  # job_name; job is Prometheus's.


def test_every_runbook_listed_in_the_index_exists():
    index = (REPO / "ops" / "runbooks" / "README.md").read_text()
    pages = re.findall(r"\]\(([a-z-]+\.md)\)", index)

    assert len(pages) == 12
    assert all((REPO / "ops" / "runbooks" / page).exists() for page in pages)


def test_the_retention_job_runs_nightly_when_set_up():
    from owl_mind.worker.jobs import RETENTION_SECONDS, jobs

    world, _ = deletion_world()

    class Nightly:
        async def run(self):
            return {}

    world.services = world.services.__class__(**{**vars(world.services), "retention": Nightly()})
    found = {name: interval for name, interval, _ in jobs(world.services)}

    assert found["retention"] == RETENTION_SECONDS == 24 * 3600
