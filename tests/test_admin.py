"""Offline: Admin routes, the fixed-reply routes, and the measuring script."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from owl_mind.core import replies, runtime, telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.intent.taxonomy import Route
from owl_mind.core.model import get_chat_model
from owl_mind.evaluation import measure_calls
from owl_mind.evaluation.measure_calls import measure, memory_services
from owl_mind.services.admin_store import safe_event
from owl_mind.services.study_store import ReviewState
from tests.fake_deepseek import USAGE, FakeDeepSeek
from tests.guard_rails import check_access_logged, check_no_model_calls
from tests.test_auth import logged_in, signup
from tests.test_filesystem import ALICE, BOB, TODAY, say
from tests.test_runtime import World, intent, parse_sse, start, stream, text_of

ADMIN_EMAIL = "owner@example.com"


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("JWT_SECRET", "s" * 32)
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CONFIGURED_ADMINS", ADMIN_EMAIL)
    monkeypatch.setenv("INVITE_CODES", "invite-code-1")
    monkeypatch.setattr(telemetry, "publish", lambda event: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def world() -> World:
    world = World()
    world.app.dependency_overrides.clear()
    return world


# -- who can reach admin data ------------------------------------------------


ADMIN_PATHS = ["/admin/requests", f"/admin/requests/{uuid.uuid4()}", "/admin/model-calls"]


@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_no_login_gets_401_and_student_gets_403(world, path):
    with world.client() as client:
        assert client.get(path).status_code == 401
        logged_in(client)
        assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_student_can_never_cause_an_access_log_row(world, path):
    world.admin.detail = {"id": "x"}
    with world.client() as client:
        logged_in(client)
        client.get(path)

    assert world.admin.views == []


def test_admin_reaches_the_pages(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get("/admin/requests").status_code == 200
        assert client.get("/admin/model-calls").status_code == 200


@pytest.mark.parametrize("query", ["page=0", "size=0", "size=101", "page=x", "q=" + "x" * 321])
def test_bad_list_parameters_are_refused(world, query):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get(f"/admin/requests?{query}").status_code == 422


@pytest.mark.parametrize("days", [0, 401, "x"])
def test_bad_window_is_refused(world, days):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get(f"/admin/model-calls?days={days}").status_code == 422


def test_unknown_request_is_404_and_logs_nothing(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get(f"/admin/requests/{uuid.uuid4()}").status_code == 404

    assert world.admin.views == []


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_admin_pages_are_read_only(world, method, path):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert getattr(client, method)(path).status_code == 405


def test_opening_a_detail_passes_the_viewer_for_the_access_log(world):
    world.admin.detail = {"id": "x"}
    run_id = uuid.uuid4()
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get(f"/admin/requests/{run_id}").status_code == 200
        admin_id = next(iter(world.accounts.accounts))

    assert world.admin.views == [(admin_id, run_id)]


def test_guard_reports_a_view_with_no_log_row():
    assert check_access_logged(3, 4) == []
    assert len(check_access_logged(3, 3)) == 1
    assert len(check_access_logged(3, 5)) == 1


# -- what admin pages can show ----------------------------------------------


def test_todo_text_never_reaches_admin_events():
    secret = "my private question"

    safe = safe_event("todo_delta", {"todos": [{"content": secret, "status": "pending"}]})

    assert safe == {"count": 1}


def test_usage_is_labeled_as_covering_the_agent_only():
    assert safe_event("usage", {"input_tokens": 10, "output_tokens": 5, "model_calls": 1}) == {
        "agent_input_tokens": 10,
        "agent_output_tokens": 5,
        "agent_model_calls": 1,
    }


def test_only_known_event_fields_are_shown():
    assert safe_event("tool_start", {"name": "build_hint", "args": "secret"}) == {
        "name": "build_hint"
    }
    assert safe_event("error", {"message": "secret"}) == {}
    assert safe_event("made_up", {"text": "secret"}) == {}


# -- routes on real runs -----------------------------------------------------


def run_of(world, started):
    return world.runs.runs[uuid.UUID(started["run_id"])]


def as_student(world) -> TestClient:
    world.app.dependency_overrides[runtime_student()] = lambda: ALICE
    return world.client()


def runtime_student():
    from owl_mind.api.deps import current_student

    return current_student


@pytest.mark.parametrize(
    ("message", "route"),
    [
        ("", None),
        ("hi", Route.TEMPLATE),
        ("thanks", Route.TEMPLATE),
        ("show my progress", Route.TEMPLATE),
        ("what's due today?", Route.RULE),
        ("I want to talk to a TA", Route.HANDOFF),
    ],
)
def test_rule_settled_message_records_its_route_and_makes_no_model_call(world, message, route):
    with as_student(world) as client:
        if route is None:
            first = start(client, "hi")
            stream(client, first["run_id"])
            response = client.post(f"/threads/{first['thread_id']}/messages", json={"message": ""})
            started = response.json()
            route = Route.RULE
        else:
            started = start(client, message)
        stream(client, started["run_id"])

    run = run_of(world, started)
    assert (run.route, run.task_kind) == (route.value, None)
    assert check_no_model_calls(world.fake.requests, route.value) == []


def test_guard_reports_a_rule_route_that_calls_a_model():
    assert len(check_no_model_calls([{"model": "x"}], "rule")) == 1


def test_orchestrated_run_records_its_route_and_kind(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    world = World(intent(concept_explain=0.9), say("answer"))
    with world.client() as client:
        started = start(client)
        stream(client, started["run_id"])

    run = run_of(world, started)
    assert (run.route, run.task_kind) == ("orchestrated", "learn")


def test_template_settled_by_the_llm_records_one_intent_call_only():
    world = World(intent(other=0.9))
    with world.client() as client:
        started = start(client, "recommend a movie")
        events = stream(client, started["run_id"])

    assert run_of(world, started).route == "template"
    assert text_of(events).startswith(replies.TEXT[runtime.Reply.OFF_TOPIC])
    assert len(world.fake.requests) == 1


def test_handoff_reply_says_the_forum_is_not_available_yet(world):
    with as_student(world) as client:
        events = stream(client, start(client, "can I speak to a tutor")["run_id"])

    assert text_of(events) == replies.HANDOFF_UNAVAILABLE


def test_progress_reply_uses_the_students_own_schedule(world):
    world.study.reviews[(ALICE, "bfs")] = ReviewState(2.5, 1, 1, TODAY)
    world.study.reviews[(ALICE, "dfs")] = ReviewState(2.5, 1, 1, date_far_ahead())
    world.study.reviews[(BOB, "other")] = ReviewState(2.5, 1, 1, TODAY)
    with as_student(world) as client:
        events = stream(client, start(client, "show my progress")["run_id"])

    assert text_of(events) == replies.progress(2, 1)
    assert world.fake.requests == []


def date_far_ahead():
    return TODAY.replace(year=TODAY.year + 50)


def test_student_with_no_schedule_gets_a_clear_progress_reply():
    assert replies.progress(0, 0) == "You have no topics scheduled for review yet."


def test_signup_helper_still_works(world):
    with world.client() as client:
        assert signup(client).status_code == 201


def test_parse_sse_ignores_comments():
    assert parse_sse(": still working\n\nid: 1-0\nevent: done\ndata: {}\n\n") == [("done", {})]


# -- the measuring script ----------------------------------------------------


def scripted(*replies_: dict):
    fake = FakeDeepSeek(list(replies_))

    def get_model(tier, *, component, max_tokens):
        model = get_chat_model(tier, component=component, max_tokens=max_tokens)
        fake.attach(model)
        return model

    return fake, memory_services(get_model)


def test_counts_match_the_scripted_calls(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    fake, services = scripted(intent(concept_explain=0.9), say("answer"))

    report = asyncio.run(
        measure(services, messages=("hello", "explain it"), cost_of=lambda event: 0.5)
    )

    assert report.messages_measured == 2
    assert report.as_dict()["groups"]["template/none"] == {
        "runs": 1,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hit_tokens": 0,
        "cost_usd": 0.0,
        "calls_with_unknown_usage": 0,
    }
    learn = report.groups["orchestrated/learn"]
    assert learn.runs == 1
    assert learn.model_calls == len(fake.requests) == 2
    assert learn.input_tokens == 2 * USAGE["prompt_tokens"]
    assert learn.output_tokens == 2 * USAGE["completion_tokens"]
    assert learn.cost_usd == 1.0


def test_unknown_cost_is_counted_and_not_treated_as_zero(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    _, services = scripted(intent(concept_explain=0.9), say("answer"))

    report = asyncio.run(measure(services, messages=("explain it",)))

    assert report.groups["orchestrated/learn"].calls_with_unknown_usage == 2


def test_it_stops_at_the_token_limit(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    fake, services = scripted(intent(concept_explain=0.9), say("answer"))

    report = asyncio.run(measure(services, messages=("first", "never sent"), token_limit=1))

    assert report.messages_measured == 1
    assert report.stopped_at_token_limit is True
    assert len(fake.requests) == 2


def test_message_set_holds_no_prompt_examples():
    from owl_mind.core.intent.taxonomy import INTENTS

    examples = {spec.example for spec in INTENTS.values()}
    assert not examples.intersection(measure_calls.MESSAGES)


def test_script_refuses_to_run_while_live_runs_are_off(monkeypatch):
    monkeypatch.delenv("LIVE_AGENT_RUNS_ENABLED", raising=False)
    get_settings.cache_clear()

    with pytest.raises(SystemExit, match="LIVE_AGENT_RUNS_ENABLED is off"):
        measure_calls.main()


def test_publish_is_restored_afterward(monkeypatch):
    original = telemetry.publish
    _, services = scripted()

    asyncio.run(measure(services, messages=("hello",)))

    assert telemetry.publish is original
