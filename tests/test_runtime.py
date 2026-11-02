"""Offline: whole turns through the real routes and agent loop, with the network replaced."""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from owl_mind.api import routes
from owl_mind.api.app import create_app
from owl_mind.api.deps import current_student
from owl_mind.core import replies, runtime, telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.events import InMemoryEventBus
from owl_mind.core.intent import classifier
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.intent.taxonomy import TaskKind
from owl_mind.core.model import get_chat_model
from owl_mind.core.runtime import RUN_LIMITS, RunLimit, RunRequest, Services, execute_run
from owl_mind.core.streaming import FLUSH_CHARS, StreamTranslator
from owl_mind.services.run_store import InMemoryRunStore
from owl_mind.services.study_store import InMemoryStudyStore, ReviewState
from tests.fake_deepseek import FakeDeepSeek
from tests.guard_rails import check_activity_event_types
from tests.test_filesystem import ALICE, BOB, TODAY, call, say, task


def intent(**confidences: float) -> dict:
    items = [{"intent": k, "confidence": v} for k, v in confidences.items()]
    return say(json.dumps({"intents": items}))


class World:
    """An app with in-memory services, a scripted model and a chosen logged-in student."""

    def __init__(self, *replies_: dict, status: int = 200) -> None:
        self.fake = FakeDeepSeek(list(replies_), status=status)
        self.runs = InMemoryRunStore()
        self.study = InMemoryStudyStore()
        self.bus = InMemoryEventBus()
        self.saver = InMemorySaver()
        self.student = ALICE

        def get_model(tier, *, component, max_tokens):
            model = get_chat_model(tier, component=component, max_tokens=max_tokens)
            self.fake.attach(model)
            return model

        self.services = Services(
            runs=self.runs,
            study=self.study,
            events=self.bus,
            store=InMemoryStore(),
            checkpointer=self.saver,
            intent=IntentStep(lambda tier, c, m: get_model(tier, component=c, max_tokens=m)),
            get_model=get_model,
        )

        @asynccontextmanager
        async def factory():
            yield self.services

        self.app = create_app(factory)
        self.app.dependency_overrides[current_student] = lambda: self.student

    def client(self) -> TestClient:
        return TestClient(self.app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
        )
        if fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def stream(client: TestClient, run_id: str, last_event_id: str | None = None):
    headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
    response = client.get(f"/runs/{run_id}/stream", headers=headers)
    assert response.status_code == 200
    return parse_sse(response.text)


def text_of(events) -> str:
    return "".join(data["text"] for kind, data in events if kind == "text_delta")


def kinds(events) -> list[str]:
    return [kind for kind, _ in events if kind != "text_delta"]


def start(client: TestClient, message: str = "help me") -> dict:
    response = client.post("/threads", json={"message": message})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def telemetry_events(monkeypatch):
    published = []
    monkeypatch.setattr(telemetry, "publish", published.append)
    return published


# -- a whole agent turn -----------------------------------------------------


def tutor_world() -> World:
    return World(
        intent(problem_help=0.9),
        task("practice_coach"),
        call("build_hint", level=1),
        say("tutor report text"),
        say("Here is a final answer that is long enough to arrive in several pieces."),
    )


def test_agent_turn_streams_events_in_order():
    world = tutor_world()
    with world.client() as client:
        started = start(client)
        events = stream(client, started["run_id"])

    assert kinds(events) == [
        "queued",
        "subagent_start",
        "tool_start",
        "tool_end",
        "subagent_end",
        "usage",
        "done",
    ]
    assert events[-1] == ("done", {"status": "completed"})
    assert world.runs.runs[uuid.UUID(started["run_id"])].status == "completed"


def test_answer_text_arrives_in_pieces_before_the_run_ends():
    world = tutor_world()
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    deltas = [data["text"] for kind, data in events if kind == "text_delta"]
    assert len(deltas) > 1
    assert all(len(piece) >= FLUSH_CHARS for piece in deltas[:-1])
    assert "".join(deltas).startswith("Here is a final answer")


def test_tutor_text_and_tool_arguments_do_not_reach_the_stream():
    world = World(
        intent(code_review=0.9),
        task("practice_coach"),
        call("analyze_complexity", code="private-student-code"),
        say("tutor report text"),
        say("final"),
    )
    with world.client() as client:
        raw = client.get(f"/runs/{start(client)['run_id']}/stream").text

    assert "tutor report text" not in raw
    assert "private-student-code" not in raw


def test_usage_event_counts_every_model_call_in_the_agent():
    world = tutor_world()
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    usage = next(data for kind, data in events if kind == "usage")
    assert usage["model_calls"] == 4
    assert usage["input_tokens"] == 40


def test_every_model_call_is_attributed_to_the_run(telemetry_events):
    world = tutor_world()
    with world.client() as client:
        started = start(client)
        stream(client, started["run_id"])

    calls = [event for event in telemetry_events if event.get("kind") != "activity"]
    assert len(calls) == len(world.fake.requests)
    assert {event["run_id"] for event in calls} == {started["run_id"]}
    assert {event["prompt_version"] for event in calls} == {runtime.PROMPT_VERSION}
    assert {event["component"] for event in calls} == {
        "intent",
        "orchestrator",
        "agent:practice_coach",
    }


def test_composite_turn_sends_todo_events():
    world = World(
        intent(concept_explain=0.9, quiz_request=0.8),
        call("write_todos", todos=[{"content": "explain", "status": "pending"}]),
        say("final"),
    )
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    todos = [data for kind, data in events if kind == "todo_delta"]
    assert todos[0]["todos"][0]["content"] == "explain"


def test_left_out_intents_are_named_after_the_answer():
    world = World(
        intent(concept_explain=0.9, quiz_request=0.85, study_plan=0.8),
        say("final"),
    )
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    assert "study plan" in text_of(events)


def test_agent_turn_has_real_tool_access_to_the_student_records():
    world = World(
        intent(study_plan=0.9),
        task("study_planner"),
        call("schedule_review", topic="bfs", quality=5),
        say("report"),
        say("final"),
    )
    with world.client() as client:
        stream(client, start(client)["run_id"])

    assert (ALICE, "bfs") in world.study.reviews


# -- routes with no agent ---------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hi", replies.TEXT[runtime.Reply.GREETING]),
        ("thanks", replies.TEXT[runtime.Reply.THANKS]),
        ("show my progress", replies.TEXT[runtime.Reply.PROGRESS]),
        ("I want to talk to a TA", replies.HANDOFF_UNAVAILABLE),
    ],
)
def test_rule_settled_message_streams_a_reply_with_no_model_call(message, expected):
    world = World()
    with world.client() as client:
        events = stream(client, start(client, message)["run_id"])

    assert text_of(events) == expected
    assert world.fake.requests == []
    assert events[-1] == ("done", {"status": "completed"})


def test_due_topics_rule_reads_the_students_schedule():
    world = World()
    world.study.reviews[(ALICE, "bfs")] = ReviewState(2.5, 1, 1, TODAY)
    world.study.reviews[(BOB, "other")] = ReviewState(2.5, 1, 1, TODAY)
    with world.client() as client:
        events = stream(client, start(client, "what's due today?")["run_id"])

    assert "bfs" in text_of(events)
    assert "other" not in text_of(events)
    assert world.fake.requests == []


def test_low_confidence_gets_a_clarifying_question_and_builds_no_agent():
    world = World(intent(concept_explain=0.4))
    with world.client() as client:
        events = stream(client, start(client, "trees")["run_id"])

    assert text_of(events) == replies.TEXT[runtime.Reply.CLARIFY]
    assert len(world.fake.requests) == 1


def test_agents_stay_off_while_live_runs_are_disabled(monkeypatch):
    monkeypatch.delenv("LIVE_AGENT_RUNS_ENABLED")
    get_settings.cache_clear()
    world = World(intent(problem_help=0.9))
    with world.client() as client:
        events = stream(client, start(client)["run_id"])

    assert text_of(events) == replies.AGENTS_UNAVAILABLE
    assert events[-1] == ("done", {"status": "completed"})
    # Only the single capped intent call, and no agent call.
    assert len(world.fake.requests) == 1


# -- input limits -----------------------------------------------------------


@pytest.mark.parametrize("message", ["", "   \n"])
def test_blank_first_message_starts_no_thread(message):
    world = World()
    with world.client() as client:
        assert client.post("/threads", json={"message": message}).status_code == 422
        assert client.get("/threads").json() == []


def test_oversized_message_is_refused_before_any_model_call():
    world = World()
    with world.client() as client:
        assert client.post("/threads", json={"message": "x" * 4001}).status_code == 422

    assert world.fake.requests == []


def test_message_at_the_limit_is_accepted():
    world = World()
    with world.client() as client:
        world.fake.replies.append(intent(concept_explain=0.4))
        assert client.post("/threads", json={"message": "x" * 4000}).status_code == 201


def test_blank_message_in_a_thread_gets_a_rule_reply_with_no_model_call():
    world = World()
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        run = client.post(f"/threads/{started['thread_id']}/messages", json={"message": " "})
        events = stream(client, run.json()["run_id"])

    assert text_of(events) == replies.TEXT[runtime.Reply.EMPTY]
    assert world.fake.requests == []


# -- threads and history ----------------------------------------------------


def test_title_is_cut_from_the_first_message_with_no_model_call():
    world = World()
    with world.client() as client:
        stream(client, start(client, "hi")["run_id"])
        threads = client.get("/threads").json()

    assert threads[0]["title"] == "hi"
    assert world.fake.requests == []


def test_long_title_is_cut():
    world = World(intent(concept_explain=0.4))
    with world.client() as client:
        stream(client, start(client, "word " * 50)["run_id"])
        title = client.get("/threads").json()[0]["title"]

    assert len(title) == 60


def test_threads_are_listed_newest_first():
    world = World()
    with world.client() as client:
        first = start(client, "hi")
        stream(client, first["run_id"])
        second = start(client, "hello")
        stream(client, second["run_id"])
        listed = [thread["id"] for thread in client.get("/threads").json()]

    assert listed == [second["thread_id"], first["thread_id"]]


def test_history_holds_template_and_agent_turns_in_one_thread():
    world = World(intent(concept_explain=0.9), say("agent answer"))
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        run = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "x"})
        stream(client, run.json()["run_id"])
        history = client.get(f"/threads/{started['thread_id']}/messages").json()

    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[0]["content"] == "hi"
    assert history[3]["content"] == "agent answer"
    # The agent received the earlier template turn as part of the conversation.
    assert "hi" in json.dumps(world.fake.requests[-1]["messages"])


def test_template_turn_after_an_agent_turn_keeps_the_history():
    world = World(intent(concept_explain=0.9), say("agent answer"))
    with world.client() as client:
        started = start(client, "explain bfs")
        stream(client, started["run_id"])
        run = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "thanks"})
        stream(client, run.json()["run_id"])
        history = client.get(f"/threads/{started['thread_id']}/messages").json()

    assert [m["content"] for m in history][:2] == ["explain bfs", "agent answer"]
    assert history[2]["content"] == "thanks"
    assert len(history) == 4


def test_history_leaves_out_tool_steps():
    world = tutor_world()
    with world.client() as client:
        started = start(client)
        stream(client, started["run_id"])
        history = client.get(f"/threads/{started['thread_id']}/messages").json()

    assert [m["role"] for m in history] == ["user", "assistant"]


# -- belonging to one student -----------------------------------------------


def test_no_login_means_every_route_refuses():
    world = World()
    del world.app.dependency_overrides[current_student]
    some_id = uuid.uuid4()
    with world.client() as client:
        responses = [
            client.post("/threads", json={"message": "hi"}),
            client.get("/threads"),
            client.post(f"/threads/{some_id}/messages", json={"message": "hi"}),
            client.get(f"/threads/{some_id}/messages"),
            client.get(f"/runs/{some_id}/stream"),
        ]

    assert {response.status_code for response in responses} == {401}


def test_another_students_thread_and_run_look_missing():
    world = World()
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        world.student = BOB
        thread, run = started["thread_id"], started["run_id"]
        responses = [
            client.get(f"/threads/{thread}/messages"),
            client.post(f"/threads/{thread}/messages", json={"message": "hi"}),
            client.get(f"/runs/{run}/stream"),
        ]
        listed = client.get("/threads").json()

    assert {response.status_code for response in responses} == {404}
    assert listed == []


# -- reconnecting -----------------------------------------------------------


def test_reconnect_gets_only_what_it_missed():
    world = tutor_world()
    with world.client() as client:
        run_id = start(client)["run_id"]
        everything = stream(client, run_id)
        rest = stream(client, run_id, last_event_id="3-0")

    assert rest == everything[3:]


def test_finished_run_with_a_lost_stream_still_ends_the_response():
    world = World()
    with world.client() as client:
        run_id = start(client, "hi")["run_id"]
        stream(client, run_id)
        world.bus.events.clear()
        events = stream(client, run_id)

    assert events == [("done", {"status": "completed"})]


# -- limits -----------------------------------------------------------------


def test_every_task_kind_has_a_limit():
    assert set(RUN_LIMITS) == set(TaskKind)
    assert RUN_LIMITS[TaskKind.SUPPORT] == RunLimit(5, 30)


def test_tool_limit_stops_the_run_and_says_so(monkeypatch):
    monkeypatch.setitem(RUN_LIMITS, TaskKind.PRACTICE, RunLimit(1, 30))
    world = World(
        intent(problem_help=0.9),
        task("practice_coach"),
        call("build_hint", level=1),
        call("build_hint", level=2),
        say("never reached"),
        say("never reached"),
    )
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    assert events[-1] == ("done", {"status": "stopped", "reason": "tool_limit"})
    assert world.runs.runs[uuid.UUID(run_id)].status == "stopped"
    assert replies.RUN_STOPPED in text_of(events)
    assert "usage" in kinds(events)


def test_time_limit_stops_the_run_and_says_so(monkeypatch):
    monkeypatch.setitem(RUN_LIMITS, TaskKind.LEARN, RunLimit(20, 0))
    world = World(intent(concept_explain=0.9), say("never reached"))
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    assert events[-1] == ("done", {"status": "stopped", "reason": "time_limit"})
    assert world.runs.runs[uuid.UUID(run_id)].status == "stopped"


def test_thread_can_continue_after_a_stopped_run(monkeypatch):
    monkeypatch.setitem(RUN_LIMITS, TaskKind.PRACTICE, RunLimit(0, 30))
    world = World(
        intent(problem_help=0.9),
        task("practice_coach"),
        intent(concept_explain=0.9),
        say("recovered"),
    )
    with world.client() as client:
        started = start(client)
        stream(client, started["run_id"])
        next_run = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "x"})
        events = stream(client, next_run.json()["run_id"])

    assert text_of(events) == "recovered"
    assert events[-1] == ("done", {"status": "completed"})


# -- one run at a time, failures and restarts --------------------------------


def test_second_message_on_a_busy_thread_gets_409():
    world = World()

    async def scenario():
        thread = await world.runs.create_thread(ALICE, "t")
        await world.runs.create_run(thread.id)
        return thread

    with world.client() as client:
        thread = client.portal.call(scenario)
        response = client.post(f"/threads/{thread.id}/messages", json={"message": "hi"})

    assert response.status_code == 409


def test_unexpected_error_fails_the_run_without_leaking_details():
    world = World()

    async def broken(*args, **kwargs):
        raise RuntimeError("private-detail-from-student-input")

    world.services.study.due_topics = broken
    with world.client() as client:
        run_id = start(client, "what's due today?")["run_id"]
        response = client.get(f"/runs/{run_id}/stream")

    events = parse_sse(response.text)
    assert events[-1] == ("error", {"message": replies.RUN_FAILED})
    assert world.runs.runs[uuid.UUID(run_id)].status == "failed"
    assert "private-detail" not in response.text


def test_failed_model_call_inside_the_agent_fails_the_run():
    world = World(intent(problem_help=0.9), say("the agent call fails before using this"))
    fake_call = world.fake.__call__
    calls = []

    def fail_after_intent(request):
        calls.append(request)
        response = fake_call(request)
        return response if len(calls) == 1 else httpx.Response(400, json={"error": {}})

    world.fake.handler = fail_after_intent
    with world.client() as client:
        run_id = start(client)["run_id"]
        events = stream(client, run_id)

    assert len(calls) == 2
    assert events[-1] == ("error", {"message": replies.RUN_FAILED})
    assert world.runs.runs[uuid.UUID(run_id)].status == "failed"
    assert "usage" in kinds(events)


def test_stuck_intent_call_gets_a_clarifying_question(monkeypatch):
    monkeypatch.setattr(classifier, "_TIMEOUT_SECONDS", 0.05)
    world = World()

    async def stuck(request):
        await asyncio.sleep(60)

    world.fake.handler = stuck
    with world.client() as client:
        events = stream(client, start(client, "trees")["run_id"])

    assert text_of(events) == replies.TEXT[runtime.Reply.CLARIFY]
    assert events[-1] == ("done", {"status": "completed"})


@pytest.mark.parametrize("bad", ["abc", "1", "1-2-3", "$", "1-x"])
def test_bad_last_event_id_is_refused(bad):
    world = World()
    with world.client() as client:
        run_id = start(client, "hi")["run_id"]
        response = client.get(f"/runs/{run_id}/stream", headers={"Last-Event-ID": bad})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("message", "kind"),
    [("hi", "done"), ("what's due today?", "error")],
    ids=["completed", "failed"],
)
def test_run_status_is_final_before_the_last_event_is_sent(message, kind):
    world = World()
    seen = []
    publish = world.bus.publish

    async def spy(run_id, event_kind, data):
        if event_kind == kind:
            seen.append(world.runs.runs[run_id].status)
        await publish(run_id, event_kind, data)

    async def broken(*args, **kwargs):
        raise RuntimeError("boom")

    world.bus.publish = spy
    world.services.study.due_topics = broken
    with world.client() as client:
        stream(client, start(client, message)["run_id"])

    assert seen and seen[0] != "running"


def test_next_message_right_after_the_last_event_is_accepted():
    world = World()
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        response = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "hi"})

    assert response.status_code == 201


def test_stream_survives_live_events_failing_and_ends_with_the_run_status(monkeypatch):
    world = World()
    monkeypatch.setattr(routes, "_RETRY_SECONDS", 0)

    async def broken_read(*args):
        raise ConnectionError("redis is down")

    world.bus.read = broken_read
    with world.client() as client:
        events = stream(client, start(client, "hi")["run_id"])

    assert events[-1] == ("done", {"status": "completed"})


def test_startup_marks_unfinished_runs_failed_and_history_stays():
    world = World()

    async def seed():
        thread = await world.runs.create_thread(ALICE, "t")
        return await world.runs.create_run(thread.id)

    run_id = asyncio.run(seed())

    with world.client() as client:
        assert world.runs.runs[run_id].status == "failed"
        assert client.get("/threads").json()[0]["title"] == "t"


def test_live_event_failure_does_not_fail_the_run():
    world = World()

    async def broken_publish(*args):
        raise ConnectionError("redis is down")

    world.bus.publish = broken_publish
    with world.client() as client:
        run_id = start(client, "hi")["run_id"]
        for _ in range(200):
            if world.runs.runs[uuid.UUID(run_id)].status != "queued":
                client.portal.call(asyncio.sleep, 0.01)
            if world.runs.runs[uuid.UUID(run_id)].status == "completed":
                break

    assert world.runs.runs[uuid.UUID(run_id)].status == "completed"


def test_shutdown_during_a_run_marks_it_failed():
    world = World()
    request_holder = {}

    async def slow_classify(message, config=None):
        await asyncio.sleep(60)

    world.services.intent.classify = slow_classify
    with world.client() as client:
        request_holder["run"] = start(client, "x")["run_id"]
        client.portal.call(asyncio.sleep, 0.01)

    assert world.runs.runs[uuid.UUID(request_holder["run"])].status == "failed"


# -- run events -------------------------------------------------------------


def test_run_events_record_what_happened_not_what_was_said():
    world = tutor_world()
    with world.client() as client:
        run_id = uuid.UUID(start(client)["run_id"])
        stream(client, run_id)

    recorded = [kind for (run, _), (kind, _) in world.runs.events.items() if run == run_id]
    assert "subagent_start" in recorded
    assert "usage" in recorded
    assert "text_delta" not in recorded


def test_event_record_failure_does_not_fail_the_run():
    world = World()

    async def broken_record(*args):
        raise ConnectionError("database is down")

    world.runs.record_event = broken_record
    with world.client() as client:
        events = stream(client, start(client, "hi")["run_id"])

    assert events[-1] == ("done", {"status": "completed"})


def test_repeated_event_key_overwrites():
    store = InMemoryRunStore()
    run_id = uuid.uuid4()

    async def scenario():
        await store.record_event(run_id, "k", "usage", {"n": 1})
        await store.record_event(run_id, "k", "usage", {"n": 2})

    asyncio.run(scenario())
    assert store.events == {(run_id, "k"): ("usage", {"n": 2})}


# -- activity ---------------------------------------------------------------


def activity(events) -> list[str]:
    return [event["event_type"] for event in events if event.get("kind") == "activity"]


def test_new_chat_records_both_activity_events_once(telemetry_events):
    world = World()
    with world.client() as client:
        stream(client, start(client, "hi")["run_id"])

    assert activity(telemetry_events) == ["session_started", "message_sent"]


def test_later_message_records_only_message_sent(telemetry_events):
    world = World()
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        telemetry_events.clear()
        run = client.post(f"/threads/{started['thread_id']}/messages", json={"message": "thanks"})
        stream(client, run.json()["run_id"])

    assert activity(telemetry_events) == ["message_sent"]


def test_viewing_history_and_reconnecting_record_nothing(telemetry_events):
    world = World()
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        telemetry_events.clear()
        client.get(f"/threads/{started['thread_id']}/messages")
        client.get("/threads")
        stream(client, started["run_id"])

    assert activity(telemetry_events) == []


def test_activity_events_hold_no_message_text(telemetry_events):
    world = World()
    with world.client() as client:
        stream(client, start(client, "hi")["run_id"])

    found = [event for event in telemetry_events if event.get("kind") == "activity"]
    assert set(found[0]) == {
        "kind",
        "event_id",
        "event_type",
        "student_id",
        "thread_id",
        "occurred_at",
    }


def test_activity_failure_does_not_fail_the_request(monkeypatch):
    def broken(event):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(telemetry, "publish", broken)
    world = World()
    with world.client() as client:
        events = stream(client, start(client, "hi")["run_id"])

    assert events[-1] == ("done", {"status": "completed"})


def test_unknown_activity_type_is_refused():
    with pytest.raises(ValueError, match="made_up"):
        telemetry.record_activity("made_up", ALICE, ALICE)


def test_guard_accepts_the_two_allowed_activity_types():
    assert check_activity_event_types(telemetry.ACTIVITY_EVENT_TYPES) == []


def test_guard_reports_an_extra_activity_type():
    extra = {*telemetry.ACTIVITY_EVENT_TYPES, "page_viewed"}

    assert len(check_activity_event_types(extra)) == 1


# -- the stream translator on its own ---------------------------------------


def test_translator_ignores_unknown_modes_and_empty_updates():
    translator = StreamTranslator()

    assert translator.translate((), "custom", {}) == []
    assert translator.translate((), "updates", {"middleware": None}) == []
    assert translator.flush() == []


def test_direct_execute_run_marks_status():
    world = World()

    async def scenario():
        thread = await world.runs.create_thread(ALICE, "t")
        run_id = await world.runs.create_run(thread.id)
        await execute_run(world.services, RunRequest(run_id, thread.id, ALICE, "hi"))
        return run_id

    assert world.runs.runs[asyncio.run(scenario())].status == "completed"
