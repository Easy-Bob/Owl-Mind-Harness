"""Offline: the M4-D experiments' code paths, each behind a setting that is off by default."""

import asyncio

import pytest

from owl_mind.core import prompts, quiz_reuse
from owl_mind.core.config import get_settings
from owl_mind.core.quiz_reuse import quiz_key
from owl_mind.evaluation.intent_tuning import OFF_TOPIC, RULES, analyze, keyword_intent
from tests.test_filesystem import call, say, task
from tests.test_integrity import QUIZ_QUESTION, integrity, run
from tests.test_runtime import World, intent, text_of
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to all.


@pytest.fixture
def setting():
    changed = {}

    def set_(key, value):
        settings = get_settings()
        changed.setdefault(key, getattr(settings, key))
        setattr(settings, key, value)

    yield set_
    for key, value in changed.items():
        setattr(get_settings(), key, value)


def system_prompt(request: dict) -> str:
    content = request["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


def test_the_kept_savings_are_on_in_the_service_and_the_rejected_ones_off(monkeypatch):
    monkeypatch.delenv("DIRECT_SINGLE_TUTOR")
    monkeypatch.delenv("QUIZ_REUSE")
    get_settings.cache_clear()
    settings = get_settings()

    assert (settings.direct_single_tutor, settings.quiz_reuse) == (True, True)
    assert settings.force_delegation is False
    assert settings.composite_answer == "join" and settings.large_model == "deepseek-flash"
    get_settings.cache_clear()


# -- forced delegation -------------------------------------------------------------------------


def test_forced_delegation_makes_the_first_orchestrator_call_call_a_tutor(setting):
    setting("force_delegation", True)
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("report"), say("Heaps."))
    events, _ = run(world, "explain heaps")

    orchestrator = [r for r in world.fake.agent_requests if "tools" in r][::2]
    first, last = world.fake.agent_requests[1], world.fake.agent_requests[-1]
    assert first["tool_choice"] == {"type": "function", "function": {"name": "task"}}
    assert "tool_choice" not in last or last["tool_choice"] in (None, "auto")
    assert orchestrator and text_of(events) == "Heaps."


def test_without_the_setting_the_orchestrator_chooses():
    world = World(intent(concept_explain=0.9), task("concept_tutor"), say("report"), say("Heaps."))
    run(world, "explain heaps")

    assert "tool_choice" not in world.fake.agent_requests[1]


# -- the direct route ---------------------------------------------------------------------------


def test_a_direct_learn_turn_asks_only_the_tutor_and_checks_its_answer(setting):
    setting("direct_single_tutor", True)
    world = World(intent(concept_explain=0.9), say("A heap keeps the smallest item on top."))
    events, saved = run(world, "explain heaps")

    [_, tutor] = world.fake.agent_requests
    assert system_prompt(tutor).startswith(prompts.CONCEPT_TUTOR[:40])
    assert text_of(events) == saved == "A heap keeps the smallest item on top."
    # No orchestrator received it first, so the answer is reviewed before it's shown.
    assert integrity(events)["layer"] == "reviewer"
    assert world.fake.auto_reviews == 1


def test_a_direct_quiz_turn_shows_the_posted_quiz_with_no_closing_call(setting):
    setting("direct_single_tutor", True)
    world = World(
        intent(quiz_request=0.9), call("post_quiz", topic="heaps", questions=[QUIZ_QUESTION])
    )
    events, saved = run(world, "quiz me on heaps")

    assert text_of(events).startswith("**Quiz: heaps**")
    assert saved == text_of(events)
    assert len(world.fake.agent_requests) == 2  # Intent, then the quiz master once.
    assert integrity(events)["layer"] == "quiz"


def test_a_two_tutor_turn_still_goes_through_the_orchestrator(setting):
    setting("direct_single_tutor", True)
    world = World(
        intent(concept_explain=0.9, quiz_request=0.8),
        task("concept_tutor"),
        say("Heaps keep the smallest item on top."),
        task("quiz_master"),
        call("post_quiz", topic="heaps", questions=[QUIZ_QUESTION]),
        say("Posted."),
    )
    events, _ = run(world, "explain heaps, then quiz me")

    assert system_prompt(world.fake.agent_requests[1]).startswith(prompts.ORCHESTRATOR[:40])
    assert "**Quiz: heaps**" in text_of(events)


# -- quiz reuse ---------------------------------------------------------------------------------


class Notes:
    """Course search with only a version, which is all reuse reads."""

    def __init__(self, version: int = 3) -> None:
        self.version = version

    async def corpus_version(self) -> int:
        return self.version


@pytest.mark.parametrize(
    ("message", "key"),
    [
        ("quiz me on heaps", "heaps"),
        ("Quiz me on consistent hashing.", "consistent hashing"),
        ("Give me a short quiz on hash tables.", "hash tables"),
        ("can you test me about the Bloom filters?", "bloom filters"),
        ("quiz me on it", None),
        ("explain heaps, then quiz me on them", None),
        ("quiz me on heaps and explain them first", "heaps and explain them first"),
    ],
)
def test_only_a_plain_quiz_request_has_a_reuse_key(message, key):
    assert quiz_key(message) == key


def quiz_world(*replies) -> World:
    world = World(*replies)
    world.services = world.services.__class__(**{**vars(world.services), "search": Notes()})
    return world


def test_a_second_student_asking_the_same_quiz_gets_it_with_no_agent_run(setting):
    setting("quiz_reuse", True)
    world = quiz_world(
        intent(quiz_request=0.9),
        task("quiz_master"),
        call("post_quiz", topic="heaps", questions=[QUIZ_QUESTION]),
        say("Posted."),
        intent(quiz_request=0.9),
    )
    first, _ = run(world, "quiz me on heaps")
    asked = len(world.fake.agent_requests)
    world.student = __import__("uuid").uuid4()
    second, saved = run(world, "Quiz me on heaps!")

    assert text_of(second) == text_of(first) == saved
    assert len(world.fake.agent_requests) == asked + 1  # Only the intent step.
    assert ("answer", {"mode": "reused_quiz", "parts": 1}) in second


def test_new_notes_or_another_topic_make_a_new_quiz(setting):
    setting("quiz_reuse", True)
    world = quiz_world()
    store = world.services.store

    async def unseen(_quiz):
        return False

    async def body():
        await quiz_reuse.keep(store, 3, "heaps", "**Quiz: heaps**")
        return (
            await quiz_reuse.find(store, 3, "heaps", unseen),
            await quiz_reuse.find(store, 4, "heaps", unseen),
            await quiz_reuse.find(store, 3, "tries", unseen),
        )

    assert asyncio.run(body()) == ("**Quiz: heaps**", None, None)


def test_a_topics_pool_holds_three_quizzes_and_skips_ones_the_student_has_seen():
    store = quiz_world().services.store

    async def body():
        for n in range(5):
            await quiz_reuse.keep(store, 1, "heaps", f"quiz {n}")
        await quiz_reuse.keep(store, 1, "heaps", "quiz 0")  # Already kept.

        async def seen_first_two(quiz):
            return quiz in ("quiz 0", "quiz 1")

        async def seen_all(_quiz):
            return True

        return (
            await quiz_reuse._pool(store, 1, "heaps"),
            await quiz_reuse.find(store, 1, "heaps", seen_first_two),
            await quiz_reuse.find(store, 1, "heaps", seen_all),
        )

    pool, unseen, none_left = asyncio.run(body())
    assert pool == ["quiz 0", "quiz 1", "quiz 2"]
    assert unseen == "quiz 2" and none_left is None


def test_the_same_student_asking_again_gets_a_new_quiz_not_the_one_they_saw(setting):
    setting("quiz_reuse", True)
    other = dict(QUIZ_QUESTION, text="Where is a max-heap's largest item?")
    world = quiz_world(
        intent(quiz_request=0.9),
        task("quiz_master"),
        call("post_quiz", topic="heaps", questions=[QUIZ_QUESTION]),
        say("Posted."),
        intent(quiz_request=0.9),
        task("quiz_master"),
        call("post_quiz", topic="heaps", questions=[other]),
        say("Posted."),
    )
    first, _ = run(world, "quiz me on heaps")
    again, _ = run(world, "quiz me on heaps")

    assert text_of(first) != text_of(again)
    assert "max-heap" in text_of(again)
    assert not world.fake.replies  # The second turn ran the agent: no reused quiz.


def test_without_the_setting_nothing_is_kept_or_reused():
    world = quiz_world(
        intent(quiz_request=0.9),
        task("quiz_master"),
        call("post_quiz", topic="heaps", questions=[QUIZ_QUESTION]),
        say("Posted."),
    )
    run(world, "quiz me on heaps")

    assert asyncio.run(world.services.store.asearch(("quizzes",))) == []


# -- intent tuning ------------------------------------------------------------------------------


def test_every_broad_rule_accepts_an_off_topic_message():
    # Why no new layer-1 rule ships: a pattern can't tell a CS topic from any other.
    for name, (_, pattern) in RULES.items():
        assert any(pattern.match(m) for m in OFF_TOPIC), name


def test_keyword_signals_and_the_analysis_run_offline():
    from owl_mind.evaluation.corpus_intent import INTENT

    assert keyword_intent("quiz me on heaps") == "quiz_request"
    assert keyword_intent("write my english essay") == "other"
    scored = [[(sorted(accepted)[0], 0.9)] for _, _, accepted in INTENT]
    report = analyze(scored)
    assert report["threshold_sweep"]["0.7"]["overall_right"] == 1.0
    assert report["gray_zone"]["cases"] == 0


# -- found while measuring ----------------------------------------------------------------------


def test_a_run_that_finishes_between_two_reads_still_sends_its_answer():
    from tests.test_runtime import stream

    world = World()

    class LateBus:
        """The first read finds nothing, as when the run publishes just after it."""

        def __init__(self, bus) -> None:
            self.bus, self.reads = bus, 0

        async def publish(self, run_id, kind, data):
            await self.bus.publish(run_id, kind, data)

        async def read(self, run_id, after, block_ms):
            self.reads += 1
            return [] if self.reads == 1 else await self.bus.read(run_id, after, block_ms)

    with world.client() as client:
        thread = asyncio.run(world.runs.create_thread(world.student, "t"))
        run_id = asyncio.run(world.runs.create_run(thread.id))
        for kind, data in (("text_delta", {"text": "Answer."}), ("done", {"status": "completed"})):
            asyncio.run(world.bus.publish(run_id, kind, data))
        asyncio.run(world.runs.set_run_status(run_id, "completed"))
        object.__setattr__(world.services, "events", LateBus(world.bus))  # Read per request.
        events = stream(client, str(run_id))

    assert events == [("text_delta", {"text": "Answer."}), ("done", {"status": "completed"})]
