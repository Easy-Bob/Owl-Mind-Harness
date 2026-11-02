"""Offline: rules, parsing and routing, with the real client and only the network replaced."""

import asyncio
import json
from dataclasses import replace

import pytest

from owl_mind.core.config import get_settings
from owl_mind.core.intent import taxonomy
from owl_mind.core.intent.classifier import MAX_INTENTS, SYSTEM_PROMPT, IntentStep, parse_reply
from owl_mind.core.intent.rules import match_rule
from owl_mind.core.intent.taxonomy import (
    INTENTS,
    Intent,
    Reply,
    Route,
    Subagent,
    TaskKind,
    check_taxonomy,
)
from owl_mind.core.model import ModelTier, get_chat_model
from tests.fake_deepseek import FakeDeepSeek


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.delenv("INTENT_LAYER3_ENABLED", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def reply(**confidences: float) -> dict:
    intents = [{"intent": k, "confidence": v} for k, v in confidences.items()]
    return {"role": "assistant", "content": json.dumps({"intents": intents})}


class Harness:
    def __init__(self, *replies: dict, status: int = 200) -> None:
        self.fake = FakeDeepSeek(list(replies), status=status)
        self.calls: list[tuple[ModelTier, str, int]] = []

    def get_model(self, tier: ModelTier, component: str, max_tokens: int):
        self.calls.append((tier, component, max_tokens))
        model = get_chat_model(tier, component=component, max_tokens=max_tokens)
        self.fake.attach(model)
        return model

    def classify(self, message: str):
        return asyncio.run(IntentStep(self.get_model).classify(message))


# -- taxonomy ---------------------------------------------------------------


def test_taxonomy_has_all_19_intents():
    assert len(Intent) == 19
    assert set(INTENTS) == set(Intent)


def test_startup_check_passes_as_shipped():
    check_taxonomy()


def test_startup_check_fails_on_a_missing_intent(monkeypatch):
    incomplete = dict(INTENTS)
    del incomplete[Intent.GREETING]
    monkeypatch.setattr(taxonomy, "INTENTS", incomplete)

    with pytest.raises(ValueError, match="greeting"):
        check_taxonomy()


@pytest.mark.parametrize(
    "change",
    [
        {"description": ""},
        {"example": ""},
        {"reply": None},
    ],
    ids=["no-description", "no-example", "template-without-reply"],
)
def test_startup_check_fails_on_an_incomplete_intent(monkeypatch, change):
    broken = dict(INTENTS)
    broken[Intent.GREETING] = replace(INTENTS[Intent.GREETING], **change)
    monkeypatch.setattr(taxonomy, "INTENTS", broken)

    with pytest.raises(ValueError, match="greeting"):
        check_taxonomy()


def test_startup_check_fails_on_an_agent_intent_with_no_task_kind(monkeypatch):
    broken = dict(INTENTS)
    broken[Intent.STUDY_PLAN] = replace(INTENTS[Intent.STUDY_PLAN], task_kind=None)
    monkeypatch.setattr(taxonomy, "INTENTS", broken)

    with pytest.raises(ValueError, match="study_plan"):
        check_taxonomy()


def test_prompt_is_stable_and_names_every_intent():
    assert all(intent.value in SYSTEM_PROMPT for intent in Intent)


# -- layer 1 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "route", "intent", "reply_kind"),
    [
        ("", Route.RULE, None, Reply.EMPTY),
        ("   \n\t", Route.RULE, None, Reply.EMPTY),
        ("hi", Route.TEMPLATE, Intent.GREETING, Reply.GREETING),
        ("Good Morning!", Route.TEMPLATE, Intent.GREETING, Reply.GREETING),
        ("thanks a lot!", Route.TEMPLATE, Intent.FEEDBACK, Reply.THANKS),
        ("what's due today?", Route.RULE, None, Reply.DUE_TOPICS),
        ("show my progress", Route.TEMPLATE, Intent.PROGRESS_QUERY, Reply.PROGRESS),
        ("I want to talk to a TA", Route.HANDOFF, Intent.HUMAN_TUTOR, None),
        ("can I speak with my professor", Route.HANDOFF, Intent.HUMAN_TUTOR, None),
        ("post this to the forum", Route.HANDOFF, Intent.HUMAN_TUTOR, None),
    ],
)
def test_rule_settles_with_no_model_call(message, route, intent, reply_kind):
    harness = Harness()

    result = harness.classify(message)

    assert harness.calls == []
    assert harness.fake.requests == []
    assert (result.route, result.reply, result.settled_by_layer) == (route, reply_kind, 1)
    assert result.intents == ((intent,) if intent else ())


@pytest.mark.parametrize(
    "message",
    [
        "hi, explain BFS",
        "thanks, now quiz me",
        "say hi to the algorithm",
        "show my progress and plan my week",
        "I don't want to talk to a TA",
        "what did the TA say about talking to a tutor? I was not there",
        "what is due process in law",
        "hola",
    ],
)
def test_rule_stays_silent_on_anything_ambiguous(message):
    assert match_rule(message) is None


# -- parsing ----------------------------------------------------------------


def test_parse_reply_sorts_and_removes_duplicates():
    text = json.dumps(
        {
            "intents": [
                {"intent": "deadline", "confidence": 0.6},
                {"intent": "study_plan", "confidence": 0.9},
                {"intent": "deadline", "confidence": 0.8},
            ]
        }
    )

    parsed = parse_reply(text)

    assert [(p.intent, p.confidence) for p in parsed] == [
        (Intent.STUDY_PLAN, 0.9),
        (Intent.DEADLINE, 0.8),
    ]


def test_parse_reply_accepts_json_inside_a_code_fence():
    assert parse_reply('```json\n{"intents": []}\n```') == []


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{broken",
        '{"intents": [{"intent": "made_up", "confidence": 0.9}]}',
        '{"intents": [{"intent": "greeting", "confidence": 7}]}',
        '{"other": 1}',
    ],
)
def test_parse_reply_refuses_bad_replies(text):
    with pytest.raises(ValueError):
        parse_reply(text)


# -- routing through layer 2 ------------------------------------------------


def expected(intent: Intent):
    spec = INTENTS[intent]
    return spec.route, spec.task_kind, (spec.subagent,) if spec.subagent else ()


@pytest.mark.parametrize("intent", list(Intent))
def test_each_intent_routes_as_the_table_says(intent):
    harness = Harness(reply(**{intent.value: 0.9}))

    result = harness.classify(INTENTS[intent].example)

    assert (result.route, result.task_kind, result.subagents) == expected(intent)
    assert result.intents == (intent,)
    assert result.settled_by_layer == 2
    assert harness.calls == [(ModelTier.SMALL, "intent", 100)]


@pytest.mark.parametrize(
    ("intent", "route", "subagent"),
    [
        (Intent.CONCEPT_EXPLAIN, Route.ORCHESTRATED, Subagent.CONCEPT_TUTOR),
        (Intent.CODE_REVIEW, Route.ORCHESTRATED, Subagent.PRACTICE_COACH),
        (Intent.DEADLINE, Route.ORCHESTRATED, Subagent.STUDY_PLANNER),
        (Intent.EXPLAIN_BACK, Route.ORCHESTRATED, Subagent.QUIZ_MASTER),
        (Intent.MOTIVATION, Route.ORCHESTRATED, None),
        (Intent.PROGRESS_QUERY, Route.TEMPLATE, None),
        (Intent.HUMAN_TUTOR, Route.HANDOFF, None),
    ],
)
def test_design_table_is_what_the_code_holds(intent, route, subagent):
    assert (INTENTS[intent].route, INTENTS[intent].subagent) == (route, subagent)


def test_two_intents_on_one_subagent_select_it_once():
    harness = Harness(reply(complexity_analysis=0.9, problem_help=0.8))

    result = harness.classify("my solution is slow, give me a hint")

    assert result.subagents == (Subagent.PRACTICE_COACH,)
    assert result.task_kind is TaskKind.PRACTICE
    assert result.intents == (Intent.COMPLEXITY_ANALYSIS, Intent.PROBLEM_HELP)


def test_two_intents_on_different_subagents_are_composite():
    harness = Harness(reply(concept_explain=0.9, quiz_request=0.8))

    result = harness.classify("explain BFS, then quiz me")

    assert result.subagents == (Subagent.CONCEPT_TUTOR, Subagent.QUIZ_MASTER)
    assert result.task_kind is TaskKind.COMPOSITE


def test_third_intent_is_kept_to_the_limit_and_named():
    harness = Harness(reply(concept_explain=0.9, quiz_request=0.85, study_plan=0.8))

    result = harness.classify("explain BFS, quiz me, plan my week")

    assert len(result.intents) == MAX_INTENTS
    assert result.intents == (Intent.CONCEPT_EXPLAIN, Intent.QUIZ_REQUEST)
    assert result.left_out == (Intent.STUDY_PLAN,)


def test_second_intent_below_the_floor_is_ignored():
    harness = Harness(reply(concept_explain=0.9, quiz_request=0.3))

    result = harness.classify("explain BFS")

    assert result.intents == (Intent.CONCEPT_EXPLAIN,)
    assert result.left_out == ()


def test_motivation_alone_has_no_subagent_and_a_support_kind():
    result = Harness(reply(motivation=0.9)).classify("I can't do this")

    assert result.route is Route.ORCHESTRATED
    assert result.task_kind is TaskKind.SUPPORT
    assert result.subagents == ()


def test_tutor_kind_wins_over_motivation_beside_it():
    result = Harness(reply(motivation=0.9, problem_help=0.8)).classify("I give up on this problem")

    assert result.task_kind is TaskKind.PRACTICE
    assert result.subagents == (Subagent.PRACTICE_COACH,)


def test_greeting_beside_a_request_is_dropped_silently():
    result = Harness(reply(greeting=0.9, concept_explain=0.85)).classify("hi, explain BFS")

    assert result.intents == (Intent.CONCEPT_EXPLAIN,)
    assert result.left_out == ()


def test_progress_beside_a_request_is_named_as_left_out():
    result = Harness(reply(concept_explain=0.9, progress_query=0.8)).classify("explain BFS, stats?")

    assert result.intents == (Intent.CONCEPT_EXPLAIN,)
    assert result.left_out == (Intent.PROGRESS_QUERY,)


def test_human_tutor_wins_and_names_what_was_left_out():
    result = Harness(reply(concept_explain=0.9, human_tutor=0.8)).classify("explain or get a TA")

    assert result.route is Route.HANDOFF
    assert result.intents == (Intent.HUMAN_TUTOR,)
    assert result.left_out == (Intent.CONCEPT_EXPLAIN,)


def test_result_holds_no_agent_steps():
    fields = set(Harness(reply(problem_help=0.9)).classify("help").__dataclass_fields__)

    assert fields == {
        "route",
        "task_kind",
        "intents",
        "subagents",
        "reply",
        "settled_by_layer",
        "left_out",
    }


def test_non_english_message_reaches_the_model():
    harness = Harness(reply(concept_explain=0.9))

    result = harness.classify("\u00bfc\u00f3mo funciona BFS?")

    assert result.intents == (Intent.CONCEPT_EXPLAIN,)
    assert len(harness.fake.requests) == 1


# -- confidence and layer 3 -------------------------------------------------


def test_low_confidence_with_layer3_off_asks_for_clarification_with_one_call():
    harness = Harness(reply(concept_explain=0.5))

    result = harness.classify("trees?")

    assert (result.route, result.reply) == (Route.TEMPLATE, Reply.CLARIFY)
    assert result.settled_by_layer == 2
    assert len(harness.fake.requests) == 1


def test_low_confidence_with_layer3_on_makes_one_large_tier_call(monkeypatch):
    monkeypatch.setenv("INTENT_LAYER3_ENABLED", "true")
    get_settings.cache_clear()
    harness = Harness(reply(concept_explain=0.5), reply(concept_explain=0.9))

    result = harness.classify("trees?")

    assert [call[:2] for call in harness.calls] == [
        (ModelTier.SMALL, "intent"),
        (ModelTier.LARGE, "intent:layer3"),
    ]
    assert result.intents == (Intent.CONCEPT_EXPLAIN,)
    assert result.settled_by_layer == 3


def test_layer3_still_unsure_asks_for_clarification(monkeypatch):
    monkeypatch.setenv("INTENT_LAYER3_ENABLED", "true")
    get_settings.cache_clear()
    harness = Harness(reply(concept_explain=0.5), reply(concept_explain=0.5))

    result = harness.classify("trees?")

    assert (result.reply, result.settled_by_layer) == (Reply.CLARIFY, 3)


@pytest.mark.parametrize(
    "bad",
    [
        {"role": "assistant", "content": "I think it is about graphs"},
        {"role": "assistant", "content": '{"intents": [{"intent": "made_up", "confidence": 1}]}'},
        {"role": "assistant", "content": '{"intents": []}'},
    ],
    ids=["prose", "unknown-intent", "empty"],
)
def test_unusable_reply_asks_for_clarification(bad):
    result = Harness(bad).classify("trees?")

    assert result.reply is Reply.CLARIFY


def test_failed_model_call_asks_for_clarification_and_does_not_raise():
    result = Harness(reply(greeting=1), status=500).classify("trees?")

    assert result.reply is Reply.CLARIFY


def test_missing_api_key_asks_for_clarification_and_rules_still_work(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    get_settings.cache_clear()
    step = IntentStep()

    assert asyncio.run(step.classify("trees?")).reply is Reply.CLARIFY
    assert asyncio.run(step.classify("hello")).reply is Reply.GREETING


def test_request_sends_prompt_and_message_with_a_token_cap():
    harness = Harness(reply(concept_explain=0.9))

    harness.classify("explain BFS")

    body = harness.fake.requests[0]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "explain BFS"
    assert body["max_tokens"] == 100
    assert body["thinking"] == {"type": "disabled"}
    assert "tools" not in body
