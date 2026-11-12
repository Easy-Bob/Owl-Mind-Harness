"""Offline: the intent step sees the previous chat only for a follow-up with no topic of its own."""

import pytest

from owl_mind.core.intent.followup import (
    EARLIER_CHARS,
    earlier_for,
    might_need_earlier,
    needs_earlier,
    previous_chat,
    with_earlier,
)
from tests.test_runtime import World, intent, start, stream
from tests.test_runtime import settings as runtime_settings  # noqa: F401  Applied to every test.


@pytest.mark.parametrize(
    "message",
    [
        "quiz me on it",
        "make them harder",
        "explain it again please",
        "give me another one",
        "what about this one",
        "Can you check that?",
    ],
)
def test_a_pointing_word_with_no_topic_of_its_own_gets_the_previous_chat(message):
    assert needs_earlier(message)


@pytest.mark.parametrize(
    "message",
    [
        "quiz me on Bloom filters",
        "thanks, now plan my week",
        "why does it fail on empty lists",
        "explain hash tables",
        "harder",
        "",
    ],
)
def test_a_message_with_its_own_topic_or_no_pointing_word_does_not(message):
    assert not needs_earlier(message)


@pytest.mark.parametrize("message", ["yes", "ok", "sure", "no"])
def test_a_bare_reply_is_classified_on_its_own(message):
    assert not needs_earlier(message)


def test_the_previous_chat_is_cut_to_200_characters_from_the_start():
    history = [
        {"role": "user", "content": "explain Bloom filters " * 20},
        {"role": "assistant", "content": "A Bloom filter is a bit array. " + "x" * 1000},
    ]

    student, tutor = previous_chat(history)

    assert len(student) == len(tutor) == EARLIER_CHARS == 200
    assert tutor.startswith("A Bloom filter is a bit array.")


def test_the_previous_chat_is_the_last_exchange_trimmed():
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "explain Bloom filters"},
        {"role": "assistant", "content": "x" * 1000},
    ]

    student, tutor = previous_chat(history)

    assert student == "explain Bloom filters"
    assert len(tutor) == EARLIER_CHARS


def test_no_previous_chat_in_a_new_thread():
    assert previous_chat([]) is None
    assert previous_chat([{"role": "user", "content": "only a question"}]) is None


def test_the_earlier_chat_is_marked_as_background():
    text = with_earlier("quiz me on it", ("explain Bloom filters", "A Bloom filter is..."))

    assert text.startswith("<earlier>\nStudent: explain Bloom filters\nTutor: A Bloom filter")
    assert "</earlier>" in text
    assert text.endswith("New message: quiz me on it")


def last_user_text(request: dict) -> str:
    return request["messages"][-1]["content"]


def test_a_follow_up_sends_the_previous_chat_in_the_one_intent_call():
    world = World(intent(quiz_request=0.3))
    with world.client() as client:
        started = start(client, "show my progress")
        stream(client, started["run_id"])
        follow_up = client.post(
            f"/threads/{started['thread_id']}/messages", json={"message": "quiz me on it"}
        ).json()
        stream(client, follow_up["run_id"])

    [request] = world.fake.requests
    sent = last_user_text(request)
    assert sent.startswith("<earlier>\nStudent: show my progress\nTutor: ")
    assert sent.endswith("New message: quiz me on it")
    assert "<earlier> block" in request["messages"][0]["content"]


def test_a_message_with_its_own_topic_sends_no_history():
    world = World(intent(quiz_request=0.3))
    with world.client() as client:
        started = start(client, "show my progress")
        stream(client, started["run_id"])
        follow_up = client.post(
            f"/threads/{started['thread_id']}/messages",
            json={"message": "quiz me on hash tables"},
        ).json()
        stream(client, follow_up["run_id"])

    [request] = world.fake.requests
    assert last_user_text(request) == "quiz me on hash tables"


def test_a_follow_up_in_a_new_thread_sends_no_history():
    world = World(intent(quiz_request=0.3))
    with world.client() as client:
        stream(client, start(client, "quiz me on it")["run_id"])

    [request] = world.fake.requests
    assert last_user_text(request) == "quiz me on it"


# A short answer to the tutor's question (issue #70).

PLANNER_ASKS = (
    "Cuckoo hashing study plan: four topics, starting with collision resolution basics. "
    + "x" * 300
    + " Two things it's missing: what you're aiming for and the date you need it by. If you "
    "tell me either, I'll adjust the ordering and pace."
)


def asked(reply: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "schedule a study plan on cuckoo hashing"},
        {"role": "assistant", "content": reply},
    ]


@pytest.mark.parametrize(
    "message", ["exam", "by next Friday", "the final on Dec 5", "5", "false positives"]
)
def test_a_short_answer_after_a_question_gets_the_end_of_the_reply(message):
    earlier = earlier_for(message, asked(PLANNER_ASKS))

    assert earlier is not None
    student, tutor = earlier
    assert student == "schedule a study plan on cuckoo hashing"
    assert len(tutor) == EARLIER_CHARS
    assert tutor.endswith("I'll adjust the ordering and pace.")


@pytest.mark.parametrize(
    "reply",
    [
        "Do you want the plan to cover rehashing too?",
        "Let me know which topic to start with.",
        "Would you like a harder version",
    ],
)
def test_a_question_is_found_with_or_without_a_question_mark(reply):
    assert earlier_for("rehashing", asked(reply)) is not None


def test_a_short_message_after_a_reply_that_asked_nothing_stands_alone():
    assert earlier_for("exam", asked("Week 1: hashing. Week 2: sketches.")) is None


def test_a_question_early_in_a_long_reply_does_not_count():
    reply = "Why do collisions happen? " + "Because keys outnumber slots. " * 20
    assert earlier_for("exam", asked(reply)) is None


@pytest.mark.parametrize("message", ["yes", "ok", "sure", "no", "Yes!", "ok thanks"])
def test_a_bare_reply_stands_alone_even_after_a_question(message):
    assert earlier_for(message, asked(PLANNER_ASKS)) is None


def test_a_long_message_after_a_question_stands_alone():
    message = "I have a midterm exam on hashing next Friday afternoon"
    assert not might_need_earlier(message)
    assert earlier_for(message, asked(PLANNER_ASKS)) is None


def test_a_pointing_follow_up_still_gets_the_start_of_the_reply():
    student, tutor = earlier_for("quiz me on it", asked(PLANNER_ASKS))

    assert tutor.startswith("Cuckoo hashing study plan")


def test_no_earlier_turn_in_a_new_thread_for_a_short_answer():
    assert earlier_for("exam", []) is None


def test_an_answer_to_a_greeting_sends_the_question_in_the_one_intent_call():
    world = World(intent(study_plan=0.3))
    with world.client() as client:
        started = start(client, "hi")
        stream(client, started["run_id"])
        answer = client.post(
            f"/threads/{started['thread_id']}/messages", json={"message": "exam"}
        ).json()
        stream(client, answer["run_id"])

    [request] = world.fake.requests
    sent = last_user_text(request)
    assert sent.startswith("<earlier>\nStudent: hi\nTutor: Hello! What would you like to study")
    assert "which question it answers" in sent
    assert sent.endswith("New message: exam")
