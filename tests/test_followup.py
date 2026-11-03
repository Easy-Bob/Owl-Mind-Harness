"""Offline: the intent step sees the previous chat only for a follow-up with no topic of its own."""

import pytest

from owl_mind.core.intent.followup import (
    EARLIER_CHARS,
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
