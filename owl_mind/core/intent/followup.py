"""Decides when the intent step should also see the previous chat, and formats it.

Two rules, both code, so they add no model call:
- Owner's rule: a message gets the previous chat when it has a word pointing back, such as "it"
  or "them", and no specific noun of its own. "Quiz me on it" does; "quiz me on Bloom filters",
  "thanks, now plan my week" and a bare "yes" don't.
- A short answer to a question: a message of a few words gets the previous chat when the tutor's
  last reply asked something. "exam", after the planner asked what the plan is for, does. Here
  the tutor's reply is cut from its end, where the question is. A bare "yes" or "ok" still
  doesn't.
"""

import re

POINTING_BACK = {"it", "them", "this", "that", "these", "those", "one", "ones", "its"}
# Words that carry no topic of their own: function words, request words and how a student asks
# for more of the same. Anything else left in the message is treated as its own topic.
_NO_TOPIC = POINTING_BACK | {
    "a", "an", "the", "me", "my", "i", "you", "your", "we", "us", "our", "on", "of", "in", "to",
    "for", "with", "about", "at", "by", "from", "and", "or", "but", "so", "then", "now", "again",
    "please", "can", "could", "would", "will", "do", "does", "did", "is", "are", "was", "be",
    "what", "how", "why", "which", "more", "another", "other", "same", "next", "instead", "too",
    "also", "just", "some", "any", "harder", "easier", "simpler", "shorter", "longer", "better",
    "explain", "quiz", "test", "check", "give", "show", "tell", "make", "try", "help", "write",
    "practice", "review", "grade", "teach", "walk", "through", "over", "up", "out", "example",
    "examples", "question", "questions", "problem", "problems", "hint", "hints", "way",
    "let", "lets", "let's", "want", "need", "like", "ok", "okay", "yes", "no", "not", "don't",
    "dont", "really", "still", "get", "go", "use", "using", "version", "bit", "little", "lot",
}  # fmt: skip
# Replies that say nothing by themselves. They stay on their own, as the owner's rule says.
_BARE = {"yes", "no", "ok", "okay", "sure", "yeah", "yep", "nope", "fine", "thanks", "thank", "you"}
_WORD = re.compile(r"[a-z']+")
EARLIER_CHARS = 200
SHORT_ANSWER_WORDS = 6
# Where a reply's question would be. Tutors don't always end with "?": the planner wrote
# "If you tell me either, I'll adjust the ordering and pace."
_ASKING_TAIL = 300
_ASKS = re.compile(
    r"\?|\b(tell me|let me know|if you (tell|give|share|say|let)|do you want|would you like)\b",
    re.IGNORECASE,
)


def needs_earlier(message: str) -> bool:
    words = _WORD.findall(message.lower())
    if not any(word in POINTING_BACK for word in words):
        return False
    return all(word in _NO_TOPIC for word in words)


def may_answer(message: str) -> bool:
    """Short enough to be an answer to a question, and not a bare yes or ok."""
    # Counted by spaces, so "Dec 5" is two words; checked for bareness by letters only.
    count = len(message.split())
    words = _WORD.findall(message.lower())
    bare = bool(words) and all(word in _BARE for word in words)
    return 0 < count <= SHORT_ANSWER_WORDS and not bare


def asks_something(reply: str) -> bool:
    return bool(_ASKS.search(reply[-_ASKING_TAIL:]))


def _last_exchange(history: list[dict[str, str]]) -> tuple[str, str] | None:
    for index in range(len(history) - 1, 0, -1):
        if history[index]["role"] == "assistant" and history[index - 1]["role"] == "user":
            return history[index - 1]["content"], history[index]["content"]
    return None


def previous_chat(history: list[dict[str, str]]) -> tuple[str, str] | None:
    """The student's last message and the tutor's reply to it, each trimmed to its start."""
    found = _last_exchange(history)
    if found is None:
        return None
    student, tutor = found
    return student[:EARLIER_CHARS], tutor[:EARLIER_CHARS]


def asked_turn(history: list[dict[str, str]]) -> tuple[str, str] | None:
    """The last exchange when the tutor's reply asked something, with the reply cut from its end."""
    found = _last_exchange(history)
    if found is None or not asks_something(found[1]):
        return None
    student, tutor = found
    return student[:EARLIER_CHARS], tutor[-EARLIER_CHARS:]


def might_need_earlier(message: str) -> bool:
    """Whether the history is worth reading at all. Most messages need no read."""
    return needs_earlier(message) or may_answer(message)


def earlier_for(message: str, history: list[dict[str, str]]) -> tuple[str, str] | None:
    """The earlier turn the intent step should see with this message, if any."""
    if needs_earlier(message):
        return previous_chat(history)
    if may_answer(message):
        return asked_turn(history)
    return None


def with_earlier(message: str, earlier: tuple[str, str]) -> str:
    student, tutor = earlier
    return (
        "<earlier>\n"
        f"Student: {student}\n"
        f"Tutor: {tutor}\n"
        "</earlier>\n"
        "Classify only the new message. Use the earlier turn only to work out what it refers "
        "to, or which question it answers.\n"
        f"New message: {message}"
    )
