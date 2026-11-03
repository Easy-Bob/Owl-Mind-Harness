"""Decides when the intent step should also see the previous chat, and formats it.

Owner's rule: a message gets the previous chat only when it has a word pointing back, such as
"it" or "them", and no specific noun of its own. "Quiz me on it" does; "quiz me on Bloom
filters" and "thanks, now plan my week" don't. This is code, so it adds no model call.
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
_WORD = re.compile(r"[a-z']+")
EARLIER_CHARS = 400


def needs_earlier(message: str) -> bool:
    words = _WORD.findall(message.lower())
    if not any(word in POINTING_BACK for word in words):
        return False
    return all(word in _NO_TOPIC for word in words)


def previous_chat(history: list[dict[str, str]]) -> tuple[str, str] | None:
    """The student's last message and the tutor's reply to it, each trimmed."""
    for index in range(len(history) - 1, 0, -1):
        if history[index]["role"] == "assistant" and history[index - 1]["role"] == "user":
            return (
                history[index - 1]["content"][:EARLIER_CHARS],
                history[index]["content"][:EARLIER_CHARS],
            )
    return None


def with_earlier(message: str, earlier: tuple[str, str]) -> str:
    student, tutor = earlier
    return (
        "<earlier>\n"
        f"Student: {student}\n"
        f"Tutor: {tutor}\n"
        "</earlier>\n"
        "Classify only the new message. Use the earlier turn only to work out what it refers "
        "to.\n"
        f"New message: {message}"
    )
