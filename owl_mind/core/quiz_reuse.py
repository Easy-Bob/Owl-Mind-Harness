"""Quiz reuse (M4-D experiment, off by default).

A plain "quiz me on X" is answered with a quiz already made from the same course notes, so the
agent doesn't run. Only code-made quizzes are kept, keyed by the topic named in the message and
the corpus version, so a new import of the notes starts afresh. A quiz is only ever kept after
a turn the intent step sent to the quiz master, so an off-topic request never finds one.

Each topic keeps a pool of up to POOL quizzes, and a student is never shown one they've seen:
asking again for practice gets a new quiz, which joins the pool.
"""

import re
from collections.abc import Awaitable, Callable

from langgraph.store.base import BaseStore

# The whole message must be the request, so "quiz me on heaps, then explain them" isn't reused.
_ASK = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could)\s+you\s+)?"
    r"(?:(?:quiz|test)\s+me|(?:give|make)\s+me\s+(?:a|an)\s+(?:short\s+|quick\s+)?quiz)"
    r"\s+(?:on|about)\s+(?P<topic>[a-z0-9][a-z0-9 '\-]{1,60}?)\s*(?:please)?[\s.!?]*$",
    re.IGNORECASE,
)


def quiz_key(message: str) -> str | None:
    match = _ASK.match(message)
    if match is None:
        return None
    topic = " ".join(match["topic"].lower().split())
    topic = re.sub(r"^(the|a|an)\s+", "", topic)
    return topic if topic not in {"it", "this", "that", "them", "these"} else None


def _namespace(version: int) -> tuple[str, ...]:
    return ("quizzes", f"corpus-{version}")


POOL = 3


async def _pool(store: BaseStore, version: int, key: str) -> list[str]:
    item = await store.aget(_namespace(version), key)
    return list((item.value or {}).get("quizzes", [])) if item else []


async def find(
    store: BaseStore, version: int, key: str, seen: Callable[[str], Awaitable[bool]]
) -> str | None:
    """A kept quiz on this topic that the student hasn't been shown, if there is one."""
    for quiz in await _pool(store, version, key):
        if not await seen(quiz):
            return quiz
    return None


async def keep(store: BaseStore, version: int, key: str, quiz: str) -> None:
    pool = await _pool(store, version, key)
    if quiz not in pool and len(pool) < POOL:
        await store.aput(_namespace(version), key, {"quizzes": [*pool, quiz]}, index=False)
