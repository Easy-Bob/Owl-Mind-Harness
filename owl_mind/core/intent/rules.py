"""Layer 1. A match skips the model entirely, so rules fire only on unmistakable messages.

All but the human tutor rule must match the whole message. Otherwise "hi, explain BFS" would
stop at the greeting and never reach a tutor.
"""

import re
from dataclasses import dataclass

from owl_mind.core.intent.taxonomy import Intent, Reply, Route


@dataclass(frozen=True)
class RuleMatch:
    route: Route
    intent: Intent | None = None
    reply: Reply | None = None


_END = r"[\s!.?]*$"

_HUMAN_TUTOR = re.compile(
    r"\b(talk|speak)\s+(to|with)\s+(a\s+|an\s+|my\s+)?"
    r"(ta|human|person|real person|tutor|instructor|professor)\b"
    r"|\bpost\s+(this|it)\s+(to|on)\s+the\s+forum\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(not|never|don'?t|do not|won'?t|didn'?t)\b", re.IGNORECASE)

_WHOLE_MESSAGE_RULES = (
    (
        re.compile(rf"^\s*(hi|hello|hey|good (morning|afternoon|evening))( there)?{_END}", re.I),
        RuleMatch(Route.TEMPLATE, Intent.GREETING, Reply.GREETING),
    ),
    (
        re.compile(rf"^\s*(thanks|thank you|thx)( so much| a lot)?{_END}", re.I),
        RuleMatch(Route.TEMPLATE, Intent.FEEDBACK, Reply.THANKS),
    ),
    (
        re.compile(rf"^\s*(what'?s|what is) (due|due today){_END}", re.I),
        RuleMatch(Route.RULE, None, Reply.DUE_TOPICS),
    ),
    (
        re.compile(rf"^\s*(show|see) my progress{_END}", re.I),
        RuleMatch(Route.TEMPLATE, Intent.PROGRESS_QUERY, Reply.PROGRESS),
    ),
)


def match_rule(message: str) -> RuleMatch | None:
    if not message.strip():
        return RuleMatch(Route.RULE, None, Reply.EMPTY)
    if _HUMAN_TUTOR.search(message) and not _NEGATION.search(message):
        return RuleMatch(Route.HANDOFF, Intent.HUMAN_TUTOR)
    for pattern, match in _WHOLE_MESSAGE_RULES:
        if pattern.match(message):
            return match
    return None
