"""M4-D experiment 1: intent tuning, measured on the M4-A intent corpus.

One live pass records the small model's scores for every case. From those, offline:
- a threshold sweep, and how many cases a gray zone (a specific question from 0.4) would catch;
- keyword signals alone, as a fallback when the model call fails, and as a cross-check;
- candidate layer-1 rules that would skip the model, with their precision, including off-topic
  probes that a rule can't tell apart from a study request.

Usage: LIVE_AGENT_RUNS_ENABLED=true uv run python -m owl_mind.evaluation.intent_tuning
Writes evals/experiments/<date>-intent_tuning.json (numbers and case IDs only).
"""

import asyncio
import json
import re
from collections import Counter
from datetime import date
from typing import Any

from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import ModelTier
from owl_mind.evaluation.corpus_intent import INTENT
from owl_mind.evaluation.experiment import EXPERIMENTS_DIR

THRESHOLDS = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
GRAY_FLOOR = 0.4

# Keyword signals: the first group whose pattern is found wins.
KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("human_tutor", re.compile(r"\b(ta|human|instructor|professor|forum)\b", re.I)),
    ("quiz_request", re.compile(r"\b(quiz|test me|questions? on)\b", re.I)),
    ("mock_exam", re.compile(r"\b(mock|practice (exam|test)|timed)\b", re.I)),
    ("study_plan", re.compile(r"\b(plan|schedule|revision|study this week)\b", re.I)),
    ("deadline", re.compile(r"\b(due|deadline)\b", re.I)),
    ("progress_query", re.compile(r"\b(progress|how am i doing|my scores?)\b", re.I)),
    ("code_review", re.compile(r"\b(review my code|my code|refactor)\b", re.I)),
    ("complexity_analysis", re.compile(r"\b(runtime|big[- ]o|complexity|o\(n)\b", re.I)),
    ("homework_check", re.compile(r"\b(check my|is my (answer|solution))\b", re.I)),
    ("problem_help", re.compile(r"\b(stuck|hint|help (me )?with|can'?t (start|solve))\b", re.I)),
    ("concept_compare", re.compile(r"\b(difference|compare|vs\.?|versus)\b", re.I)),
    ("material_search", re.compile(r"\b(notes|slides|lecture|where .* covered)\b", re.I)),
    ("motivation", re.compile(r"\b(give up|can'?t do this|failing|stressed|hopeless)\b", re.I)),
    ("greeting", re.compile(r"^\s*(hi|hello|hey)\b", re.I)),
    ("feedback", re.compile(r"\b(thanks|thank you|that helped)\b", re.I)),
    ("concept_explain", re.compile(r"\b(explain|what is|what's|how does|what does)\b", re.I)),
]

# Layer-1 candidates: whole-message patterns that would skip the model.
RULES: dict[str, tuple[str, re.Pattern]] = {
    "quiz_me_on": ("quiz_request", re.compile(r"^\s*(quiz|test) me on [\w\s'-]+[.!?]*$", re.I)),
    "what_is": (
        "concept_explain",
        re.compile(r"^\s*what (is|are) (a |an |the )?[\w\s'-]+\?*$", re.I),
    ),
    "explain": ("concept_explain", re.compile(r"^\s*explain [\w\s'-]+[.!?]*$", re.I)),
    "difference": (
        "concept_compare",
        re.compile(r"^\s*what'?s? (is )?the difference between [\w\s'-]+\?*$", re.I),
    ),
}
# Off-topic messages a student could send. Each is "other": the service refuses non-CS requests.
OFF_TOPIC = [
    "Quiz me on the French Revolution.",
    "Test me on Spanish verbs.",
    "What is photosynthesis?",
    "What is the capital of France?",
    "Explain the causes of World War I.",
    "Explain how to bake bread.",
    "What's the difference between a crocodile and an alligator?",
    "What is 1 + 1?",
]


def keyword_intent(message: str) -> str:
    return next((name for name, pattern in KEYWORDS if pattern.search(message)), "other")


async def scores() -> list[list[tuple[str, float]]]:
    step, limit = IntentStep(), asyncio.Semaphore(8)

    async def one(message: str):
        async with limit:
            found = await step._ask(ModelTier.SMALL, "intent", message, None)
            return [(s.intent.value, s.confidence) for s in found]

    return await asyncio.gather(*[one(m) for _, m, _ in INTENT])


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def analyze(scored: list[list[tuple[str, float]]]) -> dict[str, Any]:
    cases = [
        (case_id, message, accepted, s)
        for (case_id, message, accepted), s in zip(INTENT, scored, strict=True)
    ]
    total = len(cases)
    sweep = {}
    for t in THRESHOLDS:
        taken = [(a, s) for _, _, a, s in cases if s and s[0][1] >= t]
        right = sum(s[0][0] in a for a, s in taken)
        sweep[str(t)] = {
            "accepted_share": _rate(len(taken), total),
            "accepted_precision": _rate(right, len(taken)),
            "overall_right": _rate(right, total),
        }
    gray = [(case_id, a, s) for case_id, _, a, s in cases if s and GRAY_FLOOR <= s[0][1] < 0.7]
    gray_right = sum(s[0][0] in a for _, a, s in gray)
    failed_calls = [case_id for case_id, _, _, s in cases if not s]

    keyword = [(keyword_intent(m), a, s) for _, m, a, s in cases]
    kw_right = sum(k in a for k, a, _ in keyword)
    disagree = [(k, a, s) for k, a, s in keyword if s and k != s[0][0]]
    kw_wins = sum(k in a and s[0][0] not in a for k, a, s in disagree)
    model_wins = sum(s[0][0] in a and k not in a for k, a, s in disagree)

    rules = {}
    for name, (intent, pattern) in RULES.items():
        hits = [(case_id, a) for case_id, m, a, _ in cases if pattern.match(m)]
        off = [m for m in OFF_TOPIC if pattern.match(m)]
        rules[name] = {
            "corpus_hits": len(hits),
            "corpus_precision": _rate(sum(intent in a for _, a in hits), len(hits)),
            "wrong_cases": [case_id for case_id, a in hits if intent not in a],
            "off_topic_accepted": len(off),
        }
    return {
        "cases": total,
        "failed_calls": failed_calls,
        "top_confidence": dict(Counter(round(s[0][1], 1) for _, _, _, s in cases if s)),
        "threshold_sweep": sweep,
        "gray_zone": {
            "cases": len(gray),
            "top_intent_right": gray_right,
            "case_ids": [case_id for case_id, _, _ in gray],
        },
        "keywords": {
            "accuracy_alone": _rate(kw_right, total),
            "disagreements": len(disagree),
            "keyword_right_model_wrong": kw_wins,
            "model_right_keyword_wrong": model_wins,
        },
        "rules": rules,
    }


def main() -> None:
    report = analyze(asyncio.run(scores()))
    report["date"] = date.today().isoformat()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{date.today().isoformat()}-intent_tuning.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
