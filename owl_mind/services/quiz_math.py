"""Quiz shapes, rubric fingerprints and mark arithmetic. The model never totals a mark."""

import hashlib
import json
from typing import Any

# A quiz "spec" is the structure, not the questions. Pinning the shape is what
# makes two quizzes on the same topic comparable -- without it, "quiz me on
# hash tables" twice yields two quizzes that cannot be scored against each
# other, and progress tracking measures the quiz rather than the student.
_MIX: dict[str, tuple[float, float, float]] = {
    # recall, application, analysis
    "easy": (0.6, 0.3, 0.1),
    "mixed": (0.3, 0.4, 0.3),
    "hard": (0.1, 0.4, 0.5),
}
_BANDS = ("recall", "application", "analysis")
MAX_QUESTIONS = 20


def allocate(count: int, difficulty: str) -> dict[str, int]:
    """Split ``count`` questions across the bands. Sums to count exactly.

    Largest-remainder, with ties broken by band order, so the same request
    always produces the same distribution -- floor-and-hope leaves a remainder
    whose placement would otherwise depend on dict ordering.
    """
    weights = _MIX.get(difficulty, _MIX["mixed"])
    exact = [count * weight for weight in weights]
    allocated = [int(value) for value in exact]
    remainder = count - sum(allocated)
    order = sorted(range(len(_BANDS)), key=lambda i: (-(exact[i] - allocated[i]), i))
    for index in order[:remainder]:
        allocated[index] += 1
    return dict(zip(_BANDS, allocated, strict=True))


def fingerprint(rubric: list[dict[str, Any]]) -> str:
    """Stable hash of a whole rubric, so a later change to any point can be detected."""
    canonical = json.dumps(
        [
            {
                "id": str(point["id"]),
                "weight": float(point["weight"]),
                "description": str(point["description"]),
            }
            for point in sorted(rubric, key=lambda point: str(point["id"]))
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def grade(rubric: list[dict[str, Any]], checks: dict[str, bool]) -> dict[str, Any]:
    """Sum the weights of the rubric points that were met. Arithmetic only."""
    met: list[str] = []
    missed: list[str] = []
    unchecked: list[str] = []
    score = 0.0
    total = 0.0

    for point in rubric:
        point_id = str(point["id"])
        weight = float(point["weight"])
        total += weight
        if point_id not in checks:
            # Reported rather than silently counted as a miss: a rubric point
            # nobody judged is a broken grading run, not a wrong answer.
            unchecked.append(point_id)
            missed.append(point_id)
        elif checks[point_id]:
            score += weight
            met.append(point_id)
        else:
            missed.append(point_id)

    return {
        "score": round(score, 4),
        "max_score": round(total, 4),
        "percentage": 0.0 if total == 0 else round(100 * score / total, 2),
        "met": met,
        "missed": missed,
        "unchecked": unchecked,
        "rubric_fingerprint": fingerprint(rubric),
        "determinism": (
            "The arithmetic is exact: the same checks against the same rubric always "
            "give the same mark. The checks themselves are model judgements and can "
            "differ between runs."
        ),
    }
