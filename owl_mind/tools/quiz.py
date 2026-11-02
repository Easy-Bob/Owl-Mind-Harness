"""generate_quiz_spec, pin_rubric and grade_answer.

The rubric is pinned once and read back from storage to grade, so the model can't change it.
What stays non-deterministic is the model's judgment of each point, and the result says so.
"""

import hashlib
import uuid
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from owl_mind.services.quiz_math import _MIX, MAX_QUESTIONS, allocate, fingerprint, grade
from owl_mind.tools.context import RunContext

MAX_RUBRIC_POINTS = 10


class RubricPoint(BaseModel):
    id: str = Field(min_length=1, max_length=50)
    weight: float = Field(gt=0, le=100)
    description: str = Field(min_length=1, max_length=500)


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


@tool
async def generate_quiz_spec(
    topic: str, runtime: ToolRuntime[RunContext], count: int = 5, difficulty: str = "mixed"
) -> dict[str, Any]:
    """Get a quiz's structure before writing questions: how many questions, split across recall,
    application and analysis. You write the questions and pin a rubric for each."""
    context = runtime.context
    count = max(1, min(MAX_QUESTIONS, count))
    difficulty = difficulty.lower() if difficulty.lower() in _MIX else "mixed"
    spec = {
        "topic": topic,
        "count": count,
        "difficulty": difficulty,
        "distribution": allocate(count, difficulty),
    }
    spec_id = await context.store.save_spec(context.student_id, spec)
    return {"spec_id": str(spec_id), **spec}


@tool
async def pin_rubric(
    spec_id: str, question: str, points: list[RubricPoint], runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Pin a question's rubric before the student answers: two to four independently checkable
    points, each with a weight. It can't be changed afterward."""
    context = runtime.context
    parsed_spec = _uuid(spec_id)
    ids = [point.id for point in points]
    if not points or len(points) > MAX_RUBRIC_POINTS or len(set(ids)) != len(ids):
        return {"error": f"A rubric needs 1 to {MAX_RUBRIC_POINTS} points with unique ids."}
    stored = [point.model_dump() for point in points]
    pinned = fingerprint(stored)
    rubric_id = (
        await context.store.pin_rubric(context.student_id, parsed_spec, question, stored, pinned)
        if parsed_spec
        else None
    )
    if rubric_id is None:
        return {"error": "Unknown spec_id."}
    return {"rubric_id": str(rubric_id), "fingerprint": pinned}


@tool
async def grade_answer(
    rubric_id: str, answer: str, checks: dict[str, bool], runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Turn your judgment of each rubric point into a mark. Judge each point separately, pass one
    true or false per point id, and report the score this returns."""
    context = runtime.context
    parsed = _uuid(rubric_id)
    rubric = await context.store.get_rubric(context.student_id, parsed) if parsed else None
    if rubric is None:
        return {"error": "Unknown rubric_id."}
    if fingerprint(rubric.points) != rubric.fingerprint:
        return {"error": "This rubric changed after it was pinned, so it can't be graded."}
    answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return await context.store.record_grade(
        context.student_id, rubric.id, answer_hash, grade(rubric.points, checks)
    )
