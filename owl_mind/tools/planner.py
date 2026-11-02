"""schedule_review, get_due_topics and propose_study_plan. Dates always come from code."""

from datetime import date
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from owl_mind.services.sm2 import DEFAULT_EASE, sm2
from owl_mind.services.study_store import ReviewState
from owl_mind.tools.context import RunContext

MAX_PLAN_ITEMS = 30


class PlanItem(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=500)


def _normalize(topic: str) -> str:
    return topic.strip().lower()


@tool
async def schedule_review(
    topic: str, quality: int, runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Record how well the student recalled a topic and get its next review date. quality is 0
    (blank), 3 (correct with effort) or up to 5 (perfect). Never work out a date yourself."""
    context = runtime.context
    topic = _normalize(topic)
    current = await context.store.get_review(context.student_id, topic)
    result = sm2(
        quality=max(0, min(5, quality)),
        ease=current.ease if current else DEFAULT_EASE,
        interval=current.interval_days if current else 0,
        repetitions=current.repetitions if current else 0,
        today=context.today,
    )
    state = ReviewState(
        ease=result["ease"],
        interval_days=result["interval"],
        repetitions=result["repetitions"],
        due_date=date.fromisoformat(result["due_date"]),
    )
    await context.store.save_review(context.student_id, topic, state)
    return {"topic": topic, **result}


@tool
async def get_due_topics(runtime: ToolRuntime[RunContext]) -> dict[str, Any]:
    """Get the topics this student should review today, most overdue first."""
    context = runtime.context
    topics = await context.store.due_topics(context.student_id, context.today)
    return {"today": context.today.isoformat(), "due": topics}


@tool
async def propose_study_plan(
    title: str, items: list[PlanItem], runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Save a proposed study plan: an ordered list of topics, each with an optional note. Review
    dates are filled in from the student's schedule. The plan is not committed."""
    context = runtime.context
    if not items or len(items) > MAX_PLAN_ITEMS:
        return {"error": f"A plan needs 1 to {MAX_PLAN_ITEMS} items."}
    planned = []
    for item in items:
        topic = _normalize(item.topic)
        state = await context.store.get_review(context.student_id, topic)
        due = state.due_date if state else context.today
        planned.append({"topic": topic, "note": item.note, "review_date": due.isoformat()})
    plan_id = await context.store.save_plan(context.student_id, title, planned)
    return {"plan_id": str(plan_id), "status": "proposed", "items": planned}
