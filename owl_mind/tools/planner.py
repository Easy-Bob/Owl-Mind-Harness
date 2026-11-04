"""The planner's tools. Dates always come from code, and changes wait for the student's approval.

commit_study_plan and reset_progress pause the run with an interrupt. The card the student sees
is written here, from the student's own records, never by the model.
"""

import uuid
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from owl_mind.tools.context import RunContext

MAX_PLAN_ITEMS = 30


class PlanItem(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=500)


def _normalize(topic: str) -> str:
    return topic.strip().lower()


@tool
async def preview_review_dates(
    topics: list[str], runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Get each topic's next review date: its current due date, or today for a new topic. Saves
    nothing. Never work out a date yourself."""
    context = runtime.context
    dates = {}
    for topic in topics[:MAX_PLAN_ITEMS]:
        state = await context.store.get_review(context.student_id, _normalize(topic))
        dates[_normalize(topic)] = (state.due_date if state else context.today).isoformat()
    return {"today": context.today.isoformat(), "dates": dates}


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


def _approved(kind: str, summary: str, target: str, items: list | None = None) -> bool:
    """Pause the run until the student answers the card. Resuming returns their decision."""
    decision = interrupt({"kind": kind, "summary": summary, "target": target, "items": items or []})
    return isinstance(decision, dict) and decision.get("approved") is True


@tool
async def commit_study_plan(plan_id: str, runtime: ToolRuntime[RunContext]) -> dict[str, Any]:
    """Commit a proposed plan, after the student approves it on a card. New topics in it join
    their review schedule. Use only when the student wants to keep the plan."""
    context = runtime.context
    parsed = _uuid(plan_id)
    plan = await context.store.get_plan(context.student_id, parsed) if parsed else None
    if plan is None or plan.status != "proposed":
        return {"error": "Unknown plan_id, or the plan isn't a proposal."}
    new = [
        item["topic"]
        for item in plan.items
        if await context.store.get_review(context.student_id, item["topic"]) is None
    ]
    summary = (
        f"Commit '{plan.title}', {len(plan.items)} items. {len(new)} new topics join your reviews."
    )
    # The card lists the saved proposal itself, so the student approves exactly what is kept.
    items = [{"topic": i["topic"], "review_date": i["review_date"]} for i in plan.items]
    if not _approved("commit_plan", summary, str(plan.id), items):
        return {"status": "not committed", "reason": "The student said no. Nothing changed."}
    if not await context.store.commit_plan(context.student_id, plan.id):
        return {"error": "The plan was already committed."}
    return {"status": "committed", "plan_id": str(plan.id), "new_topics": new}


@tool
async def reset_progress(
    runtime: ToolRuntime[RunContext], topic: str = "", all_topics: bool = False
) -> dict[str, Any]:
    """Clear the student's review progress for one topic, or for all topics with all_topics,
    after they approve it on a card. Use only when the student asks to start over."""
    context = runtime.context
    if all_topics:
        scheduled, _ = await context.store.review_counts(context.student_id, context.today)
        if not scheduled:
            return {"error": "There's no progress to reset."}
        summary, target = f"Reset progress for all {scheduled} topics.", "all"
    else:
        target = _normalize(topic)
        if not target or await context.store.get_review(context.student_id, target) is None:
            return {"error": "That topic has no progress to reset."}
        summary = f"Reset progress for '{target}'."
    if not _approved("reset_progress", summary, target):
        return {"status": "not reset", "reason": "The student said no. Nothing changed."}
    cleared = await context.store.reset_progress(context.student_id, None if all_topics else target)
    return {"status": "reset", "topics": cleared}


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
