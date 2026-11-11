"""Admin mode's data. Read-only, apart from recording who opened a user's detail."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import Depends, HTTPException, Query

from owl_mind.api.auth import admin_router
from owl_mind.api.deps import CurrentUser, get_services, require_admin
from owl_mind.core.config import get_settings
from owl_mind.core.intent.taxonomy import Route
from owl_mind.core.runtime import Services
from owl_mind.evaluation.check_baseline import RESULTS
from owl_mind.services.admin_reports import eval_results
from owl_mind.worker.runs import worker_status

__all__ = ["admin_router"]

ServicesDep = Annotated[Services, Depends(get_services)]
Admin = Annotated[CurrentUser, Depends(require_admin)]


@admin_router.get("/requests")
async def list_requests(
    services: ServicesDep,
    q: Annotated[str, Query(max_length=320)] = "",
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 25,
    route: Annotated[Route | Literal[""], Query()] = "",
) -> dict[str, Any]:
    return await services.admin.list_requests(q, page, size, str(route))


@admin_router.get("/requests/{run_id}")
async def get_request(run_id: uuid.UUID, admin: Admin, services: ServicesDep) -> dict[str, Any]:
    detail = await services.admin.get_request(run_id, admin.id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return detail


@admin_router.get("/model-calls")
async def model_calls(
    services: ServicesDep, days: Annotated[int, Query(ge=1, le=400)] = 30
) -> dict[str, Any]:
    return await services.admin.model_call_summary(days)


async def _budget_share(services: Services, student_id: uuid.UUID) -> dict[str, float | None]:
    """Today's share of the student's and everyone's budget, from the counters that enforce
    them. None when the counters can't be read."""
    settings = get_settings()
    try:
        spend = await services.budgets.spent(student_id, datetime.now(UTC).date())
    except Exception:
        return {"student": None, "global": None}
    return {
        "student": spend.student_tokens / settings.student_daily_tokens,
        "global": spend.global_micro_usd / (settings.global_daily_usd * 1_000_000),
    }


@admin_router.get("/overview")
async def overview(services: ServicesDep) -> dict[str, Any]:
    result = await services.admin.overview()
    result["global_budget_used"] = (await _budget_share(services, uuid.UUID(int=0)))["global"]
    return result


@admin_router.get("/users")
async def users(
    services: ServicesDep, q: Annotated[str, Query(max_length=320)] = ""
) -> dict[str, Any]:
    rows = await services.admin.users(q)
    for row in rows:
        row["budget_used_today"] = (await _budget_share(services, uuid.UUID(row["student_id"])))[
            "student"
        ]
    return {"users": rows, "daily_tokens": get_settings().student_daily_tokens}


@admin_router.get("/users/{student_id}")
async def user_usage(student_id: uuid.UUID, admin: Admin, services: ServicesDep) -> dict[str, Any]:
    usage = await services.admin.user_usage(student_id, admin.id)
    if usage is None:
        raise HTTPException(status_code=404, detail="Student not found")
    usage["budget_used_today"] = (await _budget_share(services, student_id))["student"]
    return usage


@admin_router.get("/quality")
async def quality(
    services: ServicesDep, days: Annotated[int, Query(ge=1, le=400)] = 30
) -> dict[str, Any]:
    """Eval results by date and prompt version, and live quality figures (M4-F)."""
    result = await services.admin.quality(days)
    result["evals"] = eval_results(RESULTS)
    return result


@admin_router.get("/reliability")
async def reliability(
    services: ServicesDep, days: Annotated[int, Query(ge=1, le=90)] = 14
) -> dict[str, Any]:
    """Failed and stopped runs, and how the workers, jobs and queue are doing (M4-F)."""
    result = await services.admin.reliability(days)
    result["workers_used"] = services.queue is not None
    if services.queue is not None:
        try:
            result.update(await worker_status(services.queue))
        except Exception:
            result["worker_status_error"] = "Worker status couldn't be read from Redis."
    return result


@admin_router.get("/savings")
async def savings(
    services: ServicesDep, days: Annotated[int, Query(ge=1, le=400)] = 30
) -> dict[str, Any]:
    return await services.admin.savings(days)
