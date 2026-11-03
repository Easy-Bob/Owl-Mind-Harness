"""Admin mode's data. Read-only, apart from recording who opened a user's detail."""

import uuid
from typing import Annotated, Any, Literal

from fastapi import Depends, HTTPException, Query

from owl_mind.api.auth import admin_router
from owl_mind.api.deps import CurrentUser, get_services, require_admin
from owl_mind.core.intent.taxonomy import Route
from owl_mind.core.runtime import Services

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
