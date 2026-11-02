"""The student's approval card. Every route acts only on the student's own drafts."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from owl_mind.api.deps import CurrentUser, current_user, get_services
from owl_mind.core.runtime import Services
from owl_mind.forum.handoff import (
    MAX_BODY_CHARS,
    MAX_TITLE_CHARS,
    HandoffError,
    NeedsReview,
    known_values,
    public,
)

router = APIRouter(prefix="/handoffs")

ServicesDep = Annotated[Services, Depends(get_services)]
User = Annotated[CurrentUser, Depends(current_user)]


class ApproveIn(BaseModel):
    title: str = Field(max_length=MAX_TITLE_CHARS)
    body: str = Field(max_length=MAX_BODY_CHARS)
    graded: bool = False


def _service(services: Services):
    if services.handoffs is None:
        raise HTTPException(status_code=503, detail="Forum handoff isn't available.")
    return services.handoffs


def _home(services: Services) -> str | None:
    forum = services.handoffs.forum if services.handoffs else None
    return forum.home_url if forum else None


@router.get("")
async def for_thread(thread_id: uuid.UUID, user: User, services: ServicesDep) -> list[dict]:
    handoffs = await _service(services).store.for_thread(user.id, thread_id)
    return [public(handoff, _home(services)) for handoff in handoffs]


@router.post("/{handoff_id}/approve")
async def approve(handoff_id: uuid.UUID, body: ApproveIn, user: User, services: ServicesDep) -> Any:
    service = _service(services)
    try:
        handoff = await service.approve(
            user.id, handoff_id, body.title, body.body, body.graded, known_values(user.email)
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Draft not found") from exc
    except NeedsReview as review:
        return JSONResponse(status_code=409, content=public(review.handoff, _home(services)))
    except HandoffError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return public(handoff, _home(services))


@router.post("/{handoff_id}/cancel")
async def cancel(handoff_id: uuid.UUID, user: User, services: ServicesDep) -> dict:
    try:
        handoff = await _service(services).cancel(user.id, handoff_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Draft not found") from exc
    except HandoffError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return public(handoff, _home(services))
