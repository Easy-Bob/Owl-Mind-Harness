"""Connecting a student's own GitHub account. No route ever returns a token."""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from owl_mind.api.deps import CurrentUser, current_user, get_services
from owl_mind.core.config import get_settings
from owl_mind.core.runtime import Services
from owl_mind.forum.accounts import OAuthFailed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forum/account")

ServicesDep = Annotated[Services, Depends(get_services)]
User = Annotated[CurrentUser, Depends(current_user)]

_STATE_AUDIENCE = "forum-connect"
_STATE_MINUTES = 10
CALLBACK_PATH = "/forum/account/callback"


class HideCallbackQuery(logging.Filter):
    """The request log would otherwise record GitHub's one-time code from the callback URL."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            if args[2].startswith(CALLBACK_PATH + "?"):
                record.args = (*args[:2], CALLBACK_PATH, *args[3:])
        return True


def _accounts(services: Services):
    if services.forum_accounts is None:
        raise HTTPException(status_code=503, detail="Connecting a GitHub account isn't set up.")
    return services.forum_accounts


def _state(student_id: uuid.UUID) -> str:
    """Signed and tied to the logged-in student, so another site can't finish a connection
    for them, or connect someone else's account to theirs."""
    now = datetime.now(UTC)
    claims = {
        "sub": str(student_id),
        "aud": _STATE_AUDIENCE,
        "exp": now + timedelta(minutes=_STATE_MINUTES),
    }
    secret = get_settings().jwt_secret.get_secret_value()
    return jwt.encode(claims, secret, algorithm="HS256")


def _state_matches(state: str, student_id: uuid.UUID) -> bool:
    secret = get_settings().jwt_secret.get_secret_value()
    try:
        claims = jwt.decode(state, secret, algorithms=["HS256"], audience=_STATE_AUDIENCE)
    except jwt.InvalidTokenError:
        return False
    return claims.get("sub") == str(student_id)


@router.get("")
async def status(user: User, services: ServicesDep) -> dict[str, Any]:
    if services.forum_accounts is None:
        return {"available": False, "connected": False, "login": None, "needs_reconnect": False}
    return {"available": True, **await services.forum_accounts.status(user.id)}


@router.get("/connect")
async def connect(user: User, services: ServicesDep) -> RedirectResponse:
    accounts = _accounts(services)
    return RedirectResponse(accounts.oauth.authorize_url(_state(user.id)), status_code=303)


@router.get(CALLBACK_PATH.removeprefix("/forum/account"))
async def callback(
    user: User, services: ServicesDep, state: str = "", code: str = "", error: str = ""
) -> RedirectResponse:
    accounts = _accounts(services)
    if error or not code or not _state_matches(state, user.id):
        return RedirectResponse("/?forum=failed", status_code=303)
    try:
        await accounts.connect(user.id, code)
    except OAuthFailed as exc:
        logger.warning("GitHub connection failed: %s", type(exc).__name__)
        return RedirectResponse("/?forum=failed", status_code=303)
    return RedirectResponse("/?forum=connected", status_code=303)


@router.post("/disconnect", status_code=204)
async def disconnect(user: User, services: ServicesDep) -> None:
    await _accounts(services).disconnect(user.id)
