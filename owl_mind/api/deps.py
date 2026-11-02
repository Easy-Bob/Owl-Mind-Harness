"""Shared request dependencies."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, Request

from owl_mind.core.config import Settings, get_settings
from owl_mind.core.runtime import Services

SESSION_COOKIE = "owl_session"
SESSION_HOURS = 4
_ALGORITHM = "HS256"


@dataclass(frozen=True)
class CurrentUser:
    id: uuid.UUID
    email: str
    role: str


def get_services(request: Request) -> Services:
    return request.app.state.services


def admin_emails(settings: Settings) -> set[str]:
    return {
        email.strip().lower() for email in settings.configured_admins.split(",") if email.strip()
    }


def role_for(email: str, settings: Settings) -> str:
    # Worked out on each request and never stored, so removing an email takes effect at once.
    return "admin" if email.lower() in admin_emails(settings) else "student"


def create_session_token(account_id: uuid.UUID, settings: Settings) -> str:
    now = datetime.now(UTC)
    claims = {"sub": str(account_id), "iat": now, "exp": now + timedelta(hours=SESSION_HOURS)}
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=_ALGORITHM)


def _account_id(token: str, settings: Settings) -> uuid.UUID:
    claims = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=[_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
    return uuid.UUID(claims["sub"])


async def current_user(
    services: Annotated[Services, Depends(get_services)],
    owl_session: Annotated[str | None, Cookie()] = None,
) -> CurrentUser:
    unauthorized = HTTPException(status_code=401, detail="Login required")
    settings = get_settings()
    if owl_session is None or settings.jwt_secret is None:
        raise unauthorized
    try:
        account_id = _account_id(owl_session, settings)
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise unauthorized from exc
    account = await services.accounts.get_by_id(account_id)
    if account is None:
        raise unauthorized
    return CurrentUser(account.id, account.email, role_for(account.email, settings))


def current_student(user: Annotated[CurrentUser, Depends(current_user)]) -> uuid.UUID:
    return user.id


def require_admin(user: Annotated[CurrentUser, Depends(current_user)]) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user
