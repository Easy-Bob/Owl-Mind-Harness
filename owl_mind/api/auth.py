"""Signup, login and logout. Passwords, tokens and invite codes never reach logs or errors."""

import asyncio
import hashlib
import hmac
import logging
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

from owl_mind.api.deps import (
    SESSION_COOKIE,
    SESSION_HOURS,
    CurrentUser,
    create_session_token,
    current_user,
    get_services,
    require_admin,
    role_for,
)
from owl_mind.core.config import ConfigError, Settings, get_settings
from owl_mind.core.runtime import Services
from owl_mind.services.accounts import EmailTaken

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth")
# At the router level, so that a new admin route can't forget the check.
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

ServicesDep = Annotated[Services, Depends(get_services)]

MIN_JWT_SECRET_CHARS = 32
MIN_PASSWORD_CHARS = 10
MAX_PASSWORD_CHARS = 128
MAX_ACCOUNT_FAILURES = 5
MAX_ADDRESS_FAILURES = 20

NOTICE = (
    "This is a test system. Your messages are sent to DeepSeek, a model provider that stores "
    "data in China, so don't include personal information. The admin can see your activity "
    "and open your messages."
)

_hasher = PasswordHasher()
# Checked when an email has no account, so that a login takes the same time either way.
_DUMMY_HASH = _hasher.hash("no account has this password")
_BAD_LOGIN = "Incorrect email or password"


class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_CHARS, max_length=MAX_PASSWORD_CHARS)
    invite_code: str = Field(max_length=200)
    accepted_notice: bool


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=MAX_PASSWORD_CHARS)


class UserOut(BaseModel):
    email: str
    role: str


class NoticeOut(BaseModel):
    notice: str


def check_jwt_secret(settings: Settings) -> None:
    secret = settings.jwt_secret
    if secret is None or len(secret.get_secret_value()) < MIN_JWT_SECRET_CHARS:
        raise ConfigError(
            f"Missing or invalid settings: JWT_SECRET needs {MIN_JWT_SECRET_CHARS}+ characters"
        )


def _valid_invite(code: str, settings: Settings) -> bool:
    codes = [c.strip() for c in settings.invite_codes.get_secret_value().split(",") if c.strip()]
    # Check every code, so that timing doesn't reveal how close a guess was.
    results = [hmac.compare_digest(code.encode(), known.encode()) for known in codes]
    return any(results)


def _account_key(email: str) -> str:
    # Hashed, so that Redis holds no email addresses.
    return f"rl:login:{hashlib.sha256(email.lower().encode()).hexdigest()}"


def _address_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    return f"rl:login-address:{hashlib.sha256(host.encode()).hexdigest()}"


async def _is_limited(services: Services, key: str, maximum: int) -> bool:
    try:
        return await services.limiter.failures(key) >= maximum
    except Exception as exc:
        # Argon2 still slows guessing, and a Redis outage shouldn't lock everyone out.
        logger.warning("Login limiter unavailable: %s", type(exc).__name__)
        return False


async def _record_failure(services: Services, *keys: str) -> None:
    try:
        for key in keys:
            await services.limiter.record_failure(key)
    except Exception as exc:
        logger.warning("Login limiter unavailable: %s", type(exc).__name__)


async def _password_matches(password_hash: str | None, password: str) -> bool:
    try:
        return await asyncio.to_thread(_hasher.verify, password_hash or _DUMMY_HASH, password)
    except VerificationError, InvalidHashError:
        return False


@auth_router.get("/notice")
async def notice() -> NoticeOut:
    return NoticeOut(notice=NOTICE)


@auth_router.post("/signup", status_code=201)
async def signup(body: SignupIn, request: Request, services: ServicesDep) -> UserOut:
    settings = get_settings()
    address = _address_key(request)
    if await _is_limited(services, address, MAX_ADDRESS_FAILURES):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    if not _valid_invite(body.invite_code, settings):
        await _record_failure(services, address)
        raise HTTPException(status_code=403, detail="Invalid invite code")
    if not body.accepted_notice:
        raise HTTPException(status_code=400, detail="The notice must be accepted")
    password_hash = await asyncio.to_thread(_hasher.hash, body.password)
    try:
        account = await services.accounts.create(body.email, password_hash)
    except EmailTaken as exc:
        raise HTTPException(status_code=409, detail="This email already has an account") from exc
    return UserOut(email=account.email, role=role_for(account.email, settings))


@auth_router.post("/login")
async def login(
    body: LoginIn, request: Request, response: Response, services: ServicesDep
) -> UserOut:
    settings = get_settings()
    account_key, address_key = _account_key(body.email), _address_key(request)
    if await _is_limited(services, account_key, MAX_ACCOUNT_FAILURES) or await _is_limited(
        services, address_key, MAX_ADDRESS_FAILURES
    ):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")

    account = await services.accounts.get_by_email(body.email)
    matches = await _password_matches(account.password_hash if account else None, body.password)
    if account is None or account.password_hash is None or not matches:
        await _record_failure(services, account_key, address_key)
        raise HTTPException(status_code=401, detail=_BAD_LOGIN)

    try:
        await services.limiter.clear(account_key)
    except Exception as exc:
        logger.warning("Login limiter unavailable: %s", type(exc).__name__)
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(account.id, settings),
        max_age=SESSION_HOURS * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )
    return UserOut(email=account.email, role=role_for(account.email, settings))


@auth_router.post("/logout", status_code=204)
async def logout(response: Response) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=get_settings().cookie_secure,
        samesite="lax",
        path="/",
    )


@auth_router.get("/me")
async def me(user: Annotated[CurrentUser, Depends(current_user)]) -> UserOut:
    return UserOut(email=user.email, role=user.role)


@admin_router.get("/me")
async def admin_me(user: Annotated[CurrentUser, Depends(require_admin)]) -> UserOut:
    """Gives the admin router a route until M1-G adds its pages."""
    return UserOut(email=user.email, role=user.role)
