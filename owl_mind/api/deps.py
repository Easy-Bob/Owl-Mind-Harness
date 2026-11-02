"""Shared request dependencies."""

import uuid

from fastapi import HTTPException, Request

from owl_mind.core.runtime import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def current_student() -> uuid.UUID:
    """Refuses everyone until M1-E adds login, so no route is open in the meantime."""
    raise HTTPException(status_code=401, detail="Login required")
