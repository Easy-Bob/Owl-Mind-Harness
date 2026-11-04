"""Request and response models for the chat API."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

# Owner's decision: short questions. Pasted code or long answers don't fit.
MAX_MESSAGE_CHARS = 200


class MessageIn(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)


class RunStarted(BaseModel):
    thread_id: uuid.UUID
    run_id: uuid.UUID


class ThreadOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime


class HistoryMessage(BaseModel):
    role: str
    content: str
