"""Chat routes. Every route needs a logged-in student, and only reaches their own data."""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from owl_mind.api.deps import current_student, get_services
from owl_mind.api.schemas import HistoryMessage, MessageIn, RunStarted, ThreadOut
from owl_mind.core.events import TERMINAL_KINDS
from owl_mind.core.runtime import RunRequest, Services, execute_run, read_history
from owl_mind.core.telemetry import record_activity
from owl_mind.services.run_store import FINISHED, ActiveRunExists

logger = logging.getLogger(__name__)

router = APIRouter()

Student = Annotated[uuid.UUID, Depends(current_student)]
ServicesDep = Annotated[Services, Depends(get_services)]

_MAX_TITLE_CHARS = 60
_IDLE_WAIT_MS = 2000
_RETRY_SECONDS = 1
_EVENT_ID = re.compile(r"\d+-\d+")


def _title(message: str) -> str:
    return " ".join(message.split())[:_MAX_TITLE_CHARS]


def _start(http: Request, services: Services, run: RunRequest) -> None:
    tasks = http.app.state.run_tasks
    task = asyncio.create_task(execute_run(services, run))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def _queue_run(
    http: Request, services: Services, student: uuid.UUID, thread_id: uuid.UUID, message: str
) -> uuid.UUID:
    run_id = await services.runs.create_run(thread_id)
    try:
        await services.events.publish(run_id, "queued", {})
    except Exception:
        pass  # Live events are allowed to be lost. The run still goes ahead.
    _start(http, services, RunRequest(run_id, thread_id, student, message))
    return run_id


@router.post("/threads", status_code=201)
async def start_thread(
    body: MessageIn, http: Request, student: Student, services: ServicesDep
) -> RunStarted:
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="A new chat needs a first message")
    thread = await services.runs.create_thread(student, _title(body.message))
    run_id = await _queue_run(http, services, student, thread.id, body.message)
    await asyncio.to_thread(record_activity, "session_started", student, thread.id)
    await asyncio.to_thread(record_activity, "message_sent", student, thread.id)
    return RunStarted(thread_id=thread.id, run_id=run_id)


@router.get("/threads")
async def list_threads(student: Student, services: ServicesDep) -> list[ThreadOut]:
    threads = await services.runs.list_threads(student)
    return [ThreadOut(id=t.id, title=t.title, created_at=t.created_at) for t in threads]


@router.post("/threads/{thread_id}/messages", status_code=201)
async def send_message(
    thread_id: uuid.UUID, body: MessageIn, http: Request, student: Student, services: ServicesDep
) -> RunStarted:
    if await services.runs.get_thread(student, thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    try:
        run_id = await _queue_run(http, services, student, thread_id, body.message)
    except ActiveRunExists as exc:
        raise HTTPException(status_code=409, detail="A run is already active") from exc
    await asyncio.to_thread(record_activity, "message_sent", student, thread_id)
    return RunStarted(thread_id=thread_id, run_id=run_id)


@router.get("/threads/{thread_id}/messages")
async def get_messages(
    thread_id: uuid.UUID, student: Student, services: ServicesDep
) -> list[HistoryMessage]:
    if await services.runs.get_thread(student, thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    history = await read_history(services.checkpointer, thread_id)
    return [HistoryMessage(**message) for message in history]


def _sse(event_id: str, kind: str, data: dict) -> str:
    return f"id: {event_id}\nevent: {kind}\ndata: {json.dumps(data)}\n\n"


async def _event_stream(
    services: Services, student: uuid.UUID, run_id: uuid.UUID, after: str
) -> AsyncIterator[str]:
    while True:
        try:
            events = await services.events.read(run_id, after, _IDLE_WAIT_MS)
        except Exception as exc:
            # Losing live events must not end the response, so fall back to the run's status.
            logger.warning("Live events unavailable: %s", type(exc).__name__)
            events = []
            await asyncio.sleep(_RETRY_SECONDS)
        for event_id, kind, data in events:
            after = event_id
            yield _sse(event_id, kind, data)
            if kind in TERMINAL_KINDS:
                return
        if events:
            continue
        run = await services.runs.get_run(student, run_id)
        # The live stream can expire or be lost, so a finished run must still end the response.
        if run is None or run.status in FINISHED:
            status = run.status if run else "failed"
            yield _sse("0-0", "done", {"status": status})
            return
        yield ": still working\n\n"


@router.get("/runs/{run_id}/stream")
async def stream_run(
    run_id: uuid.UUID,
    student: Student,
    services: ServicesDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    if last_event_id is not None and not _EVENT_ID.fullmatch(last_event_id):
        raise HTTPException(status_code=422, detail="Invalid Last-Event-ID")
    if await services.runs.get_run(student, run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        _event_stream(services, student, run_id, last_event_id or "0"),
        media_type="text/event-stream",
    )
