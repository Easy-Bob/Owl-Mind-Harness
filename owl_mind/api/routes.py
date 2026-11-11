"""Chat routes. Every route needs a logged-in student, and only reaches their own data."""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from owl_mind.api.deps import current_student, get_services
from owl_mind.api.schemas import HistoryMessage, MessageIn, RunStarted, ThreadOut
from owl_mind.core import replies
from owl_mind.core.events import TERMINAL_KINDS
from owl_mind.core.runtime import RunRequest, Services, course_topics, execute_run, read_history
from owl_mind.core.telemetry import record_activity
from owl_mind.services.memory import Memory
from owl_mind.services.run_store import FINISHED, WAITING, ActiveRunExists

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


async def _start(http: Request, services: Services, run: RunRequest) -> None:
    if services.queue is not None:
        # A worker runs it (M4-B), so a busy or restarting API never holds a run up.
        await services.queue.enqueue(run)
        return
    tasks = http.app.state.run_tasks
    task = asyncio.create_task(execute_run(services, run))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


# One answer at a time per student, in any chat, so a second tab can't double what they spend.
_BUSY = "An answer is still being written. Wait for it to finish, then send this again."


async def _queue_run(
    http: Request, services: Services, student: uuid.UUID, thread_id: uuid.UUID, message: str
) -> uuid.UUID:
    run_id = await services.runs.create_run(thread_id)
    try:
        await services.events.publish(run_id, "queued", {})
    except Exception:
        pass  # Live events are allowed to be lost. The run still goes ahead.
    await _start(http, services, RunRequest(run_id, thread_id, student, message))
    return run_id


@router.post("/threads", status_code=201)
async def start_thread(
    body: MessageIn, http: Request, student: Student, services: ServicesDep
) -> RunStarted:
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="A new chat needs a first message")
    thread = await services.runs.create_thread(student, _title(body.message))
    try:
        run_id = await _queue_run(http, services, student, thread.id, body.message)
    except ActiveRunExists as exc:
        # No empty chat is left behind in the student's list.
        await services.runs.delete_empty_thread(thread.id)
        raise HTTPException(status_code=409, detail=_BUSY) from exc
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
        raise HTTPException(status_code=409, detail=_BUSY) from exc
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
        if run is None or run.status in (*FINISHED, WAITING):
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


# -- approvals ----------------------------------------------------------------


def _approval_out(approval) -> dict:
    return {
        "id": str(approval.id),
        "run_id": str(approval.run_id),
        "kind": approval.kind,
        "summary": approval.summary,
        "items": approval.items,
        "expires_at": approval.expires_at.isoformat(),
    }


@router.get("/threads/{thread_id}/approval")
async def pending_approval(thread_id: uuid.UUID, student: Student, services: ServicesDep) -> dict:
    """The chat's card waiting for an answer, so it shows again after a reload."""
    found = None
    if services.approvals is not None and await services.runs.get_thread(student, thread_id):
        found = await services.approvals.pending_for_thread(student, thread_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Nothing waiting for approval")
    return _approval_out(found)


async def _decide(
    run_id: uuid.UUID, approved: bool, http: Request, student: uuid.UUID, services: Services
) -> RunStarted:
    found = None
    if services.approvals is not None:
        found = await services.approvals.pending_for_run(student, run_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Nothing waiting for approval")
    step = "approved" if approved else "rejected"
    # Claimed once, so a double click can't apply the change twice.
    if not await services.approvals.decide(found.id, step):
        raise HTTPException(status_code=409, detail="This has already been answered.")
    await services.runs.record_event(
        run_id, f"approval:{step}", "approval", {"step": step, "kind": found.kind}
    )
    await services.runs.set_run_status(run_id, "completed")
    # A new run continues the paused turn from its checkpoint, with the student's answer.
    resume_run = await services.runs.create_run(found.thread_id)
    await services.approvals.set_resume_run(found.id, resume_run)
    await _start(
        http,
        services,
        RunRequest(
            resume_run,
            found.thread_id,
            student,
            "",
            resume={"approved": approved},
            setup=found.setup,
        ),
    )
    return RunStarted(thread_id=found.thread_id, run_id=resume_run)


@router.post("/runs/{run_id}/approve")
async def approve(
    run_id: uuid.UUID, http: Request, student: Student, services: ServicesDep
) -> RunStarted:
    return await _decide(run_id, True, http, student, services)


@router.post("/runs/{run_id}/reject")
async def reject(
    run_id: uuid.UUID, http: Request, student: Student, services: ServicesDep
) -> RunStarted:
    return await _decide(run_id, False, http, student, services)


# -- citations ----------------------------------------------------------------

_MAX_CITATIONS = 20


@router.get("/citations")
async def citations(
    ids: Annotated[str, Query(max_length=2000)], student: Student, services: ServicesDep
) -> dict:
    """Where each [cite:ID] in an answer comes from. Every student can search the same notes."""
    wanted = list(dict.fromkeys(part.strip() for part in ids.split(",") if part.strip()))
    if len(wanted) > _MAX_CITATIONS:
        raise HTTPException(status_code=422, detail=f"At most {_MAX_CITATIONS} citations")
    found = await services.search.resolve(wanted) if services.search is not None else {}
    return {
        citation_id: (
            {"title": c.title, "location": c.location, "url": c.source_url} if c else None
        )
        for citation_id, c in ((i, found.get(i)) for i in wanted)
    }


@router.get("/help")
async def help_page(student: Student, services: ServicesDep) -> dict:
    """What a student can ask, and the course notes' topics, which change with each import."""
    return {
        "can_help_with": [{"what": w, "example": e} for w, e in replies.CAN_HELP_WITH],
        "cannot": replies.CANNOT,
        "topics": await course_topics(services),
    }


# -- long-term memory -------------------------------------------------------------


@router.get("/memory")
async def my_memory(student: Student, services: ServicesDep) -> dict:
    """What the tutors remember about this student, for them to check."""
    memory = Memory(services.store)
    return {
        "profile": await memory.profile(student),
        "episodes": [
            {"key": e.key, "summary": e.summary, "written_at": e.written_at}
            for e in await memory.episodes(student)
        ],
    }


@router.delete("/memory", status_code=204)
async def forget_me(student: Student, services: ServicesDep) -> None:
    """Delete the student's profile and every past-session summary."""
    await Memory(services.store).delete_all(student)
