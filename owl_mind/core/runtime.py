"""Runs one turn: classify, answer or build an agent, stream, finish."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.store.base import BaseStore

from owl_mind.core import replies, telemetry
from owl_mind.core.events import EventBus
from owl_mind.core.factory import LiveAgentRunsDisabled, build_owl_agent
from owl_mind.core.intent.classifier import IntentResult, IntentStep
from owl_mind.core.intent.taxonomy import Reply, Route, TaskKind
from owl_mind.core.model import get_chat_model
from owl_mind.core.prompts import PROMPT_VERSION
from owl_mind.core.streaming import STREAM_MODES, StreamTranslator
from owl_mind.forum.handoff import known_values
from owl_mind.services.accounts import AccountStore, LoginLimiter
from owl_mind.services.admin_store import AdminStore
from owl_mind.services.run_store import RunStore
from owl_mind.services.study_store import StudyStore
from owl_mind.tools.context import RunContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunLimit:
    tool_calls: int
    seconds: int


# Starting values, to be tuned from eval traces.
RUN_LIMITS: dict[TaskKind, RunLimit] = {
    TaskKind.LEARN: RunLimit(20, 120),
    TaskKind.PRACTICE: RunLimit(20, 120),
    TaskKind.PLAN: RunLimit(15, 90),
    TaskKind.ASSESS: RunLimit(30, 180),
    TaskKind.COMPOSITE: RunLimit(35, 200),
    TaskKind.SUPPORT: RunLimit(5, 30),
}


@dataclass(frozen=True)
class Services:
    admin: AdminStore
    accounts: AccountStore
    limiter: LoginLimiter
    runs: RunStore
    study: StudyStore
    events: EventBus
    store: BaseStore
    checkpointer: BaseCheckpointSaver
    intent: IntentStep
    get_model: Callable[..., Any] = get_chat_model
    ingestor: Any = None
    search: Any = None
    handoffs: Any = None
    forum_accounts: Any = None


@dataclass(frozen=True)
class RunRequest:
    run_id: uuid.UUID
    thread_id: uuid.UUID
    student_id: uuid.UUID
    message: str


class _RunStopped(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Emitter:
    """Sends live events and records them. Neither failure may fail the run."""

    def __init__(self, services: Services, run_id: uuid.UUID) -> None:
        self._services = services
        self._run_id = run_id
        self._count = 0

    async def __call__(self, kind: str, data: dict[str, Any]) -> None:
        self._count += 1
        try:
            await self._services.events.publish(self._run_id, kind, data)
        except Exception as exc:
            logger.warning("Live event lost: %s", type(exc).__name__)
        # Answer text stays in the checkpoint. The event record holds what happened, not what
        # was said.
        if kind == "text_delta":
            return
        try:
            key = f"{self._count:06d}:{kind}"
            await self._services.runs.record_event(self._run_id, key, kind, data)
        except Exception as exc:
            logger.warning("Run event not recorded: %s", type(exc).__name__)


def _history_graph(checkpointer: BaseCheckpointSaver):
    """Reads and writes a thread's messages with the same state as the agent."""
    return (
        StateGraph(AgentState)
        .add_node("record", lambda state: {"messages": []})
        .add_edge(START, "record")
        .compile(checkpointer=checkpointer)
    )


def _thread_config(thread_id: uuid.UUID) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(thread_id)}}


async def read_history(
    checkpointer: BaseCheckpointSaver, thread_id: uuid.UUID
) -> list[dict[str, str]]:
    state = await _history_graph(checkpointer).aget_state(_thread_config(thread_id))
    history = []
    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage) and message.text and not message.tool_calls:
            history.append({"role": "assistant", "content": message.text})
    return history


async def _record_turn(
    checkpointer: BaseCheckpointSaver, thread_id: uuid.UUID, message: str, answer: str
) -> None:
    await _history_graph(checkpointer).ainvoke(
        {"messages": [HumanMessage(message), AIMessage(answer)]},
        config=_thread_config(thread_id),
    )


async def _draft_handoff(services: Services, request: RunRequest, emit: _Emitter) -> str:
    """Draft a forum post from a template, with no model. The student reviews it on a card."""
    history = await read_history(services.checkpointer, request.thread_id)
    earlier = [m["content"] for m in history if m["role"] == "user" and m["content"]][-3:]
    account = await services.accounts.get_by_id(request.student_id)
    handoff, removed, similar = await services.handoffs.draft(
        request.student_id,
        request.thread_id,
        request.message,
        earlier,
        known_values(account.email) if account else [],
    )
    await emit(
        "handoff_draft",
        {
            "id": str(handoff.id),
            "removed": removed,
            "similar": [{"title": s.title, "url": s.url} for s in similar],
        },
    )
    return replies.HANDOFF_DRAFTED


def _kind(result: IntentResult) -> str | None:
    return result.task_kind.value if result.task_kind else None


async def _fixed_reply(
    services: Services, request: RunRequest, result: IntentResult, emit: _Emitter
) -> str:
    if result.route is Route.HANDOFF:
        if services.handoffs is None:
            text = replies.HANDOFF_UNAVAILABLE
        else:
            text = await _draft_handoff(services, request, emit)
    elif result.reply is Reply.DUE_TOPICS:
        today = datetime.now(UTC).date()
        text = replies.due_topics(await services.study.due_topics(request.student_id, today))
    elif result.reply is Reply.PROGRESS:
        counts = await services.study.review_counts(request.student_id, datetime.now(UTC).date())
        text = replies.progress(*counts)
    else:
        text = replies.TEXT[result.reply]
    return text + (replies.left_out_note(result.left_out) if result.left_out else "")


async def _run_agent(
    services: Services,
    request: RunRequest,
    result: IntentResult,
    config: dict[str, Any],
    emit: _Emitter,
) -> None:
    agent = build_owl_agent(
        result.task_kind,
        result.subagents,
        store=services.store,
        checkpointer=services.checkpointer,
        get_model=services.get_model,
    )
    limit = RUN_LIMITS[result.task_kind]
    translator = StreamTranslator()
    context = RunContext(
        student_id=request.student_id,
        store=services.study,
        today=datetime.now(UTC).date(),
        search=services.search,
    )
    stream = agent.astream(
        {"messages": [HumanMessage(request.message)]},
        config=config,
        context=context,
        stream_mode=STREAM_MODES,
        subgraphs=True,
    )
    try:
        async with asyncio.timeout(limit.seconds):
            async for namespace, mode, data in stream:
                for kind, event in translator.translate(namespace, mode, data):
                    # Text comes first so that the screen stays in order.
                    if kind != "text_delta":
                        for text_kind, text in translator.flush():
                            await emit(text_kind, text)
                    await emit(kind, event)
                if translator.tool_calls > limit.tool_calls:
                    raise _RunStopped("tool_limit")
    except TimeoutError as exc:
        raise _RunStopped("time_limit") from exc
    finally:
        await stream.aclose()
        for kind, event in translator.flush():
            await emit(kind, event)
        await emit("usage", translator.usage)
    if result.left_out:
        await emit("text_delta", {"text": replies.left_out_note(result.left_out)})


async def execute_run(services: Services, request: RunRequest) -> None:
    emit = _Emitter(services, request.run_id)
    config = {
        **_thread_config(request.thread_id),
        "metadata": {
            telemetry.RUN_ID_KEY: str(request.run_id),
            telemetry.PROMPT_VERSION_KEY: PROMPT_VERSION,
        },
    }
    try:
        await services.runs.set_run_status(request.run_id, "running")
        result = await services.intent.classify(request.message, config=config)
        await services.runs.set_run_setup(request.run_id, result.route.value, _kind(result))
        if result.route is Route.ORCHESTRATED:
            try:
                await _run_agent(services, request, result, config, emit)
            except LiveAgentRunsDisabled:
                await _answer(services, request, replies.AGENTS_UNAVAILABLE, emit)
        else:
            text = await _fixed_reply(services, request, result, emit)
            await _answer(services, request, text, emit)
    except _RunStopped as stopped:
        await emit("text_delta", {"text": replies.RUN_STOPPED})
        await _finish(services, request, "stopped", emit, "done", reason=stopped.reason)
    except asyncio.CancelledError:
        await services.runs.set_run_status(request.run_id, "failed")
        raise
    except Exception as exc:
        # Details stay in the log, because error text can hold a student's input.
        logger.warning("Run failed: %s", type(exc).__name__)
        await _finish(services, request, "failed", emit, "error", message=replies.RUN_FAILED)
    else:
        await _finish(services, request, "completed", emit, "done")


async def _finish(
    services: Services, request: RunRequest, status: str, emit: _Emitter, kind: str, **extra
) -> None:
    # Status first: a client that sees the last event may send its next message straight away.
    await services.runs.set_run_status(request.run_id, status)
    data = {"message": extra["message"]} if kind == "error" else {"status": status, **extra}
    await emit(kind, data)


async def _answer(services: Services, request: RunRequest, text: str, emit: _Emitter) -> None:
    await _record_turn(services.checkpointer, request.thread_id, request.message, text)
    await emit("text_delta", {"text": text})
