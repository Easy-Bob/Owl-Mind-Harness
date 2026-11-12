"""Runs one turn: classify, answer or build an agent, stream, finish."""

import asyncio
import logging
import re
import time
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
from langgraph.types import Command

from owl_mind.core import metrics, quiz_reuse, replies, telemetry
from owl_mind.core.budget import BudgetExceeded, BudgetStop, RunMeter
from owl_mind.core.config import get_settings
from owl_mind.core.events import EventBus
from owl_mind.core.factory import LiveAgentRunsDisabled, build_owl_agent, goes_direct
from owl_mind.core.integrity import CHECKED_TUTORS, IntegrityCheck, Verdict, fallback, needs_check
from owl_mind.core.intent.classifier import IntentResult, IntentStep
from owl_mind.core.intent.followup import earlier_for, might_need_earlier
from owl_mind.core.intent.taxonomy import Intent, Reply, Route, Subagent, TaskKind
from owl_mind.core.limits import RUN_LIMITS, RunLimit  # noqa: F401  RunLimit is imported from here.
from owl_mind.core.model import get_chat_model
from owl_mind.core.prompts import PROMPT_VERSION
from owl_mind.core.streaming import STREAM_MODES, StreamTranslator
from owl_mind.forum.handoff import known_values
from owl_mind.services.accounts import AccountStore, LoginLimiter
from owl_mind.services.admin_store import AdminStore
from owl_mind.services.cohort import allowed
from owl_mind.services.run_store import WAITING, RunStore
from owl_mind.services.study_store import StudyStore
from owl_mind.tools.context import RunContext

logger = logging.getLogger(__name__)


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
    improver: Any = None
    # A BudgetLedger. Without one no agent runs, because budgets fail closed.
    budgets: Any = None
    # An ApprovalStore. Without one, a change that needs approval fails the run.
    approvals: Any = None
    # A MemoryWriter, for the background job. Reading memory needs only the store.
    memory: Any = None
    # A RedisRunQueue: runs go to workers (M4-B). None runs them in this process, as in tests.
    queue: Any = None
    # Retention and account deletion (M4-G). None in tests that don't need them.
    retention: Any = None


@dataclass(frozen=True)
class RunRequest:
    run_id: uuid.UUID
    thread_id: uuid.UUID
    student_id: uuid.UUID
    message: str
    # Set when the run resumes a paused one: the student's decision and the paused run's setup.
    resume: dict[str, Any] | None = None
    setup: dict[str, Any] | None = None
    # When the API queued it (Unix seconds), for the queue-wait metric (M4-E).
    queued_at: float | None = None
    # Set when a worker takes over a run another worker left: the agent goes on from the
    # thread's last checkpoint with no new input, so the question isn't added twice (M4-C).
    from_checkpoint: bool = False


class _RunStopped(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _budget_stop(stop: BudgetStop) -> tuple[str, str]:
    """The student's message and the recorded reason for a budget stop."""
    scope = stop.scope if isinstance(stop, BudgetExceeded) else "unavailable"
    if scope == "run":
        return replies.RUN_STOPPED, "token_limit"
    if scope == "student":
        return replies.STUDENT_BUDGET, "student_budget"
    return replies.AGENTS_PAUSED, f"{scope}_budget" if scope == "global" else "budget_unavailable"


_CITATION = re.compile(r"\[cite:[^\]\s]+\]")


class _Emitter:
    """Sends live events and records them. Neither failure may fail the run."""

    def __init__(self, services: Services, run_id: uuid.UUID) -> None:
        self._services = services
        self._run_id = run_id
        self._count = 0
        # The run's route and kind once the intent step has chosen them, for metric labels.
        self.route, self.task_kind = "unknown", "none"
        # The answer's citation markers, counted for the Quality page (M4-F). Only the count
        # is ever recorded, never the text.
        self._text: list[str] = []

    async def __call__(self, kind: str, data: dict[str, Any]) -> None:
        self._count += 1
        metrics.observe_event(kind, data)
        try:
            await self._services.events.publish(self._run_id, kind, data)
        except Exception as exc:
            logger.warning("Live event lost: %s", type(exc).__name__)
        # Answer text stays in the checkpoint. The event record holds what happened, not what
        # was said.
        if kind == "text_delta":
            self._text.append(str(data.get("text", "")))
            return
        await self._record(kind, data)

    def citations(self) -> int:
        return len(_CITATION.findall("".join(self._text)))

    async def note(self, kind: str, data: dict[str, Any]) -> None:
        """Record an event for Admin without sending it to the student's screen."""
        self._count += 1
        await self._record(kind, data)

    async def _record(self, kind: str, data: dict[str, Any]) -> None:
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
        run_id=request.run_id,
    )
    # Metadata only, for the Admin timeline: counts, never the draft or what was removed.
    await emit("handoff", {"step": "drafted", "removed": removed})
    if similar:
        await emit("handoff", {"step": "similar_found", "count": len(similar)})
    await emit(
        "handoff_draft",
        {
            "id": str(handoff.id),
            "removed": removed,
            "similar": [{"title": s.title, "url": s.url} for s in similar],
        },
    )
    return replies.HANDOFF_DRAFTED


async def _earlier_for(services: Services, request: RunRequest) -> tuple[str, str] | None:
    """The previous chat, only for a follow-up or a short answer to the tutor's question."""
    if not might_need_earlier(request.message):
        return None
    return earlier_for(
        request.message, await read_history(services.checkpointer, request.thread_id)
    )


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
        if result.reply in replies.WITH_HELP:
            text += "\n\n" + replies.help_text(await course_topics(services))
    return text + (replies.left_out_note(result.left_out) if result.left_out else "")


async def course_topics(services: Services) -> list[str]:
    """The topics of the imported course notes. A lookup failure only shortens the help."""
    store = getattr(services.ingestor, "store", None)
    if store is None:
        return []
    try:
        return await store.list_topics()
    except Exception as exc:
        logger.warning("Topics unavailable: %s", type(exc).__name__)
        return []


async def _run_agent(
    services: Services,
    request: RunRequest,
    result: IntentResult,
    config: dict[str, Any],
    emit: _Emitter,
    meter: RunMeter,
) -> bool:
    """Run the agent. True if it paused for the student's approval."""
    cohort = get_settings().agent_cohort.get_secret_value()
    if cohort and not allowed(await services.accounts.get_by_id(request.student_id), cohort):
        raise LiveAgentRunsDisabled("not in the cohort with live agent runs")
    limit = RUN_LIMITS[result.task_kind]
    meter.run_tokens = limit.tokens
    agent = build_owl_agent(
        result.task_kind,
        result.subagents,
        store=services.store,
        meter=meter,
        checkpointer=services.checkpointer,
        get_model=services.get_model,
    )
    joins = result.task_kind is TaskKind.ASSESS or (
        result.task_kind is TaskKind.COMPOSITE and get_settings().composite_answer == "join"
    )
    # Practice and quiz answers are held until the integrity check passes, so a leak never
    # reaches the screen first.
    # A direct turn has no orchestrator to receive the tutor's reply first, so it's checked.
    checked = needs_check(result.subagents) or goes_direct(result.task_kind, result.subagents)
    translator = StreamTranslator(hold_until_answer=joins or checked, keep_answer=checked)
    context = RunContext(
        student_id=request.student_id,
        store=services.study,
        today=datetime.now(UTC).date(),
        search=services.search,
        record=emit,
        run_id=request.run_id,
    )
    # A quiz posted before a takeover or an approval still counts: one per turn, and its
    # answers are left out of the review.
    for run_id in _turn_runs(request):
        context.posted_quizzes.extend(await services.study.quiz_posts(run_id))
    start = await _start_input(agent, config, request)
    stream = agent.astream(
        start,
        config=config,
        context=context,
        stream_mode=STREAM_MODES,
        subgraphs=True,
    )
    try:
        async with asyncio.timeout(limit.seconds):
            async for namespace, mode, data in stream:
                if context.flags.get("check_answer") and not checked:
                    # A solution was asked for: hold the rest of the answer and check it.
                    checked = True
                    translator.hold_until_answer = translator.keep_answer = True
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
    if translator.interrupt is not None:
        await _ask_approval(services, request, result, emit, translator.interrupt)
        return True
    if checked:
        await _show_checked(
            services, request, result, config, emit, meter, translator, context.posted_quizzes
        )
    if result.left_out:
        await emit("text_delta", {"text": replies.left_out_note(result.left_out)})
    await _keep_quiz(services, request, result, context.posted_quizzes)
    return False


async def _reuse_target(
    services: Services, request: RunRequest, result: IntentResult
) -> tuple[int, str] | None:
    """The corpus version and topic a quiz turn could reuse a quiz for, if reuse applies."""
    if not get_settings().quiz_reuse or services.search is None or request.setup is not None:
        return None
    if result.task_kind is not TaskKind.ASSESS or result.subagents != (Subagent.QUIZ_MASTER,):
        return None
    key = quiz_reuse.quiz_key(request.message)
    return (await services.search.corpus_version(), key) if key else None


async def _keep_quiz(
    services: Services, request: RunRequest, result: IntentResult, quizzes: list[str]
) -> None:
    target = await _reuse_target(services, request, result)
    if target is not None and len(quizzes) == 1:
        await quiz_reuse.keep(services.store, *target, quizzes[0])


async def _reused_quiz(
    services: Services, request: RunRequest, result: IntentResult, emit: _Emitter
) -> bool:
    """Show a quiz already made from the same notes. True if there was one."""
    target = await _reuse_target(services, request, result)
    if target is None:
        return False

    async def seen(quiz: str) -> bool:
        return await services.study.has_seen_quiz(request.student_id, quiz)

    quiz = await quiz_reuse.find(services.store, *target, seen)
    if quiz is None:
        return False
    await services.study.save_quiz_post(request.student_id, request.run_id, target[1], quiz)
    await emit("answer", {"mode": "reused_quiz", "parts": 1})
    await _answer(services, request, quiz, emit)
    return True


async def _start_input(agent: Any, config: dict[str, Any], request: RunRequest) -> Any:
    """What the agent is called with. A taken-over run goes on from where the checkpoint is."""
    question = {"messages": [HumanMessage(request.message)]}
    if not request.from_checkpoint:
        return Command(resume=request.resume) if request.resume is not None else question
    state = await agent.aget_state(config)
    if any(task.interrupts for task in state.tasks):
        # Still paused: give the student's answer if there is one, else ask again.
        return Command(resume=request.resume) if request.resume is not None else None
    if state.next:
        return None  # Mid-turn: carry on with no new input.
    asked = [m for m in state.values.get("messages", []) if isinstance(m, HumanMessage)]
    if request.resume is None and (not asked or asked[-1].content != request.message):
        return question  # Stopped before the question reached the thread.
    return None


def _turn_runs(request: RunRequest) -> list[uuid.UUID]:
    """This run and, when it resumes a paused one, that run too: together they are one turn."""
    paused = (request.setup or {}).get("paused_run")
    return [request.run_id, *([uuid.UUID(paused)] if paused else [])]


def _setup(result: IntentResult) -> dict[str, Any]:
    return {
        "task_kind": result.task_kind.value,
        "subagents": [s.value for s in result.subagents],
        "left_out": [i.value for i in result.left_out],
    }


def _from_setup(setup: dict[str, Any]) -> IntentResult:
    return IntentResult(
        route=Route.ORCHESTRATED,
        task_kind=TaskKind(setup["task_kind"]),
        subagents=tuple(Subagent(s) for s in setup["subagents"]),
        left_out=tuple(Intent(i) for i in setup.get("left_out", [])),
    )


async def _ask_approval(
    services: Services,
    request: RunRequest,
    result: IntentResult,
    emit: _Emitter,
    asked: dict[str, Any],
) -> None:
    """Save what the tool asked, and show the student the card. The run then waits."""
    if services.approvals is None:
        raise RuntimeError("No approval store, so no change can be approved")
    paused = (request.setup or {}).get("paused_run") or str(request.run_id)
    approval = await services.approvals.create(
        request.run_id,
        request.student_id,
        request.thread_id,
        asked,
        {**_setup(result), "paused_run": paused},
    )
    await emit("approval", {"step": "requested", "kind": approval.kind})
    await emit(
        "approval_request",
        {
            "id": str(approval.id),
            "kind": approval.kind,
            "summary": approval.summary,
            "items": approval.items,
        },
    )


async def _review(
    check: IntegrityCheck, meter: RunMeter, request: RunRequest, answer: str, config
) -> Verdict:
    try:
        await meter.check()
    except BudgetStop:
        # No budget left to check with, so the answer isn't shown.
        return Verdict(leak=True, reason="unclear", layer="budget")
    return await check.review(request.message, answer, config)


async def _show_checked(
    services: Services,
    request: RunRequest,
    result: IntentResult,
    config: dict[str, Any],
    emit: _Emitter,
    meter: RunMeter,
    translator: StreamTranslator,
    quizzes: list[str] = (),
) -> None:
    """Show the answer if it passes, else the tutor's one rewrite if that passes, else a hint."""
    await emit("integrity_check", {})
    check = IntegrityCheck(services.get_model)
    answer = translator.answer
    # A code-made quiz shows its answers by the owner's decision, so only the rest is reviewed.
    rest = answer
    for quiz in quizzes:
        rest = rest.replace(quiz, "")
    quiz_text = "\n\n".join(quizzes)
    if quizzes and not rest.strip():
        verdict = Verdict(leak=False, layer="quiz")
    else:
        verdict = await _review(check, meter, request, rest.strip(), config)
    shown, outcome = answer, "passed"
    if verdict.leak:
        shown, outcome = fallback(request.message), "blocked"
        # On a learn turn held after a withheld reply, the concept tutor does the rewrite.
        tutor = next((t for t in result.subagents if t in CHECKED_TUTORS), result.subagents[0])
        if verdict.layer != "budget":
            rewrite = await check.rewrite(
                tutor, request.message, rest.strip(), verdict.reason or "unclear", config
            )
            if rewrite and not (await _review(check, meter, request, rewrite, config)).leak:
                # The quiz stays as code made it, after the rewritten part.
                shown = f"{rewrite}\n\n{quiz_text}" if quizzes else rewrite
                outcome = "retried"
    await emit("integrity", {"result": outcome, "reason": verdict.reason, "layer": verdict.layer})
    if shown != answer:
        # The thread's history keeps what the student saw, so a blocked answer never returns.
        await _replace_answer(services.checkpointer, request.thread_id, translator.answer_id, shown)
    await emit("text_delta", {"text": shown})


async def _replace_answer(
    checkpointer: BaseCheckpointSaver, thread_id: uuid.UUID, answer_id: str | None, text: str
) -> None:
    message = AIMessage(text, id=answer_id) if answer_id else AIMessage(text)
    await _history_graph(checkpointer).aupdate_state(
        _thread_config(thread_id), {"messages": [message]}, as_node="record"
    )


async def _classify(
    services: Services, request: RunRequest, config: dict[str, Any], meter: RunMeter
) -> tuple[IntentResult | None, BudgetStop | None]:
    """With no budget left, only the rules run: a study question then gets no model call."""
    try:
        await meter.daily()
    except BudgetStop as stop:
        return services.intent.match(request.message), stop
    earlier = await _earlier_for(services, request)
    return await services.intent.classify(request.message, config=config, earlier=earlier), None


async def execute_run(services: Services, request: RunRequest) -> None:
    emit = _Emitter(services, request.run_id)
    meter = RunMeter(services.budgets, request.student_id, record=emit, run_id=request.run_id)
    config = {
        **_thread_config(request.thread_id),
        "metadata": {
            telemetry.RUN_ID_KEY: str(request.run_id),
            telemetry.PROMPT_VERSION_KEY: PROMPT_VERSION,
        },
        "callbacks": [meter],
    }
    waiting = False
    started = time.monotonic()
    metrics.RUNS_STARTED.inc()
    metrics.RUNS_RUNNING.inc()
    try:
        await services.runs.set_run_status(request.run_id, "running")
        # What this turn spent before a takeover or an approval still counts.
        await meter.restore(*_turn_runs(request))
        if request.setup is not None:
            # A resumed run continues the paused turn, so the intent step isn't asked again.
            result, closed = _from_setup(request.setup), None
            await meter.daily()
        else:
            result, closed = await _classify(services, request, config, meter)
        if result is None or (closed and result.route is Route.ORCHESTRATED):
            raise closed
        setup = None
        if result.route is Route.ORCHESTRATED:
            # Saved so a worker that takes this run over resumes it without the intent step.
            setup = {
                **_setup(result),
                **{k: v for k, v in (request.setup or {}).items() if k == "paused_run"},
            }
        await services.runs.set_run_setup(request.run_id, result.route.value, _kind(result), setup)
        emit.route, emit.task_kind = result.route.value, _kind(result) or "none"
        if result.route is Route.ORCHESTRATED:
            try:
                if not await _reused_quiz(services, request, result, emit):
                    waiting = await _run_agent(services, request, result, config, emit, meter)
            except LiveAgentRunsDisabled:
                await _answer(services, request, replies.AGENTS_UNAVAILABLE, emit)
        else:
            text = await _fixed_reply(services, request, result, emit)
            await _answer(services, request, text, emit)
            # Which fixed reply was sent, so the Quality page can count clarifying questions.
            reply = result.reply.value if result.reply else result.route.value
            await emit.note("reply", {"reply": reply})
    except _RunStopped as stopped:
        await emit("text_delta", {"text": replies.RUN_STOPPED})
        await _finish(services, request, "stopped", emit, "done", reason=stopped.reason)
    except BudgetStop as stop:
        text, reason = _budget_stop(stop)
        await emit("text_delta", {"text": text})
        await _finish(services, request, "stopped", emit, "done", reason=reason)
    except asyncio.CancelledError:
        await services.runs.set_run_status(request.run_id, "failed")
        _count_finish(emit, "failed", "cancelled")
        raise
    except Exception as exc:
        # Details stay in the log, because error text can hold a student's input.
        logger.warning("Run failed: %s", type(exc).__name__)
        await _finish(services, request, "failed", emit, "error", message=replies.RUN_FAILED)
    else:
        status = WAITING if waiting else "completed"
        if emit.route == Route.ORCHESTRATED.value:
            await emit.note("answer_stats", {"citations": emit.citations()})
        await _finish(services, request, status, emit, "done")
    finally:
        metrics.RUNS_RUNNING.dec()
        metrics.RUN_SECONDS.labels(emit.route, emit.task_kind).observe(time.monotonic() - started)


def _count_finish(emit: _Emitter, status: str, reason: str) -> None:
    metrics.RUNS_FINISHED.labels(emit.route, emit.task_kind, status, reason).inc()


async def _finish(
    services: Services, request: RunRequest, status: str, emit: _Emitter, kind: str, **extra
) -> None:
    # Status first: a client that sees the last event may send its next message straight away.
    await services.runs.set_run_status(request.run_id, status)
    _count_finish(emit, status, str(extra.get("reason") or ("error" if kind == "error" else "")))
    data = {"message": extra["message"]} if kind == "error" else {"status": status, **extra}
    await emit(kind, data)


async def _answer(services: Services, request: RunRequest, text: str, emit: _Emitter) -> None:
    await _record_turn(services.checkpointer, request.thread_id, request.message, text)
    await emit("text_delta", {"text": text})
