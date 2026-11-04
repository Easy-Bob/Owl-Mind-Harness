"""Checks around every agent step, attached by the factory to the orchestrator and every tutor.

Each one is code, so a model can't talk its way past it, and none makes a model call.
"""

import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from owl_mind.core import replies
from owl_mind.core.budget import BudgetStop, RunMeter
from owl_mind.core.integrity import MAX_CODE_LINES, code_lines
from owl_mind.core.intent.taxonomy import Subagent, TaskKind

logger = logging.getLogger(__name__)

MAX_TASK_CALLS = 3
DEFAULT_ARGUMENT_CHARS = 2000
# Longer limits where a tool needs room, such as a student's code. Shorter where a long value
# could only be a way to stuff text into a search.
ARGUMENT_CHARS: dict[tuple[str, str], int] = {
    ("materials_search", "query"): 300,
    ("task", "description"): 4000,
    ("analyze_complexity", "code"): 8000,
    ("write_file", "content"): 8000,
    ("edit_file", "new_string"): 8000,
    ("edit_file", "old_string"): 8000,
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DEFAULT_RESULT_CHARS = 6000
# A tutor's report is the whole of its work, so it gets more room than one tool's result.
RESULT_CHARS: dict[str, int] = {"task": 8000}
# Tools the run can do without once its budget runs low: each one leads to more model calls.
_WIND_DOWN_TOOLS = frozenset({"task", "write_todos"})
_WIND_DOWN_NOTE = (
    "## Budget\nThis request has used most of its budget. Don't ask a tutor or plan further: "
    "answer with what you have."
)
_FINAL_NOTE = (
    "## Budget\nThis request's budget is nearly used up. No tools now: give your answer with "
    "what you have, and say briefly if anything is left out."
)


def _with_note(system: SystemMessage | None, note: str) -> SystemMessage:
    if system is None:
        return SystemMessage(note)
    if isinstance(system.content, str):
        return SystemMessage(f"{system.content}\n\n{note}")
    return SystemMessage([*system.content, {"type": "text", "text": f"\n\n{note}"}])


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or tool.get("function", {}).get("name")
    return getattr(tool, "name", None)


def _refuse(request: ToolCallRequest, text: str) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(content=text, tool_call_id=call["id"], name=call["name"], status="error")


class ToolErrorMiddleware(AgentMiddleware):
    """A failing tool becomes a fixed message to the model, and the run carries on.

    The exception's text stays out of the message and the logs, because it can hold the
    student's input.
    """

    def _message(self, request: ToolCallRequest, exc: Exception) -> ToolMessage:
        name = request.tool_call["name"]
        logger.warning("Tool %s failed: %s", name, type(exc).__name__)
        return _refuse(
            request,
            f"{name} failed ({type(exc).__name__}). Answer without it, or tell the student "
            "it isn't available right now.",
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> Any:
        try:
            return handler(request)
        except GraphBubbleUp, BudgetStop:
            raise  # An interrupt or a stop, not a failure.
        except Exception as exc:
            return self._message(request, exc)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        try:
            return await handler(request)
        except GraphBubbleUp, BudgetStop:
            raise
        except Exception as exc:
            return self._message(request, exc)


def _capped(message: ToolMessage, limit: int) -> ToolMessage:
    if not isinstance(message.content, str) or len(message.content) <= limit:
        return message
    cut = message.content[:limit]
    note = f"\n\n[Cut to {limit} of {len(message.content)} characters.]"
    return message.model_copy(update={"content": cut + note})


class ResultCapMiddleware(AgentMiddleware):
    """Cut every tool result and tutor report to a fixed size before the model reads it."""

    def _cap(self, request: ToolCallRequest, result: Any) -> Any:
        limit = RESULT_CHARS.get(request.tool_call["name"], DEFAULT_RESULT_CHARS)
        if isinstance(result, ToolMessage):
            return _capped(result, limit)
        # The task tool answers with a Command that carries the tutor's report.
        update = getattr(result, "update", None) if isinstance(result, Command) else None
        if isinstance(update, dict) and isinstance(update.get("messages"), list):
            messages = [
                _capped(m, limit) if isinstance(m, ToolMessage) else m for m in update["messages"]
            ]
            return Command(
                graph=result.graph,
                update={**update, "messages": messages},
                resume=result.resume,
                goto=result.goto,
            )
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> Any:
        return self._cap(request, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        return self._cap(request, await handler(request))


class ToolArgumentMiddleware(AgentMiddleware):
    """Remove control characters from every text argument, and refuse values that are too long."""

    def _check(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        call = request.tool_call
        cleaned: dict[str, Any] = {}
        for name, value in (call.get("args") or {}).items():
            if isinstance(value, str):
                value = _CONTROL.sub("", value)
                limit = ARGUMENT_CHARS.get((call["name"], name), DEFAULT_ARGUMENT_CHARS)
                if len(value) > limit:
                    return _refuse(
                        request,
                        f"The {name} argument of {call['name']} is too long: at most {limit} "
                        "characters. Shorten it and try again.",
                    )
            cleaned[name] = value
        return request.override(tool_call={**call, "args": cleaned})

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> Any:
        checked = self._check(request)
        return checked if isinstance(checked, ToolMessage) else handler(checked)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        checked = self._check(request)
        return checked if isinstance(checked, ToolMessage) else await handler(checked)


class FanOutMiddleware(AgentMiddleware):
    """The orchestrator may call only the tutors the intent step chose, and at most a few times.

    Retries can't use up another tutor's turn: a call is refused if it would leave fewer calls
    than the chosen tutors that haven't run yet. One instance serves one turn, because the
    factory builds a new agent for every turn.
    """

    def __init__(self, allowed: frozenset[str], maximum: int = MAX_TASK_CALLS) -> None:
        super().__init__()
        self.allowed = allowed
        self.maximum = maximum
        self.calls = 0
        self.called: dict[str, int] = {}

    def _check(self, request: ToolCallRequest) -> ToolMessage | None:
        call = request.tool_call
        if call["name"] != "task":
            return None
        tutor = (call.get("args") or {}).get("subagent_type")
        if tutor not in self.allowed:
            allowed = ", ".join(sorted(self.allowed)) or "none"
            return _refuse(request, f"{tutor} isn't available for this request. Use: {allowed}.")
        if self.calls >= self.maximum:
            return _refuse(
                request,
                f"This request has used its {self.maximum} tutor calls. Answer with what you have.",
            )
        waiting = sorted(t for t in self.allowed if t != tutor and not self.called.get(t))
        if self.maximum - self.calls - 1 < len(waiting):
            return _refuse(
                request,
                f"Keep a call for {', '.join(waiting)}, which hasn't been asked yet. Use what "
                f"{tutor} has given so far.",
            )
        self.calls += 1
        self.called[tutor] = self.called.get(tutor, 0) + 1
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> Any:
        return self._check(request) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        return self._check(request) or await handler(request)


class ContextMiddleware(AgentMiddleware):
    """Adds facts about the turn, written by code, after the agent's fixed prompt.

    It never holds student text. It changes from turn to turn, so it goes after the part of the
    prompt that stays the same and can be cached.
    """

    def __init__(
        self, task_kind: TaskKind, tutors: tuple[str, ...], tool_calls: int, seconds: int
    ) -> None:
        super().__init__()
        self.task_kind = task_kind
        self.tutors = tutors
        self.tool_calls = tool_calls
        self.seconds = seconds

    def block(self, request: ModelRequest) -> str:
        context = getattr(request.runtime, "context", None)
        today = getattr(context, "today", None)
        lines = [
            "## This request",
            f"- Today: {today.isoformat() if today else 'unknown'}",
            f"- Kind of request: {self.task_kind.value}",
            f"- Tutors available: {', '.join(self.tutors) or 'none'}",
            f"- Limits: at most {self.tool_calls} tool calls and {self.seconds} seconds",
        ]
        return "\n".join(lines)

    def _with_block(self, request: ModelRequest) -> ModelRequest:
        block = self.block(request)
        current = request.system_message
        if current is None:
            return request.override(system_message=SystemMessage(block))
        if isinstance(current.content, str):
            content: Any = f"{current.content}\n\n{block}"
        else:
            content = [*current.content, {"type": "text", "text": f"\n\n{block}"}]
        return request.override(system_message=SystemMessage(content))

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        return handler(self._with_block(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._with_block(request))


class BudgetMiddleware(AgentMiddleware):
    """Checks every budget before each model call, and steps down as the run's budget runs low.

    At 70% the run can't ask a tutor or plan. At 90% the model gets one call with no tools to
    answer with what it has. At 100% no call starts. Async only, like every run.
    """

    def __init__(self, meter: RunMeter) -> None:
        super().__init__()
        self.meter = meter

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        step = await self.meter.check()
        if step == "final":
            request = request.override(
                tool_choice="none",
                system_message=_with_note(request.system_message, _FINAL_NOTE),
            )
        elif step == "wind_down":
            request = request.override(
                tools=[t for t in request.tools if _tool_name(t) not in _WIND_DOWN_TOOLS],
                system_message=_with_note(request.system_message, _WIND_DOWN_NOTE),
            )
        return await handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        # Hidden tools are refused too, in case the model calls one anyway.
        if self.meter.step and request.tool_call["name"] in _WIND_DOWN_TOOLS:
            return _refuse(
                request, "This request's budget is nearly used. Answer with what you have."
            )
        return await handler(request)


class JoinAnswerMiddleware(AgentMiddleware):
    """On a two-tutor turn, the answer is the tutors' replies joined in code, once all have come.

    The orchestrator's last model call is skipped, so every [cite:ID] stays exactly as the tutor
    wrote it and no tokens go on rewriting. If a tutor failed, the orchestrator answers as usual.
    It goes outside the budget check, because joining costs nothing.
    """

    def __init__(self, tutors: tuple[str, ...]) -> None:
        super().__init__()
        self.tutors = tutors

    def _replies(self, messages: Sequence[Any]) -> dict[str, str]:
        """Each tutor's latest successful reply in this turn."""
        start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
        asked: dict[str, str] = {}
        found: dict[str, str] = {}
        for message in messages[start:]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call["name"] == "task":
                        asked[call["id"]] = (call.get("args") or {}).get("subagent_type", "")
            elif isinstance(message, ToolMessage) and message.name == "task":
                tutor = asked.get(message.tool_call_id)
                text = message.text.strip()
                if tutor and message.status != "error" and text:
                    found[tutor] = text
        return found

    def joined(self, messages: Sequence[Any], quizzes: Sequence[str] = ()) -> str | None:
        found = self._replies(messages)
        if not all(tutor in found for tutor in self.tutors):
            return None
        if quizzes and "quiz_master" in found:
            # The code-made quiz, exactly, in place of the tutor's own line about it.
            found["quiz_master"] = "\n\n".join(quizzes)
        parts = [found[self.tutors[0]]]
        parts += [f"{replies.JOIN_LEADS.get(t, '')}\n\n{found[t]}".strip() for t in self.tutors[1:]]
        return "\n\n".join(parts)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        # The state holds every message, even ones a summary has replaced in the request.
        context = getattr(request.runtime, "context", None)
        quizzes = getattr(context, "posted_quizzes", None) or ()
        text = self.joined(request.state.get("messages", request.messages), quizzes)
        if text is None:
            return await handler(request)
        record = getattr(getattr(request.runtime, "context", None), "record", None)
        if record is not None:
            await record("answer", {"mode": "joined", "parts": len(self.tutors)})
        # An id of its own, so the integrity check can replace it in the thread's history.
        return ModelResponse(result=[AIMessage(content=text, id=f"joined-{uuid.uuid4()}")])


_WITHHELD = (
    "The tutor's reply was withheld because it solved the student's exercise ({reason}). Help "
    "with hints and questions instead, and write no working solution yourself."
)


class ReportCheckMiddleware(AgentMiddleware):
    """On turns whose answer isn't checked, a tutor's reply with a long code block is reviewed
    before the orchestrator reads it, so a solution asked for as an explanation can't reach the
    student. This runs before anything streams. Most replies have no long code and cost nothing.
    """

    def __init__(self, check: Any) -> None:
        super().__init__()
        self.check = check

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        result = await handler(request)
        if request.tool_call["name"] != "task":
            return result
        update = getattr(result, "update", None) if isinstance(result, Command) else None
        messages = update.get("messages") if isinstance(update, dict) else None
        reply = next((m for m in messages or [] if isinstance(m, ToolMessage)), None)
        if reply is None or code_lines(reply.text) <= MAX_CODE_LINES:
            return result
        state = request.state or {}
        asked = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        # The tool's config carries the run's budget meter, so the review counts toward it.
        config = getattr(request.runtime, "config", None) or {}
        request_text = asked[-1].text if asked else ""
        verdict = await self.check.review(request_text, reply.text, config, guard=False)
        record = getattr(getattr(request.runtime, "context", None), "record", None)
        if record is not None:
            outcome = "withheld" if verdict.leak else "passed"
            await record(
                "integrity", {"result": outcome, "reason": verdict.reason, "layer": "report"}
            )
        if not verdict.leak:
            return result
        withheld = reply.model_copy(update={"content": _WITHHELD.format(reason=verdict.reason)})
        return Command(
            graph=result.graph,
            update={**update, "messages": [withheld if m is reply else m for m in messages]},
            resume=result.resume,
            goto=result.goto,
        )


def turn_middleware(
    task_kind: TaskKind,
    tutors: Sequence[Subagent],
    tool_calls: int,
    seconds: int,
    meter: RunMeter,
    *,
    orchestrator: bool,
) -> list[AgentMiddleware]:
    """The checks for one agent, outermost first: errors are caught around everything else."""
    names = tuple(tutor.value for tutor in tutors)
    stack: list[AgentMiddleware] = [
        BudgetMiddleware(meter),
        ToolErrorMiddleware(),
        ResultCapMiddleware(),
        ToolArgumentMiddleware(),
    ]
    if orchestrator:
        stack.append(FanOutMiddleware(allowed=frozenset(names)))
    stack.append(ContextMiddleware(task_kind, names, tool_calls, seconds))
    return stack
