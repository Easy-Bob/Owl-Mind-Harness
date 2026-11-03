"""Checks around every agent step, attached by the factory to the orchestrator and every tutor.

Each one is code, so a model can't talk its way past it, and none makes a model call.
"""

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

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
    ("grade_answer", "answer"): 4000,
    ("write_file", "content"): 8000,
    ("edit_file", "new_string"): 8000,
    ("edit_file", "old_string"): 8000,
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
        except GraphBubbleUp:
            raise  # An interrupt or a stop, not a failure.
        except Exception as exc:
            return self._message(request, exc)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._message(request, exc)


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


def turn_middleware(
    task_kind: TaskKind,
    tutors: Sequence[Subagent],
    tool_calls: int,
    seconds: int,
    *,
    orchestrator: bool,
) -> list[AgentMiddleware]:
    """The checks for one agent, outermost first: errors are caught around everything else."""
    names = tuple(tutor.value for tutor in tutors)
    stack: list[AgentMiddleware] = [ToolErrorMiddleware(), ToolArgumentMiddleware()]
    if orchestrator:
        stack.append(FanOutMiddleware(allowed=frozenset(names)))
    stack.append(ContextMiddleware(task_kind, names, tool_calls, seconds))
    return stack
