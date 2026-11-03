"""Turns DeepAgents stream items into our SSE events.

This is the only module that knows the stream format, because DeepAgents 0.x can change it.
Only the orchestrator's text reaches the student: a tutor's text is its report to the orchestrator.
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

FLUSH_CHARS = 24
STREAM_MODES = ["messages", "updates"]
_TASK_TOOL = "task"
_TODO_TOOL = "write_todos"


@dataclass
class StreamTranslator:
    """Keeps a run's counts and buffered text between stream items."""

    tool_calls: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
    )
    # Hold the orchestrator's text until its call ends, and drop it if the call handed out work.
    # For a joined answer, where every orchestrator call only delegates and code writes the
    # answer, so anything it says on the way would reach the student as a stray preamble.
    hold_until_answer: bool = False
    # Keep the answer here instead of streaming it, so it can be checked first (M3-B).
    keep_answer: bool = False
    answer: str = ""
    answer_id: str | None = None
    _buffer: str = ""
    _held: str = ""

    def flush(self) -> list[tuple[str, dict[str, Any]]]:
        if not self._buffer:
            return []
        text, self._buffer = self._buffer, ""
        return [("text_delta", {"text": text})]

    def translate(self, namespace: tuple, mode: str, data: Any) -> list[tuple[str, dict[str, Any]]]:
        if mode == "messages":
            return self._message(namespace, *data)
        if mode == "updates":
            return self._updates(namespace, data)
        return []

    def _message(self, namespace, message, metadata) -> list[tuple[str, dict[str, Any]]]:
        # Pieces of a streamed reply, or a whole message made in code, such as a joined answer.
        is_answer_text = (
            not namespace
            and isinstance(message, AIMessage)
            and metadata.get("langgraph_node") == "model"
            and isinstance(message.content, str)
        )
        if not is_answer_text:
            return []
        if self.hold_until_answer:
            self._held += message.content
            return []
        self._buffer += message.content
        return self.flush() if len(self._buffer) >= FLUSH_CHARS else []

    def _updates(self, namespace, data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        for node, update in data.items():
            if not isinstance(update, dict):
                continue
            if node == "model":
                events += self._model_update(namespace, update)
            elif node == "tools":
                events += self._tools_update(update)
        return events

    def _model_update(self, namespace, update) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        agent = "tutor" if namespace else "orchestrator"
        for message in _messages(update):
            if not isinstance(message, AIMessage):
                continue
            if not namespace and self.hold_until_answer:
                held, self._held = self._held, ""
                if message.tool_calls:
                    pass
                elif self.keep_answer:
                    self.answer, self.answer_id = held, message.id
                else:
                    self._buffer += held
                    events += self.flush()
            usage = message.usage_metadata or {}
            self.usage["input_tokens"] += usage.get("input_tokens", 0)
            self.usage["output_tokens"] += usage.get("output_tokens", 0)
            self.usage["model_calls"] += 1
            for call in message.tool_calls:
                self.tool_calls += 1
                # Names only: arguments can hold the student's own text.
                if call["name"] == _TASK_TOOL:
                    tutor = call["args"].get("subagent_type", "")
                    events.append(("subagent_start", {"name": tutor}))
                elif call["name"] != _TODO_TOOL:
                    events.append(("tool_start", {"name": call["name"], "agent": agent}))
        return events

    def _tools_update(self, update) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        if "todos" in update:
            events.append(("todo_delta", {"todos": update["todos"]}))
        for message in _messages(update):
            if not isinstance(message, ToolMessage):
                continue
            if message.name == _TASK_TOOL:
                events.append(("subagent_end", {"status": message.status}))
            elif message.name != _TODO_TOOL:
                events.append(("tool_end", {"name": message.name, "status": message.status}))
        return events


def _messages(update: dict[str, Any]) -> list:
    messages = update.get("messages")
    # Some nodes wrap their messages in an Overwrite, which holds them in .value.
    messages = getattr(messages, "value", messages)
    return messages if isinstance(messages, list) else []
