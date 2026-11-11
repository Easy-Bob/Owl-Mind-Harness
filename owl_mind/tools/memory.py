"""search_memory: past sessions of this student, found by meaning."""

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from owl_mind.services.memory import Memory
from owl_mind.tools.context import RunContext


@tool
async def search_memory(query: str, runtime: ToolRuntime[RunContext]) -> dict[str, Any]:
    """Find this student's past study sessions about a topic: what was covered and how it went.
    Their profile is in /memories/profile.md."""
    if runtime.store is None:
        return {"sessions": []}
    found = await Memory(runtime.store).search(runtime.context.student_id, query)
    return {"sessions": [{"date": e.written_at[:10], "summary": e.summary} for e in found]}
