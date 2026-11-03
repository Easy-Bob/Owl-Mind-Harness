"""Run limits per kind of request, set from the M3-E baseline with room above the largest turn."""

from dataclasses import dataclass

from owl_mind.core.intent.taxonomy import TaskKind


@dataclass(frozen=True)
class RunLimit:
    tool_calls: int
    seconds: int
    # Input and output tokens of every model call in the run, the intent step's included.
    tokens: int


RUN_LIMITS: dict[TaskKind, RunLimit] = {
    TaskKind.LEARN: RunLimit(20, 120, 60_000),
    TaskKind.PRACTICE: RunLimit(20, 120, 60_000),
    TaskKind.PLAN: RunLimit(30, 120, 80_000),
    TaskKind.ASSESS: RunLimit(30, 180, 100_000),
    TaskKind.COMPOSITE: RunLimit(35, 200, 180_000),
    TaskKind.SUPPORT: RunLimit(5, 30, 15_000),
}
