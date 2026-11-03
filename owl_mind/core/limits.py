"""Run limits per kind of request. Starting values, to be tuned from the M3-E baseline."""

from dataclasses import dataclass

from owl_mind.core.intent.taxonomy import TaskKind


@dataclass(frozen=True)
class RunLimit:
    tool_calls: int
    seconds: int


RUN_LIMITS: dict[TaskKind, RunLimit] = {
    TaskKind.LEARN: RunLimit(20, 120),
    TaskKind.PRACTICE: RunLimit(20, 120),
    TaskKind.PLAN: RunLimit(15, 90),
    TaskKind.ASSESS: RunLimit(30, 180),
    TaskKind.COMPOSITE: RunLimit(35, 200),
    TaskKind.SUPPORT: RunLimit(5, 30),
}
