"""What a tool knows about the turn. The model can't set any of this through a tool argument."""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from owl_mind.services.study_store import StudyStore


@dataclass(frozen=True)
class RunContext:
    student_id: uuid.UUID
    store: StudyStore
    today: date
    # MaterialsSearch, or None where there are no materials, such as in some tests.
    search: Any = None
    # Records a metadata-only event on the run, for the Admin timeline. Never fails a tool.
    record: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    # Quizzes post_quiz laid out in this turn. The student sees them exactly as written here.
    posted_quizzes: list[str] = field(default_factory=list)
    # Set when a tutor's reply was withheld as a solution: the student asked for one, so the rest
    # of the turn's answer is held and checked too (found by the M4-A evals).
    flags: dict[str, bool] = field(default_factory=dict)
