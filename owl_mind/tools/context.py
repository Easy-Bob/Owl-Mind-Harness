"""What a tool knows about the turn. The model can't set any of this through a tool argument."""

import uuid
from dataclasses import dataclass
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
