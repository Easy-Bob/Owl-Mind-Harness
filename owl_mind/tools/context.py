"""What a tool knows about the turn. The model can't set any of this through a tool argument."""

import uuid
from dataclasses import dataclass
from datetime import date

from owl_mind.services.study_store import StudyStore


@dataclass(frozen=True)
class RunContext:
    student_id: uuid.UUID
    store: StudyStore
    today: date
