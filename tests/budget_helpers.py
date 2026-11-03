"""A run meter for tests that build an agent directly, with in-memory counters."""

import uuid

from owl_mind.core.budget import InMemoryBudgetLedger, RunMeter

ALICE = uuid.UUID(int=1)


def meter(student_id: uuid.UUID = ALICE, run_tokens: int = 60_000) -> RunMeter:
    return RunMeter(InMemoryBudgetLedger(), student_id, run_tokens)
