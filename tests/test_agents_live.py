"""Real tutors on DeepSeek, one turn per kind of request. Costs a little, and never runs in CI.

Run only on request, with the course materials loaded and agent runs on:
LIVE_AGENT_RUNS_ENABLED=true uv run pytest -m live -k agents

Each turn is capped by its run budget, so the whole file can't spend more than the sum of them.
"""

import asyncio
import dataclasses
import re
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.budget import InMemoryBudgetLedger
from owl_mind.core.config import ConfigError, get_settings
from owl_mind.core.intent.taxonomy import TaskKind
from owl_mind.core.limits import RUN_LIMITS
from owl_mind.core.runtime import RunRequest, execute_run
from owl_mind.evaluation.measure_calls import memory_services
from owl_mind.materials.embedding import FastEmbedEmbedder
from owl_mind.materials.search import InMemorySearchCache, MaterialsSearch
from owl_mind.materials.store import PostgresMaterialsStore
from owl_mind.services.models import sqlalchemy_url

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]

TURNS = [
    (TaskKind.LEARN, "How does a Bloom filter decide that an item is probably in the set?"),
    (TaskKind.PRACTICE, "I'm stuck starting a recursive function that sums a list. A hint?"),
    (TaskKind.PLAN, "Help me plan two weeks of revision for my algorithms final."),
    (TaskKind.ASSESS, "Give me a short quiz on hash tables."),
    (TaskKind.COMPOSITE, "Explain linear probing in hash tables, then quiz me on it."),
    (TaskKind.SUPPORT, "I don't think I can pass this class, I keep failing the quizzes."),
]
_CITATION = re.compile(r"\[cite:[^\]]+\]")


@pytest.fixture(autouse=True)
def needs_live_runs():
    try:
        settings = get_settings()
    except ConfigError as exc:
        pytest.skip(str(exc))
    if settings.deepseek_api_key is None or not settings.live_agent_runs_enabled:
        pytest.skip("Needs DEEPSEEK_API_KEY and LIVE_AGENT_RUNS_ENABLED=true")


async def one_turn(message: str):
    settings = get_settings()
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    search = MaterialsSearch(
        sessions, PostgresMaterialsStore(sessions), embedder, InMemorySearchCache()
    )
    ledger = InMemoryBudgetLedger()
    services = dataclasses.replace(memory_services(), search=search, budgets=ledger)
    student = uuid.uuid4()
    try:
        thread = await services.runs.create_thread(student, "live")
        run_id = await services.runs.create_run(thread.id)
        await execute_run(services, RunRequest(run_id, thread.id, student, message))
        info = await services.runs.get_run(student, run_id)
        events = services.events.events.get(run_id, [])
    finally:
        await engine.dispose()
    spent = ledger.students.get((student, datetime.now(UTC).date()), 0)
    return info, events, spent


@pytest.mark.parametrize(("task_kind", "message"), TURNS, ids=[kind.value for kind, _ in TURNS])
def test_agents_one_turn_per_kind_finishes_within_budget(task_kind, message):
    info, events, spent = asyncio.run(one_turn(message))

    kinds = [kind for _, kind, _ in events]
    answer = "".join(data["text"] for _, kind, data in events if kind == "text_delta")
    steps = [k for k in kinds if k != "text_delta"]
    print(f"\n{task_kind.value}: {spent} tokens, {steps}, {events[-1][2]}\n{answer[:300]}")
    assert info.task_kind == task_kind.value
    assert info.status == "completed", kinds
    assert answer.strip()
    assert spent <= RUN_LIMITS[task_kind].tokens
    found = [data for _, kind, data in events if kind == "retrieval" and data["result_count"]]
    if found and task_kind in (TaskKind.LEARN, TaskKind.COMPOSITE):
        assert _CITATION.search(answer), "The tutor found course facts but the answer cites none"
