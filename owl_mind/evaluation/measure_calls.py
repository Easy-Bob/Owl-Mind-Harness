"""Measures model calls, tokens and cost per route and task kind on a fixed message set.

Live, it uses the real course search, so tutors read real notes. Offline tests run it against
scripted replies, which proves that it counts correctly, not what real turns cost.

Usage: LIVE_AGENT_RUNS_ENABLED=true uv run python -m owl_mind.evaluation.measure_calls out.json
"""

import asyncio
import dataclasses
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core import telemetry
from owl_mind.core.budget import InMemoryBudgetLedger
from owl_mind.core.config import get_settings
from owl_mind.core.events import InMemoryEventBus
from owl_mind.core.intent.classifier import IntentStep
from owl_mind.core.model import get_chat_model
from owl_mind.core.runtime import RunRequest, Services, execute_run
from owl_mind.materials.embedding import FastEmbedEmbedder
from owl_mind.materials.search import InMemorySearchCache, MaterialsSearch
from owl_mind.materials.store import PostgresMaterialsStore
from owl_mind.services.accounts import InMemoryAccountStore, InMemoryLoginLimiter
from owl_mind.services.models import sqlalchemy_url
from owl_mind.services.run_store import InMemoryRunStore
from owl_mind.services.study_store import InMemoryStudyStore
from owl_mind.worker.telemetry import calculate_cost

# About $0.15 at the peak price. Every agent turn is capped by its run budget as well.
TOKEN_LIMIT = 500_000

MESSAGES = (
    "hello",
    "thanks",
    "what's due today?",
    "show my progress",
    "I want to talk to a TA",
    "Can you walk me through how a binary heap keeps its order?",
    "Should I reach for an array or a linked list when inserting often?",
    "I can't work out how to begin this recursion exercise.",
    "My loop over the list runs far too slowly, what might be wrong?",
    "Help me plan two weeks of revision for my algorithms final.",
    "Give me a short quiz on hash tables.",
    "Teach me Dijkstra's algorithm, then test me on it.",
)


@dataclass
class Group:
    runs: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cost_usd: float = 0.0
    calls_with_unknown_usage: int = 0

    def add(self, event: dict[str, Any], cost: Any) -> None:
        self.model_calls += 1
        self.input_tokens += event["input_tokens"] or 0
        self.output_tokens += event["output_tokens"] or 0
        self.cache_hit_tokens += event["cache_hit_tokens"] or 0
        if cost is None:
            self.calls_with_unknown_usage += 1
        else:
            self.cost_usd += float(cost)


@dataclass
class Report:
    groups: dict[str, Group] = field(default_factory=dict)
    messages_measured: int = 0
    stopped_at_token_limit: bool = False
    total_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages_measured": self.messages_measured,
            "stopped_at_token_limit": self.stopped_at_token_limit,
            "total_tokens": self.total_tokens,
            "groups": {name: vars(group) for name, group in sorted(self.groups.items())},
        }


async def measure(
    services: Services,
    messages: tuple[str, ...] = MESSAGES,
    cost_of: Callable[[dict[str, Any]], Any] = lambda event: None,
    token_limit: int = TOKEN_LIMIT,
    run: Callable[[Services, RunRequest], Awaitable[None]] = execute_run,
) -> Report:
    """Run each message once in its own thread. Stops before a message once over the limit."""
    report = Report()
    student = uuid.uuid4()
    original_publish = telemetry.publish
    events: list[dict[str, Any]] = []
    telemetry.publish = events.append
    try:
        for message in messages:
            if report.total_tokens >= token_limit:
                report.stopped_at_token_limit = True
                break
            thread = await services.runs.create_thread(student, "measure")
            run_id = await services.runs.create_run(thread.id)
            events.clear()
            await run(services, RunRequest(run_id, thread.id, student, message))
            info = await services.runs.get_run(student, run_id)
            name = f"{info.route}/{info.task_kind or 'none'}"
            group = report.groups.setdefault(name, Group())
            group.runs += 1
            for event in events:
                group.add(event, cost_of(event))
                report.total_tokens += (event["input_tokens"] or 0) + (event["output_tokens"] or 0)
            report.messages_measured += 1
    finally:
        telemetry.publish = original_publish
    return report


def memory_services(get_model: Callable[..., Any] = get_chat_model) -> Services:
    return Services(
        admin=None,
        accounts=InMemoryAccountStore(),
        limiter=InMemoryLoginLimiter(),
        runs=InMemoryRunStore(),
        study=InMemoryStudyStore(),
        events=InMemoryEventBus(),
        store=InMemoryStore(),
        checkpointer=InMemorySaver(),
        intent=IntentStep(lambda tier, c, m: get_model(tier, component=c, max_tokens=m)),
        get_model=get_model,
        budgets=InMemoryBudgetLedger(),
    )


async def _measure_live(db: psycopg.Connection) -> Report:
    settings = get_settings()
    engine = create_async_engine(sqlalchemy_url(settings.database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    search = MaterialsSearch(
        sessions, PostgresMaterialsStore(sessions), embedder, InMemorySearchCache()
    )
    try:
        services = dataclasses.replace(memory_services(), search=search)
        return await measure(services, cost_of=lambda event: calculate_cost(event, db))
    finally:
        await engine.dispose()


def main() -> None:
    settings = get_settings()
    if not settings.live_agent_runs_enabled:
        sys.exit("LIVE_AGENT_RUNS_ENABLED is off, so there's nothing to measure.")
    with psycopg.connect(settings.database_url) as db:
        report = asyncio.run(_measure_live(db))
    Path(sys.argv[1]).write_text(json.dumps(report.as_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
