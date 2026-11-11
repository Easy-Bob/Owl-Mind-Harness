"""Needs the compose services: docker compose up -d --wait

Student isolation and /scratch/ against real Postgres, because a namespace mistake there would
show one student's memory to another.
"""

import asyncio
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from owl_mind.core import telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.factory import build_owl_agent
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.model import get_chat_model
from owl_mind.core.persistence import open_persistence
from owl_mind.services.study_store import InMemoryStudyStore
from owl_mind.tools.context import RunContext
from tests.budget_helpers import meter
from tests.fake_deepseek import FakeDeepSeek
from tests.persistence_probe import fake_embed
from tests.test_filesystem import TODAY, call, memory_item, say, task

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    monkeypatch.setattr(telemetry, "publish", lambda event: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def run_agent(persistence, subagents, replies, student, thread) -> list[str]:
    fake = FakeDeepSeek(replies)

    def get_model(tier, *, component, max_tokens):
        model = get_chat_model(tier, component=component, max_tokens=max_tokens)
        fake.attach(model)
        return model

    agent = build_owl_agent(
        TaskKind.LEARN,
        subagents,
        store=persistence.store,
        meter=meter(student),
        checkpointer=persistence.checkpointer,
        get_model=get_model,
    )
    await agent.ainvoke(
        {"messages": [HumanMessage("help")]},
        config={"configurable": {"thread_id": thread}},
        context=RunContext(student_id=student, store=InMemoryStudyStore(), today=TODAY),
    )
    requests = fake.agent_requests
    index = len(requests) - (2 if subagents else 1)
    return [m["content"] for m in requests[index]["messages"] if m["role"] == "tool"]


def with_persistence(body):
    async def main():
        async with open_persistence(get_settings().database_url, fake_embed) as persistence:
            return await body(persistence)

    return asyncio.run(main())


def test_postgres_store_keeps_students_apart():
    alice, bob = uuid.uuid4(), uuid.uuid4()
    secret = f"secret-{alice}"

    async def body(persistence):
        await persistence.store.aput(
            ("memories", str(alice)), "/profile.md", memory_item(secret), index=False
        )
        read = call("read_file", file_path="/memories/profile.md")
        replies = [task(Subagent.STUDY_PLANNER.value), read, say(), say()]
        own = await run_agent(persistence, [Subagent.STUDY_PLANNER], replies, alice, str(alice))
        other = await run_agent(persistence, [Subagent.STUDY_PLANNER], replies, bob, str(bob))
        return own, other

    own, other = with_persistence(body)

    assert secret in own[0]
    assert secret not in other[0]


def test_scratch_survives_a_restart_in_postgres():
    student, thread = uuid.uuid4(), str(uuid.uuid4())

    async def write(persistence):
        step = call("write_file", file_path="/scratch/draft.md", content="kept-text")
        await run_agent(persistence, [], [step, say()], student, thread)

    async def read(persistence):
        step = call("read_file", file_path="/scratch/draft.md")
        return await run_agent(persistence, [], [step, say()], student, thread)

    with_persistence(write)
    results = with_persistence(read)

    assert "kept-text" in results[-1]
