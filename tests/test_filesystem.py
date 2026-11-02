"""Offline: every agent and path through the real file tools, with only the network replaced."""

import asyncio
import json
import os
import uuid
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from owl_mind.backends.composite import (
    MATERIALS_DIR,
    SKILLS_DIR,
    ReadOnlyFilesystemBackend,
    build_backend,
)
from owl_mind.backends.permissions import ORCHESTRATOR_PERMISSIONS, TUTOR_PERMISSIONS
from owl_mind.core import telemetry
from owl_mind.core.config import get_settings
from owl_mind.core.factory import build_owl_agent
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.core.model import get_chat_model
from owl_mind.services.study_store import InMemoryStudyStore
from owl_mind.tools.context import RunContext
from tests.fake_deepseek import FakeDeepSeek

ALICE = uuid.UUID(int=1)
BOB = uuid.UUID(int=2)
TODAY = date(2026, 11, 2)
ORCHESTRATOR = "orchestrator"
DENIED = "permission denied"
SECRET = "alice-private-profile-text"

# Agent -> (paths it may read, paths it may write), as in the M1-C plan's table.
EXPECTED = {
    ORCHESTRATOR: ({"memories", "skills", "scratch"}, {"scratch"}),
    Subagent.CONCEPT_TUTOR: ({"materials", "skills"}, set()),
    Subagent.PRACTICE_COACH: ({"materials", "skills"}, set()),
    Subagent.STUDY_PLANNER: ({"memories", "skills"}, set()),
    Subagent.QUIZ_MASTER: ({"materials", "skills", "scratch/quiz"}, {"scratch/quiz"}),
}
AREAS = ("materials", "memories", "skills", "scratch", "scratch/quiz", "reviews", "other")
AGENTS = list(EXPECTED)


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LIVE_AGENT_RUNS_ENABLED", "true")
    monkeypatch.setattr(telemetry, "publish", lambda event: None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def call(name: str, **arguments) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def task(subagent: str) -> dict:
    return call("task", description="Help", subagent_type=subagent)


def say(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


def memory_item(text: str) -> dict:
    return {"content": text, "encoding": "utf-8", "created_at": "", "modified_at": ""}


class Harness:
    def __init__(self, store=None, checkpointer=None, student=ALICE) -> None:
        self.store = store if store is not None else InMemoryStore()
        self.checkpointer = checkpointer
        self.student = student

    def run(self, agent_name, *steps: dict, thread="t1", run_async=asyncio.run):
        """Run scripted tool calls as one agent. Returns every tool result that agent received."""
        selected = [] if agent_name == ORCHESTRATOR else [agent_name]
        if agent_name == ORCHESTRATOR:
            replies = [*steps, say()]
        else:
            replies = [task(agent_name.value), *steps, say("tutor done"), say()]
        fake = FakeDeepSeek(replies)

        def get_model(tier, *, component, max_tokens):
            model = get_chat_model(tier, component=component, max_tokens=max_tokens)
            fake.attach(model)
            return model

        agent = build_owl_agent(
            TaskKind.LEARN,
            selected,
            store=self.store,
            checkpointer=self.checkpointer,
            get_model=get_model,
        )
        context = RunContext(student_id=self.student, store=InMemoryStudyStore(), today=TODAY)
        config = {"configurable": {"thread_id": thread}}
        run_async(
            agent.ainvoke({"messages": [HumanMessage("help")]}, config=config, context=context)
        )
        # The last request to this agent holds all its tool results.
        index = len(fake.requests) - (1 if agent_name == ORCHESTRATOR else 2)
        self.last_request = fake.requests[index]
        self.first_request = fake.requests[0 if agent_name == ORCHESTRATOR else 1]
        return [m["content"] for m in fake.requests[index]["messages"] if m["role"] == "tool"]


# -- the full permission table ---------------------------------------------


@pytest.mark.parametrize("agent", AGENTS, ids=str)
@pytest.mark.parametrize("area", AREAS)
def test_read_matches_the_table(agent, area):
    results = Harness().run(agent, call("read_file", file_path=f"/{area}/missing.md"))

    readable, _ = EXPECTED[agent]
    allowed = any(area == root or area.startswith(f"{root}/") for root in readable)
    assert (DENIED not in results[0].lower()) is allowed, results[0]


@pytest.mark.parametrize("agent", AGENTS, ids=str)
@pytest.mark.parametrize("area", AREAS)
def test_write_matches_the_table(agent, area):
    path = f"/{area}/new-{uuid.uuid4().hex[:6]}.md"

    results = Harness().run(agent, call("write_file", file_path=path, content="x"))

    _, writable = EXPECTED[agent]
    allowed = any(area == root or area.startswith(f"{root}/") for root in writable)
    assert (DENIED not in results[0].lower()) is allowed, results[0]


@pytest.mark.parametrize("agent", AGENTS, ids=str)
@pytest.mark.parametrize("tool", ["edit_file", "delete"])
def test_changing_tools_are_denied_outside_writable_paths(agent, tool):
    path = "/materials/graph-traversal.md"
    arguments = {"file_path": path}
    if tool == "edit_file":
        arguments |= {"old_string": "Graph", "new_string": "Changed"}

    results = Harness().run(agent, call(tool, **arguments))

    assert DENIED in results[0].lower()
    assert "Graph traversal" in (MATERIALS_DIR / "graph-traversal.md").read_text()


def test_every_rule_list_ends_by_denying_everything():
    for rules in [ORCHESTRATOR_PERMISSIONS, *TUTOR_PERMISSIONS.values()]:
        last = rules[-1]
        assert (last.mode, last.paths, set(last.operations)) == (
            "deny",
            ["/**"],
            {"read", "write"},
        )


def test_every_tutor_has_permissions():
    assert set(TUTOR_PERMISSIONS) == set(Subagent)


# -- reading real files ----------------------------------------------------


def test_tutor_reads_the_seed_notes():
    results = Harness().run(
        Subagent.CONCEPT_TUTOR, call("read_file", file_path="/materials/graph-traversal.md")
    )

    assert "Breadth-first search" in results[0]


def test_tutor_lists_only_what_it_may_see():
    results = Harness().run(Subagent.CONCEPT_TUTOR, call("ls", path="/materials/"))

    assert "graph-traversal.md" in results[0]


@pytest.mark.parametrize(
    "steps",
    [
        call("glob", pattern="**/*", path="/"),
        call("grep", pattern=SECRET, path="/"),
        call("ls", path="/"),
    ],
    ids=["glob", "grep", "ls"],
)
def test_bulk_tools_from_the_root_do_not_reveal_forbidden_files(steps):
    store = InMemoryStore()
    asyncio.run(store.aput(("memories", str(ALICE)), "/profile.md", memory_item(SECRET)))

    results = Harness(store=store).run(Subagent.CONCEPT_TUTOR, steps)

    assert SECRET not in results[0]
    assert "profile.md" not in results[0]


# -- path escape, with permissions and in the backend alone -----------------


@pytest.mark.parametrize(
    "path",
    [
        "/materials/../memories/profile.md",
        "/materials/../../pyproject.toml",
        "/etc/passwd",
        "~/.ssh/id_rsa",
        "../pyproject.toml",
        "/materials/%2e%2e/%2e%2e/pyproject.toml",
    ],
)
def test_tools_refuse_path_escape(path):
    store = InMemoryStore()
    asyncio.run(store.aput(("memories", str(ALICE)), "/profile.md", memory_item(SECRET)))

    results = Harness(store=store).run(Subagent.CONCEPT_TUTOR, call("read_file", file_path=path))

    assert SECRET not in results[0]
    assert "owl-mind-harness" not in results[0]
    assert "root:" not in results[0]


@pytest.mark.parametrize(
    "path",
    [
        "/../pyproject.toml",
        "/../../pyproject.toml",
        "/../../../../../../etc/passwd",
        "~/secret",
    ],
)
def test_backend_refuses_path_escape_with_no_permissions_involved(tmp_path, path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text("outside-the-root")
    backend = ReadOnlyFilesystemBackend(root_dir=root, virtual_mode=True)

    try:
        result = backend.read(path)
    except ValueError:
        return
    assert result.error
    assert "outside-the-root" not in str(result)


def test_symlink_inside_a_read_only_root_cannot_read_outside_it(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside-the-root")
    os.symlink(outside, root / "link.md")
    backend = ReadOnlyFilesystemBackend(root_dir=root, virtual_mode=True)

    with pytest.raises(ValueError, match="outside root"):
        backend.read("/link.md")


# -- read-only backends, with no permissions involved ----------------------


def read_only(tmp_path: Path) -> ReadOnlyFilesystemBackend:
    (tmp_path / "note.md").write_text("original")
    return ReadOnlyFilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def test_read_only_backend_refuses_every_change(tmp_path):
    backend = read_only(tmp_path)

    assert backend.write("/new.md", "x").error
    assert backend.edit("/note.md", "original", "changed").error
    assert backend.delete("/note.md").error
    assert backend.upload_files([("/new.md", b"x")])[0].error
    assert (tmp_path / "note.md").read_text() == "original"
    assert not (tmp_path / "new.md").exists()


def test_read_only_backend_refuses_async_changes_too(tmp_path):
    backend = read_only(tmp_path)

    async def scenario():
        return [
            (await backend.awrite("/new.md", "x")).error,
            (await backend.aedit("/note.md", "original", "changed")).error,
            (await backend.adelete("/note.md")).error,
            (await backend.aupload_files([("/new.md", b"x")]))[0].error,
        ]

    assert all(asyncio.run(scenario()))
    assert (tmp_path / "note.md").read_text() == "original"


def test_read_only_backend_still_reads(tmp_path):
    assert "original" in str(read_only(tmp_path).read("/note.md"))


def test_routes_use_the_read_only_backend():
    routes = build_backend().routes

    assert isinstance(routes["/materials/"], ReadOnlyFilesystemBackend)
    assert isinstance(routes["/skills/"], ReadOnlyFilesystemBackend)


# -- students stay apart ----------------------------------------------------


def test_each_student_sees_only_their_own_memory():
    store = InMemoryStore()
    asyncio.run(store.aput(("memories", str(ALICE)), "/profile.md", memory_item(SECRET)))
    read = call("read_file", file_path="/memories/profile.md")

    alice = Harness(store=store, student=ALICE).run(Subagent.STUDY_PLANNER, read)
    bob = Harness(store=store, student=BOB).run(Subagent.STUDY_PLANNER, read)

    assert SECRET in alice[0]
    assert SECRET not in bob[0]


def test_memory_cannot_be_written_in_m1():
    store = InMemoryStore()

    results = Harness(store=store).run(
        ORCHESTRATOR, call("write_file", file_path="/memories/profile.md", content="invented")
    )

    assert DENIED in results[0].lower()
    assert asyncio.run(store.aget(("memories", str(ALICE)), "/profile.md")) is None


# -- /scratch/ --------------------------------------------------------------


def test_scratch_survives_a_new_agent_on_the_same_thread():
    saver = InMemorySaver()
    harness = Harness(checkpointer=saver)

    harness.run(ORCHESTRATOR, call("write_file", file_path="/scratch/draft.md", content="v1"))
    results = harness.run(ORCHESTRATOR, call("read_file", file_path="/scratch/draft.md"))

    assert "v1" in results[-1]


def test_scratch_is_kept_per_thread():
    saver = InMemorySaver()
    harness = Harness(checkpointer=saver)

    harness.run(
        ORCHESTRATOR, call("write_file", file_path="/scratch/draft.md", content="v1"), thread="a"
    )
    results = harness.run(
        ORCHESTRATOR, call("read_file", file_path="/scratch/draft.md"), thread="b"
    )

    assert "v1" not in results[-1]


def test_writing_a_scratch_file_twice_keeps_the_latest():
    harness = Harness(checkpointer=InMemorySaver())

    harness.run(ORCHESTRATOR, call("write_file", file_path="/scratch/d.md", content="old"))
    harness.run(
        ORCHESTRATOR,
        call("edit_file", file_path="/scratch/d.md", old_string="old", new_string="new"),
    )
    results = harness.run(ORCHESTRATOR, call("read_file", file_path="/scratch/d.md"))

    assert "new" in results[-1]


# -- skills -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["academic-integrity", "socratic-method", "study-planning", "quiz-authoring"]
)
def test_tutor_can_load_each_skill_but_cannot_change_it(name):
    results = Harness().run(
        Subagent.PRACTICE_COACH,
        call("read_file", file_path=f"/skills/{name}/SKILL.md", limit=1000),
        call("write_file", file_path=f"/skills/{name}/SKILL.md", content="changed"),
    )

    assert name.split("-")[0] in results[0].lower()
    assert DENIED in results[1].lower()


def test_skill_body_is_not_in_the_prompt_until_it_is_loaded():
    harness = Harness()

    harness.run(Subagent.PRACTICE_COACH)

    system = json.dumps(harness.first_request["messages"][0]["content"])
    assert "academic-integrity" in system
    assert "Use when a student asks for an answer" in system
    assert "Find the first place their reasoning goes wrong" not in system


def test_missing_file_gives_an_error_the_agent_can_recover_from():
    results = Harness().run(
        Subagent.CONCEPT_TUTOR,
        call("read_file", file_path="/materials/missing.md"),
        call("read_file", file_path="/materials/hash-tables.md"),
    )

    assert "not found" in results[0].lower()
    assert "Load factor" in results[1]


def test_seed_notes_and_skills_exist_and_are_small():
    notes = list(MATERIALS_DIR.glob("*.md"))
    skills = list(SKILLS_DIR.glob("*/SKILL.md"))

    assert len(notes) >= 2
    assert {path.parent.name for path in skills} == {
        "academic-integrity",
        "socratic-method",
        "study-planning",
        "quiz-authoring",
    }
    assert all(len(path.read_text()) < 1500 for path in skills)
