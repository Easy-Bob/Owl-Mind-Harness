"""Offline: long-term memory is written once per changed chat, kept private, and readable."""

import asyncio
import json
import uuid

import pytest
from langgraph.store.memory import InMemoryStore

from owl_mind.core.budget import InMemoryBudgetLedger
from owl_mind.core.config import get_settings
from owl_mind.core.intent.taxonomy import Subagent, TaskKind
from owl_mind.services.admin_store import safe_event
from owl_mind.services.memory import (
    COMPONENT,
    PROMPT,
    DueChat,
    InMemoryMemoryLog,
    Memory,
    MemoryWriter,
    clean,
    namespace,
    profile_line,
)
from tests.test_factory import Run
from tests.test_factory import settings as factory_settings  # noqa: F401  Applied to all.
from tests.test_filesystem import ALICE, BOB, call, say

EMAIL = "jordan.lee@school.edu"
HISTORY = [
    {"role": "user", "content": "explain how a bloom filter decides membership please"},
    {"role": "assistant", "content": "A Bloom filter sets k bits per item..."},
    {"role": "user", "content": "i keep mixing up false positives and false negatives honestly"},
    {"role": "assistant", "content": "A false positive means..."},
]
REPLY = {
    "profile": {"strengths": ["Hash functions"], "weak_topics": ["False positives"], "style": ""},
    "episode": "Studied Bloom filters and their error types. Confused positives and negatives.",
}
VOCAB = ["bloom", "filter", "hash", "markov", "chain", "heap", "positive", "graph"]


def toy_embed(texts: list[str]) -> list[list[float]]:
    """Counts of a few words, so texts about the same topic land close together."""
    vectors = []
    for text in texts:
        words = text.lower().split()
        vectors.append([float(sum(w.startswith(v) for w in words)) + 0.01 for v in VOCAB])
    return vectors


class FakeModel:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list = []

    def __call__(self, tier, *, component, max_tokens):
        assert (tier, component, max_tokens) == ("small", COMPONENT, 400)
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append((messages, config))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return type("Reply", (), {"text": json.dumps(reply)})()


class Setup:
    def __init__(self, *replies, history=HISTORY, store=None) -> None:
        self.store = store or InMemoryStore()
        self.log = InMemoryMemoryLog()
        self.model = FakeModel(list(replies))
        self.budgets = InMemoryBudgetLedger()
        self.history = {}
        self.thread = uuid.uuid4()
        self.history[self.thread] = list(history)
        self.events: list = []
        self.chat = DueChat(self.thread, ALICE, EMAIL, uuid.uuid4())
        self.log.due_chats = [self.chat]

        async def read(thread_id):
            return self.history[thread_id]

        async def record(*args):
            self.events.append(args)

        self.writer = MemoryWriter(
            Memory(self.store), self.log, read, self.model, self.budgets, record=record
        )

    def run(self):
        return asyncio.run(self.writer.run_due())


# -- writing -------------------------------------------------------------------------


def test_an_idle_changed_chat_is_written_once_and_an_unchanged_one_costs_nothing():
    setup = Setup(REPLY)

    assert setup.run() == {"written": 1}
    assert setup.run() == {"unchanged": 1}
    assert len(setup.model.calls) == 1
    memory = Memory(setup.store)
    assert asyncio.run(memory.profile(ALICE))["weak_topics"] == ["False positives"]
    [episode] = asyncio.run(memory.episodes(ALICE))
    assert episode.summary.startswith("Studied Bloom filters")
    # An unchanged chat records nothing, so the timeline isn't filled with checks.
    assert [event[2:] for event in setup.events] == [("memory", {"step": "written", "episodes": 1})]


def test_new_turns_after_a_write_are_written_again_with_the_profile_so_far():
    setup = Setup(REPLY, REPLY)
    setup.run()
    setup.history[setup.thread].append({"role": "user", "content": "now quiz me on heaps"})

    assert setup.run() == {"written": 1}
    messages, config = setup.model.calls[1]
    assert messages[0].content == PROMPT
    assert '"weak_topics": ["False positives"]' in messages[1].content
    assert "<session>" in messages[1].content
    assert config["callbacks"]  # The run meter, so the call counts toward the daily budget.


def test_two_checks_at_once_write_once():
    setup = Setup(REPLY, REPLY)

    async def both():
        return await asyncio.gather(setup.writer.write(setup.chat), setup.writer.write(setup.chat))

    assert sorted(asyncio.run(both())) == ["unchanged", "written"]
    assert len(setup.model.calls) == 1


def test_a_chat_with_nothing_from_the_student_makes_no_call():
    setup = Setup(history=[{"role": "assistant", "content": "Hello!"}])

    assert setup.run() == {"empty": 1}
    assert setup.model.calls == []


def test_a_student_over_their_daily_budget_gets_no_memory_call():
    setup = Setup(REPLY)
    setup.budgets.students[
        (ALICE, __import__("datetime").datetime.now(__import__("datetime").UTC).date())
    ] = get_settings().student_daily_tokens

    assert setup.run() == {"budget": 1}
    assert setup.model.calls == []
    assert setup.log.hashes == {}  # Tried again on a later check.


def test_a_failed_call_is_retried_on_the_next_check():
    setup = Setup(ConnectionError("down"), REPLY)

    assert setup.run() == {"failed": 1}
    assert setup.run() == {"written": 1}


# -- what is kept -----------------------------------------------------------------------


def test_emails_names_from_the_email_and_quotes_of_the_student_are_never_kept():
    raw = {
        "profile": {"strengths": ["Hashing", "Jordan is good at sorting"], "weak_topics": []},
        "episode": (
            "Reviewed Bloom filters. The student said i keep mixing up false positives and "
            "false negatives honestly. Contact jordan.lee@school.edu for more."
        ),
    }

    profile, episode = clean(raw, " ".join(m["content"] for m in HISTORY), [EMAIL, "jordan"])

    assert "keep mixing up" not in episode
    assert "@" not in episode and "jordan" not in episode.lower()
    assert episode.startswith("Reviewed Bloom filters.")
    assert all("jordan" not in s.lower() for s in profile["strengths"])


def test_long_profile_items_are_cut_at_a_word():
    long = "distinguishing false positives from false negatives in Bloom filters and more"
    profile, _ = clean({"profile": {"weak_topics": [long]}, "episode": ""}, "", [])

    [item] = profile["weak_topics"]
    assert len(item) <= 60 and long.startswith(item) and long[len(item)] == " "


def test_a_malformed_reply_keeps_nothing_odd():
    profile, episode = clean({"profile": "nope", "episode": 7}, "", [])

    assert profile == {"strengths": [], "weak_topics": [], "style": ""}
    assert episode == ""


# -- reading, privacy and deleting ---------------------------------------------------------


def save(store, student, episode, profile=None):
    asyncio.run(Memory(store).save(student, uuid.uuid4(), profile or REPLY["profile"], episode))


def test_one_students_memory_is_never_found_for_another():
    store = InMemoryStore(index={"dims": len(VOCAB), "embed": toy_embed, "fields": ["content"]})
    save(store, ALICE, "Bloom filter false positive practice.")

    assert asyncio.run(Memory(store).search(BOB, "bloom filter")) == []
    assert asyncio.run(Memory(store).episodes(BOB)) == []
    assert asyncio.run(Memory(store).profile(BOB)) is None


def test_past_sessions_are_found_by_meaning():
    store = InMemoryStore(index={"dims": len(VOCAB), "embed": toy_embed, "fields": ["content"]})
    save(store, ALICE, "Worked through Markov chain transition matrices.")
    save(store, ALICE, "Bloom filter false positive rates with k hash functions.")
    save(store, ALICE, "Heap insertion and sift down.")

    [first, *_] = asyncio.run(Memory(store).search(ALICE, "bloom filter hash"))

    assert first.summary.startswith("Bloom filter")


def test_memory_is_stored_as_files_an_agent_can_read():
    store = InMemoryStore()
    save(store, ALICE, "Studied heaps.")

    profile = asyncio.run(store.aget(namespace(ALICE), "/profile.md"))
    assert profile.value["content"].startswith("# Student profile\n- Strengths: Hash functions")
    assert profile.value["encoding"] == "utf-8"


def test_deleting_removes_the_profile_and_every_episode():
    store = InMemoryStore()
    save(store, ALICE, "One.")
    save(store, ALICE, "Two.")
    save(store, BOB, "Bob's.")

    assert asyncio.run(Memory(store).delete_all(ALICE)) == 3
    assert asyncio.run(store.asearch(namespace(ALICE))) == []
    assert len(asyncio.run(Memory(store).episodes(BOB))) == 1


def test_the_profile_becomes_one_short_line():
    line = profile_line({"strengths": ["Hashing"], "weak_topics": ["Chernoff bounds"], "style": ""})

    assert line == "struggles with Chernoff bounds; handles Hashing well"
    assert profile_line(None) is None
    assert profile_line({"strengths": [], "weak_topics": [], "style": ""}) is None


# -- inside the agent ----------------------------------------------------------------------


def memory_run(*replies) -> Run:
    run = Run(*replies)
    run.memory_store = InMemoryStore()
    save(run.memory_store, ALICE, "Bloom filters, false positives.")
    return run


def invoke(run: Run, task_kind, subagents):
    from langchain_core.messages import HumanMessage

    from owl_mind.core.factory import build_owl_agent
    from owl_mind.tools.context import RunContext
    from tests.test_factory import TODAY

    agent = build_owl_agent(
        task_kind, subagents, store=run.memory_store, meter=run.meter, get_model=run.get_model
    )
    context = RunContext(student_id=ALICE, store=run.store, today=TODAY)
    return asyncio.run(agent.ainvoke({"messages": [HumanMessage("help")]}, context=context))


def test_every_agent_sees_the_students_profile_in_its_context_block():
    run = memory_run(say("final"))

    invoke(run, TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    system = run.fake.requests[0]["messages"][0]["content"]
    system = system if isinstance(system, str) else json.dumps(system)
    assert "- This student: struggles with False positives; handles Hash functions well" in system


def test_the_orchestrator_can_search_past_sessions():
    run = memory_run(call("search_memory", query="bloom filters"), say("final"))

    invoke(run, TaskKind.LEARN, [Subagent.CONCEPT_TUTOR])

    [result] = [m["content"] for m in run.fake.requests[1]["messages"] if m["role"] == "tool"]
    assert "Bloom filters, false positives." in result


# -- routes and Admin ------------------------------------------------------------------------


@pytest.fixture
def world():
    from tests.test_runtime import World

    return World()


def test_a_student_can_see_and_delete_their_memory(world, monkeypatch):
    from tests.test_runtime import settings as runtime_settings  # noqa: F401

    save(world.services.store, ALICE, "Studied heaps.")
    with world.client() as client:
        shown = client.get("/memory").json()
        deleted = client.delete("/memory")
        after = client.get("/memory").json()

    assert shown["profile"]["weak_topics"] == ["False positives"]
    assert [e["summary"] for e in shown["episodes"]] == ["Studied heaps."]
    assert deleted.status_code == 204
    assert after == {"profile": None, "episodes": []}


def test_admin_sees_only_the_step_and_count():
    payload = {"step": "written", "episodes": 3, "summary": "Studied heaps."}

    assert safe_event("memory", payload) == {"step": "written", "episodes": 3}
