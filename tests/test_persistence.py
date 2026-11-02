"""Offline: the same graph and store calls the app makes, on the in-memory backends."""

import asyncio

from owl_mind.core.persistence import in_memory_persistence
from owl_mind.services.models import Base
from tests.thread_graph import build_graph, config


def test_in_memory_saver_keeps_thread_messages():
    async def scenario():
        graph = build_graph(in_memory_persistence().checkpointer)
        await graph.ainvoke({"messages": [("user", "hello")]}, config("t1"))
        return await graph.aget_state(config("t1"))

    state = asyncio.run(scenario())

    assert [message.content for message in state.values["messages"]] == ["hello", "reply 1"]


def test_unknown_thread_is_empty():
    async def scenario():
        graph = build_graph(in_memory_persistence().checkpointer)
        return await graph.aget_state(config("missing"))

    assert asyncio.run(scenario()).values == {}


def test_in_memory_store_keeps_values():
    async def scenario():
        store = in_memory_persistence().store
        await store.aput(("memories", "s1"), "profile", {"text": "likes graphs"})
        return await store.aget(("memories", "s1"), "profile")

    assert asyncio.run(scenario()).value == {"text": "likes graphs"}


def test_app_tables_are_in_the_app_schema():
    assert {"app.students", "app.threads", "app.runs"} <= set(Base.metadata.tables)
    assert all(table.schema == "app" for table in Base.metadata.tables.values())
