"""Offline: each imported file's topics are named once, and the list follows the materials."""

import asyncio
import json
import uuid

from owl_mind.materials.ingest import Ingestor
from owl_mind.materials.store import InMemoryMaterialsStore
from owl_mind.materials.topics import PROMPT, TopicNamer, backfill, clean
from tests.test_materials import UPLOADER, FakeEmbedder

NOTES = b"# Bloom filters\n\nA Bloom filter is a bit array.\n\n# Chernoff bounds\n\nTail bounds."


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def __call__(self, tier, *, component, max_tokens):
        assert (tier, component, max_tokens) == ("small", "materials:topics", 120)
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise ConnectionError("no reply")
        return type("Reply", (), {"text": self.replies.pop(0)})()


def topics_reply(*topics: str) -> str:
    return json.dumps({"topics": list(topics)})


def imported(model: FakeModel, *files: tuple[str, bytes]):
    store = InMemoryMaterialsStore()
    ingestor = Ingestor(store, FakeEmbedder(), 30, TopicNamer(model))

    async def scenario():
        jobs = [await ingestor.submit(name, data, UPLOADER) for name, data in files]
        await ingestor.wait()
        return [store.jobs[job] for job in jobs]

    return store, asyncio.run(scenario())


def test_an_import_names_its_topics_with_one_call_and_reading_them_costs_none():
    model = FakeModel([topics_reply("Bloom filters", "Chernoff bounds")])
    store, [job] = imported(model, ("week3.md", NOTES))

    assert job.status == "done"
    assert len(model.calls) == 1
    system, user = model.calls[0]
    assert system.content == PROMPT
    assert user.content.startswith("<material>\nBloom filters\nChernoff bounds")
    for _ in range(3):
        assert asyncio.run(store.list_topics()) == ["Bloom filters", "Chernoff bounds"]
    assert len(model.calls) == 1


def test_the_list_merges_files_and_drops_a_deleted_files_topics():
    model = FakeModel(
        [topics_reply("Bloom filters", "Hashing"), topics_reply("hashing", "Markov chains")]
    )
    store, jobs = imported(model, ("a.md", NOTES), ("b.md", NOTES + b"\n\n# Markov chains\n"))

    assert asyncio.run(store.list_topics()) == ["Hashing", "Bloom filters", "Markov chains"]
    asyncio.run(store.delete_document(jobs[1].document_id))
    assert asyncio.run(store.list_topics()) == ["Bloom filters", "Hashing"]


def test_a_failed_naming_call_still_imports_the_file():
    store, [job] = imported(FakeModel([]), ("week3.md", NOTES))

    assert job.status == "done"
    assert asyncio.run(store.list_topics()) == []


def test_a_reimported_file_is_not_named_again():
    model = FakeModel([topics_reply("Bloom filters")])
    imported_store, _ = imported(model, ("week3.md", NOTES), ("copy.md", NOTES))

    assert len(model.calls) == 1


def test_the_backfill_names_only_files_without_topics():
    store, _ = imported(FakeModel([]), ("a.md", NOTES), ("b.md", NOTES + b"\n# Tries\n"))
    first = next(iter(store.documents))
    asyncio.run(store.set_topics(first, ["Already named"]))
    model = FakeModel([topics_reply("Tries")])

    named = asyncio.run(backfill(store, TopicNamer(model)))

    assert named == 1 and len(model.calls) == 1
    assert sorted(asyncio.run(store.list_topics())) == ["Already named", "Tries"]


def test_unsafe_or_long_model_text_is_dropped():
    raw = [
        "Bloom filters",
        "bloom filters",
        "x" * 41,
        "see http://evil.example",
        "<script>",
        42,
        "Count-min sketch",
        "A",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
    ]

    assert clean(raw) == ["Bloom filters", "Count-min sketch", "One", "Two", "Three", "Four"]
    assert clean("not a list") == []


def test_topics_of_an_unknown_document_are_ignored():
    store = InMemoryMaterialsStore()
    asyncio.run(store.set_topics(uuid.uuid4(), ["Nothing"]))

    assert asyncio.run(store.list_topics()) == []
