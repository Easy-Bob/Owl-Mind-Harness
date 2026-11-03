"""Needs the compose services: docker compose up -d --wait

Search on real Postgres with the real embedding model. Every test adds its own documents with
unique words, so the course materials in a local database can't decide the outcome.
"""

import asyncio
import functools
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.materials.chunking import Chunk
from owl_mind.materials.embedding import FastEmbedEmbedder
from owl_mind.materials.search import (
    MAX_PER_DOCUMENT,
    MAX_RESULT_CHARS,
    MAX_RESULTS,
    InMemorySearchCache,
    MaterialsSearch,
)
from owl_mind.materials.store import PostgresMaterialsStore
from owl_mind.services.models import sqlalchemy_url

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOADER = uuid.UUID(int=7)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def own_documents_only():
    """Each test starts and ends with none of the test uploader's documents, so earlier runs
    can't crowd results, and a local database keeps only real course files."""

    def delete():
        with psycopg.connect(get_settings().database_url) as connection:
            connection.execute(
                "DELETE FROM materials.documents WHERE uploaded_by = %s", (UPLOADER,)
            )

    delete()
    yield
    delete()


class CountingEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.real = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return self.real.embed(texts)


@functools.cache
def embedder() -> CountingEmbedder:
    """Created on first use, so importing this file needs no settings or model."""
    return CountingEmbedder()


def scenario(body):
    async def main():
        engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = PostgresMaterialsStore(sessions)
        search = MaterialsSearch(sessions, store, embedder(), InMemorySearchCache())
        try:
            return await body(store, search)
        finally:
            await engine.dispose()

    return asyncio.run(main())


async def add(store, *texts: str, title: str = "Test notes") -> list[str]:
    chunks = [Chunk(i, f"page {i + 1}", title, text) for i, text in enumerate(texts)]
    document = await store.save_document(
        title,
        ".md",
        uuid.uuid4().hex * 2,
        None,
        UPLOADER,
        chunks,
        embedder().real.embed(list(texts)),
    )
    return [document]


def keyword_matches(sessions, query: str) -> set[str]:
    """Chunks the exact-word ranking alone finds, with no help from search by meaning."""

    async def run():
        async with sessions() as session:
            rows = await session.execute(
                text(
                    "SELECT text FROM materials.chunks"
                    " WHERE search_text @@ websearch_to_tsquery('english', :query)"
                ),
                {"query": query},
            )
            return {row[0] for row in rows}

    return run()


@pytest.mark.parametrize("searched", ["whole", "module", "function"])
def test_each_part_of_a_dotted_name_is_found_by_keyword(searched):
    module, function = f"zqx{uuid.uuid4().hex[:8]}", f"push{uuid.uuid4().hex[:8]}"
    query = {"whole": f"{module}.{function}", "module": module, "function": function}[searched]
    passage = f"Call {module}.{function}(heap, item) to add an item to the heap."

    async def body(store, search):
        await add(store, passage)
        engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
        try:
            found = await keyword_matches(async_sessionmaker(engine), query)
        finally:
            await engine.dispose()
        return found, await search.search(query, limit=MAX_RESULTS)

    found, results = scenario(body)

    assert found == {passage}
    assert any(passage == r.text for r in results)


def test_text_with_no_dotted_names_is_indexed_as_before():
    marker = f"zqx{uuid.uuid4().hex[:8]}"
    passage = f"A {marker} heap keeps its smallest item at the root, with heap_push to add."

    async def body(store, search):
        await add(store, passage)
        engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
        try:
            sessions = async_sessionmaker(engine)
            return [
                await keyword_matches(sessions, f"{marker} {word}")
                for word in ("root", "push", "heaps")
            ]
        finally:
            await engine.dispose()

    assert scenario(body) == [{passage}, {passage}, {passage}]


def test_a_paraphrase_is_found_by_meaning_with_no_shared_words():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        await add(
            store, f"Plants convert sunlight into chemical energy inside chloroplasts. {marker}"
        )
        return await search.search("how do leaves make food from light", limit=MAX_RESULTS)

    assert any(marker in r.text for r in scenario(body))


def test_results_carry_ids_that_resolve_and_made_up_ids_do_not():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        await add(store, f"Reservoir sampling keeps a uniform sample of a stream {marker}.")
        results = await search.search(f"reservoir sampling {marker}", limit=3)
        ids = [r.citation_id for r in results]
        return results, await search.resolve([*ids, str(uuid.uuid4()), "not-an-id"])

    results, resolved = scenario(body)
    mine = next(r for r in results if marker in r.text)
    assert resolved[mine.citation_id].location == "page 1"
    assert list(resolved.values())[-2:] == [None, None]


def test_a_repeated_query_is_answered_from_the_cache_without_embedding():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        await add(store, f"Cuckoo hashing moves items between two tables {marker}.")
        first = await search.search(f"cuckoo {marker}")
        calls = embedder().calls
        second = await search.search(f"  Cuckoo   {marker} ")
        return first, second, embedder().calls - calls

    first, second, extra_calls = scenario(body)
    assert first == second
    assert extra_calls == 0


def test_a_new_import_misses_the_cache():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        await add(store, f"Linear counting estimates distinct items {marker}.")
        await search.search(f"linear counting {marker}")
        calls = embedder().calls
        await add(store, f"Another note about linear counting {marker}.")
        after = await search.search(f"linear counting {marker}", limit=MAX_RESULTS)
        return embedder().calls - calls, after

    extra_calls, after = scenario(body)
    assert extra_calls == 1
    assert sum(marker in r.text for r in after) == 2


def test_result_count_and_text_size_are_capped():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        for _ in range(4):
            await add(store, f"{marker} skip list " + "levels " * 400)
        return await search.search(f"{marker} skip list", limit=500)

    results = scenario(body)
    assert len(results) <= MAX_RESULTS
    assert all(len(r.text) <= MAX_RESULT_CHARS for r in results)


def test_one_document_cannot_crowd_out_the_rest():
    marker = uuid.uuid4().hex[:8]

    async def body(store, search):
        await add(store, *[f"{marker} treap rotation detail {i}" for i in range(8)])
        await add(store, f"{marker} treap overview", title="Other notes")
        return await search.search(f"{marker} treap", limit=MAX_RESULTS)

    results = scenario(body)
    mine = [r for r in results if marker in r.text]
    assert sum(r.title == "Test notes" for r in mine) <= MAX_PER_DOCUMENT
    assert any(r.title == "Other notes" for r in mine)


def test_a_blank_query_returns_nothing_and_embeds_nothing():
    async def body(store, search):
        calls = embedder().calls
        return await search.search("   "), embedder().calls - calls

    assert scenario(body) == ([], 0)


def test_stopwords_alone_still_search_by_meaning():
    async def body(store, search):
        return await search.search("the and of it")

    assert isinstance(scenario(body), list)
