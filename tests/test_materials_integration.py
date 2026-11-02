"""Needs the compose services: docker compose up -d --wait

The same store checks as the in-memory version, plus the real embedding model.
"""

import asyncio
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from owl_mind.core.config import get_settings
from owl_mind.materials.chunking import Chunk
from owl_mind.materials.embedding import EMBEDDING_DIMS, FastEmbedEmbedder
from owl_mind.materials.ingest import Ingestor
from owl_mind.materials.store import (
    DuplicateDocument,
    InMemoryMaterialsStore,
    PostgresMaterialsStore,
)
from owl_mind.services.models import sqlalchemy_url
from tests import material_fixtures as fixtures
from tests.persistence_probe import fake_embed

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


@asynccontextmanager
async def postgres_store():
    engine = create_async_engine(sqlalchemy_url(get_settings().database_url))
    yield PostgresMaterialsStore(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


@asynccontextmanager
async def memory_store():
    yield InMemoryMaterialsStore()


@pytest.fixture(params=[memory_store, postgres_store], ids=["memory", "postgres"])
def store(request):
    return request.param


def chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(index, f"page {index + 1}", "Heading", text) for index, text in enumerate(texts)]


def unique_hash() -> str:
    return uuid.uuid4().hex * 2


def test_save_find_delete_and_versions(store):
    content_hash = unique_hash()
    texts = ["Bloom filters", "Count-min sketch"]

    async def body():
        async with store() as materials:
            before = await materials.corpus_version()
            document = await materials.save_document(
                "Lecture 1",
                ".pdf",
                content_hash,
                "https://example.edu/a.pdf",
                UPLOADER,
                chunks(texts),
                fake_embed(texts),
            )
            found = await materials.find_by_hash(content_hash)
            with pytest.raises(DuplicateDocument):
                await materials.save_document(
                    "Again", ".pdf", content_hash, None, UPLOADER, chunks(texts), fake_embed(texts)
                )
            middle = await materials.corpus_version()
            listed = [d for d in await materials.list_documents() if d.id == document]
            deleted = await materials.delete_document(document)
            again = await materials.delete_document(document)
            return (
                before,
                document,
                found,
                middle,
                listed,
                deleted,
                again,
                await materials.corpus_version(),
            )

    before, document, found, middle, listed, deleted, again, after = asyncio.run(body())
    assert found == document
    assert middle == before + 1
    assert listed[0].chunk_count == 2
    assert (deleted, again) == (True, False)
    assert after == before + 2


def test_jobs_round_trip_and_unfinished_ones_fail(store):
    async def body():
        async with store() as materials:
            job = await materials.create_job("a.pdf", UPLOADER)
            await materials.update_job(job, "running")
            await materials.fail_unfinished_jobs()
            failed = await materials.get_job(job)
            done = await materials.create_job("b.pdf", UPLOADER)
            await materials.update_job(done, "done", document_id=job)
            return failed, await materials.get_job(done), await materials.get_job(uuid.uuid4())

    failed, done, missing = asyncio.run(body())
    assert (failed.status, failed.reason) == ("failed", "The app restarted during this import.")
    assert (done.status, done.filename) == ("done", "b.pdf")
    assert missing is None


def rows(sql: str, params) -> list[tuple]:
    with psycopg.connect(get_settings().database_url) as db:
        return db.execute(sql, params).fetchall()


def test_postgres_stores_vectors_and_searchable_text_and_deletes_chunks_with_documents():
    content_hash = unique_hash()
    texts = ["Bloom filters have false positives", "Count-min sketch estimates frequency"]

    async def save():
        async with postgres_store() as materials:
            return await materials.save_document(
                "L", ".pdf", content_hash, None, UPLOADER, chunks(texts), fake_embed(texts)
            )

    document = asyncio.run(save())
    found = rows(
        "SELECT position, vector_dims(embedding) FROM materials.chunks"
        " WHERE document_id = %s AND search_text @@ plainto_tsquery('english', 'frequencies')",
        (document,),
    )
    assert found == [(1, EMBEDDING_DIMS)]

    async def delete():
        async with postgres_store() as materials:
            await materials.delete_document(document)

    asyncio.run(delete())
    assert rows("SELECT count(*) FROM materials.chunks WHERE document_id = %s", (document,)) == [
        (0,)
    ]


def test_a_failure_part_way_leaves_no_document_or_chunks():
    content_hash = unique_hash()

    async def body():
        async with postgres_store() as materials:
            # One vector for two chunks makes the chunk insert fail after the document insert.
            with pytest.raises(ValueError):
                await materials.save_document(
                    "L", ".pdf", content_hash, None, UPLOADER, chunks(["a", "b"]), fake_embed(["a"])
                )
            return await materials.find_by_hash(content_hash)

    assert asyncio.run(body()) is None


def test_text_with_nul_characters_imports_into_postgres():
    async def body():
        async with postgres_store() as materials:
            ingestor = Ingestor(materials, type("E", (), {"embed": staticmethod(fake_embed)})(), 30)
            data = f"# Bounds\n\nMarkov\x00 inequality {uuid.uuid4()}".encode()
            job = await ingestor.submit("bounds.md", data, UPLOADER)
            await ingestor.wait()
            return await materials.get_job(job)

    job = asyncio.run(body())
    assert job.status == "done", job.reason
    text = rows("SELECT text FROM materials.chunks WHERE document_id = %s", (job.document_id,))
    assert text[0][0].startswith("Markov inequality")


def test_real_model_imports_a_file_into_postgres(tmp_path):
    """Downloads the embedding model on first run, about 90 MB."""
    settings = get_settings()
    data = fixtures.pptx(
        [("Bloom filters", f"False positives, never false negatives {uuid.uuid4()}")]
    )

    async def body():
        async with postgres_store() as materials:
            ingestor = Ingestor(
                materials,
                FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir),
                timeout_seconds=60,
            )
            job = await ingestor.submit("Lecture_9.pptx", data, UPLOADER)
            await ingestor.wait()
            return await materials.get_job(job)

    job = asyncio.run(body())
    assert job.status == "done", job.reason
    near = rows(
        "SELECT c.location FROM materials.chunks c WHERE c.document_id = %s"
        " ORDER BY c.embedding <=> (SELECT embedding FROM materials.chunks"
        " WHERE document_id = %s AND location = 'slide 1') LIMIT 1",
        (job.document_id, job.document_id),
    )
    assert near == [("slide 1",)]
