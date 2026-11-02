"""Runs imports in the background, one at a time, so they never hold up a chat."""

import asyncio
import hashlib
import logging
import uuid
from pathlib import PurePath

from owl_mind.materials.chunking import chunk_sections, looks_like_solution
from owl_mind.materials.embedding import Embedder
from owl_mind.materials.extract import ExtractionError, extract_isolated, file_format
from owl_mind.materials.store import DuplicateDocument, MaterialsStore

logger = logging.getLogger(__name__)

SOLUTION_REFUSED = "This looks like a solution or answer key, which can't be imported."


class Ingestor:
    def __init__(self, store: MaterialsStore, embedder: Embedder, timeout_seconds: float) -> None:
        self.store = store
        self._embedder = embedder
        self._timeout = timeout_seconds
        self._one_at_a_time = asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task] = set()

    async def submit(
        self, filename: str, data: bytes, uploaded_by: uuid.UUID, source_url: str | None = None
    ) -> uuid.UUID:
        job_id = await self.store.create_job(filename, uploaded_by)
        task = asyncio.create_task(self._run(job_id, filename, data, uploaded_by, source_url))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def wait(self) -> None:
        """Let every queued import finish. Used by tests and scripts."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run(self, job_id, filename, data, uploaded_by, source_url) -> None:
        async with self._one_at_a_time:
            try:
                await self.store.update_job(job_id, "running")
                document_id = await self._import(filename, data, uploaded_by, source_url)
            except ExtractionError as exc:
                await self.store.update_job(job_id, "failed", str(exc))
            except asyncio.CancelledError:
                await self.store.update_job(job_id, "failed", "The app stopped during this import.")
                raise
            except Exception as exc:
                logger.warning("Import failed: %s", type(exc).__name__)
                await self.store.update_job(job_id, "failed", "The import failed. Try again.")
            else:
                await self.store.update_job(job_id, "done", document_id=document_id)

    async def _import(self, filename, data, uploaded_by, source_url) -> uuid.UUID:
        file_type = file_format(filename)
        content_hash = hashlib.sha256(data).hexdigest()
        if existing := await self.store.find_by_hash(content_hash):
            return existing
        sections = await asyncio.to_thread(extract_isolated, data, file_type, self._timeout)
        if looks_like_solution(filename, sections):
            raise ExtractionError(SOLUTION_REFUSED)
        chunks = chunk_sections(sections)
        vectors = await asyncio.to_thread(self._embedder.embed, [chunk.text for chunk in chunks])
        title = PurePath(filename).stem.replace("_", " ")
        try:
            return await self.store.save_document(
                title, file_type, content_hash, source_url, uploaded_by, chunks, vectors
            )
        except DuplicateDocument as duplicate:
            return duplicate.document_id
