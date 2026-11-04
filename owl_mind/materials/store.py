"""Course materials and import jobs in the materials schema."""

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.materials.chunking import Chunk


class DuplicateDocument(Exception):
    def __init__(self, document_id: uuid.UUID) -> None:
        self.document_id = document_id


@dataclass(frozen=True)
class DocumentInfo:
    id: uuid.UUID
    title: str
    file_type: str
    chunk_count: int
    source_url: str | None


@dataclass(frozen=True)
class JobInfo:
    id: uuid.UUID
    filename: str
    status: str
    reason: str | None = None
    document_id: uuid.UUID | None = None


@runtime_checkable
class MaterialsStore(Protocol):
    async def find_by_hash(self, content_hash: str) -> uuid.UUID | None: ...

    async def save_document(
        self,
        title: str,
        file_type: str,
        content_hash: str,
        source_url: str | None,
        uploaded_by: uuid.UUID,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> uuid.UUID:
        """Save a document and all its chunks at once, and raise the corpus version.

        Raise DuplicateDocument if the same content was saved first.
        """

    async def delete_document(self, document_id: uuid.UUID) -> bool: ...

    async def set_topics(self, document_id: uuid.UUID, topics: list[str]) -> None: ...

    async def list_topics(self, limit: int = 40) -> list[str]:
        """Every document's topics together, most common first. A deleted file's topics go."""

    async def documents_without_topics(self) -> list[tuple[uuid.UUID, list[str], list[str]]]:
        """Each such document's id, headings and opening chunk texts, for the backfill."""

    async def list_documents(self) -> list[DocumentInfo]: ...

    async def corpus_version(self) -> int: ...

    async def create_job(self, filename: str, uploaded_by: uuid.UUID) -> uuid.UUID: ...

    async def update_job(
        self,
        job_id: uuid.UUID,
        status: str,
        reason: str | None = None,
        document_id: uuid.UUID | None = None,
    ) -> None: ...

    async def get_job(self, job_id: uuid.UUID) -> JobInfo | None: ...

    async def fail_unfinished_jobs(self) -> int: ...


class InMemoryMaterialsStore:
    """The same behavior for unit tests, with no database."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, dict[str, Any]] = {}
        self.jobs: dict[uuid.UUID, JobInfo] = {}
        self.version = 0

    async def find_by_hash(self, content_hash: str) -> uuid.UUID | None:
        return next((i for i, d in self.documents.items() if d["hash"] == content_hash), None)

    async def save_document(
        self, title, file_type, content_hash, source_url, uploaded_by, chunks, vectors
    ) -> uuid.UUID:
        existing = await self.find_by_hash(content_hash)
        if existing:
            raise DuplicateDocument(existing)
        document_id = uuid.uuid4()
        self.documents[document_id] = {
            "title": title,
            "type": file_type,
            "hash": content_hash,
            "source_url": source_url,
            "chunks": list(zip(chunks, vectors, strict=True)),
            "topics": [],
        }
        self.version += 1
        return document_id

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        if self.documents.pop(document_id, None) is None:
            return False
        self.version += 1
        return True

    async def set_topics(self, document_id: uuid.UUID, topics: list[str]) -> None:
        if document_id in self.documents:
            self.documents[document_id]["topics"] = list(topics)

    async def list_topics(self, limit: int = 40) -> list[str]:
        return _most_common([d["topics"] for d in self.documents.values()], limit)

    async def documents_without_topics(self) -> list[tuple[uuid.UUID, list[str], list[str]]]:
        return [
            (i, [c.heading for c, _ in d["chunks"]], [c.text for c, _ in d["chunks"][:3]])
            for i, d in self.documents.items()
            if not d["topics"]
        ]

    async def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(i, d["title"], d["type"], len(d["chunks"]), d["source_url"])
            for i, d in self.documents.items()
        ]

    async def corpus_version(self) -> int:
        return self.version

    async def create_job(self, filename: str, uploaded_by: uuid.UUID) -> uuid.UUID:
        job_id = uuid.uuid4()
        self.jobs[job_id] = JobInfo(job_id, filename, "queued")
        return job_id

    async def update_job(self, job_id, status, reason=None, document_id=None) -> None:
        self.jobs[job_id] = replace(
            self.jobs[job_id], status=status, reason=reason, document_id=document_id
        )

    async def get_job(self, job_id: uuid.UUID) -> JobInfo | None:
        return self.jobs.get(job_id)

    async def fail_unfinished_jobs(self) -> int:
        unfinished = [j for j in self.jobs.values() if j.status in ("queued", "running")]
        for job in unfinished:
            await self.update_job(job.id, "failed", "The app restarted during this import.")
        return len(unfinished)


def _most_common(per_document: list[list[str]], limit: int) -> list[str]:
    """Merge topics case-insensitively, keeping the first spelling seen."""
    counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for topics in per_document:
        for topic in dict.fromkeys(t.lower() for t in topics):
            counts[topic] = counts.get(topic, 0) + 1
        for topic in topics:
            spelling.setdefault(topic.lower(), topic)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [spelling[key] for key in ranked[:limit]]


_BUMP_VERSION = """
INSERT INTO materials.corpus (id, version) VALUES (1, 1)
ON CONFLICT (id) DO UPDATE SET version = materials.corpus.version + 1
"""


def _vector(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in values) + "]"


class PostgresMaterialsStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def find_by_hash(self, content_hash: str) -> uuid.UUID | None:
        async with self._sessions() as session:
            return await session.scalar(
                text("SELECT id FROM materials.documents WHERE content_hash = :hash"),
                {"hash": content_hash},
            )

    async def save_document(
        self, title, file_type, content_hash, source_url, uploaded_by, chunks, vectors
    ) -> uuid.UUID:
        document_id = uuid.uuid4()
        # One transaction: a failure part way leaves no document and no chunks.
        async with self._sessions.begin() as session:
            inserted = await session.scalar(
                text(
                    "INSERT INTO materials.documents"
                    " (id, title, file_type, content_hash, source_url, uploaded_by, chunk_count)"
                    " VALUES (:id, :title, :type, :hash, :url, :by, :count)"
                    " ON CONFLICT (content_hash) DO NOTHING RETURNING id"
                ),
                {
                    "id": document_id,
                    "title": title,
                    "type": file_type,
                    "hash": content_hash,
                    "url": source_url,
                    "by": uploaded_by,
                    "count": len(chunks),
                },
            )
            if inserted is None:
                raise DuplicateDocument(await self.find_by_hash(content_hash))
            await session.execute(
                text(
                    "INSERT INTO materials.chunks"
                    " (id, document_id, position, location, heading, text, embedding)"
                    " VALUES (:id, :doc, :position, :location, :heading, :text,"
                    " CAST(:embedding AS vector))"
                ),
                [
                    {
                        "id": uuid.uuid4(),
                        "doc": document_id,
                        "position": chunk.position,
                        "location": chunk.location,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "embedding": _vector(vector),
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            await session.execute(text(_BUMP_VERSION))
        return document_id

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        async with self._sessions.begin() as session:
            deleted = await session.scalar(
                text("DELETE FROM materials.documents WHERE id = :id RETURNING id"),
                {"id": document_id},
            )
            if deleted is not None:
                await session.execute(text(_BUMP_VERSION))
        return deleted is not None

    async def set_topics(self, document_id: uuid.UUID, topics: list[str]) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE materials.documents SET topics = CAST(:topics AS jsonb) WHERE id = :id"
                ),
                {"id": document_id, "topics": json.dumps(topics)},
            )

    async def list_topics(self, limit: int = 40) -> list[str]:
        async with self._sessions() as session:
            rows = await session.scalars(text("SELECT topics FROM materials.documents"))
            return _most_common([list(t) for t in rows], limit)

    async def documents_without_topics(self) -> list[tuple[uuid.UUID, list[str], list[str]]]:
        async with self._sessions() as session:
            documents = (
                await session.scalars(
                    text("SELECT id FROM materials.documents WHERE topics = '[]'::jsonb")
                )
            ).all()
            found = []
            for document_id in documents:
                rows = (
                    await session.execute(
                        text(
                            "SELECT heading, text FROM materials.chunks WHERE document_id = :id"
                            " ORDER BY position"
                        ),
                        {"id": document_id},
                    )
                ).all()
                found.append(
                    (document_id, [r.heading or "" for r in rows], [r.text for r in rows[:3]])
                )
        return found

    async def list_documents(self) -> list[DocumentInfo]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(
                    "SELECT id, title, file_type, chunk_count, source_url"
                    " FROM materials.documents ORDER BY created_at DESC"
                )
            )
        return [DocumentInfo(*row) for row in rows]

    async def corpus_version(self) -> int:
        async with self._sessions() as session:
            version = await session.scalar(text("SELECT version FROM materials.corpus"))
        return version or 0

    async def create_job(self, filename: str, uploaded_by: uuid.UUID) -> uuid.UUID:
        job_id = uuid.uuid4()
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "INSERT INTO materials.ingest_jobs (id, filename, status, uploaded_by)"
                    " VALUES (:id, :filename, 'queued', :by)"
                ),
                {"id": job_id, "filename": filename, "by": uploaded_by},
            )
        return job_id

    async def update_job(self, job_id, status, reason=None, document_id=None) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                text(
                    "UPDATE materials.ingest_jobs SET status = :status, reason = :reason,"
                    " document_id = :doc, updated_at = now() WHERE id = :id"
                ),
                {"id": job_id, "status": status, "reason": reason, "doc": document_id},
            )

    async def get_job(self, job_id: uuid.UUID) -> JobInfo | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT id, filename, status, reason, document_id"
                        " FROM materials.ingest_jobs WHERE id = :id"
                    ),
                    {"id": job_id},
                )
            ).first()
        return None if row is None else JobInfo(*row)

    async def fail_unfinished_jobs(self) -> int:
        async with self._sessions.begin() as session:
            result = await session.execute(
                text(
                    "UPDATE materials.ingest_jobs SET status = 'failed',"
                    " reason = 'The app restarted during this import.', updated_at = now()"
                    " WHERE status IN ('queued', 'running')"
                )
            )
        return result.rowcount
