"""Hybrid search over imported materials: by meaning and by exact words, merged by rank."""

import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from owl_mind.materials.embedding import Embedder
from owl_mind.materials.store import MaterialsStore

MAX_RESULTS = 8
MAX_RESULT_CHARS = 1200
MAX_PER_DOCUMENT = 3
_CANDIDATES = 40
# Reciprocal rank fusion: each ranking adds 1 / (K + rank). K = 60 is the usual starting value.
_RANK_K = 60
_CACHE_SECONDS = 600

_SEARCH = f"""
WITH by_meaning AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:vector AS vector)) AS rank
    FROM materials.chunks
    ORDER BY embedding <=> CAST(:vector AS vector)
    LIMIT {_CANDIDATES}
),
by_words AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(search_text, query) DESC) AS rank
    FROM materials.chunks, websearch_to_tsquery('english', :query) AS query
    WHERE search_text @@ query
    ORDER BY ts_rank_cd(search_text, query) DESC
    LIMIT {_CANDIDATES}
)
SELECT c.id, c.document_id, d.title, d.source_url, c.location, c.heading, c.text,
    COALESCE(1.0 / ({_RANK_K} + m.rank), 0) + COALESCE(1.0 / ({_RANK_K} + w.rank), 0) AS score
FROM by_meaning m
FULL JOIN by_words w USING (id)
JOIN materials.chunks c ON c.id = COALESCE(m.id, w.id)
JOIN materials.documents d ON d.id = c.document_id
ORDER BY score DESC, c.id
LIMIT {_CANDIDATES}
"""

_RESOLVE = """
SELECT c.id, d.title, d.source_url, c.location FROM materials.chunks c
JOIN materials.documents d ON d.id = c.document_id
WHERE c.id = ANY(CAST(:ids AS uuid[]))
"""


@dataclass(frozen=True)
class SearchResult:
    citation_id: str
    title: str
    location: str
    heading: str
    text: str
    score: float
    source_url: str | None


@dataclass(frozen=True)
class Citation:
    title: str
    location: str
    source_url: str | None


@runtime_checkable
class SearchCache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, seconds: int) -> None: ...


class RedisSearchCache:
    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        value = await self._client.get(key)
        return value.decode() if value is not None else None

    async def set(self, key: str, value: str, seconds: int) -> None:
        await self._client.set(key, value, ex=seconds)


class InMemorySearchCache:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, seconds: int) -> None:
        self.values[key] = value


def _vector(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in values) + "]"


def _cache_key(version: int, query: str, limit: int) -> str:
    # The corpus version is part of the key, so a new import can never reuse old results.
    digest = hashlib.sha256(f"{version}\n{limit}\n{query}".encode()).hexdigest()
    return f"cache:materials:{digest}"


class MaterialsSearch:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        store: MaterialsStore,
        embedder: Embedder,
        cache: SearchCache,
    ) -> None:
        self._sessions = sessions
        self._store = store
        self._embedder = embedder
        self._cache = cache

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query = " ".join(query.split())
        limit = max(1, min(MAX_RESULTS, limit))
        if not query:
            return []
        key = _cache_key(await self._store.corpus_version(), query.lower(), limit)
        try:
            cached = await self._cache.get(key)
        except Exception:
            cached = None  # A cache outage only costs speed.
        if cached is not None:
            return [SearchResult(**item) for item in json.loads(cached)]
        # Embedding is CPU work, so it runs off the event loop.
        [vector] = await asyncio.to_thread(self._embedder.embed, [query])
        results = await self._query(query, vector, limit)
        try:
            await self._cache.set(key, json.dumps([asdict(r) for r in results]), _CACHE_SECONDS)
        except Exception:
            pass
        return results

    async def _query(self, query: str, vector: list[float], limit: int) -> list[SearchResult]:
        async with self._sessions() as session:
            rows = (
                await session.execute(text(_SEARCH), {"query": query, "vector": _vector(vector)})
            ).all()
        results, per_document = [], {}
        for row in rows:
            # Neighboring chunks of one document overlap, so they mustn't crowd out the rest.
            if per_document.get(row.document_id, 0) >= MAX_PER_DOCUMENT:
                continue
            per_document[row.document_id] = per_document.get(row.document_id, 0) + 1
            results.append(
                SearchResult(
                    citation_id=str(row.id),
                    title=row.title,
                    location=row.location,
                    heading=row.heading,
                    text=row.text[:MAX_RESULT_CHARS],
                    score=round(float(row.score), 6),
                    source_url=row.source_url,
                )
            )
            if len(results) == limit:
                break
        return results

    async def resolve(self, citation_ids: list[str]) -> dict[str, Citation | None]:
        """Map each ID to its document and position, or None if it isn't a real chunk."""
        valid = []
        for citation_id in citation_ids:
            try:
                valid.append(str(uuid.UUID(citation_id)))
            except ValueError:
                pass
        found: dict[str, Citation] = {}
        if valid:
            async with self._sessions() as session:
                rows = await session.execute(text(_RESOLVE), {"ids": valid})
                found = {str(r.id): Citation(r.title, r.location, r.source_url) for r in rows}
        return {citation_id: found.get(citation_id) for citation_id in citation_ids}
