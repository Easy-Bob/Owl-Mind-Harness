"""LangGraph checkpoints and store, each kept in its own Postgres schema."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.base.embed import EmbeddingsFunc
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# The dimension of all-MiniLM-L6-v2, the embedding model chosen in Q14.
EMBEDDING_DIMS = 384


@dataclass(frozen=True)
class Persistence:
    checkpointer: BaseCheckpointSaver
    store: BaseStore


def _pool(database_url: str, schema: str) -> AsyncConnectionPool:
    # The saver and store have no schema option, so search_path decides where their tables go.
    # public stays on the path because the pgvector extension lives there.
    return AsyncConnectionPool(
        database_url,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={schema},public",
        },
    )


@asynccontextmanager
async def open_persistence(database_url: str, embed: EmbeddingsFunc) -> AsyncIterator[Persistence]:
    """Open the Postgres saver and store. Run the Alembic migrations first."""
    async with _pool(database_url, "lg_checkpoint") as checkpoint_pool:
        async with _pool(database_url, "lg_store") as store_pool:
            checkpointer = AsyncPostgresSaver(checkpoint_pool)
            store = AsyncPostgresStore(
                store_pool,
                index={"dims": EMBEDDING_DIMS, "embed": embed, "distance_type": "cosine"},
            )
            await checkpointer.setup()
            await store.setup()
            yield Persistence(checkpointer=checkpointer, store=store)


def in_memory_persistence() -> Persistence:
    """The same interface for unit tests, with no database."""
    return Persistence(checkpointer=InMemorySaver(), store=InMemoryStore())
