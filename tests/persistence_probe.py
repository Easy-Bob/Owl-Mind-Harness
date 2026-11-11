"""Write or read persisted data in its own process, so a restart test is a real restart."""

import asyncio
import hashlib
import json
import sys

from owl_mind.core.config import get_settings
from owl_mind.core.persistence import EMBEDDING_DIMS, open_persistence
from tests.thread_graph import build_graph, config


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic vectors with no model. Equal text gets an equal vector."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vectors.append([digest[i % len(digest)] / 255 + 0.001 for i in range(EMBEDDING_DIMS)])
    return vectors


async def main(action: str, thread_id: str) -> None:
    namespace = ("memories", thread_id)
    async with open_persistence(get_settings().database_url, fake_embed) as persistence:
        graph = build_graph(persistence.checkpointer)
        if action == "write":
            await graph.ainvoke({"messages": [("user", "hello")]}, config(thread_id))
            # The file format memory uses: the index embeds "content".
            await persistence.store.aput(namespace, thread_id, {"content": "weak on recursion"})
        else:
            state = await graph.aget_state(config(thread_id))
            item = await persistence.store.aget(namespace, thread_id)
            hits = await persistence.store.asearch(namespace, query="weak on recursion", limit=1)
            result = {
                "messages": [message.content for message in state.values.get("messages", [])],
                "store": item.value if item else None,
                "search": [hit.key for hit in hits],
            }
            print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
