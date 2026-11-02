"""The one embedding model, used by every part that stores or searches vectors."""

import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

EMBEDDING_DIMS = 384
_BATCH_SIZE = 64


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Loads the model on first use, because loading takes seconds and most runs never embed."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self._model_name, cache_dir=str(self._cache_dir))
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = [vector.tolist() for vector in model.embed(texts, batch_size=_BATCH_SIZE)]
        if vectors and len(vectors[0]) != EMBEDDING_DIMS:
            raise ValueError(f"Expected {EMBEDDING_DIMS} dimensions, got {len(vectors[0])}")
        return vectors
