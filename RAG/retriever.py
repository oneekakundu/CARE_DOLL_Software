"""Similarity retrieval only; no answer generation is performed here."""

from __future__ import annotations

from time import perf_counter

from .embedder import Embedder
from .logger import get_logger
from .utils import RetrievalResult
from .vector_store import FaissVectorStore

LOGGER = get_logger(__name__)


class Retriever:
    def __init__(self, store: FaissVectorStore, embedder: Embedder) -> None:
        self.store, self.embedder = store, embedder

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Question cannot be empty")
        started = perf_counter()
        index, chunks = self.store.load()
        scores, ids = index.search(self.embedder.embed([query], query=True), min(top_k, len(chunks)))
        results = [RetrievalResult(float(score), chunks[int(idx)]) for score, idx in zip(scores[0], ids[0]) if idx >= 0]
        LOGGER.info("Retrieval Time: %.3fs | Retrieved Chunks: %d", perf_counter() - started, len(results))
        return results
