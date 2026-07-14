"""Local Sentence Transformers embedding adapter."""

from __future__ import annotations

import numpy as np

from .config import RAGConfig
from .logger import get_logger

LOGGER = get_logger(__name__)


class Embedder:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.embedding_model, device="cpu")
            except Exception as exc:
                raise RuntimeError(f"Could not load local embedding model {self.config.embedding_model}: {exc}") from exc
        return self._model

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        prefix = "query: " if query else "passage: "
        vectors = self.model.encode([prefix + text for text in texts], normalize_embeddings=True, show_progress_bar=False)
        LOGGER.info("Embeddings Generated: %d", len(texts))
        return np.asarray(vectors, dtype=np.float32)
