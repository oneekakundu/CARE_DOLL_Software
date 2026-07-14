"""Persistent FAISS index and metadata manifest."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import RAGConfig
from .logger import get_logger
from .utils import Chunk

LOGGER = get_logger(__name__)


class FaissVectorStore:
    def __init__(self, config: RAGConfig) -> None:
        self.directory = config.vector_store_path
        self.index_path = self.directory / "index.faiss"
        self.metadata_path = self.directory / "metadata.pkl"

    def is_current(self, manifest: dict[str, str], signature: dict[str, object]) -> bool:
        if not (self.index_path.exists() and self.metadata_path.exists()):
            return False
        try:
            with self.metadata_path.open("rb") as handle:
                stored = pickle.load(handle)
            return stored["manifest"] == manifest and stored["signature"] == signature
        except (OSError, KeyError, pickle.UnpicklingError):
            return False

    def build(self, vectors: np.ndarray, chunks: list[Chunk], manifest: dict[str, str], signature: dict[str, object]) -> None:
        import faiss
        if not len(vectors):
            raise ValueError("Cannot build an empty FAISS index")
        self.directory.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))
        with self.metadata_path.open("wb") as handle:
            pickle.dump({"chunks": chunks, "manifest": manifest, "signature": signature}, handle)
        LOGGER.info("FAISS Built: %d vectors", len(chunks))

    def load(self):
        import faiss
        if not (self.index_path.exists() and self.metadata_path.exists()):
            raise FileNotFoundError("FAISS index has not been built")
        with self.metadata_path.open("rb") as handle:
            data = pickle.load(handle)
        LOGGER.info("FAISS Loaded: %d vectors", len(data["chunks"]))
        return faiss.read_index(str(self.index_path)), data["chunks"]
