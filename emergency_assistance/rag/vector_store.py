from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rag.schemas import DocumentChunk, RetrievedChunk

LOGGER = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence-transformer embedding model loading, caching, and vector generation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def load_model(self) -> None:
        """Loads the SentenceTransformer model into memory if not already loaded."""
        if self._model is not None:
            return
        LOGGER.info("Initializing embedding model: '%s'", self.model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            LOGGER.info("Embedding model '%s' loaded successfully.", self.model_name)
        except Exception as error:
            LOGGER.error("Failed to load embedding model '%s': %s", self.model_name, error)
            raise RuntimeError(f"Could not load embedding model: {error}") from error

    def embed_query(self, query: str) -> np.ndarray:
        """Generates a normalized embedding vector for a single query string."""
        if not query.strip():
            raise ValueError("Query text cannot be empty.")
        self.load_model()
        assert self._model is not None
        LOGGER.info("Generating embedding for query: '%s...'", query[:30])
        try:
            vector = self._model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vector, dtype=np.float32)
        except Exception as error:
            LOGGER.error("Query embedding generation failed: %s", error)
            raise RuntimeError(f"Query embedding generation failed: {error}") from error

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Generates normalized embedding vectors for a list of document chunk texts."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.load_model()
        assert self._model is not None
        LOGGER.info("Generating embeddings for %d document chunks.", len(texts))
        try:
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)
        except Exception as error:
            LOGGER.error("Document embedding generation failed: %s", error)
            raise RuntimeError(f"Document embedding generation failed: {error}") from error

    def get_status(self) -> dict[str, object]:
        """Returns the status of the embedding model manager."""
        return {
            "model_name": self.model_name,
            "loaded": self._model is not None,
        }


# Maintain backward compatibility alias
EmbeddingGenerator = EmbeddingManager


class FaissStore:
    """FAISS-based vector database store for saving and retrieving document chunks."""

    def __init__(self, directory: Path, embedder: EmbeddingManager) -> None:
        self._directory = directory
        self._embedder = embedder
        self._index = None
        self._chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> int:
        """Builds a new FAISS index from the given document chunks and persists it."""
        if not chunks:
            raise ValueError("No extractable document text was found to index.")
        
        LOGGER.info("Building FAISS vector index with %d chunks.", len(chunks))
        import faiss
        
        try:
            vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            
            self._directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self._directory / "index.faiss"))
            (self._directory / "chunks.json").write_text(
                json.dumps([c.to_dict() for c in chunks], indent=2), encoding="utf-8"
            )
            self._index, self._chunks = index, chunks
            LOGGER.info("Successfully built and saved FAISS index with %d chunks.", len(chunks))
            return len(chunks)
        except Exception as error:
            LOGGER.error("Failed to build FAISS index: %s", error)
            raise RuntimeError(f"FAISS index build failed: {error}") from error

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieves the top-k document chunks closest to the search query."""
        if not query.strip():
            return []
        
        self._ensure_loaded()
        if self._index is None or not self._chunks:
            LOGGER.warning("Retrieval skipped: FAISS index is empty or not loaded.")
            return []

        LOGGER.info("Retrieving context from FAISS index for query: '%s...'", query[:40])
        try:
            query_vector = self._embedder.embed_query(query)
            # Expand dimensions to fit FAISS search expectations (1, D)
            query_vector = np.expand_dims(query_vector, axis=0)
            
            k = min(top_k, len(self._chunks))
            scores, ids = self._index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:
                    results.append(RetrievedChunk(self._chunks[int(idx)], float(score)))
            
            LOGGER.info("Retrieved %d matches from vector store.", len(results))
            return results
        except Exception as error:
            LOGGER.error("FAISS query retrieval failed: %s", error)
            return []

    def get_status(self) -> dict[str, object]:
        """Returns the current state and metrics of the vector store."""
        try:
            self._ensure_loaded()
            available = self._index is not None
            chunk_count = len(self._chunks)
        except Exception:
            available = False
            chunk_count = 0

        return {
            "available": available,
            "directory": str(self._directory),
            "chunks_count": chunk_count,
        }

    def _ensure_loaded(self) -> None:
        """Helper to lazy-load the FAISS index files if they exist."""
        if self._index is not None:
            return
            
        index_file = self._directory / "index.faiss"
        chunks_file = self._directory / "chunks.json"
        
        if not index_file.exists() or not chunks_file.exists():
            LOGGER.warning(
                "FAISS index files not found in '%s'. RAG capability will be disabled until manual ingestion is run.",
                self._directory
            )
            return
            
        try:
            import faiss
            LOGGER.info("Loading FAISS index from '%s'", index_file)
            self._index = faiss.read_index(str(index_file))
            self._chunks = [
                DocumentChunk.from_dict(item) 
                for item in json.loads(chunks_file.read_text(encoding="utf-8"))
            ]
            if self._index.ntotal != len(self._chunks):
                raise ValueError(
                    f"Index total ({self._index.ntotal}) does not match metadata chunk count ({len(self._chunks)})"
                )
            LOGGER.info("FAISS vector store loaded successfully with %d chunks.", len(self._chunks))
        except Exception as error:
            self._index = None
            self._chunks = []
            LOGGER.error("Failed to load FAISS vector store: %s", error)
            # Log warning and handle gracefully rather than raising a crash-inducing error during startup
            LOGGER.warning("FAISS store initialization skipped due to load failure.")
