"""All configuration for the offline RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Change this one value to compare implementations: semantic, hybrid, fixed,
# sentence_window, or hierarchical.
CHUNKER = "semantic"


@dataclass(frozen=True, slots=True)
class RAGConfig:
    document_path: Path = PROJECT_ROOT / "documents" / "extracted"
    question_path: Path = PROJECT_ROOT / "data" / "text"
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunker: str = CHUNKER
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    supported_extensions: tuple[str, ...] = (".md", ".txt")


DEFAULT_CONFIG = RAGConfig()
