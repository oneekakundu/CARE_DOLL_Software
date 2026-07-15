"""Coordinates the retrieval-only pipeline and terminal presentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output encoding to prevent Windows cp1252 charmap print errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .chunker import get_chunker
from .config import DEFAULT_CONFIG, RAGConfig
from .document_loader import DocumentLoader
from .embedder import Embedder
from .logger import get_logger
from .parser import DoclingParser
from .retriever import Retriever
from .utils import RetrievalResult
from .vector_store import FaissVectorStore

LOGGER = get_logger(__name__)


class RAGPipeline:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.store = FaissVectorStore(config)
        self.embedder = Embedder(config)

    def ensure_index(self) -> None:
        files = DocumentLoader(self.config).load()
        manifest = {str(item.path.relative_to(self.config.document_path)): item.content_hash for item in files}
        signature = {"model": self.config.embedding_model, "chunker": self.config.chunker, "size": self.config.chunk_size, "overlap": self.config.chunk_overlap}
        if self.store.is_current(manifest, signature):
            self.store.load()
            return
        parser, chunker = DoclingParser(), get_chunker(self.config)
        chunks = [chunk for item in files for chunk in chunker.chunk(parser.parse(item))]
        if not chunks:
            raise RuntimeError("No chunks were created from the configured document directory")
        self.store.build(self.embedder.embed([chunk.text for chunk in chunks]), chunks, manifest, signature)
        LOGGER.info("Chunks Created: %d", len(chunks))

    def retrieve(self, question: str) -> list[RetrievalResult]:
        self.ensure_index()
        return Retriever(self.store, self.embedder).retrieve(question, self.config.top_k)

    def read_latest_question(self) -> str:
        files = list(self.config.question_path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No question text files found in {self.config.question_path}")
        question = max(files, key=lambda path: path.stat().st_mtime).read_text(encoding="utf-8").strip()
        if not question:
            raise ValueError("Latest question file is empty")
        return question

    @staticmethod
    def display(question: str, results: list[RetrievalResult]) -> None:
        print("-" * 49 + f"\nQUESTION\n\n{question}\n" + "-" * 49)
        for number, result in enumerate(results, 1):
            chunk = result.chunk
            print(f"\nRetrieved Chunk {number}\n\nSimilarity Score: {result.score:.3f}\nDocument: {chunk.source}\nSection: {chunk.section_heading}\nChunk ID: {chunk.chunk_id}\nText:\n\n{chunk.text}\n" + "-" * 49)

    def run_from_question_file(self) -> list[RetrievalResult]:
        question = self.read_latest_question()
        results = self.retrieve(question)
        self.display(question, results)
        return results


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    RAGPipeline().run_from_question_file()
