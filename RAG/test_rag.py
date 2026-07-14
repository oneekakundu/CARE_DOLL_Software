"""Runnable smoke tests for each retrieval stage (no LLM required)."""

from __future__ import annotations

from .config import DEFAULT_CONFIG
from .document_loader import DocumentLoader
from .rag_pipeline import RAGPipeline


def main() -> None:
    files = DocumentLoader(DEFAULT_CONFIG).load()
    assert files, "No source documents found"
    print("PASS: Document loading")
    pipeline = RAGPipeline()
    pipeline.ensure_index()
    print("PASS: Docling parsing, chunking, embedding, and FAISS indexing")
    results = pipeline.retrieve("What should I do if an EV produces smoke?")
    assert results, "No chunks retrieved"
    print("PASS: Retrieval")


if __name__ == "__main__":
    main()
