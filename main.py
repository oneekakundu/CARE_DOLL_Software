#!/usr/bin/env python3
"""Run the CARE DOLL pipeline from microphone input through RAG retrieval.

This entry point deliberately ends after displaying retrieved document chunks.
It does not create an answer or call an LLM/Ollama.
"""

from __future__ import annotations

from pathlib import Path

from RAG.rag_pipeline import RAGPipeline
from Voice_Input.pipeline import run_pipeline


def main() -> None:
    """Record audio, transcribe it, then retrieve relevant emergency information."""
    print("=" * 60)
    print("CARE DOLL: Voice Input → Speech-to-Text → RAG Retrieval")
    print("=" * 60)

    _, question_path = run_pipeline()
    question = Path(question_path).read_text(encoding="utf-8").strip()
    if not question:
        raise RuntimeError(f"Speech-to-text produced an empty question: {question_path}")

    rag = RAGPipeline()
    results = rag.retrieve(question)
    rag.display(question, results)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc
