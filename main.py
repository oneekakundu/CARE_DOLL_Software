#!/usr/bin/env python3
"""Run the CARE DOLL pipeline from microphone input through RAG retrieval.

This entry point deliberately ends after displaying retrieved document chunks.
It does not create an answer or call an LLM/Ollama.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding to prevent Windows cp1252 charmap print errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RAG.rag_pipeline import RAGPipeline
from Voice_Input.pipeline import run_pipeline


def main() -> None:
    """Record audio, transcribe it, then retrieve relevant emergency information."""
    parser = argparse.ArgumentParser(description="Run the CARE DOLL pipeline from microphone or text input.")
    parser.add_argument("--query", "-q", type=str, help="Text query to bypass voice input and transcription.")
    args = parser.parse_args()

    print("=" * 60)
    print("CARE DOLL: Voice/Text Input -> RAG Retrieval")
    print("=" * 60)

    if args.query:
        question = args.query.strip()
        print(f"Using query: {question}")
    else:
        question = ""
        try:
            _, question_path = run_pipeline()
            question = Path(question_path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"Voice Input pipeline failed or was skipped: {exc}")
            print("Falling back to text input.")

        if not question:
            question = input("Enter your question: ").strip()

        if not question:
            raise RuntimeError("No query provided (speech transcription and text input were both empty).")

    rag = RAGPipeline()
    results = rag.retrieve(question)
    rag.display(question, results)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc
