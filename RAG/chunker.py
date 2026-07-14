"""Pluggable, heading-aware chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re

from .config import RAGConfig
from .utils import Chunk, ParsedDocument, Section, sentences, token_count


class BaseChunker(ABC):
    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...

    def _pack(self, document: ParsedDocument, sections: list[Section], *, semantic: bool = False) -> list[Chunk]:
        output: list[Chunk] = []
        for section in sections:
            units = sentences(section.text)
            if not units:
                continue
            groups: list[list[str]] = [[]]
            for unit in units:
                candidate = " ".join(groups[-1] + [unit])
                split = token_count(candidate) > self.config.chunk_size
                if semantic and groups[-1] and self._topic_shift(groups[-1][-1], unit):
                    split = True
                if split:
                    overlap = self._overlap(groups[-1])
                    groups.append(overlap + [unit])
                else:
                    groups[-1].append(unit)
            for group in groups:
                text = " ".join(group).strip()
                if text:
                    output.append(Chunk(str(len(output) + 1), text, document.source, section.heading, section.page_number))
        return output

    def _overlap(self, units: list[str]) -> list[str]:
        selected: list[str] = []
        for unit in reversed(units):
            selected.insert(0, unit)
            if token_count(" ".join(selected)) >= self.config.chunk_overlap:
                break
        return selected

    @staticmethod
    def _topic_shift(left: str, right: str) -> bool:
        words = lambda value: {word.lower() for word in re.findall(r"[A-Za-z]{3,}", value)}
        a, b = words(left), words(right)
        return bool(a and b) and len(a & b) / len(a | b) < 0.04


class SemanticChunker(BaseChunker):
    """Split at locally embedded sentence topic boundaries, never across headings."""

    def __init__(self, config: RAGConfig) -> None:
        super().__init__(config)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.embedding_model, device="cpu")
            except Exception as exc:
                raise RuntimeError(f"Could not load semantic chunking model {self.config.embedding_model}: {exc}") from exc
        return self._model

    def _topic_shift(self, left: str, right: str) -> bool:
        vectors = self.model.encode([left, right], normalize_embeddings=True, show_progress_bar=False)
        similarity = float(vectors[0] @ vectors[1])
        return similarity < 0.42

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        return self._pack(document, document.sections, semantic=True)


class HybridChunker(SemanticChunker):
    """Heading-aware semantic chunks with the configured token overlap."""


class FixedChunker(BaseChunker):
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        return self._pack(document, document.sections)


class SentenceWindowChunker(BaseChunker):
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        output: list[Chunk] = []
        window = 3
        for section in document.sections:
            units = sentences(section.text)
            for start in range(0, len(units), window):
                text = " ".join(units[start : start + window]).strip()
                if text:
                    output.append(Chunk(str(len(output) + 1), text, document.source, section.heading, section.page_number))
        return output


class HierarchicalChunker(BaseChunker):
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks = self._pack(document, document.sections, semantic=True)
        for chunk in chunks:
            chunk.text = f"Section: {chunk.section_heading}\n{chunk.text}"
        return chunks


def get_chunker(config: RAGConfig) -> BaseChunker:
    choices = {"semantic": SemanticChunker, "hybrid": HybridChunker, "fixed": FixedChunker, "sentence_window": SentenceWindowChunker, "hierarchical": HierarchicalChunker}
    try:
        return choices[config.chunker.lower()](config)
    except KeyError as exc:
        raise ValueError(f"Unknown chunker '{config.chunker}'. Choose from: {', '.join(choices)}") from exc
