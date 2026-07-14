"""Data contracts and small side-effect-free helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentFile:
    path: Path
    content_hash: str


@dataclass(slots=True)
class Section:
    heading: str
    text: str
    page_number: int | None = None
    heading_level: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ParsedDocument:
    source: str
    sections: list[Section]
    metadata: dict[str, Any]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    section_heading: str
    page_number: int | None

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    score: float
    chunk: Chunk


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def sentences(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?])\s+", text.strip()) if value.strip()]
