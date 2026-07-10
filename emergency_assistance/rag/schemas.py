from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    text: str
    document_name: str
    page: int
    section: str
    source_file: str
    original_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DocumentChunk:
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            document_name=str(value.get("document_name", value.get("source", "Unknown"))),
            page=int(value["page"]),
            section=str(value.get("section", "General")),
            source_file=str(value.get("source_file", value.get("source", "Unknown"))),
            original_text=str(value.get("original_text", value.get("text", ""))),
        )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
