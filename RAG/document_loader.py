"""Discover source files only; parsing belongs to parser.py."""

from __future__ import annotations

from pathlib import Path

from .config import RAGConfig
from .logger import get_logger
from .utils import DocumentFile, sha256_file

LOGGER = get_logger(__name__)


class DocumentLoader:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    def load(self) -> list[DocumentFile]:
        root = self.config.document_path
        if not root.exists():
            raise FileNotFoundError(f"Document directory does not exist: {root}")
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in self.config.supported_extensions]
        documents = [DocumentFile(path=path, content_hash=sha256_file(path)) for path in sorted(files)]
        LOGGER.info("Documents Loaded: %d", len(documents))
        return documents
