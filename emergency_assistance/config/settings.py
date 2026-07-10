from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings loaded once at application startup."""

    base_dir: Path
    host: str
    port: int
    ollama_url: str
    ollama_model: str
    embedding_model: str
    top_k: int
    enable_ocr: bool
    piper_command: str
    log_level: str

    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def vector_store_dir(self) -> Path:
        return self.base_dir / "data" / "vector_store"

    @property
    def database_path(self) -> Path:
        return self.base_dir / "database" / "care.db"

    @property
    def log_dir(self) -> Path:
        return self.base_dir / "logs"

    @classmethod
    def from_environment(cls, base_dir: Path | None = None) -> Settings:
        load_dotenv()
        root = base_dir or Path(__file__).resolve().parents[1]
        
        settings = cls(
            base_dir=root,
            host=os.getenv("CARE_HOST", "127.0.0.1"),
            port=int(os.getenv("CARE_PORT", "8000")),
            ollama_url=os.getenv("CARE_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("CARE_OLLAMA_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("CARE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            top_k=max(1, int(os.getenv("CARE_TOP_K", "4"))),
            enable_ocr=os.getenv("CARE_ENABLE_OCR", "true").lower() == "true",
            piper_command=os.getenv("CARE_PIPER_COMMAND", "piper"),
            log_level=os.getenv("CARE_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validates all required configuration values. Raises ValueError on failure."""
        errors: list[str] = []

        if not self.ollama_url:
            errors.append("CARE_OLLAMA_URL is empty.")
        elif not (self.ollama_url.startswith("http://") or self.ollama_url.startswith("https://")):
            errors.append(f"CARE_OLLAMA_URL must be a valid HTTP/HTTPS URL: '{self.ollama_url}'")

        if not self.ollama_model:
            errors.append("CARE_OLLAMA_MODEL is empty.")

        if not self.embedding_model:
            errors.append("CARE_EMBEDDING_MODEL is empty.")

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"CARE_LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL: '{self.log_level}'")

        # Validate that path directories can be created / are writeable
        for path_name, path in [
            ("RAW_DATA_PATH", self.raw_data_dir),
            ("VECTOR_STORE_PATH", self.vector_store_dir),
            ("DATABASE_PATH", self.database_path.parent),
            ("LOG_DIR", self.log_dir),
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as error:
                errors.append(f"{path_name} directory '{path}' could not be created or is not writable: {error}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f" - {err}" for err in errors))
