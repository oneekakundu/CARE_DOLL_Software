from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ConversationRepository:
    """Small SQLite audit log; only text supplied to this local app is stored."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path

    def initialize(self) -> None:
        """Initializes the database schema if not already present."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self._path) as connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL, query TEXT NOT NULL, response TEXT NOT NULL
                )""")
            LOGGER.info("SQLite database initialized successfully at '%s'", self._path)
        except Exception as error:
            LOGGER.error("Failed to initialize SQLite database: %s", error)
            raise RuntimeError(f"Database initialization failed: {error}") from error

    def save(self, query: str, response: str) -> None:
        """Saves a conversation record to the SQLite database."""
        try:
            with sqlite3.connect(self._path) as connection:
                connection.execute(
                    "INSERT INTO conversations(created_at, query, response) VALUES (?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), query, response),
                )
            LOGGER.info("Saved conversation to SQLite database.")
        except Exception as error:
            LOGGER.error("Failed to save conversation record to SQLite: %s", error)
            raise RuntimeError(f"Failed to write to SQLite: {error}") from error

    def check_connection(self) -> bool:
        """Verifies if the SQLite database is reachable and writable."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._path, timeout=2.0) as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations';")
                return True
        except Exception as error:
            LOGGER.error("SQLite connection check failed: %s", error)
            return False
