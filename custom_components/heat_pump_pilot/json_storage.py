"""Shared JSON storage helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class AtomicJsonStorage:
    """Simple JSON file persistence with atomic replace and fallback write."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, Any] | None:
        """Load persisted state from disk."""
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, payload: dict[str, Any]) -> None:
        """Persist state atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                json.dump(payload, file, ensure_ascii=True)
                file.flush()
                os.fsync(file.fileno())
            temp_path.replace(self._path)
        except FileNotFoundError as exc:
            LOGGER.warning("Temporary JSON file missing during save for %s: %s", self._path, exc)
            self._path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("JSON save failed for %s (%s). Falling back to direct write.", self._path, exc)
            self._path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
