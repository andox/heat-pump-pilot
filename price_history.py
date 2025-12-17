"""Price history persistence helpers for mpc_heat_pump.

This module is intentionally free of Home Assistant imports so it can be
unit-tested in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PriceHistoryStorage:
    """Simple JSON file persistence for the price history."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, Any] | None:
        """Load persisted history from disk."""
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, payload: dict[str, Any]) -> None:
        """Persist history atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True)
        temp_path.replace(self._path)
