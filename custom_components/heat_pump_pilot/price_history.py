"""Price history persistence helpers for heat_pump_pilot."""

from __future__ import annotations

try:
    from .json_storage import AtomicJsonStorage
except ImportError:  # pragma: no cover - direct test imports
    from json_storage import AtomicJsonStorage  # type: ignore


class PriceHistoryStorage(AtomicJsonStorage):
    """Simple JSON file persistence for the price history."""
