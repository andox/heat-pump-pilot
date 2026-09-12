"""Helpers for config entry identity and persisted mode state.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

from typing import Any, Iterable

HVAC_MODE_HEAT = "heat"
HVAC_MODE_OFF = "off"
VALID_HVAC_MODES = {HVAC_MODE_HEAT, HVAC_MODE_OFF}


def build_unique_id(
    *,
    controlled_entity: str | None,
    indoor_temp_entity: str,
    outdoor_temp_entity: str,
) -> str:
    """Build the config entry unique id."""
    controlled = controlled_entity.strip() if isinstance(controlled_entity, str) else None
    if controlled:
        return controlled
    return f"{indoor_temp_entity}_{outdoor_temp_entity}"


def normalize_hvac_mode(value: Any, default: str = HVAC_MODE_HEAT) -> str:
    """Normalize a persisted HVAC mode string."""
    fallback = default if default in VALID_HVAC_MODES else HVAC_MODE_HEAT
    if not isinstance(value, str):
        return fallback
    normalized = value.strip().lower()
    if normalized in VALID_HVAC_MODES:
        return normalized
    return fallback


def find_conflicting_entry_id(
    entries: Iterable[Any],
    *,
    candidate_unique_id: str,
    current_entry_id: str,
) -> str | None:
    """Return the conflicting entry id for a candidate unique id, if any."""
    for entry in entries:
        entry_id = getattr(entry, "entry_id", None)
        unique_id = getattr(entry, "unique_id", None)
        if entry_id == current_entry_id:
            continue
        if unique_id == candidate_unique_id:
            return entry_id
    return None
