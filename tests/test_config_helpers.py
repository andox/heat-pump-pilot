"""Unit tests for config helper utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from config_helpers import (  # noqa: E402
    HVAC_MODE_HEAT,
    HVAC_MODE_OFF,
    build_unique_id,
    find_conflicting_entry_id,
    normalize_hvac_mode,
)


@dataclass
class _Entry:
    entry_id: str
    unique_id: str | None


def test_build_unique_id_prefers_controlled_entity() -> None:
    unique_id = build_unique_id(
        controlled_entity=" number.virtual_outdoor ",
        indoor_temp_entity="sensor.indoor",
        outdoor_temp_entity="sensor.outdoor",
    )
    assert unique_id == "number.virtual_outdoor"


def test_build_unique_id_falls_back_to_sensor_pair() -> None:
    unique_id = build_unique_id(
        controlled_entity=None,
        indoor_temp_entity="sensor.indoor",
        outdoor_temp_entity="sensor.outdoor",
    )
    assert unique_id == "sensor.indoor_sensor.outdoor"


def test_find_conflicting_entry_id_ignores_current_entry() -> None:
    entries = [
        _Entry(entry_id="one", unique_id="number.virtual_outdoor"),
        _Entry(entry_id="two", unique_id="number.virtual_outdoor"),
    ]
    conflict = find_conflicting_entry_id(
        entries,
        candidate_unique_id="number.virtual_outdoor",
        current_entry_id="two",
    )
    assert conflict == "one"


def test_find_conflicting_entry_id_returns_none_when_unique() -> None:
    entries = [
        _Entry(entry_id="one", unique_id="number.virtual_outdoor"),
        _Entry(entry_id="two", unique_id="sensor.indoor_sensor.outdoor"),
    ]
    conflict = find_conflicting_entry_id(
        entries,
        candidate_unique_id="switch.heater",
        current_entry_id="two",
    )
    assert conflict is None


def test_find_conflicting_entry_id_detects_conflict_after_options_edit() -> None:
    entries = [
        _Entry(entry_id="one", unique_id="switch.heater"),
        _Entry(entry_id="two", unique_id="number.virtual_outdoor"),
    ]
    conflict = find_conflicting_entry_id(
        entries,
        candidate_unique_id="switch.heater",
        current_entry_id="two",
    )
    assert conflict == "one"


def test_normalize_hvac_mode_accepts_known_values() -> None:
    assert normalize_hvac_mode("HEAT") == HVAC_MODE_HEAT
    assert normalize_hvac_mode(" off ") == HVAC_MODE_OFF


def test_normalize_hvac_mode_falls_back_for_invalid_values() -> None:
    assert normalize_hvac_mode("cool") == HVAC_MODE_HEAT
    assert normalize_hvac_mode(None) == HVAC_MODE_HEAT
    assert normalize_hvac_mode("cool", default=HVAC_MODE_OFF) == HVAC_MODE_OFF
