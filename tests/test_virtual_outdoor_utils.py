"""Unit tests for planned virtual outdoor temperature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing the integration modules directly from this directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from virtual_outdoor_utils import compute_planned_virtual_outdoor_temperatures  # noqa: E402


def test_planned_virtual_outdoor_heating_is_colder_by_offset() -> None:
    planned = compute_planned_virtual_outdoor_temperatures(
        [True, False],
        outdoor_forecast=[10.0, 10.0],
        price_forecast=[1.0, 1.0],
        base_outdoor_fallback=10.0,
        virtual_heat_offset=5.0,
        price_comfort_weight=0.9,
        price_baseline=1.0,
    )
    assert planned == [5.0, 10.0]


def test_planned_virtual_outdoor_idle_warm_shift_clamped_to_offset() -> None:
    planned = compute_planned_virtual_outdoor_temperatures(
        [False],
        outdoor_forecast=[6.0],
        price_forecast=[3.0],
        base_outdoor_fallback=6.0,
        virtual_heat_offset=10.0,
        price_comfort_weight=1.0,
        price_baseline=1.0,
    )
    # ratio=3, boost = 1*(3-1)*10 = 20 clamped to 10 => 6+10
    assert planned == [16.0]


def test_planned_virtual_outdoor_respects_max_virtual_outdoor() -> None:
    planned = compute_planned_virtual_outdoor_temperatures(
        [False],
        outdoor_forecast=[24.0],
        price_forecast=[2.0],
        base_outdoor_fallback=24.0,
        virtual_heat_offset=10.0,
        price_comfort_weight=1.0,
        price_baseline=1.0,
        max_virtual_outdoor=25.0,
    )
    assert planned == [25.0]

