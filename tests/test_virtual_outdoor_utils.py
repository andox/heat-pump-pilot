"""Unit tests for planned virtual outdoor temperature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing the integration modules directly from this directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from virtual_outdoor_utils import (  # noqa: E402
    compute_overshoot_warm_bias,
    compute_planned_virtual_outdoor_temperatures,
)


def test_planned_virtual_outdoor_heating_is_colder_by_offset() -> None:
    planned = compute_planned_virtual_outdoor_temperatures(
        [True, False],
        outdoor_forecast=[10.0, 10.0],
        price_forecast=[1.0, 1.0],
        base_outdoor_fallback=10.0,
        virtual_heat_offset=5.0,
        price_comfort_weight=0.9,
        price_baseline=1.0,
        comfort_temperature_tolerance=0.5,
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
        comfort_temperature_tolerance=0.5,
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
        comfort_temperature_tolerance=0.5,
        max_virtual_outdoor=25.0,
    )
    assert planned == [25.0]


def test_planned_virtual_outdoor_overshoot_warm_bias() -> None:
    planned = compute_planned_virtual_outdoor_temperatures(
        [False],
        outdoor_forecast=[6.0],
        price_forecast=[1.0],
        predicted_temperatures=[21.0],
        base_outdoor_fallback=6.0,
        virtual_heat_offset=10.0,
        price_comfort_weight=0.0,
        price_baseline=1.0,
        comfort_temperature_tolerance=0.5,
        target_temperature=20.0,
        overshoot_warm_bias_enabled=True,
        overshoot_warm_bias_curve="linear",
    )
    # overshoot=1.0, tolerance=0.5 => scale=1.0, bias=max (offset=10)
    assert planned == [16.0]


def test_overshoot_warm_bias_curves_shape() -> None:
    bias_linear, _, min_bias, max_bias = compute_overshoot_warm_bias(
        overshoot=1.0,
        tolerance=0.5,
        virtual_heat_offset=8.0,
        curve="linear",
    )
    bias_quadratic, _, _, _ = compute_overshoot_warm_bias(
        overshoot=0.75,
        tolerance=0.5,
        virtual_heat_offset=8.0,
        curve="quadratic",
    )
    bias_sqrt, _, _, _ = compute_overshoot_warm_bias(
        overshoot=0.75,
        tolerance=0.5,
        virtual_heat_offset=8.0,
        curve="sqrt",
    )

    assert min_bias == 4.0
    assert max_bias == 8.0
    assert bias_linear == max_bias
    assert min_bias < bias_quadratic < bias_linear
    assert bias_sqrt > bias_quadratic
