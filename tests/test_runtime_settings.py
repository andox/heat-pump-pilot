"""Unit tests for runtime settings normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from const import (  # noqa: E402
    CONF_COMFORT_TEMPERATURE_TOLERANCE,
    CONF_CONTINUOUS_CONTROL_WINDOW_HOURS,
    CONF_CONTROL_INTERVAL_MINUTES,
    CONF_HVAC_MODE,
    CONF_LEARNING_MODEL,
    CONF_PRICE_ABSOLUTE_LOW_THRESHOLD,
    CONF_PRICE_BASELINE_WINDOW_HOURS,
    CONF_PRICE_COMFORT_WEIGHT,
)
from runtime_settings import build_thermal_model_from_options, merge_climate_options  # noqa: E402


def test_merge_climate_options_normalizes_ranges_and_enums() -> None:
    merged = merge_climate_options(
        {
            CONF_CONTROL_INTERVAL_MINUTES: "0",
            CONF_PRICE_COMFORT_WEIGHT: "2.0",
            CONF_PRICE_BASELINE_WINDOW_HOURS: "999",
            CONF_PRICE_ABSOLUTE_LOW_THRESHOLD: "-5",
            CONF_CONTINUOUS_CONTROL_WINDOW_HOURS: "9",
            CONF_COMFORT_TEMPERATURE_TOLERANCE: "bad",
            CONF_HVAC_MODE: "OFF",
            CONF_LEARNING_MODEL: "invalid",
        }
    )
    assert merged[CONF_CONTROL_INTERVAL_MINUTES] == 1
    assert merged[CONF_PRICE_COMFORT_WEIGHT] == pytest.approx(1.0)
    assert merged[CONF_PRICE_BASELINE_WINDOW_HOURS] == 24
    assert merged[CONF_PRICE_ABSOLUTE_LOW_THRESHOLD] == "auto"
    assert merged[CONF_CONTINUOUS_CONTROL_WINDOW_HOURS] == pytest.approx(2.0)
    assert merged[CONF_COMFORT_TEMPERATURE_TOLERANCE] == pytest.approx(1.0)
    assert merged[CONF_HVAC_MODE] == "off"
    assert merged[CONF_LEARNING_MODEL] == "ekf"


def test_build_thermal_model_from_options_uses_rls_when_requested() -> None:
    model = build_thermal_model_from_options(
        merge_climate_options(
            {
                CONF_LEARNING_MODEL: "rls",
            }
        )
    )
    assert type(model).__name__ == "ThermalModelRlsEstimator"
