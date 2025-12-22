"""Unit tests for learning helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from const import (  # noqa: E402
    CONF_CONTROL_INTERVAL_MINUTES,
    CONF_HEAT_LOSS_COEFFICIENT,
    CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
    CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
    CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
    CONF_INITIAL_HEAT_LOSS_OVERRIDE,
    CONF_LEARNING_MODEL,
    CONF_RLS_FORGETTING_FACTOR,
    CONF_THERMAL_RESPONSE_SEED,
    LEARNING_MODEL_EKF,
    LEARNING_MODEL_RLS,
)
from learning_utils import should_reseed_thermal_model  # noqa: E402


def test_should_reseed_false_for_unrelated_option_changes() -> None:
    previous = {
        CONF_THERMAL_RESPONSE_SEED: 0.5,
        CONF_HEAT_LOSS_COEFFICIENT: 0.05,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE: None,
        CONF_LEARNING_MODEL: LEARNING_MODEL_EKF,
        CONF_RLS_FORGETTING_FACTOR: 0.99,
        CONF_CONTROL_INTERVAL_MINUTES: 15,
        CONF_HEATING_SUPPLY_TEMP_THRESHOLD: 29,
        CONF_HEATING_SUPPLY_TEMP_HYSTERESIS: 1,
        CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS: 60,
    }
    current = dict(previous)
    current[CONF_CONTROL_INTERVAL_MINUTES] = 10
    current[CONF_HEATING_SUPPLY_TEMP_THRESHOLD] = 30
    current[CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS] = 0
    assert should_reseed_thermal_model(previous, current) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (CONF_THERMAL_RESPONSE_SEED, 0.9),
        (CONF_HEAT_LOSS_COEFFICIENT, 0.02),
        (CONF_INITIAL_HEAT_LOSS_OVERRIDE, 0.03),
        (CONF_LEARNING_MODEL, LEARNING_MODEL_RLS),
        (CONF_RLS_FORGETTING_FACTOR, 0.97),
    ],
)
def test_should_reseed_true_for_estimator_initialization_changes(key: str, value) -> None:
    previous = {
        CONF_THERMAL_RESPONSE_SEED: 0.5,
        CONF_HEAT_LOSS_COEFFICIENT: 0.05,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE: None,
        CONF_LEARNING_MODEL: LEARNING_MODEL_EKF,
        CONF_RLS_FORGETTING_FACTOR: 0.99,
    }
    current = dict(previous)
    current[key] = value
    assert should_reseed_thermal_model(previous, current) is True
