"""Helpers for learning/estimation logic.

This module is intentionally Home Assistant free to keep unit tests simple.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised indirectly in Home Assistant
    from .const import (
        CONF_HEAT_LOSS_COEFFICIENT,
        CONF_INITIAL_HEAT_GAIN,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE,
        CONF_INITIAL_INDOOR_TEMP,
        CONF_THERMAL_RESPONSE_SEED,
    )
except ImportError:  # pragma: no cover - allows unit tests without package context
    from const import (  # type: ignore
        CONF_HEAT_LOSS_COEFFICIENT,
        CONF_INITIAL_HEAT_GAIN,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE,
        CONF_INITIAL_INDOOR_TEMP,
        CONF_THERMAL_RESPONSE_SEED,
    )

_THERMAL_RESEED_KEYS: tuple[str, ...] = (
    CONF_THERMAL_RESPONSE_SEED,
    CONF_INITIAL_INDOOR_TEMP,
    CONF_INITIAL_HEAT_GAIN,
    CONF_INITIAL_HEAT_LOSS_OVERRIDE,
    # Treat the base heat loss coefficient as a "reseed" signal; otherwise a user change
    # will be overwritten by the next EKF step.
    CONF_HEAT_LOSS_COEFFICIENT,
)


def should_reseed_thermal_model(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True if the thermal estimator should be reseeded.

    Reseeding resets the estimator state/covariance. We only want to do this when the
    user changes options that explicitly affect the estimator initialization, not when
    they tweak unrelated options like control interval or heating detection thresholds.
    """

    for key in _THERMAL_RESEED_KEYS:
        if _normalized(previous.get(key)) != _normalized(current.get(key)):
            return True
    return False


def _normalized(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
