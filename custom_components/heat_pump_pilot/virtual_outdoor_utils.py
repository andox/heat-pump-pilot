"""Helpers for building a virtual outdoor temperature trajectory.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _coerce_float_iterable(values: Iterable[object] | None) -> list[float]:
    if values is None:
        return []
    coerced: list[float] = []
    for value in values:
        try:
            coerced.append(float(value))
        except (TypeError, ValueError):
            continue
    return coerced


def _pad_to_length(values: list[float], length: int, fallback: float) -> list[float]:
    if length <= 0:
        return []
    if not values:
        return [fallback] * length
    if len(values) < length:
        return values + [values[-1]] * (length - len(values))
    return values[:length]


def compute_planned_virtual_outdoor_temperatures(
    sequence: Sequence[bool] | None,
    outdoor_forecast: Sequence[float] | None,
    price_forecast: Sequence[float] | None,
    *,
    predicted_temperatures: Sequence[float] | None = None,
    base_outdoor_fallback: float,
    virtual_heat_offset: float,
    price_comfort_weight: float,
    price_baseline: float | None,
    comfort_temperature_tolerance: float,
    target_temperature: float | None = None,
    overshoot_warm_bias_enabled: bool = False,
    overshoot_warm_bias_curve: str = "linear",
    max_virtual_outdoor: float = 25.0,
) -> list[float] | None:
    """Return a per-step virtual outdoor temperature plan.

    This is derived from:
    - the MPC on/off `sequence`
    - `outdoor_forecast` per step
    - `price_forecast` per step (used only for warm-shifting during idle)

    It mirrors the integration's current control mapping:
    - If heating is requested, the pump sees `base - virtual_heat_offset`.
    - If idle, the pump sees `base` plus a price-based warm shift capped by `virtual_heat_offset`.
    - Optionally, when indoor temperature is predicted above the target temperature, an
      overshoot warm-bias can add an additional warm shift (also capped by `virtual_heat_offset`).
    - Always clamp warmer than `max_virtual_outdoor`.
    """
    if not sequence:
        return None

    steps = len(sequence)
    outdoor = _pad_to_length(_coerce_float_iterable(outdoor_forecast), steps, base_outdoor_fallback)
    prices = _pad_to_length(_coerce_float_iterable(price_forecast), steps, 0.0)
    predicted = _pad_to_length(_coerce_float_iterable(predicted_temperatures), steps, target_temperature or outdoor[0])

    try:
        offset = float(virtual_heat_offset)
    except (TypeError, ValueError):
        offset = 0.0

    try:
        weight = float(price_comfort_weight)
    except (TypeError, ValueError):
        weight = 0.0

    baseline = None
    if price_baseline is not None:
        try:
            baseline = float(price_baseline)
        except (TypeError, ValueError):
            baseline = None

    target = None
    if target_temperature is not None:
        try:
            target = float(target_temperature)
        except (TypeError, ValueError):
            target = None

    try:
        tolerance = max(0.0, float(comfort_temperature_tolerance))
    except (TypeError, ValueError):
        tolerance = 0.0

    planned: list[float] = []
    for idx, heat_on in enumerate(sequence):
        base = outdoor[idx]
        value = base - offset if heat_on else base

        if not heat_on and offset > 0:
            boost_total = 0.0

            if baseline and baseline > 0:
                price = prices[idx]
                ratio = price / baseline
                if ratio > 1.0:
                    boost_total += weight * (ratio - 1.0) * max(1.0, offset)

            if overshoot_warm_bias_enabled and target is not None:
                overshoot = predicted[idx] - target
                if overshoot > tolerance:
                    bias, _, _, _ = compute_overshoot_warm_bias(
                        overshoot,
                        tolerance,
                        offset,
                        overshoot_warm_bias_curve,
                    )
                    boost_total += bias

            boost_total = min(max(0.0, boost_total), offset)
            value += boost_total

        planned.append(min(value, max_virtual_outdoor))

    return planned


def compute_overshoot_warm_bias(
    overshoot: float,
    tolerance: float,
    virtual_heat_offset: float,
    curve: str,
) -> tuple[float, float, float, float]:
    """Return (bias, multiplier, min_bias, max_bias) for overshoot warm-bias."""
    try:
        offset = max(0.0, float(virtual_heat_offset))
    except (TypeError, ValueError):
        offset = 0.0

    if offset <= 0:
        return 0.0, 1.0, 0.0, 0.0

    try:
        tol = max(0.0, float(tolerance))
    except (TypeError, ValueError):
        tol = 0.0

    min_bias = 0.5 * offset
    max_bias = offset

    if overshoot <= tol:
        return 0.0, 1.0, min_bias, max_bias

    scale = (overshoot - tol) / max(tol, 0.1)
    scale = min(1.0, max(0.0, scale))

    shaped = _curve_scale(scale, curve)
    bias = min_bias + (max_bias - min_bias) * shaped
    multiplier = 1.0 + (bias / offset)
    return bias, multiplier, min_bias, max_bias


def _curve_scale(scale: float, curve: str) -> float:
    if curve == "quadratic":
        return scale * scale
    if curve == "cubic":
        return scale * scale * scale
    if curve == "sqrt":
        return math.sqrt(scale)
    return scale
