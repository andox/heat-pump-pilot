"""Helpers for building a virtual outdoor temperature trajectory.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

try:
    from .const import DEFAULT_PRICE_PENALTY_CURVE, DEFAULT_PRICE_RATIO_CAP, PRICE_PENALTY_CURVES
except ImportError:  # pragma: no cover - allow direct imports in tests
    from const import (  # type: ignore
        DEFAULT_PRICE_PENALTY_CURVE,
        DEFAULT_PRICE_RATIO_CAP,
        PRICE_PENALTY_CURVES,
    )


def _coerce_float_iterable(values: Iterable[object] | None) -> list[float]:
    if values is None:
        return []
    coerced: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        coerced.append(numeric)
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
    virtual_outdoor_min_temp: float | None = None,
    allow_below_min_when_outdoor_lower: bool = True,
    price_comfort_weight: float,
    price_baseline: float | None,
    comfort_temperature_tolerance: float,
    target_temperature: float | None = None,
    price_penalty_curve: str = DEFAULT_PRICE_PENALTY_CURVE,
    price_ratio_cap: float = DEFAULT_PRICE_RATIO_CAP,
    overshoot_warm_bias_enabled: bool = False,
    overshoot_warm_bias_curve: str = "linear",
    continuous_control_enabled: bool = False,
    continuous_control_window_steps: int | None = None,
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
        min_temp = float(virtual_outdoor_min_temp) if virtual_outdoor_min_temp is not None else None
    except (TypeError, ValueError):
        min_temp = None

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
    duty_ratios: list[float] | None = None
    if continuous_control_enabled:
        window_steps = continuous_control_window_steps or 1
        window_steps = max(1, int(window_steps))
        duty_ratios = _compute_duty_ratios(sequence, window_steps)
    for idx, heat_on in enumerate(sequence):
        base = outdoor[idx]
        if continuous_control_enabled:
            ratio = duty_ratios[idx] if duty_ratios is not None else 0.0
            value = compute_continuous_virtual_outdoor(
                base,
                ratio,
                virtual_heat_offset=offset,
                price=prices[idx] if prices else None,
                price_baseline=baseline,
                price_comfort_weight=weight,
                price_penalty_curve=price_penalty_curve,
                price_ratio_cap=price_ratio_cap,
                predicted_temp=predicted[idx] if predicted else None,
                target_temperature=target,
                comfort_temperature_tolerance=tolerance,
                overshoot_warm_bias_enabled=overshoot_warm_bias_enabled,
                overshoot_warm_bias_curve=overshoot_warm_bias_curve,
                max_virtual_outdoor=max_virtual_outdoor,
            )
            if min_temp is not None and (not allow_below_min_when_outdoor_lower or base >= min_temp):
                value = max(min_temp, value)
            planned.append(value)
            continue

        value = base - offset if heat_on else base

        if not heat_on and offset > 0:
            boost_total = compute_idle_warm_bias(
                price=prices[idx] if prices else None,
                price_baseline=baseline,
                price_comfort_weight=weight,
                price_penalty_curve=price_penalty_curve,
                price_ratio_cap=price_ratio_cap,
                predicted_temp=predicted[idx] if predicted else None,
                target_temperature=target,
                comfort_temperature_tolerance=tolerance,
                overshoot_warm_bias_enabled=overshoot_warm_bias_enabled,
                overshoot_warm_bias_curve=overshoot_warm_bias_curve,
                virtual_heat_offset=offset,
            )
            value += boost_total

        value = min(value, max_virtual_outdoor)
        if min_temp is not None and (not allow_below_min_when_outdoor_lower or base >= min_temp):
            value = max(min_temp, value)
        planned.append(value)

    return planned


def compute_continuous_virtual_outdoor(
    base_outdoor: float,
    duty_ratio: float,
    *,
    virtual_heat_offset: float,
    price: float | None,
    price_baseline: float | None,
    price_comfort_weight: float,
    price_penalty_curve: str = DEFAULT_PRICE_PENALTY_CURVE,
    price_ratio_cap: float = DEFAULT_PRICE_RATIO_CAP,
    predicted_temp: float | None,
    target_temperature: float | None,
    comfort_temperature_tolerance: float,
    overshoot_warm_bias_enabled: bool = False,
    overshoot_warm_bias_curve: str = "linear",
    max_virtual_outdoor: float = 25.0,
) -> float:
    """Compute a continuous virtual outdoor temperature from a duty ratio."""
    try:
        offset = max(0.0, float(virtual_heat_offset))
    except (TypeError, ValueError):
        offset = 0.0
    if offset <= 0:
        return min(float(base_outdoor), max_virtual_outdoor)

    ratio = max(0.0, min(1.0, float(duty_ratio)))
    base_shift = offset * (1.0 - 2.0 * ratio)
    value = float(base_outdoor) + base_shift

    warm_bias = compute_idle_warm_bias(
        price=price,
        price_baseline=price_baseline,
        price_comfort_weight=price_comfort_weight,
        price_penalty_curve=price_penalty_curve,
        price_ratio_cap=price_ratio_cap,
        predicted_temp=predicted_temp,
        target_temperature=target_temperature,
        comfort_temperature_tolerance=comfort_temperature_tolerance,
        overshoot_warm_bias_enabled=overshoot_warm_bias_enabled,
        overshoot_warm_bias_curve=overshoot_warm_bias_curve,
        virtual_heat_offset=offset,
    )
    if warm_bias > 0.0:
        warm_bias *= max(0.0, 1.0 - ratio)
        value = min(value + warm_bias, float(base_outdoor) + offset)

    value = max(float(base_outdoor) - offset, value)
    return min(value, max_virtual_outdoor)


def compute_duty_ratio(sequence: Sequence[bool], start: int, window_steps: int) -> float:
    """Compute the fraction of heat-on steps in a window."""
    if not sequence:
        return 0.0
    if window_steps <= 0:
        return 0.0
    start = max(0, int(start))
    end = min(len(sequence), start + int(window_steps))
    if end <= start:
        return 0.0
    on_steps = sum(1 for value in sequence[start:end] if value)
    return on_steps / (end - start)


def _compute_duty_ratios(sequence: Sequence[bool], window_steps: int) -> list[float]:
    return [compute_duty_ratio(sequence, idx, window_steps) for idx in range(len(sequence))]


def compute_idle_warm_bias(
    *,
    price: float | None,
    price_baseline: float | None,
    price_comfort_weight: float,
    price_penalty_curve: str,
    price_ratio_cap: float,
    predicted_temp: float | None,
    target_temperature: float | None,
    comfort_temperature_tolerance: float,
    overshoot_warm_bias_enabled: bool,
    overshoot_warm_bias_curve: str,
    virtual_heat_offset: float,
) -> float:
    try:
        offset = max(0.0, float(virtual_heat_offset))
    except (TypeError, ValueError):
        return 0.0
    if offset <= 0:
        return 0.0

    boost_total = 0.0
    if price is not None and price_baseline and price_baseline > 0:
        try:
            ratio = float(price) / float(price_baseline)
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None and ratio > 1.0:
            shaped_ratio = _apply_price_penalty_curve(ratio, price_penalty_curve, price_ratio_cap)
            try:
                weight = float(price_comfort_weight)
            except (TypeError, ValueError):
                weight = 0.0
            boost_total += weight * (shaped_ratio - 1.0) * max(1.0, offset)

    if overshoot_warm_bias_enabled and target_temperature is not None and predicted_temp is not None:
        try:
            overshoot = float(predicted_temp) - float(target_temperature)
            tolerance = max(0.0, float(comfort_temperature_tolerance))
        except (TypeError, ValueError):
            overshoot = None
            tolerance = 0.0
        if overshoot is not None and overshoot > tolerance:
            bias, _, _, _ = compute_overshoot_warm_bias(
                overshoot,
                tolerance,
                offset,
                overshoot_warm_bias_curve,
            )
            boost_total += bias

    return min(max(0.0, boost_total), offset)


def _apply_price_penalty_curve(ratio: float, curve: str, ratio_cap: float) -> float:
    """Apply the configured price curve above the baseline ratio."""
    if ratio <= 1.0:
        return ratio
    try:
        cap = max(1.0, float(ratio_cap))
    except (TypeError, ValueError):
        cap = DEFAULT_PRICE_RATIO_CAP
    capped_ratio = min(ratio, cap)
    x = max(0.0, capped_ratio - 1.0)
    curve = curve if curve in PRICE_PENALTY_CURVES else DEFAULT_PRICE_PENALTY_CURVE
    if curve == "sqrt":
        adjusted = math.sqrt(x)
    elif curve == "quadratic":
        adjusted = x * x
    else:
        adjusted = x
    return 1.0 + adjusted


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

    min_bias = 0.0
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
