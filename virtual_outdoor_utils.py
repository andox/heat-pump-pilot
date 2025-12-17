"""Helpers for building a virtual outdoor temperature trajectory.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

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
    base_outdoor_fallback: float,
    virtual_heat_offset: float,
    price_comfort_weight: float,
    price_baseline: float | None,
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
    - Always clamp warmer than `max_virtual_outdoor`.
    """
    if not sequence:
        return None

    steps = len(sequence)
    outdoor = _pad_to_length(_coerce_float_iterable(outdoor_forecast), steps, base_outdoor_fallback)
    prices = _pad_to_length(_coerce_float_iterable(price_forecast), steps, 0.0)

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

    planned: list[float] = []
    for idx, heat_on in enumerate(sequence):
        base = outdoor[idx]
        value = base - offset if heat_on else base

        if not heat_on and baseline and baseline > 0:
            price = prices[idx]
            ratio = price / baseline
            if ratio > 1.0:
                boost = weight * (ratio - 1.0) * max(1.0, offset)
                boost = min(boost, max(0.0, offset))
                value += boost

        planned.append(min(value, max_virtual_outdoor))

    return planned

