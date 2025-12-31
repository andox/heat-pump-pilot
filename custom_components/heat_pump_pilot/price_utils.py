"""Helpers for price baseline calculations.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable
import math

try:  # pragma: no cover - allow direct imports in tests
    from .forecast_utils import expand_to_steps
except ImportError:  # pragma: no cover
    from forecast_utils import expand_to_steps  # type: ignore


def _coerce_float_iterable(values: Iterable[object] | None) -> list[float]:
    if not values:
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


def compute_price_baseline(
    *,
    history: Iterable[object] | None,
    forecast: Iterable[object] | None,
    time_step_hours: float,
    window_hours: int,
    baseline_floor: float,
) -> tuple[float, dict[str, int]]:
    """Compute a baseline from recent history plus available forecast."""
    steps_per_hour = int(round(1 / time_step_hours)) if time_step_hours > 0 else 1
    steps_per_hour = max(1, steps_per_hour)
    max_history_samples = max(0, int(window_hours) * steps_per_hour)

    history_values = _coerce_float_iterable(history)
    if max_history_samples:
        history_values = history_values[-max_history_samples:]
    history_positive = [value for value in history_values if value > 0]

    forecast_values = _coerce_float_iterable(forecast)
    forecast_expanded: list[float] = []
    if forecast_values:
        expected_step_samples = steps_per_hour * min(int(window_hours), 24)
        forecast_is_step = time_step_hours < 1 and len(forecast_values) >= expected_step_samples
        if forecast_is_step:
            forecast_expanded = list(forecast_values)
        else:
            forecast_expanded = expand_to_steps(
                forecast_values,
                len(forecast_values) * steps_per_hour,
                time_step_hours,
            )
    forecast_positive = [value for value in forecast_expanded if value > 0]

    baseline_pool = history_positive + forecast_positive
    baseline = median(baseline_pool) if baseline_pool else baseline_floor
    if baseline <= baseline_floor:
        baseline = baseline_floor

    details = {
        "history_samples": len(history_positive),
        "forecast_samples": len(forecast_positive),
    }
    return baseline, details


def compute_absolute_low_price_threshold(
    *,
    history: Iterable[object] | None,
    time_step_hours: float,
    window_hours: int,
) -> tuple[float | None, dict[str, int]]:
    """Compute an absolute low-price threshold from recent history."""
    steps_per_hour = int(round(1 / time_step_hours)) if time_step_hours > 0 else 1
    steps_per_hour = max(1, steps_per_hour)
    max_history_samples = max(0, int(window_hours) * steps_per_hour)

    history_values = _coerce_float_iterable(history)
    if max_history_samples:
        history_values = history_values[-max_history_samples:]
    history_positive = [value for value in history_values if value > 0]

    if not history_positive:
        return None, {"history_samples": 0}

    return median(history_positive), {"history_samples": len(history_positive)}


_PRICE_LABEL_ORDER = ("very_low", "low", "normal", "high", "very_high", "extreme")


def _cap_price_label(label: str, max_label: str) -> str:
    try:
        label_idx = _PRICE_LABEL_ORDER.index(label)
        max_idx = _PRICE_LABEL_ORDER.index(max_label)
    except ValueError:
        return label
    if label_idx > max_idx:
        return max_label
    return label


def price_label_from_ratio(ratio: float) -> str:
    """Map a price ratio to a human-friendly category."""
    if ratio < 0.75:
        return "very_low"
    if ratio < 0.9:
        return "low"
    if ratio < 1.1:
        return "normal"
    if ratio < 1.3:
        return "high"
    if ratio < 1.6:
        return "very_high"
    return "extreme"


def classify_price(
    current_price: float | None,
    baseline: float | None,
    *,
    absolute_low_threshold: float | None = None,
) -> tuple[float | None, str | None]:
    """Classify a price against a given baseline."""
    if current_price is None or baseline is None:
        return None, None
    if baseline <= 0:
        return None, None
    ratio = current_price / baseline
    label = price_label_from_ratio(ratio)
    if absolute_low_threshold is not None and current_price <= absolute_low_threshold:
        label = _cap_price_label(label, "normal")
    return ratio, label
