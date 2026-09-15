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
    forecast_is_step: bool = False,
) -> tuple[float, dict[str, int]]:
    """Median of known prices, including zero/negative, with a positive scale floor.

    Callers supply history already restricted by timestamp. Untimed forecasts
    are hourly unless explicitly marked as controller-step data.
    """
    steps_per_hour = int(round(1 / time_step_hours)) if time_step_hours > 0 else 1
    steps_per_hour = max(1, steps_per_hour)
    max_history_samples = max(0, int(window_hours) * steps_per_hour)
    max_forecast_samples = max_history_samples

    history_values = _coerce_float_iterable(history)
    if max_history_samples:
        history_values = history_values[-max_history_samples:]
    history_known = history_values

    forecast_raw = list(forecast) if forecast is not None else []
    if forecast_is_step is True:
        forecast_raw = forecast_raw[:max_forecast_samples]
    forecast_values = _coerce_float_iterable(forecast_raw)
    forecast_expanded: list[float] = []
    if forecast_values:
        if forecast_is_step:
            forecast_expanded = list(forecast_values)
        else:
            forecast_expanded = expand_to_steps(
                forecast_values,
                len(forecast_values) * steps_per_hour,
                time_step_hours,
            )
        if max_forecast_samples:
            forecast_expanded = forecast_expanded[:max_forecast_samples]
        else:
            forecast_expanded = []
    forecast_known = forecast_expanded

    baseline_pool = history_known + forecast_known
    baseline = median(baseline_pool) if baseline_pool else baseline_floor
    if baseline <= baseline_floor:
        baseline = baseline_floor

    details = {
        "history_samples": len(history_known),
        "forecast_samples": len(forecast_known),
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
    history_known = history_values

    if not history_known:
        return None, {"history_samples": 0}

    return median(history_known), {"history_samples": len(history_known)}


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
