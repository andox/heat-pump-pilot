"""Helpers for performance/quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class PerformanceSample:
    """Point-in-time sample used for performance summaries."""

    when: Any
    indoor_temp: float
    target_temp: float
    heating_detected: bool | None
    price: float | None
    prediction_error: float | None
    suggested_heat_on: bool | None = None


def compute_comfort_score(samples: Iterable[PerformanceSample], tolerance: float) -> tuple[float | None, dict[str, Any]]:
    """Compute comfort score based on time within tolerance."""
    errors = []
    temps = []
    targets = []
    for sample in samples:
        try:
            error = abs(float(sample.indoor_temp) - float(sample.target_temp))
        except (TypeError, ValueError):
            continue
        errors.append(error)
        temps.append(float(sample.indoor_temp))
        targets.append(float(sample.target_temp))
    if not errors:
        return None, {"samples": 0, "tolerance": tolerance}
    within = sum(1 for err in errors if err <= tolerance)
    score = 100 * within / len(errors)
    details = {
        "samples": len(errors),
        "within_tolerance_pct": score,
        "mean_abs_error": mean(errors),
        "max_abs_error": max(errors),
        "mean_indoor_temp": mean(temps),
        "mean_target_temp": mean(targets),
        "tolerance": tolerance,
    }
    return score, details


def compute_price_score(samples: Iterable[PerformanceSample]) -> tuple[float | None, dict[str, Any]]:
    """Compute a price score based on heating occurring during lower prices."""
    prices: list[float] = []
    heating_prices: list[float] = []
    for sample in samples:
        if sample.price is None:
            continue
        try:
            price = float(sample.price)
        except (TypeError, ValueError):
            continue
        prices.append(price)
        if sample.heating_detected:
            heating_prices.append(price)
    if not prices:
        return None, {"samples": 0}
    min_price = min(prices)
    max_price = max(prices)
    avg_price = mean(prices)
    avg_price_heating = mean(heating_prices) if heating_prices else None
    if avg_price_heating is None:
        score = None
    else:
        span = max(max_price - min_price, 1e-6)
        score = 100 * (1 - (avg_price_heating - min_price) / span)
        score = max(0.0, min(100.0, score))
    details = {
        "samples": len(prices),
        "heating_samples": len(heating_prices),
        "heating_ratio": len(heating_prices) / len(prices) if prices else 0.0,
        "avg_price": avg_price,
        "avg_price_when_heating": avg_price_heating,
        "min_price": min_price,
        "max_price": max_price,
    }
    return score, details


def compute_prediction_accuracy(
    samples: Iterable[PerformanceSample],
) -> tuple[float | None, dict[str, Any]]:
    """Compute prediction accuracy metrics from stored errors."""
    errors = []
    for sample in samples:
        if sample.prediction_error is None:
            continue
        try:
            errors.append(float(sample.prediction_error))
        except (TypeError, ValueError):
            continue
    if not errors:
        return None, {"samples": 0}
    abs_errors = [abs(err) for err in errors]
    mae = mean(abs_errors)
    rmse = sqrt(mean(err * err for err in errors))
    details = {
        "samples": len(errors),
        "mae": mae,
        "rmse": rmse,
        "bias": mean(errors),
        "max_abs_error": max(abs_errors),
        "last_error": errors[-1],
    }
    return mae, details


def compute_curve_recommendation(
    samples: Iterable[PerformanceSample],
    *,
    min_samples: int,
    idle_ratio_threshold: float,
    active_ratio_threshold: float,
) -> tuple[str, dict[str, Any]]:
    """Recommend curve adjustments based on heating detected vs MPC intent."""
    idle_samples = 0
    idle_heating = 0
    active_samples = 0
    active_heating = 0

    for sample in samples:
        if sample.suggested_heat_on is None or sample.heating_detected is None:
            continue
        if sample.suggested_heat_on:
            active_samples += 1
            if sample.heating_detected:
                active_heating += 1
        else:
            idle_samples += 1
            if sample.heating_detected:
                idle_heating += 1

    idle_ratio = (idle_heating / idle_samples) if idle_samples else None
    active_ratio = (active_heating / active_samples) if active_samples else None
    active_miss_ratio = ((active_samples - active_heating) / active_samples) if active_samples else None

    recommendation = "insufficient_data"
    if idle_samples >= min_samples or active_samples >= min_samples:
        recommendation = "ok"
        if idle_samples >= min_samples and idle_ratio is not None and idle_ratio >= idle_ratio_threshold:
            recommendation = "lower_curve"
        elif (
            active_samples >= min_samples
            and active_miss_ratio is not None
            and active_miss_ratio >= active_ratio_threshold
        ):
            recommendation = "raise_curve"

    details = {
        "idle_samples": idle_samples,
        "idle_heating_samples": idle_heating,
        "idle_heating_ratio": idle_ratio,
        "active_samples": active_samples,
        "active_heating_samples": active_heating,
        "active_heating_ratio": active_ratio,
        "active_miss_ratio": active_miss_ratio,
        "min_samples": min_samples,
        "idle_ratio_threshold": idle_ratio_threshold,
        "active_ratio_threshold": active_ratio_threshold,
    }
    return recommendation, details
