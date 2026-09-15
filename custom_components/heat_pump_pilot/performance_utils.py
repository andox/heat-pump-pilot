"""Helpers for performance/quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
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
    requested_duty_ratio: float | None = None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _score_intervals(samples, *, now=None, window_start=None, max_gap_minutes=15):
    """Hold each observation forward, capped at one expected control interval.

    Missing time is not extrapolated across outages. Duplicate timestamps use
    the latest observation. With no time bounds, retain equal-sample weighting
    for callers comparing already uniformly sampled data.
    """
    if now is None:
        return [(s, 1.0) for s in samples], {"weighting": "samples"}
    end = now.timestamp()
    start = window_start.timestamp() if window_start is not None else float("-inf")
    gap = _finite(max_gap_minutes)
    if gap is None or gap <= 0:
        raise ValueError("max_gap_minutes must be finite and positive")
    by_time = {}
    for sample in samples:
        try:
            when = sample.when.timestamp()
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if isfinite(when) and when <= end:
            by_time[when] = sample
    ordered = sorted(by_time.items())
    intervals = []
    for i, (when, sample) in enumerate(ordered):
        next_time = ordered[i + 1][0] if i + 1 < len(ordered) else end
        seconds = min(next_time, when + gap * 60, end) - max(when, start)
        if seconds > 0:
            intervals.append((sample, seconds))
    return intervals, {"weighting": "elapsed_time", "max_gap_minutes": gap}


def _coverage(details, weight, *, now, window_start):
    if now is not None:
        details["covered_hours"] = weight / 3600
        if window_start is not None:
            window = (now - window_start).total_seconds()
            details["coverage_pct"] = 100 * weight / window if window > 0 else 0.0


def compute_comfort_score(
    samples: Iterable[PerformanceSample], tolerance: float, *,
    now=None, window_start=None, max_gap_minutes=15,
) -> tuple[float | None, dict[str, Any]]:
    """Time in the comfort band, with separate cold and warm diagnostics."""
    intervals, details = _score_intervals(
        samples, now=now, window_start=window_start, max_gap_minutes=max_gap_minutes)
    tolerance = _finite(tolerance)
    if tolerance is None or tolerance < 0:
        return None, {**details, "samples": 0, "reason": "invalid_tolerance"}
    rows = []
    for sample, weight in intervals:
        temp, target = _finite(sample.indoor_temp), _finite(sample.target_temp)
        if temp is not None and target is not None:
            rows.append((temp, target, temp - target, weight))
    total = sum(r[3] for r in rows)
    details.update(samples=len(rows), tolerance=tolerance)
    _coverage(details, total, now=now, window_start=window_start)
    if not total:
        return None, {**details, "reason": "insufficient_data"}
    # Tiny numerical noise at the boundary must not turn exactly 20.2 into
    # discomfort at a target of 20 and tolerance of 0.2.
    cold = sum(w for _, _, e, w in rows if e < -tolerance - 1e-9)
    warm = sum(w for _, _, e, w in rows if e > tolerance + 1e-9)
    score = 100 * (total - cold - warm) / total
    details.update({
        "within_tolerance_pct": score,
        "too_cold_pct": 100 * cold / total,
        "too_warm_pct": 100 * warm / total,
        "mean_abs_error": sum(abs(e) * w for _, _, e, w in rows) / total,
        "mean_signed_error": sum(e * w for _, _, e, w in rows) / total,
        "max_abs_error": max(abs(e) for _, _, e, _ in rows),
        "mean_indoor_temp": sum(t * w for t, _, _, w in rows) / total,
        "mean_target_temp": sum(t * w for _, t, _, w in rows) / total,
    })
    if now is not None:
        details["cold_degree_hours"] = sum(max(0, -e - tolerance) * w for _, _, e, w in rows) / 3600
        details["warm_degree_hours"] = sum(max(0, e - tolerance) * w for _, _, e, w in rows) / 3600
    return score, details


def compute_price_score(
    samples: Iterable[PerformanceSample], *, now=None, window_start=None, max_gap_minutes=15,
) -> tuple[float | None, dict[str, Any]]:
    """Score observed heat timing against equal-runtime price opportunities.

    50 means the average available price, 100 the cheapest possible allocation
    of the same heating duration, and 0 the most expensive. These theoretical
    bounds ignore thermal constraints and are not an estimate of money saved.
    """
    intervals, details = _score_intervals(
        samples, now=now, window_start=window_start, max_gap_minutes=max_gap_minutes)
    rows = []
    for sample, weight in intervals:
        price = _finite(sample.price)
        if price is not None and isinstance(sample.heating_detected, bool):
            rows.append((price, weight, sample.heating_detected))
    total = sum(w for _, w, _ in rows)
    heating = sum(w for _, w, on in rows if on)
    details.update({"samples": len(rows), "heating_samples": sum(on for _, _, on in rows),
                    "method": "equal_runtime_price_opportunity_v2"})
    _coverage(details, total, now=now, window_start=window_start)
    if not total:
        return None, {**details, "reason": "no_known_heating_and_price"}
    avg = sum(p * w for p, w, _ in rows) / total
    heat_avg = sum(p * w for p, w, on in rows if on) / heating if heating else None
    details.update({
        "heating_ratio": heating / total, "avg_price": avg,
        "avg_price_when_heating": heat_avg,
        "min_price": min(p for p, _, _ in rows), "max_price": max(p for p, _, _ in rows),
    })
    if now is not None:
        details["heating_hours"] = heating / 3600
    if not heating:
        return None, {**details, "reason": "no_heating"}
    details["price_advantage_per_kwh_proxy"] = avg - heat_avg
    if heating >= total:
        return None, {**details, "reason": "no_idle_comparison"}

    def allocation_average(reverse):
        remaining, cost = heating, 0.0
        for price, weight, _ in sorted(rows, key=lambda r: r[0], reverse=reverse):
            used = min(weight, remaining)
            cost += price * used
            remaining -= used
            if remaining <= 0:
                break
        return cost / heating

    best, worst = allocation_average(False), allocation_average(True)
    details.update(best_possible_heating_price=best, worst_possible_heating_price=worst)
    if worst - best <= 1e-9 * max(1.0, abs(best), abs(worst)):
        return None, {**details, "reason": "no_price_variation"}
    if heat_avg <= avg:
        score = 50 + 50 * (avg - heat_avg) / (avg - best) if avg > best else 50.0
    else:
        score = 50 - 50 * (heat_avg - avg) / (worst - avg) if worst > avg else 50.0
    return max(0.0, min(100.0, score)), {**details, "reason": "ok"}


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
    idle_request_threshold: float = 0.1,
) -> tuple[str, dict[str, Any]]:
    """Recommend curve adjustments based on heating detected vs effective heat request.

    In continuous-control mode, a boolean `suggested_heat_on=False` does not
    necessarily mean "idle" if the requested duty ratio is still above zero.
    This function therefore prefers `requested_duty_ratio` when available and
    falls back to legacy boolean intent for older samples.
    """
    idle_samples = 0
    idle_heating = 0
    active_samples = 0
    active_heating = 0
    request_threshold = max(0.0, min(1.0, float(idle_request_threshold)))

    for sample in samples:
        if sample.heating_detected is None:
            continue
        requested = sample.requested_duty_ratio
        if requested is None:
            if sample.suggested_heat_on is None:
                continue
            requested = 1.0 if sample.suggested_heat_on else 0.0
        try:
            requested = max(0.0, min(1.0, float(requested)))
        except (TypeError, ValueError):
            continue

        if requested > request_threshold:
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
        "idle_request_threshold": request_threshold,
    }
    return recommendation, details
