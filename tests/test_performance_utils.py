"""Unit tests for performance metrics helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from performance_utils import (  # noqa: E402
    PerformanceSample,
    compute_comfort_score,
    compute_curve_recommendation,
    compute_prediction_accuracy,
    compute_price_score,
)


def test_comfort_score_within_tolerance() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 21.0, 20.0, False, 0.2, None),
        PerformanceSample(now, 20.4, 20.0, False, 0.2, None),
        PerformanceSample(now, 19.6, 20.0, False, 0.2, None),
    ]
    score, details = compute_comfort_score(samples, tolerance=1.0)
    assert score == pytest.approx(100.0)
    assert details["samples"] == 3
    assert details["max_abs_error"] == pytest.approx(1.0)


def test_price_score_prefers_low_price_heating() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, True, 1.0, None),
        PerformanceSample(now, 20.0, 20.0, False, 2.0, None),
        PerformanceSample(now, 20.0, 20.0, False, 3.0, None),
    ]
    score, details = compute_price_score(samples)
    assert score == pytest.approx(100.0)
    assert details["heating_samples"] == 1


def test_prediction_accuracy_metrics() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, False, 0.2, 1.0),
        PerformanceSample(now, 20.0, 20.0, False, 0.2, -1.0),
    ]
    mae, details = compute_prediction_accuracy(samples)
    assert mae == pytest.approx(1.0)
    assert details["rmse"] == pytest.approx(1.0)
    assert details["bias"] == pytest.approx(0.0)
    assert details["max_abs_error"] == pytest.approx(1.0)
    assert details["last_error"] == pytest.approx(-1.0)


def test_curve_recommendation_lower_curve_when_heating_while_idle() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False),
        PerformanceSample(now, 20.0, 20.0, False, 0.2, None, suggested_heat_on=False),
    ]
    recommendation, details = compute_curve_recommendation(
        samples,
        min_samples=4,
        idle_ratio_threshold=0.5,
        active_ratio_threshold=0.2,
    )
    assert recommendation == "lower_curve"
    assert details["idle_heating_ratio"] == pytest.approx(0.75)


def test_curve_recommendation_raise_curve_when_no_heating_on_request() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, False, 0.2, None, suggested_heat_on=True),
        PerformanceSample(now, 20.0, 20.0, False, 0.2, None, suggested_heat_on=True),
        PerformanceSample(now, 20.0, 20.0, False, 0.2, None, suggested_heat_on=True),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=True),
    ]
    recommendation, details = compute_curve_recommendation(
        samples,
        min_samples=4,
        idle_ratio_threshold=0.3,
        active_ratio_threshold=0.2,
    )
    assert recommendation == "raise_curve"
    assert details["active_heating_ratio"] == pytest.approx(0.25)


def test_curve_recommendation_insufficient_data() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=True),
    ]
    recommendation, details = compute_curve_recommendation(
        samples,
        min_samples=3,
        idle_ratio_threshold=0.5,
        active_ratio_threshold=0.2,
    )
    assert recommendation == "insufficient_data"
    assert details["idle_samples"] == 1
    assert details["active_samples"] == 1
