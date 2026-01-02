"""Unit tests for price baseline utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing the integration modules directly from this directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from const import PRICE_BASELINE_FLOOR  # noqa: E402
from price_utils import (
    classify_price,
    compute_absolute_low_price_threshold,
    compute_price_baseline,
    price_label_from_ratio,
)  # noqa: E402


def test_price_baseline_uses_window_and_forecast() -> None:
    history = [100.0] * 100 + [10.0] * 100
    baseline, details = compute_price_baseline(
        history=history,
        forecast=[20.0, 20.0],
        time_step_hours=0.25,
        window_hours=24,
        baseline_floor=PRICE_BASELINE_FLOOR,
    )
    assert baseline == pytest.approx(10.0)
    assert details["history_samples"] == 96
    assert details["forecast_samples"] == 8


def test_price_baseline_ignores_non_positive_values() -> None:
    baseline, details = compute_price_baseline(
        history=[0.0, -1.0],
        forecast=[0.0, -0.5],
        time_step_hours=1.0,
        window_hours=24,
        baseline_floor=PRICE_BASELINE_FLOOR,
    )
    assert baseline == pytest.approx(PRICE_BASELINE_FLOOR)
    assert details["history_samples"] == 0
    assert details["forecast_samples"] == 0


def test_price_baseline_uses_forecast_when_no_history() -> None:
    baseline, details = compute_price_baseline(
        history=[],
        forecast=[0.5, 1.5],
        time_step_hours=1.0,
        window_hours=48,
        baseline_floor=PRICE_BASELINE_FLOOR,
    )
    assert baseline == pytest.approx(1.0)
    assert details["history_samples"] == 0
    assert details["forecast_samples"] == 2


def test_price_baseline_does_not_expand_subhourly_forecast() -> None:
    forecast = [0.2] * 96  # 24h of 15-minute data
    baseline, details = compute_price_baseline(
        history=[],
        forecast=forecast,
        time_step_hours=0.25,
        window_hours=48,
        baseline_floor=PRICE_BASELINE_FLOOR,
    )
    assert baseline == pytest.approx(0.2)
    assert details["forecast_samples"] == len(forecast)


def test_price_classification_labels() -> None:
    assert price_label_from_ratio(0.7) == "very_low"
    assert price_label_from_ratio(0.85) == "low"
    assert price_label_from_ratio(1.0) == "normal"
    assert price_label_from_ratio(1.2) == "high"
    assert price_label_from_ratio(1.4) == "very_high"
    assert price_label_from_ratio(2.0) == "extreme"

    ratio, label = classify_price(0.2, 0.1)
    assert ratio == pytest.approx(2.0)
    assert label == "extreme"


def test_absolute_low_threshold_from_history() -> None:
    threshold, details = compute_absolute_low_price_threshold(
        history=[0.0, -1.0, 0.2, 0.3, 0.4],
        time_step_hours=1.0,
        window_hours=24,
    )
    assert threshold == pytest.approx(0.3)
    assert details["history_samples"] == 3


def test_absolute_low_threshold_respects_window() -> None:
    history = list(range(1, 101))
    threshold, details = compute_absolute_low_price_threshold(
        history=history,
        time_step_hours=1.0,
        window_hours=24,
    )
    assert threshold == pytest.approx(88.5)
    assert details["history_samples"] == 24


def test_classification_caps_at_normal_when_absolute_low() -> None:
    ratio, label = classify_price(0.5, 0.25, absolute_low_threshold=0.6)
    assert ratio == pytest.approx(2.0)
    assert label == "normal"


def test_classification_no_cap_when_threshold_none() -> None:
    ratio, label = classify_price(0.5, 0.25, absolute_low_threshold=None)
    assert ratio == pytest.approx(2.0)
    assert label == "extreme"
