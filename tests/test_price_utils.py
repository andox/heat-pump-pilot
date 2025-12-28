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
from price_utils import classify_price, compute_price_baseline, price_label_from_ratio  # noqa: E402


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
