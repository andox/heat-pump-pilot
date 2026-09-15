"""Unit tests for performance metrics helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
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


ORIGIN = datetime(2026, 9, 13, tzinfo=timezone.utc)


def observation(minute, *, temp=20, price=1, heating=False):
    return PerformanceSample(ORIGIN + timedelta(minutes=minute), temp, 20, heating, price, None)


def bounds(minutes=60):
    return dict(now=ORIGIN + timedelta(minutes=minutes), window_start=ORIGIN, max_gap_minutes=15)


def test_comfort_weights_elapsed_time_not_update_count():
    samples = [observation(i) for i in range(15)]
    samples += [observation(i, temp=22) for i in (15, 30, 45, 60)]
    score, details = compute_comfort_score(samples, 0.2, **bounds())
    assert score == pytest.approx(25)
    assert details['too_warm_pct'] == pytest.approx(75)
    assert details['too_cold_pct'] == 0
    assert details['warm_degree_hours'] == pytest.approx(1.35)
    assert details['coverage_pct'] == pytest.approx(100)


def test_comfort_preserves_gap_and_window_boundary():
    samples = [observation(-5), observation(40, temp=19), observation(50, temp=20.2)]
    score, details = compute_comfort_score(samples, 0.2, **bounds())
    assert details['covered_hours'] == pytest.approx(0.5)
    assert details['coverage_pct'] == pytest.approx(50)
    assert score == pytest.approx(200 / 3)
    assert details['cold_degree_hours'] == pytest.approx(0.8 / 6)


def test_comfort_duplicate_and_future_observations_do_not_add_time():
    samples = [observation(0, temp=18), observation(0), observation(15, temp=22), observation(90)]
    score, details = compute_comfort_score(samples, 0.2, **bounds(30))
    assert score == pytest.approx(50)
    assert details['covered_hours'] == pytest.approx(0.5)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None, 'unavailable'])
def test_scores_skip_invalid_values_without_filling_the_gap(value):
    samples = [observation(0, temp=value, price=value), observation(15)]
    _, comfort = compute_comfort_score(samples, 0.2, **bounds(30))
    _, price = compute_price_score(samples, **bounds(30))
    assert comfort['coverage_pct'] == pytest.approx(50)
    assert price['coverage_pct'] == pytest.approx(50)


@pytest.mark.parametrize('tolerance', [-1, float('nan'), float('inf')])
def test_invalid_comfort_tolerance(tolerance):
    score, details = compute_comfort_score([observation(0)], tolerance)
    assert score is None
    assert details['reason'] == 'invalid_tolerance'


@pytest.mark.parametrize(('prices', 'heating', 'expected', 'reason'), [
    ([1, 1], [True, False], None, 'no_price_variation'),
    ([1, 2], [False, False], None, 'no_heating'),
    ([1, 2], [True, True], None, 'no_idle_comparison'),
    ([1, 2], [None, None], None, 'no_known_heating_and_price'),
    ([1, 2, 3, 4], [True, True, False, False], 100, 'ok'),
    ([1, 2, 3, 4], [False, False, True, True], 0, 'ok'),
    ([1, 2, 3, 4], [True, False, False, True], 50, 'ok'),
    ([-3, -2, 0, 1], [True, True, False, False], 100, 'ok'),
])
def test_price_score_meaning(prices, heating, expected, reason):
    samples = [observation(i * 15, price=p, heating=h) for i, (p, h) in enumerate(zip(prices, heating))]
    score, details = compute_price_score(samples, **bounds(len(samples) * 15))
    assert score == expected
    assert details['reason'] == reason


def test_price_excludes_unknown_heating_from_comparison():
    samples = [observation(0, price=1, heating=True), observation(15, price=2),
               observation(30, price=100, heating=None)]
    score, details = compute_price_score(samples, **bounds(45))
    assert score == 100
    assert details['max_price'] == 2
    assert details['coverage_pct'] == pytest.approx(200 / 3)


def test_price_score_ignores_extra_updates_with_same_state():
    normal = [observation(i * 15, price=p, heating=h) for i, (p, h) in enumerate(
        [(1, True), (2, True), (3, False), (4, True)])]
    busy = normal + [observation(i, price=1, heating=True) for i in range(1, 15)]
    score, details = compute_price_score(normal, **bounds())
    busy_score, busy_details = compute_price_score(busy, **bounds())
    assert busy_score == pytest.approx(score)
    assert busy_details['avg_price_when_heating'] == pytest.approx(details['avg_price_when_heating'])
    assert busy_details['heating_hours'] == pytest.approx(0.75)


def test_price_bounds_allow_partial_price_intervals():
    samples = [observation(0, price=1, heating=True), observation(10, price=2), observation(25, price=3)]
    score, details = compute_price_score(samples, **bounds(30))
    assert score == 100
    assert details['best_possible_heating_price'] == 1
    assert details['worst_possible_heating_price'] == pytest.approx(2.5)


def test_only_new_observation_has_no_elapsed_coverage():
    for function, args in [(compute_price_score, ()), (compute_comfort_score, (0.2,))]:
        score, details = function([observation(60)], *args, **bounds())
        assert score is None
        assert details['coverage_pct'] == 0


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


def test_curve_recommendation_uses_requested_duty_ratio_in_continuous_mode() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.25),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.25),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.25),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.25),
    ]
    recommendation, details = compute_curve_recommendation(
        samples,
        min_samples=4,
        idle_ratio_threshold=0.3,
        active_ratio_threshold=0.2,
        idle_request_threshold=0.1,
    )
    assert recommendation == "ok"
    assert details["idle_samples"] == 0
    assert details["active_samples"] == 4


def test_curve_recommendation_treats_low_duty_ratio_as_idle() -> None:
    now = datetime(2025, 12, 20, 12, 0, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.05),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.05),
        PerformanceSample(now, 20.0, 20.0, True, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.05),
        PerformanceSample(now, 20.0, 20.0, False, 0.2, None, suggested_heat_on=False, requested_duty_ratio=0.05),
    ]
    recommendation, details = compute_curve_recommendation(
        samples,
        min_samples=4,
        idle_ratio_threshold=0.5,
        active_ratio_threshold=0.2,
        idle_request_threshold=0.1,
    )
    assert recommendation == "lower_curve"
    assert details["idle_samples"] == 4
    assert details["idle_heating_ratio"] == pytest.approx(0.75)
