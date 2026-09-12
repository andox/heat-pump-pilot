"""Correctness checks for the offline experiment, independent of private exports."""

from dataclasses import replace

import pytest

from tools.analyze_learning import (
    Interval,
    Series,
    STEP,
    build_candidates,
    evaluate,
    fit_bounded,
    hourly_observations,
)


def test_series_is_causal_and_time_weighted():
    series = Series([(0, 0.0), (600, 1.0), (1200, 9.0)], max_age=3600)
    assert series.at(599) == 0.0
    assert series.mean(0, 900) == pytest.approx(1 / 3)
    assert series.at(-1) is None


def test_unknown_or_expired_coverage_is_not_zero_or_normalized_away():
    unknown = Series([(0, 1.0), (300, None), (600, 1.0)])
    assert unknown.mean(0, 900) is None
    expired = Series([(0, 1.0)], max_age=600)
    assert expired.mean(0, 900) is None
    assert expired.mean(0, 600) == 1.0


@pytest.mark.parametrize('bias', [False, True])
def test_regression_recovers_known_parameters(bias):
    background = 0.12 if bias else 0.0
    observations = [((difference, duty), 0.02 * difference + 0.6 * duty + background)
                    for difference in (-20, -10, -5) for duty in (0, 0.5, 1)]
    assert fit_bounded(observations, bias) == pytest.approx((0.02, 0.6, background), abs=1e-6)


def synthetic_intervals(hours=96):
    intervals = {}
    temp = 21.0
    for i in range(hours * 4):
        t = i * STEP
        outdoor = 8.0 + (i % 48) / 8
        duty = 1.0 if i % 24 < 8 else 0.0
        next_temp = temp + 0.25 * (0.02 * (outdoor - temp) + 0.6 * duty)
        intervals[t] = Interval(t, temp, next_temp, outdoor, duty, (0.02, 0.6))
        temp = next_temp
    return intervals


def test_hourly_fit_rejects_incomplete_intervals():
    rows = synthetic_intervals(2)
    del rows[STEP]
    observations = hourly_observations(rows, {t: row.heat for t, row in rows.items()})
    assert [end for end, _, _ in observations] == [7200]


def test_future_observations_cannot_change_earlier_parameters():
    original = synthetic_intervals()
    cutoff = 60 * 3600
    changed = {t: replace(r, indoor=r.indoor + 5, next_indoor=r.next_indoor + 5, outdoor=-10, heat=0.0)
               if t >= cutoff else r for t, r in original.items()}
    before, _ = build_candidates(original, train_end=48 * 3600)
    after, _ = build_candidates(changed, train_end=48 * 3600)
    for name in before:
        for t in before[name]:
            if t <= cutoff or name == 'frozen_regression':
                assert before[name][t] == after[name][t]


def test_evaluation_matches_origins_and_never_crosses_split_or_gap():
    rows = synthetic_intervals()
    del rows[70 * 3600]
    states, inputs = build_candidates(rows, train_end=24 * 3600)
    predictions = evaluate(rows, states, inputs, 24 * 3600, 48 * 3600, 96 * 3600)
    from tools.analyze_learning import timestamp
    grouped = {}
    for p in predictions:
        t = timestamp(p['origin'])
        end = t + p['horizon_hours'] * 3600
        assert not (t <= 70 * 3600 < end)
        assert end <= (48 * 3600 if p['split'] == 'validation' else 96 * 3600)
        key = (p['origin'], p['horizon_hours'])
        grouped.setdefault(key, set()).add(p['model'])
    assert all(names == set(states) for names in grouped.values())
