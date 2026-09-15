"""Exercise the real summary method without requiring Home Assistant."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from performance_utils import (
    PerformanceSample, compute_comfort_score, compute_price_score,
    compute_prediction_accuracy,
)


def test_summary_weights_history_and_includes_window_boundary():
    source = Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot/climate.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_compute_performance_summary')
    namespace = dict(
        Any=object, timedelta=timedelta,
        dt_util=SimpleNamespace(as_utc=lambda t: t.astimezone(timezone.utc)),
        compute_comfort_score=compute_comfort_score,
        compute_price_score=compute_price_score,
        compute_prediction_accuracy=compute_prediction_accuracy,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    now = datetime(2026, 9, 13, 12, tzinfo=timezone.utc)
    samples = [
        PerformanceSample(now - timedelta(minutes=65), 20, 20, True, 1, None),
        PerformanceSample(now - timedelta(minutes=10), 22, 20, False, 3, None),
        PerformanceSample(now, 20, 20, False, 2, None),
    ]
    entity = SimpleNamespace(_performance_window_hours=1, _control_interval=15,
                             _comfort_tolerance=0.2, _performance_history=samples)
    summary = namespace['_compute_performance_summary'](entity, now)
    assert summary['comfort_score'] == pytest.approx(50)
    assert summary['comfort_details']['covered_hours'] == pytest.approx(1 / 3)
    assert summary['price_score'] == 100
    assert summary['price_details']['heating_hours'] == pytest.approx(1 / 6)
