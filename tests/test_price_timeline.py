"""Price timing, missing coverage, and dated-history regression tests."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from forecast_utils import TimedValue, price_grid
from forecast_service import ForecastService
from price_observations import PriceObservations
from price_utils import compute_price_baseline, compute_absolute_low_price_threshold

NOW = datetime(2026, 9, 13, 7, 30, tzinfo=timezone.utc)


def service(rows, current='1'):
    state = SimpleNamespace(state=current, attributes={'raw_today': rows})
    hass = SimpleNamespace(states=SimpleNamespace(get=lambda entity: state))
    return ForecastService(hass, price_entity='sensor.price', weather_entity='',
        outdoor_temp_entity='', prediction_horizon=24, weather_forecast_cache_seconds=60,
        weather_forecast_service_type='hourly', state_to_float=lambda x: float(x))


@pytest.mark.parametrize('count', [1, 58, 95, 96, 192])
def test_quarter_prices_never_become_hourly_when_forecast_is_short(count):
    rows = [{'start': NOW + timedelta(minutes=15*i),
             'end': NOW + timedelta(minutes=15*(i+1)), 'value': i}
            for i in range(count)]
    svc = service(rows)
    planned, known = svc.build_price_grid(NOW, 96, 0.25)
    for i in range(min(count, 96)):
        assert planned[i] == known[i] == i
    if count < 96:
        assert known[count:] == [None] * (96-count)
        assert planned[count:] == [count-1] * (96-count)


def test_gap_is_not_shifted_or_included_in_baseline():
    rows = [TimedValue(NOW, NOW + timedelta(minutes=15), 0),
            TimedValue(NOW + timedelta(minutes=30), NOW + timedelta(hours=1), -1)]
    grid = price_grid(rows, NOW, 5, .25)
    assert grid == [0, None, -1, -1, None]
    baseline, details = compute_price_baseline(history=[2], forecast=grid,
        time_step_hours=.25, window_hours=24, baseline_floor=.01, forecast_is_step=True)
    assert baseline == .01
    assert details == {'history_samples': 1, 'forecast_samples': 3}


def test_hourly_prices_follow_actual_boundaries_and_expire():
    rows = [TimedValue(NOW, NOW + timedelta(hours=1), 1),
            TimedValue(NOW + timedelta(hours=1), NOW + timedelta(hours=2), 2)]
    assert price_grid(rows, NOW + timedelta(minutes=45), 6, .25) == [1, 2, 2, 2, 2, None]


def test_future_price_is_not_used_before_its_start():
    rows = [{'start': NOW + timedelta(hours=1), 'end': NOW + timedelta(hours=2), 'value': -1}]
    planned, known = service(rows).build_price_grid(NOW, 8, .25)
    assert planned == [1] * 4 + [-1] * 4
    assert known == [None] * 4 + [-1] * 4


def test_dst_repeated_hour_has_distinct_prices():
    first = datetime.fromisoformat('2026-10-25T02:00:00+02:00')
    second = datetime.fromisoformat('2026-10-25T02:00:00+01:00')
    rows = [TimedValue(first, second, 1), TimedValue(second, second + timedelta(hours=1), 2)]
    assert price_grid(rows, first, 8, .25) == [1]*4 + [2]*4


def test_history_windows_use_dates_and_keep_zero_and_negative_prices():
    history = PriceObservations()
    history.add(NOW - timedelta(hours=50), 100)
    history.add(NOW - timedelta(hours=20), -2)
    history.add(NOW - timedelta(hours=1), 0)
    assert history.values(NOW, 48) == [-2, 0]
    restored = PriceObservations()
    restored.restore(list(reversed(history.dump())), NOW)
    assert restored.values(NOW, 48) == [-2, 0]
    threshold, _ = compute_absolute_low_price_threshold(history=restored.values(NOW, 48),
        time_step_hours=.25, window_hours=48)
    assert threshold == -1


def test_recorder_resamples_hourly_changes_and_preserves_unknowns():
    history = PriceObservations()
    history.backfill([(NOW, 1), (NOW+timedelta(hours=1), 2),
                      (NOW+timedelta(minutes=90), None)], NOW+timedelta(hours=3))
    assert list(history.buckets.values()) == [1]*4 + [2]*2


def test_live_history_wins_over_backfill_and_old_prices_expire():
    history = PriceObservations()
    history.add(NOW, 9)
    history.backfill([(NOW, 1)], NOW+timedelta(hours=1))
    assert history.buckets[NOW] == 9
    history.prune(NOW+timedelta(days=31))
    assert history.dump() == []


def test_legacy_history_and_invalid_dates_are_not_invented():
    history = PriceObservations()
    history.restore([1, 2, {'time': 'bad'}, {'time': NOW.isoformat(), 'price': float('nan')},
                     {'time': (NOW+timedelta(days=1)).isoformat(), 'price': 2}], NOW)
    assert history.dump() == []


def test_missing_forecast_slots_do_not_pull_prices_from_outside_window():
    baseline, details = compute_price_baseline(history=[], forecast=[None]*96+[100]*96,
        time_step_hours=.25, window_hours=24, baseline_floor=.01, forecast_is_step=True)
    assert baseline == .01
    assert details['forecast_samples'] == 0


def test_actual_climate_price_storage_migrates_without_inventing_dates():
    import ast
    import asyncio
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot/climate.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
               and n.name in ('_load_price_history', '_persist_price_history')]
    namespace = dict(dt_util=SimpleNamespace(utcnow=lambda: NOW),
                     PRICE_HISTORY_BACKFILL_MIN_SAMPLES=2)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    saved = []
    payload = {'version': 1, 'bucket_minutes': 15, 'history': [100]*100}

    async def execute(func, *args):
        return func(*args)

    entity = SimpleNamespace(
        hass=SimpleNamespace(async_add_executor_job=execute),
        _price_store=SimpleNamespace(load=lambda: payload, save=saved.append),
        _controller=SimpleNamespace(time_step_hours=.25),
        _price_observations=PriceObservations(), _price_backfill_done=False)
    asyncio.run(namespace['_load_price_history'](entity))
    assert entity._price_history == []
    assert entity._price_backfill_done is False
    entity._price_observations.add(NOW-timedelta(minutes=30), -1)
    entity._price_observations.add(NOW-timedelta(minutes=15), 0)
    asyncio.run(namespace['_persist_price_history'](entity))
    payload = saved[0]
    assert payload['version'] == 2
    entity._price_observations = PriceObservations()
    asyncio.run(namespace['_load_price_history'](entity))
    assert entity._price_history == [-1, 0]
    assert entity._price_backfill_done is True
