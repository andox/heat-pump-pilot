"""Sensor warnings use report freshness rather than temperature changes."""
import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entity():
    source = Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot/climate.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name in ('_entity_problem', '_collect_sensor_issues')]
    namespace = dict(timedelta=timedelta, STATE_UNKNOWN='unknown', STATE_UNAVAILABLE='unavailable',
                     dt_util=SimpleNamespace(as_utc=lambda t: t.astimezone(timezone.utc)))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    readings = {e: SimpleNamespace(state='20', last_updated=now, last_reported=now)
                for e in ('indoor', 'outdoor', 'supply', 'price', 'weather')}
    cls = type('Entity', (), {name: namespace[name] for name in ('_entity_problem', '_collect_sensor_issues')})
    obj = cls()
    obj.hass = SimpleNamespace(states=SimpleNamespace(get=readings.get))
    obj._indoor_temp_entity, obj._outdoor_temp_entity = 'indoor', 'outdoor'
    obj._heating_supply_temp_entity = 'supply'
    obj._price_entity, obj._weather_entity = 'price', 'weather'
    obj._heating_detection_active = lambda: True
    obj._monitor_only = True
    obj._options = {}
    return obj, readings, now


def test_unchanged_temperature_with_recent_report_is_fresh(entity):
    obj, readings, now = entity
    for name in ('indoor', 'outdoor', 'supply'):
        readings[name].last_updated = now - timedelta(days=2)
    assert obj._collect_sensor_issues(now) == []


def test_supply_allows_quiet_period_but_warns_after_six_hours(entity):
    obj, readings, now = entity
    readings['supply'].last_reported = now - timedelta(minutes=70)
    assert obj._collect_sensor_issues(now) == []
    readings['supply'].last_reported = now - timedelta(hours=7)
    assert obj._collect_sensor_issues(now) == [('supply', 'stale', 7 * 3600)]


@pytest.mark.parametrize('state', ['unknown', 'unavailable'])
def test_unavailable_is_reported_even_with_recent_timestamp(entity, state):
    obj, readings, now = entity
    readings['supply'].state = state
    assert obj._collect_sensor_issues(now) == [('supply', 'unavailable', None)]


@pytest.mark.parametrize('missing', [True, False])
def test_fallback_when_last_reported_is_absent_or_none(entity, missing):
    obj, readings, now = entity
    if missing:
        del readings['supply'].last_reported
    else:
        readings['supply'].last_reported = None
    readings['supply'].last_updated = now - timedelta(hours=7)
    assert obj._collect_sensor_issues(now) == [('supply', 'stale', 7 * 3600)]


def test_missing_sensor_is_still_reported(entity):
    obj, readings, now = entity
    del readings['supply']
    assert obj._collect_sensor_issues(now) == [('supply', 'missing', None)]


def test_each_ufh_switch_is_checked_and_recovery_clears_issue(entity):
    obj, readings, now = entity
    obj._options = {'ufh_enabled': True, 'ufh_switches': ['switch.a', 'switch.b']}
    readings['switch.a'] = SimpleNamespace(state='unavailable')
    assert obj._collect_sensor_issues(now) == [
        ('switch.a', 'unavailable', None), ('switch.b', 'missing', None)]
    readings['switch.a'].state = 'off'
    readings['switch.b'] = SimpleNamespace(state='unknown')
    assert obj._collect_sensor_issues(now) == [('switch.b', 'unavailable', None)]
    readings['switch.b'].state = 'on'
    assert obj._collect_sensor_issues(now) == []


def test_stable_pump_switch_does_not_expire(entity):
    obj, readings, now = entity
    obj._options = {'ufh_enabled': True, 'ufh_switches': ['switch.a']}
    readings['switch.a'] = SimpleNamespace(state='off', last_updated=now-timedelta(days=30),
                                          last_reported=now-timedelta(days=30))
    assert obj._collect_sensor_issues(now) == []


def test_disabled_ufh_does_not_warn_about_its_switches(entity):
    obj, readings, now = entity
    obj._options = {'ufh_enabled': False, 'ufh_switches': ['switch.a']}
    assert obj._collect_sensor_issues(now) == []


def test_shared_controlled_switch_is_not_listed_twice(entity):
    obj, readings, now = entity
    obj._monitor_only = False
    obj._controlled_entity = 'switch.a'
    obj._options = {'ufh_enabled': True, 'ufh_switches': ['switch.a', 'switch.a']}
    assert obj._collect_sensor_issues(now) == [('switch.a', 'missing', None)]
