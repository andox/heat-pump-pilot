"""Sensor warnings use report freshness rather than temperature changes."""
import ast
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

COMPONENT_DIR = Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot'
if str(COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENT_DIR))
from notification_utils import NotificationTracker, NotificationUpdate


@pytest.fixture
def entity():
    source = Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot/climate.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method_names = ('_entity_problem', '_collect_sensor_issues', '_async_update_notifications')
    methods = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in method_names]
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id.startswith(('STARTUP_', 'NOTIFY_'))
                         for target in n.targets)]
    namespace = dict(timedelta=timedelta, STATE_UNKNOWN='unknown', STATE_UNAVAILABLE='unavailable',
                     dt_util=SimpleNamespace(as_utc=lambda t: t.astimezone(timezone.utc)))
    exec(compile(ast.Module(body=constants + methods, type_ignores=[]), str(source), 'exec'), namespace)
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    readings = {e: SimpleNamespace(state='20', last_updated=now, last_reported=now)
                for e in ('indoor', 'outdoor', 'supply', 'price', 'weather')}
    cls = type('Entity', (), {name: namespace[name] for name in method_names})
    obj = cls()
    obj.hass = SimpleNamespace(states=SimpleNamespace(get=readings.get))
    obj._indoor_temp_entity, obj._outdoor_temp_entity = 'indoor', 'outdoor'
    obj._heating_supply_temp_entity = 'supply'
    obj._price_entity, obj._weather_entity = 'price', 'weather'
    obj._heating_detection_active = lambda: True
    obj._monitor_only = True
    obj._options = {}
    obj._startup_time = now - timedelta(hours=2)
    obj._attr_name = 'Heat Pump Pilot'
    obj._notification_tracker = NotificationTracker()
    obj._compute_health = lambda _: ('healthy', [])
    obj._last_price_forecast_source = 'sensor'
    obj._last_outdoor_forecast_source = 'weather'
    obj._format_age = lambda age: f'{age / 3600:.1f}h'
    return obj, readings, now


def test_unchanged_temperature_with_recent_report_is_fresh(entity):
    obj, readings, now = entity
    for name in ('indoor', 'outdoor', 'supply'):
        readings[name].last_updated = now - timedelta(days=2)
    assert obj._collect_sensor_issues(now) == []


@pytest.mark.parametrize('sensor', ['indoor', 'outdoor', 'supply'])
def test_temperature_allows_quiet_day_but_warns_after_24_hours(entity, sensor):
    obj, readings, now = entity
    readings[sensor].last_reported = now - timedelta(hours=7)
    assert obj._collect_sensor_issues(now) == []
    readings[sensor].last_reported = now - timedelta(hours=24)
    assert obj._collect_sensor_issues(now) == []
    readings[sensor].last_reported = now - timedelta(hours=24, seconds=1)
    assert obj._collect_sensor_issues(now) == [(sensor, 'stale', 24 * 3600 + 1)]
    # A new report of the same value is enough to clear the stale issue.
    readings[sensor].last_reported = now
    assert obj._collect_sensor_issues(now) == []


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
    readings['supply'].last_updated = now - timedelta(hours=25)
    assert obj._collect_sensor_issues(now) == [('supply', 'stale', 25 * 3600)]


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


@pytest.mark.parametrize('sensor', ['indoor', 'outdoor', 'supply'])
def test_stale_temperature_notification_created_and_cleared_by_same_value_report(entity, sensor):
    obj, readings, now = entity
    readings[sensor].last_reported = now - timedelta(hours=25)
    readings[sensor].last_updated = readings[sensor].last_reported
    calls = []

    async def apply(updates, messages, when):
        calls.append((updates, messages))

    obj._async_apply_notification_updates = apply

    async def exercise():
        await obj._async_update_notifications(now)
        assert calls[-1][0] == []
        await obj._async_update_notifications(now + timedelta(minutes=5))
        assert calls[-1][0] == [NotificationUpdate('create', 'sensors')]
        title, message = calls[-1][1]['sensors']
        assert title == 'Heat Pump Pilot: Sensor inputs'
        assert f'- {sensor}: stale (last report/update' in message
        # No changed temperature or last_updated timestamp is needed.
        readings[sensor].last_reported = now + timedelta(minutes=6)
        await obj._async_update_notifications(now + timedelta(minutes=6))
        assert calls[-1][0] == [NotificationUpdate('dismiss', 'sensors')]

    asyncio.run(exercise())


def test_restart_grace_does_not_hide_old_sensor_timestamp_indefinitely(entity):
    obj, readings, now = entity
    obj._startup_time = now
    readings['indoor'].last_reported = now - timedelta(hours=25)
    calls = []

    async def apply(updates, messages, when):
        calls.append(updates)

    obj._async_apply_notification_updates = apply

    async def exercise():
        await obj._async_update_notifications(now)
        assert calls[-1] == []
        await obj._async_update_notifications(now + timedelta(minutes=2))
        assert calls[-1] == []
        await obj._async_update_notifications(now + timedelta(minutes=7))
        assert calls[-1] == [NotificationUpdate('create', 'sensors')]

    asyncio.run(exercise())
