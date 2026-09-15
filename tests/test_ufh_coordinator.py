"""Run the actual coordinator with small HA service/event stand-ins."""
import ast
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]/'custom_components/heat_pump_pilot'


def harness(tmp_path):
    source=ROOT/'ufh_coordinator.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module.startswith('homeassistant'))]
    for n in tree.body:
        if isinstance(n,ast.ImportFrom):n.level=0
    now=[datetime(2026,9,13,12,tzinfo=timezone.utc)]
    callbacks=[]
    def track(*args):
        cancel=Mock();callbacks.append((args,cancel));return cancel
    namespace=dict(EVENT_HOMEASSISTANT_STOP='stop',callback=lambda f:f,
        async_dispatcher_send=Mock(),async_track_state_change_event=track,async_track_time_interval=track,
        dt_util=SimpleNamespace(utcnow=lambda:now[0],as_local=lambda x:x),
        TemperatureConverter=SimpleNamespace(convert=lambda v,unit,target:(v-32)*5/9 if unit=='°F' else v))
    exec(compile(tree,str(source),'exec'),namespace)
    def state(value,**attrs):
        return SimpleNamespace(state=value,attributes=attrs,last_updated=now[0],last_reported=now[0])
    states={'sensor.supply':state('35',unit_of_measurement='°C'),
        'switch.a':state('off'),'switch.b':state('off')}
    async def execute(job,*args):return job(*args)
    services=AsyncMock()
    hass=SimpleNamespace(states=SimpleNamespace(get=states.get),services=SimpleNamespace(async_call=services),
        config=SimpleNamespace(path=lambda *args:str(tmp_path/args[-1])),
        async_add_executor_job=execute,async_create_task=asyncio.create_task,
        bus=SimpleNamespace(async_listen_once=Mock(return_value=Mock())))
    entry=SimpleNamespace(entry_id='test',options={'ufh_enabled':True,'ufh_supply_entity':'sensor.supply',
        'ufh_switches':['switch.a','switch.b'],'ufh_on_hold_seconds':0,'ufh_min_off_minutes':0,
        'hvac_mode':'off','summer_heat_window_enabled':False})
    return namespace['UfhCoordinator'](hass,entry),now,states,services,callbacks,state


def test_multiple_switches_failure_isolated_and_retried(tmp_path):
    c,now,states,services,callbacks,state=harness(tmp_path)
    async def run():
        services.side_effect=[RuntimeError('offline'),None,None,None]
        await c.async_start()
        assert services.await_count==2
        assert c.diagnostics['switch_errors']=={'switch.a':'RuntimeError'}
        now[0]+=timedelta(seconds=10)
        await c.async_refresh()
        assert services.await_count==4
        assert not c.diagnostics['switch_errors']
        await c.async_stop()
        assert all(cancel.called for _,cancel in callbacks)
        await c.async_refresh()
        assert services.await_count==4
    asyncio.run(run())


def test_monitor_mode_holds_states_even_with_hot_supply(tmp_path):
    c,_,_,services,_,_=harness(tmp_path)
    c.entry.options['monitor_only']=True
    asyncio.run(c.async_start())
    services.assert_not_awaited()
    assert c.diagnostics['state']=='monitor_only'
    asyncio.run(c.async_stop())


def test_freshness_uses_reports_and_stale_supply_does_not_command(tmp_path):
    c,now,states,services,_,_=harness(tmp_path)
    states['sensor.supply'].last_updated=now[0]-timedelta(days=1)
    async def run():
        await c.async_start()
        assert services.await_count==2  # Unchanged value with recent report is healthy.
        services.reset_mock()
        now[0]+=timedelta(hours=7)
        await c.async_refresh()
        services.assert_not_awaited()
        assert c.diagnostics['state']=='sensor_unavailable_or_stale'
        await c.async_stop()
    asyncio.run(run())


def test_restored_sensor_is_not_live_evidence(tmp_path):
    c,_,states,services,_,_=harness(tmp_path)
    states['sensor.supply'].attributes['restored']=True
    asyncio.run(c.async_start())
    services.assert_not_awaited()
    asyncio.run(c.async_stop())


def test_options_release_removed_switches_and_preserve_other_timers(tmp_path):
    c,now,states,services,callbacks,_=harness(tmp_path)
    async def run():
        await c.async_start()
        since=c.model.pumps['switch.a'].since
        services.reset_mock()
        c.entry.options={**c.entry.options,'ufh_switches':['switch.a']}
        now[0]+=timedelta(seconds=10)
        await c.async_update_options()
        assert c.model.pumps['switch.a'].since==since
        assert 'switch.b' not in c.model.pumps
        assert all(call.args[2]['entity_id']=='switch.a' for call in services.await_args_list)
        services.reset_mock()
        c.entry.options={**c.entry.options,'ufh_enabled':False}
        await c.async_update_options()
        services.assert_not_awaited()
        await c.async_stop()
    asyncio.run(run())


def test_coordinator_serializes_event_refreshes(tmp_path):
    c,now,states,services,_,_=harness(tmp_path)
    async def run():
        await c.async_start()
        active=0
        maximum=0
        async def service(*args,**kwargs):
            nonlocal active,maximum
            active+=1;maximum=max(maximum,active)
            await asyncio.sleep(0)
            active-=1
        services.side_effect=service
        await asyncio.gather(c.async_refresh(),c.async_refresh())
        assert maximum==1
        await c.async_stop()
    asyncio.run(run())
