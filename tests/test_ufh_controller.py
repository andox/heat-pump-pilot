"""Temperature-only UFH control, timers and persistence."""
from datetime import datetime, timedelta, timezone

import pytest
from ufh_controller import UfhController
from ufh_settings import validate_ufh

START = datetime(2026, 9, 13, 12, 59, tzinfo=timezone.utc)


def controller(**changes):
    return UfhController(dict(ufh_enabled=True, ufh_supply_entity='sensor.supply',
        ufh_switches=['switch.a','switch.b'], ufh_on_hold_seconds=30,
        ufh_min_on_minutes=1, ufh_min_off_minutes=1, ufh_overrun_minutes=1, **changes))


def tick(c, seconds, temp=20, a='off', b='off', **kwargs):
    date = START + timedelta(seconds=seconds)
    return c.evaluate(date.timestamp(), date, temp, {'switch.a':a,'switch.b':b}, **kwargs)


def test_disabled_by_default_and_monitor_only_never_command():
    c = UfhController({})
    assert tick(c,0,40)=={}
    c = controller()
    for t in range(0,181,10):
        assert tick(c,t,40,monitor_only=True)=={}
    assert c.status=='monitor_only'


def test_warm_supply_starts_both_without_any_mpc_or_summer_inputs():
    c=controller()
    assert tick(c,0,35)=={}
    assert tick(c,30,35)=={}
    assert tick(c,60,35)=={'switch.a':True,'switch.b':True}


def test_hot_hold_resets_after_a_temperature_dip():
    c=controller()
    tick(c,0,20);tick(c,60,35);tick(c,80,29)
    assert tick(c,90,35)=={}
    assert tick(c,119,35)=={}
    assert tick(c,120,35)=={'switch.a':True,'switch.b':True}


def test_individual_switch_timers_and_unknown_member():
    c=controller()
    tick(c,0,35,a='on',b='off')
    assert tick(c,60,35,a='on',b='off')=={'switch.b':True}
    assert tick(c,70,35,a='on',b='unavailable')=={}
    assert c.reasons['switch.b']=='switch_unavailable'
    assert tick(c,80,35,a='on',b='off')=={}
    assert tick(c,140,35,a='on',b='off')=={'switch.b':True}


def test_stop_requires_both_minimum_run_and_continuous_cold():
    c=controller()
    tick(c,0,35,a='on',b='on')
    tick(c,10,20,a='on',b='on')
    assert tick(c,60,20,a='on',b='on')=={}
    tick(c,65,26,a='on',b='on')
    tick(c,70,20,a='on',b='on')
    assert tick(c,129,20,a='on',b='on')=={}
    assert tick(c,130,20,a='on',b='on')=={'switch.a':False,'switch.b':False}


@pytest.mark.parametrize('temp,fresh',[(None,True),(float('nan'),True),(float('inf'),True),(20,False)])
def test_sensor_fault_never_stops_or_starts_and_resets_cold_timer(temp,fresh):
    c=controller()
    tick(c,0,20,a='on',b='off')
    assert tick(c,60,temp,a='on',b='off',sensor_fresh=fresh)=={}
    assert c.status=='sensor_unavailable_or_stale'
    assert tick(c,70,20,a='on',b='off')=={}
    assert tick(c,130,20,a='on',b='off')=={'switch.a':False}


def test_long_observation_gap_does_not_count_as_continuous_cold():
    c=controller();tick(c,0,20,a='on',b='on')
    assert tick(c,300,20,a='on',b='on')=={}
    assert tick(c,360,20,a='on',b='on')=={'switch.a':False,'switch.b':False}


def test_restart_restores_minimum_runtime_but_not_temperature_qualification():
    c=controller();c.options['ufh_min_on_minutes']=5
    tick(c,0,20,a='on',b='on');tick(c,60,20,a='on',b='on')
    other=controller();other.options['ufh_min_on_minutes']=5
    other.restore(c.export_state(),START.timestamp()+70)
    for t in range(70,300,10):assert tick(other,t,20,a='on',b='on')=={}
    assert tick(other,300,20,a='on',b='on')=={'switch.a':False,'switch.b':False}


def exercise_controller():
    c=controller(ufh_exercise_enabled=True,ufh_exercise_idle_hours=0,ufh_exercise_minutes=1)
    c.options['ufh_min_on_minutes']=60
    return c


def start_exercise(c):
    tick(c,0,20)
    assert tick(c,60,20)=={'switch.a':True,'switch.b':True}
    tick(c,61,20,a='on',b='on')


def test_exercise_ends_independently_of_normal_minimum_run():
    c=exercise_controller();start_exercise(c)
    assert tick(c,119,20,a='on',b='on')=={}
    assert tick(c,120,20,a='on',b='on')=={'switch.a':False,'switch.b':False}
    assert tick(c,121,20)=={}


def test_hot_supply_takes_over_exercise():
    c=exercise_controller();start_exercise(c)
    tick(c,70,35,a='on',b='on');tick(c,100,35,a='on',b='on')
    assert tick(c,120,35,a='on',b='on')=={}
    assert c.pumps['switch.a'].exercise_until is None
    assert tick(c,130,20,a='on',b='on')=={}
    assert tick(c,190,20,a='on',b='on')=={}  # Normal minimum runtime applies.


def test_exercise_keeps_useful_warm_water_without_reaching_start_threshold():
    c=exercise_controller();start_exercise(c)
    assert tick(c,120,27,a='on',b='on')=={}
    assert c.pumps['switch.a'].exercise_until is None


def test_exercise_sensor_failure_prevents_blind_shutdown():
    c=exercise_controller();start_exercise(c)
    assert tick(c,120,None,a='on',b='on')=={}
    assert tick(c,130,35,a='on',b='on')=={}


def test_exercise_restart_and_daily_guard():
    c=exercise_controller();start_exercise(c)
    other=exercise_controller();other.restore(c.export_state(),START.timestamp()+80)
    assert tick(other,80,20,a='on',b='on')=={}
    assert tick(other,120,20,a='on',b='on')=={'switch.a':False,'switch.b':False}
    tick(other,121,20)
    # Repeated local clock hour (DST) must not repeat exercise on the same date.
    assert other.evaluate(START.timestamp()+500,START+timedelta(seconds=60),20,{'switch.a':'off','switch.b':'off'})=={}


def test_failed_exercise_start_is_retried_without_resetting_deadline():
    c=exercise_controller();tick(c,0)
    tick(c,60)
    assert tick(c,70)=={'switch.a':True,'switch.b':True}
    assert c.pumps['switch.a'].exercise_until==START.timestamp()+120


def test_disabling_leaves_states_alone_and_clears_exercise_ownership():
    c=exercise_controller();start_exercise(c)
    c.configure({**c.options,'ufh_enabled':False})
    assert tick(c,120,20,a='on',b='on')=={}
    assert all(p.exercise_until is None for p in c.pumps.values())


def test_corrupt_persistence_is_ignored():
    c=controller();c.restore({'version':1,'pumps':{'switch.a':{'state':'on','since':float('nan')}}},START.timestamp())
    assert c.pumps=={}


@pytest.mark.parametrize('patch,error',[
    ({'ufh_enabled':True},'ufh_missing_entities'),
    ({'ufh_on_temperature':20,'ufh_off_temperature':25},'ufh_invalid_thresholds'),
    ({'ufh_exercise_time':'25:00'},'ufh_invalid_settings'),
    ({'ufh_switches':None},'ufh_invalid_settings'),
    ({'ufh_min_on_minutes':float('nan')},'ufh_invalid_settings'),
])
def test_configuration_rejects_invalid_control_settings(patch,error):
    assert validate_ufh(patch)==error


def test_exercise_only_runs_individually_idle_pumps():
    c=exercise_controller()
    c.options['ufh_exercise_idle_hours']=1
    tick(c,0,20,a='off',b='on')
    later=START+timedelta(days=1,seconds=60)
    result=c.evaluate(later.timestamp(),later,20,{'switch.a':'off','switch.b':'on'})
    assert result=={'switch.a':True}
    assert c.pumps['switch.b'].exercise_until is None


def test_missed_daily_exercise_is_not_replayed_after_restart():
    c=exercise_controller()
    tick(c,0)
    assert tick(c,180)=={}


def test_invalid_saved_options_do_not_generate_commands():
    c=UfhController({'ufh_enabled':True,'ufh_switches':None})
    assert tick(c,0,35)=={}
    assert c.status=='ufh_invalid_settings'


def test_restart_unavailable_off_preserves_daily_exercise_eligibility():
    c=controller(ufh_exercise_enabled=True,ufh_exercise_idle_hours=24)
    tick(c,0)
    saved=c.export_state()
    other=controller(ufh_exercise_enabled=True,ufh_exercise_idle_hours=24)
    other.restore(saved,START.timestamp()+86400-120)
    assert tick(other,86400-120,a=None,b='unavailable')=={}
    assert tick(other,86400-110)=={}
    assert all(p.since==START.timestamp() for p in other.pumps.values())
    # At 13:00 the existing 24h history still qualifies both pumps.
    assert tick(other,86400+60)=={'switch.a':True,'switch.b':True}


def test_reconnection_preserves_history_but_actual_run_resets_idle():
    c=controller()
    tick(c,0)
    tick(c,400,a='unavailable')
    tick(c,410)
    assert c.pumps['switch.a'].since==START.timestamp()
    assert c.pumps['switch.a'].control_since==START.timestamp()+410
    tick(c,420,a='on')
    tick(c,500)
    assert c.pumps['switch.a'].since==START.timestamp()+500
    assert c.pumps['switch.b'].since==START.timestamp()


def test_restart_during_exercise_with_unavailable_switch_preserves_deadline():
    c=exercise_controller();start_exercise(c)
    other=exercise_controller();other.restore(c.export_state(),START.timestamp()+80)
    assert tick(other,80,a=None,b=None)=={}
    assert tick(other,90,a='on',b='on')=={}
    assert other.pumps['switch.a'].since==START.timestamp()+61
    assert tick(other,120,a='on',b='on')=={'switch.a':False,'switch.b':False}
    tick(other,121)
    assert tick(other,122)=={}


def test_legacy_saved_pump_state_restores_without_control_since():
    c=controller()
    c.restore({'version':1,'pumps':{'switch.a':{'state':'off','since':START.timestamp()}}},START.timestamp()+10)
    tick(c,10,a=None)
    tick(c,20)
    assert c.pumps['switch.a'].since==START.timestamp()
