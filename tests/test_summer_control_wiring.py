"""Exercise the merged summer override in the actual async control method."""
import ast
import asyncio
import logging
from copy import copy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from control_request_utils import resolve_effective_heat_request
from forecast_service import ForecastService
from forecast_utils import expand_to_steps
from price_utils import compute_price_baseline
from virtual_outdoor_utils import compute_duty_ratio


@pytest.mark.parametrize('continuous,response', [(False,False),(True,False),(True,True)])
def test_summer_override_reaches_actuator_and_records_actual_request(continuous,response):
    path=Path(__file__).resolve().parents[1]/'custom_components/heat_pump_pilot/climate.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_async_run_control')
    now=datetime(2026,7,1,12,tzinfo=timezone.utc)
    replay=Mock(return_value=([1.0],[20,20.1],[1.0],[4],0.1))
    namespace=dict(asyncio=asyncio,copy=copy,partial=partial,
        HVACMode=SimpleNamespace(OFF='off'),dt_util=SimpleNamespace(utcnow=lambda:now),
        _LOGGER=logging.getLogger(__name__),PRICE_BASELINE_FLOOR=0.01,
        compute_price_baseline=compute_price_baseline,expand_to_steps=expand_to_steps,
        compute_duty_ratio=compute_duty_ratio,resolve_effective_heat_request=resolve_effective_heat_request,
        replay_response=replay)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    result=SimpleNamespace(sequence=[False],duty_sequence=[0.0] if response else [],
        predicted_temperatures=[20,20],price_baseline=0.2)
    forecast=SimpleNamespace(build_price_grid=lambda n,s,h:([0.2]*s,[0.2]*s),last_price_timed_values=[],
        last_price_forecast_source='test',last_outdoor_forecast_source='test',
        build_outdoor_forecast=AsyncMock(return_value=[20]),normalize_series=ForecastService.normalize_series)
    entity=SimpleNamespace(_hvac_mode='heat',_control_lock=asyncio.Lock(),
        _indoor_temp_entity='sensor.indoor',_outdoor_temp_entity='sensor.outdoor',
        _get_state_as_float=lambda e:20,_compute_prediction_error=lambda *a:None,
        _update_thermal_model=Mock(),_prediction_horizon=1,_price_history=[],
        _price_observations=SimpleNamespace(values=lambda *a:[]),
        _price_baseline_window_hours=24,_forecast_service=forecast,_update_price_history=Mock(),
        _controller=SimpleNamespace(update_comfort_diagnostics=Mock(),time_step_hours=0.25,suggest_control=lambda **kw:(False,result)),
        _options={},_raw_mpc_sequence_head=lambda s:s,_update_summer_heat_window=Mock(return_value=True),
        _continuous_control_enabled=continuous,_continuous_control_window_steps=lambda:4,
        _summer_heat_window_virtual_heat_offset=16,
        _target_temperature=21,_comfort_tolerance=1,
        _build_effective_heat_request=lambda **kw:SimpleNamespace(raw_requested_duty_ratio=0,effective_requested_duty_ratio=0),
        _commit_effective_heat_request=Mock(),_compute_virtual_outdoor=Mock(return_value=4),
        _apply_control=AsyncMock(),_get_heating_detected=lambda n:False,
        _record_performance_sample=Mock(),_publish_decision=Mock(),
        _async_update_notifications=AsyncMock(),_persist_thermal_state=AsyncMock(),
        _persist_performance_history=AsyncMock(),_persist_summer_heat_window_state=AsyncMock(),
        async_write_ha_state=Mock())
    actuator=SimpleNamespace(clamp=lambda v:v)
    entity._response_context=lambda n:('params',0,actuator) if response else None
    async def execute(job):return job()
    entity.hass=SimpleNamespace(async_add_executor_job=execute)
    asyncio.run(MethodType(namespace['_async_run_control'],entity)())
    entity._apply_control.assert_awaited_once_with(True)
    assert entity._commit_effective_heat_request.call_args.args[0].effective_requested_duty_ratio==1
    args=entity._compute_virtual_outdoor.call_args
    assert args.args[0] is True and args.kwargs['heat_offset']==16
    sample=entity._record_performance_sample.call_args.kwargs
    assert sample['suggested_heat_on'] is False  # Normal MPC demand stays separate.
    assert sample['requested_duty_ratio']==1
    assert sample['heating_detected'] is False  # A request does not become measured heat.
    if response:
        assert replay.call_args.kwargs['first_virtual']==4
        assert replay.call_args.args[5][0]==1
