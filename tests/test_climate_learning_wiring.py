"""Exercise the climate entity's learning methods without loading HA on Windows."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import const
from adaptive_model import AdaptiveThermalModel
from learning_manager import LearningManager
from learning_utils import resolve_estimator_initial_temp, should_reseed_thermal_model
from mpc_controller import MpcController
from pump_response import virtual_request
from response_optimizer import VirtualActuator
from runtime_settings import (
    build_runtime_settings,
    build_thermal_model_from_options,
    merge_climate_options,
)
from config_helpers import normalize_hvac_mode


def harness():
    source = (
        Path(__file__).resolve().parents[1]
        / "custom_components/heat_pump_pilot/climate.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    entity = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {
        "_get_heat_on_for_model",
        "_get_heating_detected",
        "_collect_learning",
        "_response_context",
        "_handle_entry_update",
        "_apply_runtime_settings",
        "_learning_context",
        "_summer_heat_window_settings_changed",
    }
    entity.body = [
        n for n in entity.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    entity.bases, entity.decorator_list = [], []
    for method in entity.body:
        method.decorator_list = []
    tree.body = [entity]
    namespace = dict(
        vars(const),
        virtual_request=virtual_request,
        VirtualActuator=VirtualActuator,
        LearningManager=LearningManager,
        normalize_hvac_mode=normalize_hvac_mode,
        HVACMode=lambda mode: mode,
        build_runtime_settings=build_runtime_settings,
        build_thermal_model_from_options=build_thermal_model_from_options,
        merge_climate_options=merge_climate_options,
        should_reseed_thermal_model=should_reseed_thermal_model,
        resolve_estimator_initial_temp=resolve_estimator_initial_temp,
        dt_util=SimpleNamespace(utcnow=lambda: datetime.now(timezone.utc)),
    )
    exec(compile(tree, str(source), "exec"), namespace)
    instance = namespace[entity.name]()
    readings = {"sensor.indoor": 20.0, "sensor.outdoor": 5.0, "number.virtual": -5.0}
    states = {"number.virtual": SimpleNamespace(state="-5", attributes={})}
    instance.hass = SimpleNamespace(states=SimpleNamespace(get=states.get))
    instance._get_state_as_float = readings.get
    instance._state_to_float = lambda value: None if value is None else float(value)
    instance._indoor_temp_entity, instance._outdoor_temp_entity = (
        "sensor.indoor",
        "sensor.outdoor",
    )
    instance._controlled_entity = "number.virtual"
    instance._settings = build_runtime_settings({})
    instance._thermal_model = AdaptiveThermalModel()
    instance._learning = LearningManager(instance._thermal_model)
    instance._controller = MpcController(21, 0.5, 1, 24)
    instance._monitor_only = False
    instance._virtual_heat_offset = 10
    instance._heating_detection_active = lambda: False
    instance._record_model_history = lambda now: None
    instance._continuous_control_enabled = True
    instance._control_interval = 15
    instance._virtual_outdoor_min_temp = -15
    instance._virtual_outdoor_smoothing_alpha = 0.5
    instance._virtual_outdoor_smoothing_enabled = True
    return instance, readings, states


@pytest.mark.parametrize(
    "change",
    [
        {"price_comfort_weight": 0.7},
        {"learning_model": "adaptive"},
        {"initial_heat_gain_coefficient": 0.9},
    ],
)
def test_option_updates_preserve_learned_coefficients_unless_explicitly_reseeded(
    change,
):
    entity, readings, states = harness()
    entity._options = merge_climate_options({"learning_model": "ekf"})
    entity._apply_runtime_settings(build_runtime_settings(entity._options))
    entity._thermal_model = build_thermal_model_from_options(entity._options)
    entity._thermal_model.reseed(
        seed=0.5, initial_heat_loss=0.02, initial_heat_gain=0.6, initial_temp=20
    )
    entity._learning = LearningManager(entity._thermal_model)
    entity._indoor_temp = 20
    entity._forecast_service = SimpleNamespace(update_entities=lambda **kwargs: None)
    entity.config_entry = SimpleNamespace(
        data={
            "indoor_temp_entity": "sensor.indoor",
            "outdoor_temp_entity": "sensor.outdoor",
            "controlled_entity": "number.virtual",
            "price_entity": "sensor.price",
            "weather_forecast_entity": "weather.home",
        },
        options={**entity._options, **change},
    )
    entity._resubscribe_sensors = entity._schedule_control_loop = (
        entity._request_control_run
    ) = lambda: None
    entity.async_write_ha_state = lambda: None
    entity._persist_thermal_state = lambda now: None
    entity.hass.async_create_task = lambda task: None
    entity._handle_entry_update()
    if "initial_heat_gain_coefficient" in change:
        assert entity._thermal_model.heat_gain_coeff == 0.9
    else:
        assert entity._thermal_model.heat_gain_coeff == 0.6
        assert entity._controller.heat_loss_coeff == 0.02
    if "learning_model" in change:
        assert isinstance(entity._thermal_model, AdaptiveThermalModel)
        assert entity._thermal_model.background_gain == 0


def test_virtual_request_is_never_substituted_for_measured_heat():
    entity, readings, states = harness()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for minute in range(61):
        entity._collect_learning(now + timedelta(minutes=minute))
    assert entity._learning.last_update is None
    assert entity._learning.house.last_status == "insufficient_coverage"
    assert entity._thermal_model.history == []


@pytest.mark.parametrize("state", ["unknown", "unavailable"])
def test_unavailable_switch_is_unknown_instead_of_off(state):
    entity, readings, states = harness()
    entity._controlled_entity = "switch.pump"
    states["switch.pump"] = SimpleNamespace(state=state, attributes={})
    assert entity._get_heat_on_for_model(datetime.now(timezone.utc)) is None


def test_measured_heat_collected_hourly_even_when_control_calls_are_frequent():
    entity, readings, states = harness()
    entity._controlled_entity = "switch.pump"
    states["switch.pump"] = SimpleNamespace(state="off", attributes={})
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    updates = []
    for second in range(0, 3601, 10):
        if second == 1800:
            states["switch.pump"].state = "on"
        if entity._collect_learning(now + timedelta(seconds=second)):
            updates.append(second)
    assert updates == [3600]
    assert entity._thermal_model.history[0].heat == pytest.approx(0.5)


def test_response_is_gated_by_operating_mode_and_measured_signal():
    entity, readings, states = harness()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pump = entity._learning.pump
    for minute in range(-15, 1):
        entity._learning.observe(now.timestamp() + minute * 60, 20, 5, 0, 0.5)
    entity._thermal_model.gain_identified = True
    pump.ready = True
    pump.history = [(now.timestamp(), 0.5, 0.5, 5.0)]
    entity._get_heat_on_for_model = lambda now: 0.0
    assert entity._response_context(now) is not None
    entity._monitor_only = True
    assert entity._response_context(now) is None
    entity._monitor_only = False
    entity._control_interval = 30
    assert entity._response_context(now) is None
    entity._control_interval = 15
    readings["sensor.outdoor"] = None
    assert entity._response_context(now) is None
