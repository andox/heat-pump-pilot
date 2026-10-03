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
from pump_response import ResponseParameters, virtual_request
from response_optimizer import VirtualActuator
from runtime_settings import (
    build_runtime_settings,
    build_thermal_model_from_options,
    merge_climate_options,
)
from virtual_outdoor_utils import compute_virtual_outdoor_from_mpc_step, resolve_virtual_heat_offset
from config_helpers import normalize_hvac_mode
from control_request_utils import elapsed_smoothing_alpha, resolve_effective_heat_request, EffectiveHeatRequest


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
        "_request_elapsed_seconds",
        "_request_increase_age",
        "_apply_virtual_outdoor_smoothing",
        "_build_effective_heat_request",
        "_commit_effective_heat_request",
        "_uses_virtual_outdoor_control",
        "_compute_virtual_outdoor",
        "_apply_virtual_outdoor_min_temp",
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
        compute_virtual_outdoor_from_mpc_step=compute_virtual_outdoor_from_mpc_step,
        resolve_virtual_heat_offset=resolve_virtual_heat_offset,
        ResponseParameters=ResponseParameters,
        elapsed_smoothing_alpha=elapsed_smoothing_alpha,
        resolve_effective_heat_request=resolve_effective_heat_request,
        EffectiveHeatRequest=EffectiveHeatRequest,
        MAX_VIRTUAL_OUTDOOR=25.0,
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
    instance._last_control_time = None
    instance._effective_request_last_increase = None
    instance._last_effective_requested_duty_ratio = None
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
    entity._get_heating_detected = lambda now: False
    assert entity._response_context(now) is not None
    assert entity._response_context(now)[2].learned_response
    entity._monitor_only = True
    assert entity._response_context(now) is None
    entity._monitor_only = False
    entity._control_interval = 30
    assert entity._response_context(now) is None
    entity._control_interval = 15
    readings["sensor.outdoor"] = None
    assert entity._response_context(now) is None


def test_pump_learning_uses_detector_while_house_margin_is_ambiguous():
    entity, readings, _ = harness()
    entity._heating_detection_active = lambda: True
    entity._heating_supply_temp_entity = "sensor.supply"
    entity._heating_supply_temp_threshold = 30
    entity._learning_supply_temp_on_margin = 1
    entity._learning_supply_temp_off_margin = 1
    entity._get_heating_detected = lambda now: False
    readings["sensor.supply"] = 29.5
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert entity._get_heat_on_for_model(now) is None
    for minute in range(61):
        entity._collect_learning(now + timedelta(minutes=minute))
    assert len(entity._learning.pump.history) == 4
    assert all(row[2] == 0 for row in entity._learning.pump.history)
    assert entity._thermal_model.history == []


def test_live_ramp_and_smoothing_match_planner_for_early_update():
    entity, _, _ = harness()
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    entity._last_control_time = now - timedelta(seconds=60)
    entity._effective_request_last_increase = now - timedelta(hours=1)
    entity._last_effective_requested_duty_ratio = 0.0
    entity._last_virtual_outdoor = 15.0
    entity._virtual_outdoor_smoothing_alpha = 0.8
    entity._target_temperature = 21
    entity._comfort_tolerance = 0.3
    entity._controller.target_temperature = 21
    entity._controller.comfort_temperature_tolerance = 0.3
    actual = entity._build_effective_heat_request(
        raw_heat_on=True, raw_duty_ratio=1, predicted_temp=21, now=now)
    actuator = VirtualActuator(10, -15, 0.8, 15, initial_duty=0,
                              first_elapsed_seconds=60, anti_chatter=True)
    expected = actuator.request(entity._controller, 1, 0, 21, 60, 3600)
    assert actual.effective_requested_duty_ratio == pytest.approx(expected)
    raw = 5 + 10 * (1 - 2 * expected)
    live = entity._apply_virtual_outdoor_smoothing(raw, base=5, offset=10, now=now)
    planned = actuator.value(entity._controller, 5, 1, 1, 21, expected, 15, 60)
    assert live == pytest.approx(planned)
    entity._commit_effective_heat_request(actual, now=now)
    assert entity._effective_request_last_increase == now
    assert entity._request_increase_age(now + timedelta(seconds=30)) == 30


def test_unlearned_fallback_still_plans_actual_virtual_commands():
    entity, _, _ = harness()
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    entity._thermal_model.gain_identified = False
    context = entity._response_context(now)
    assert context is not None
    assert context[2].learned_response is False
    assert context[2].anti_chatter is False
    assert context[0] == ResponseParameters()
    assert not entity._learning.pump.ready


def test_main_planner_request_is_not_rewritten_by_legacy_anti_chatter():
    entity, _, _ = harness()
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    entity._last_result = SimpleNamespace(duty_sequence=[0.0])
    entity._last_effective_requested_duty_ratio = 1.0
    entity._effective_request_last_increase = now
    entity._last_control_time = now - timedelta(seconds=20)
    entity._target_temperature = 21
    entity._comfort_tolerance = 1.2
    result = entity._build_effective_heat_request(raw_heat_on=False, raw_duty_ratio=0,
                                                 predicted_temp=21, now=now)
    assert result.effective_requested_duty_ratio == 0
    assert result.anti_chatter_limited is False


def test_live_output_uses_exact_planned_command_without_extra_backoff():
    entity, _, _ = harness()
    entity._outdoor_temp, entity._indoor_temp = 10, 21.3
    entity._last_result = SimpleNamespace(predicted_temperatures=[21.3,21.4],
        price_baseline=1, planned_virtual_outdoor=[3.5], duty_sequence=[0.75])
    entity._last_price_forecast = [0.1]
    entity._price_comfort_weight, entity._price_penalty_curve = 0.8, "linear"
    entity._target_temperature, entity._comfort_tolerance = 20.5, 1.2
    entity._overshoot_warm_bias_enabled, entity._overshoot_warm_bias_curve = True, "linear"
    entity._apply_virtual_outdoor_smoothing = lambda *a, **kw: pytest.fail("Double smoothing")
    assert entity._compute_virtual_outdoor(True, [10], duty_ratio=0.75) == 3.5
