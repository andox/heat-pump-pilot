"""Realistic observation gaps and quiet periods must not erase useful learning."""
import math
import random

import pytest

from adaptive_model import AdaptiveThermalModel
from learning_interval import LearningInterval
from learning_manager import LearningManager
from pump_response import PumpResponseModel, ResponseParameters


def trained_model():
    rng = random.Random(42)
    model = PumpResponseModel()
    heat, queue = 0.0, [0.0, 0.0]
    retention = math.exp(-0.5)
    for i in range(180):
        request = rng.choice((0.0, 0.25, 0.75, 1.0))
        delayed = queue.pop(0)
        queue.append(request)
        heat = retention * heat + (1 - retention) * 0.8 * delayed
        model.add_interval(LearningInterval((i + 1) * 900, .25, 20, 20, 5, heat, request, 1))
    assert model.ready
    return model, heat, queue


def coast(model, heat, queue, hours):
    start = model.history[-1][0]
    for i in range(hours * 4):
        delayed = queue.pop(0)
        queue.append(0.0)
        heat = math.exp(-0.5) * heat + (1 - math.exp(-0.5)) * 0.8 * delayed
        model.add_interval(LearningInterval(start + (i + 1) * 900, .25, 20, 20, 5, heat, 0, 1))
    return heat, queue


def test_short_heating_cycles_survive_house_supply_margin_gaps():
    manager = LearningManager(AdaptiveThermalModel())
    for minute in range(96 * 60 + 1):
        phase = minute % 240
        request = float(phase < 60)
        detected = float(15 <= phase < 75)
        # The house's extra confidence margin excludes transition minutes.
        # The debounced pump detector remains measured and known throughout.
        house_heat = None if phase in (15, 75) else detected
        manager.observe(minute * 60, 20, 5, house_heat, request, pump_heat=detected)
    pump = manager.pump
    assert pump.ready, pump.diagnostics()
    assert pump.parameters.delay_steps == 1
    assert pump.validation_mae < .01
    assert sum(pump.history[i][2] for i in pump._usable_indices(pump.history)) > 0


@pytest.mark.parametrize("indoor", [None, "spike"])
def test_pump_observations_do_not_depend_on_indoor_sensor(indoor):
    manager = LearningManager(AdaptiveThermalModel())
    for minute in range(61):
        temperature = None if indoor is None else 20 + minute / 10
        manager.observe(minute * 60, temperature, 5, 0, 0, pump_heat=0)
    assert len(manager.pump.history) == 4
    assert manager.model.history == []
    assert manager.house.last_status in ("temperature_jump", "insufficient_coverage")


def test_unknown_measured_pump_activity_is_not_filled_from_house_signal():
    manager = LearningManager(AdaptiveThermalModel())
    for minute in range(16):
        manager.observe(minute * 60, 20, 5, 0, 0, pump_heat=None)
    assert manager.pump.history == []
    assert manager.pump_intervals.last_status == "insufficient_coverage"


def test_validated_response_survives_quiet_period_longer_than_training_window():
    model, heat, queue = trained_model()
    heat, queue = coast(model, heat, queue, 10)
    assert model.ready, model.diagnostics()
    parameters = model.parameters
    heat, queue = coast(model, heat, queue, 80)
    assert model.ready, model.diagnostics()
    assert model.reason == "ready_retained_low_variation"
    assert model.parameters == parameters
    assert model.validation_mae < .02


def test_retained_response_survives_restart_but_requires_fresh_state():
    original, heat, queue = trained_model()
    coast(original, heat, queue, 90)
    restored = PumpResponseModel()
    restored.restore(original.export_state())
    start = original.history[-1][0]
    assert not restored.ready
    assert restored.initial_state(start) is None
    for i in range(1, 10):
        # Restart mid-interval leaves a genuine gap; no invented observations.
        restored.add_interval(LearningInterval(start + 420 + i * 900, .25, 20, 20, 5, 0, 0, 1))
        if i < max(1, restored.parameters.delay_steps):
            assert restored.initial_state(start + 420 + i * 900) is None
    assert restored.ready, restored.diagnostics()
    assert restored.parameters == original.parameters
    assert restored.initial_state(restored.history[-1][0]) is not None


def test_quiet_data_cannot_activate_unvalidated_or_invalid_saved_parameters():
    original, heat, queue = trained_model()
    coast(original, heat, queue, 90)
    for mode in ("unvalidated", "invalid"):
        payload = original.export_state()
        if mode == "unvalidated":
            payload.pop("validated_at")
        else:
            payload["parameters"] = {"delay_steps": 99}
        model = PumpResponseModel()
        model.restore(payload)
        model._fit()
        assert not model.ready


def test_quiet_but_wrong_saved_model_is_rejected():
    model, heat, queue = trained_model()
    coast(model, heat, queue, 90)
    # A previously trusted model predicting 30% idle heat disagrees with reality.
    model.parameters = ResponseParameters(idle=.3, slope=.7)
    model._fit()
    assert not model.ready
    assert model.validated_at is None


@pytest.mark.parametrize("actual_gain, expected_ready", [(0.8, True), (0.2, False)])
def test_response_revalidated_when_heating_resumes_after_long_coast(actual_gain, expected_ready):
    model, heat, queue = trained_model()
    heat, queue = coast(model, heat, queue, 90)
    parameters = model.parameters
    start = model.history[-1][0]
    for i in range(1, 17):
        delayed = queue.pop(0)
        queue.append(1.0)
        # Independent plant, not predictions generated by the fitted model.
        heat = math.exp(-0.5) * heat + (1 - math.exp(-0.5)) * actual_gain * delayed
        model.add_interval(LearningInterval(start + i * 900, .25, 20, 20, 5, heat, 1, 1))
        if expected_ready:
            assert model.ready, model.diagnostics()
            assert model.parameters == parameters
            assert model.initial_state(start + i * 900) is not None
    assert model.ready == expected_ready, model.diagnostics()
    if expected_ready:
        assert model.reason == "ready_retained_validation"
        assert model.validation_mae < .01
    else:
        assert model.validated_at is None
