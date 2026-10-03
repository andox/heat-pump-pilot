"""Weather-dependent pump delivery, validation gates and persisted history."""
import math
import random
from dataclasses import replace

import pytest

from learning_interval import LearningInterval
from pump_response import PumpResponseModel, ResponseParameters
from response_optimizer import VirtualActuator, replay
from mpc_controller import MpcController


def samples(count=192, gain=0.25, weather="varying"):
    rng = random.Random(731)
    heat = 0.3
    requests = [rng.choice((0, 0.25, 0.75, 1)) for _ in range(count)]
    for i, request in enumerate(requests):
        outdoor = 5 + 6 * math.sin(i / 7)
        if weather == "constant":
            outdoor = 5
        elif weather == "confounded":
            outdoor = 10 - 10 * request
        delayed = requests[i - 2] if i >= 2 else 0
        # Independent generator, including cold-weather residual heat at zero duty.
        target = max(0, min(1, 0.25 + 0.55 * delayed + gain * (5 - outdoor) / 10))
        retention = math.exp(-0.5)
        heat = retention * heat + (1 - retention) * target
        yield LearningInterval((i + 1) * 900, 0.25, 21, 21, outdoor, heat, request, 1)


def fit(rows):
    model = PumpResponseModel()
    for row in rows:
        model.add_interval(row)
    model._fit()
    return model


def test_identifies_outdoor_effect_and_delay_with_independent_variation():
    model = fit(samples())
    assert model.ready, model.diagnostics()
    assert model.diagnostics()['outdoor_active']
    assert model.parameters.outdoor_gain == pytest.approx(0.25, abs=0.01)
    assert model.parameters.delay_steps == 2
    assert model.validation_mae < model.simple_validation_mae * 0.9
    assert model.simple_validation_mae - model.validation_mae >= 0.01


@pytest.mark.parametrize('weather,reason', [
    ('constant', 'insufficient_outdoor_variation'),
    ('confounded', 'outdoor_request_confounded'),
])
def test_weather_requires_separate_evidence(weather, reason):
    model = fit(samples(weather=weather))
    assert model.outdoor_reason == reason
    assert not model.diagnostics()['outdoor_active']


def test_simple_model_retained_when_weather_adds_no_predictive_value():
    model = fit(samples(gain=0))
    assert model.ready
    assert model.outdoor_reason == 'no_validation_improvement'
    assert model.parameters.outdoor_gain == 0
    assert model.validation_mae == model.simple_validation_mae


def test_weather_fit_must_work_on_held_out_data():
    rows = list(samples())
    # Keep training intact, reverse the weather effect only on held-out outcomes.
    reversed_rows = list(samples(gain=-0.25))
    model = fit(rows[:-16] + reversed_rows[-16:])
    assert model.outdoor_reason == 'no_validation_improvement'
    assert not model.diagnostics()['outdoor_active']


def test_no_heat_cannot_identify_weather_response():
    model = fit(replace(r, heat=0) for r in samples())
    assert not model.ready
    assert model.reason == 'insufficient_heating_variation'


def test_colder_weather_adds_residual_heat_but_does_not_extrapolate():
    p = ResponseParameters(slope=0.6, idle=0.2, outdoor_gain=0.3,
                           outdoor_reference=5, outdoor_min=-5, outdoor_max=15)
    assert p.advance(0, 0, (), -5)[0] == pytest.approx(0.5)
    assert p.advance(0, 0, (), 15)[0] == 0
    assert p.target(0, -40) == p.target(0, -5)
    assert p.target(0, 40) == p.target(0, 15)
    assert p.target(1, -5) == 1
    for temperature in (-5, 0, 5, 15):
        assert p.target(0, temperature) <= p.target(0.5, temperature) <= p.target(1, temperature)


def test_legacy_history_preserved_without_inventing_outdoor_data():
    rows = list(samples(gain=0))
    payload = {'history': [(r.end, r.request, r.heat) for r in rows]}
    model = PumpResponseModel()
    model.restore(payload)
    assert len(model.history) == len(rows)
    assert model.diagnostics()['outdoor_samples'] == 0
    assert not model.ready
    model._fit()
    assert model.ready
    assert model.outdoor_reason == 'missing_outdoor_history'
    assert model.parameters.outdoor_gain == 0


def test_restart_retains_weather_history_but_requires_fresh_validation():
    rows = list(samples(196))
    model = fit(rows[:192])
    restored = PumpResponseModel()
    restored.restore(model.export_state())
    assert restored.history == model.history
    assert not restored.ready
    assert restored.initial_state(rows[191].end) is None
    restored.add_interval(rows[192])
    assert restored.ready
    assert restored.diagnostics()['outdoor_active']


def test_missing_weather_preserves_simple_fit():
    model = fit(replace(r, outdoor=None) for r in samples(gain=0))
    assert model.ready
    assert model.outdoor_reason == 'missing_outdoor_history'


def test_optimizer_and_replay_use_forecast_outdoor_for_delivery():
    controller = MpcController(target_temperature=21, price_comfort_weight=0.5,
                               comfort_temperature_tolerance=0.2,
                               prediction_horizon_hours=3, heat_loss_coeff=0,
                               heat_gain_coeff=1, background_gain=0)
    p = ResponseParameters(slope=0.5, idle=0.2, outdoor_gain=0.3,
                           outdoor_reference=5, outdoor_min=-5, outdoor_max=15)
    context = (p, (0, ()), VirtualActuator(10, -20, 1, 5))
    cold = replay(controller, 21, [-5] * 12, [1] * 12, 1, [0] * 12, *context)
    warm = replay(controller, 21, [15] * 12, [1] * 12, 1, [0] * 12, *context)
    assert cold[2] == pytest.approx([0.5] * 12)
    assert warm[2] == pytest.approx([0] * 12)
    assert cold[1][-1] == pytest.approx(22.5)
    outdoor = [-5] * 6 + [15] * 6
    _, plan = controller.suggest_control(21, outdoor, [1] * 12,
                                        price_baseline_override=1, response_context=context)
    result = replay(controller, 21, outdoor, [1] * 12, 1, plan.duty_sequence, *context)
    assert plan.predicted_heating == pytest.approx(result[2])
    assert plan.predicted_temperatures == pytest.approx(result[1])
    assert plan.cost == pytest.approx(result[4])


def test_restart_retains_parameters_and_rebuilds_live_state_with_real_timing():
    from adaptive_model import AdaptiveThermalModel
    from learning_manager import LearningManager
    model = fit(samples())
    saved = model.export_state()
    manager = LearningManager(AdaptiveThermalModel())
    manager.pump.restore(saved)
    assert manager.pump.parameters == model.parameters
    assert not manager.pump.ready
    # Restart mid-bin: a fresh collector cannot produce a contiguous next bin.
    start = model.history[-1][0] + 420
    for minute in range(46):
        manager.observe(start + minute * 60, 21, 5, 0.4, 0.5)
        if minute < 30:
            assert manager.response_state(start + minute * 60) is None
    assert manager.pump.history[:len(model.history)] == model.history
    assert manager.pump.ready, manager.pump.diagnostics()
    assert manager.response_state(start + 45 * 60) is not None


def test_fitting_never_uses_lags_or_heat_transitions_across_gaps():
    first = list(samples(120))
    second = [replace(r, end=r.end + first[-1].end + 7200) for r in samples(72)]
    model = fit(first + second)
    assert len(model.history) == 192
    usable = model._usable_indices(model.history)
    assert not set(range(120, 128)).intersection(usable)
    assert 119 in usable and 128 in usable
    assert model.ready, model.diagnostics()
    assert model.parameters.delay_steps == 2
    assert model.parameters.outdoor_gain == pytest.approx(0.25, abs=0.02)


def test_error_reseeds_heat_after_gap():
    rows = [(i * 900, 0, 0, 5) for i in range(20)]
    rows += [(100000 + i * 900, 1, 1, 5) for i in range(20)]
    assert PumpResponseModel._error(ResponseParameters(retention=0.9), rows, 8) == 0


@pytest.mark.parametrize('parameters', [None, {'delay_steps': 99},
    {'retention': 1}, {'slope': float('nan')}, {'idle': -1}])
def test_invalid_saved_parameters_are_not_activated(parameters):
    model = PumpResponseModel()
    model.restore({'parameters': parameters})
    assert model.parameters == ResponseParameters()
    assert model.initial_state(0) is None


def test_duplicate_observation_does_not_count_as_fresh_evidence():
    row = next(samples())
    model = PumpResponseModel()
    model.add_interval(row)
    model.add_interval(row)
    assert len(model.history) == model.fresh_samples == 1
