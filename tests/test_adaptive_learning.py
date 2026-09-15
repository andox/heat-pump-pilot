"""Learning cadence, causal interval alignment, and observability safeguards."""

import math
import random

import pytest

from adaptive_model import AdaptiveThermalModel
from learning_interval import IntervalCollector, LearningInterval
from learning_manager import LearningManager
from learning_math import bounded_fit
from pump_response import PumpResponseModel, ResponseParameters, virtual_request
from runtime_settings import build_runtime_settings, build_thermal_model_from_options


def interval(i, start=20.0, end=20.0, outdoor=5.0, heat=0.0, request=None, hours=1.0):
    return LearningInterval(
        (i + 1) * hours * 3600, hours, start, end, outdoor, heat, request, 1.0
    )


def test_collector_integrates_old_heat_not_new_edge():
    collector = IntervalCollector()
    for minute in range(61):
        sample = collector.observe(
            minute * 60, 20 + minute / 600, 5, float(minute >= 30)
        )
        if minute < 60:
            assert sample is None
    assert sample.heat == pytest.approx(0.5)
    assert sample.hours == 1
    assert sample.start_temp == 20
    assert sample.end_temp == 20.1


@pytest.mark.parametrize("unknown", [None, float("nan"), float("inf")])
def test_unknown_time_does_not_become_zero_heat(unknown):
    collector = IntervalCollector()
    for minute in range(61):
        sample = collector.observe(
            minute * 60, 20, 5, unknown if 20 <= minute < 30 else 0
        )
    assert sample is None
    assert collector.last_status == "insufficient_coverage"
    assert collector.last_coverage == pytest.approx(50 / 60)


def test_outage_discards_partial_interval_and_never_learns_across_restart():
    collector = IntervalCollector()
    for minute in range(31):
        collector.observe(minute * 60, 20, 5, 0)
    assert collector.observe(7200, 19, 5, 0) is None
    assert collector.last_status == "observation_gap"
    for minute in range(1, 61):
        sample = collector.observe(7200 + minute * 60, 19, 5, 0)
    assert sample.start_temp == 19
    assert sample.hours == 1


@pytest.mark.parametrize("kind", ["adaptive", "ekf", "rls"])
def test_more_observations_do_not_mean_more_coefficient_updates(kind):
    def run(seconds):
        model = build_thermal_model_from_options({"learning_model": kind})
        manager = LearningManager(model)
        updates = []
        for t in range(0, 7201, seconds):
            if manager.observe(t, 20 - t / 36000, 5, 0):
                updates.append(t)
        return model, updates

    sparse, sparse_updates = run(60)
    frequent, frequent_updates = run(10)
    assert sparse_updates == frequent_updates == [3600, 7200]
    assert sparse.heat_loss_coeff == pytest.approx(frequent.heat_loss_coeff)
    assert sparse.heat_gain_coeff == pytest.approx(frequent.heat_gain_coeff)


def train_house(model, heat_function, count=168):
    temperature = 20.0
    for i in range(count):
        outdoor = 8 + 5 * math.sin(i / 7)
        heat = heat_function(i)
        next_temp = temperature + 0.02 * (outdoor - temperature) + 0.6 * heat + 0.15
        model.add_interval(interval(i, temperature, next_temp, outdoor, heat))
        temperature = next_temp


def test_house_recovers_joint_terms_with_excitation():
    model = AdaptiveThermalModel(initial_heat_loss=0.03, initial_heat_gain=0.8)
    train_house(model, lambda i: float(i % 7 < 3))
    assert model.gain_identified
    assert model.heat_loss_coeff == pytest.approx(0.02, abs=0.001)
    assert model.heat_gain_coeff == pytest.approx(0.6, abs=0.015)
    assert model.background_gain == pytest.approx(0.15, abs=0.015)


def test_sparse_heating_keeps_gain_seed_and_learns_background_jointly():
    model = AdaptiveThermalModel(initial_heat_loss=0.03, initial_heat_gain=0.8)
    train_house(model, lambda i: 0.0)
    assert model.status == "gain_frozen"
    assert not model.gain_identified
    assert model.heat_gain_coeff == 0.8
    assert model.background_gain == pytest.approx(0.15, abs=0.015)


def test_constant_weather_difference_freezes_all_terms():
    model = AdaptiveThermalModel(initial_heat_loss=0.03, initial_heat_gain=0.8)
    for i in range(50):
        model.add_interval(interval(i, 20, 20, 10, 0.5))
    assert model.status == "insufficient_weather_variation"
    assert model.heat_loss_coeff == 0.03
    assert model.background_gain == 0


def test_weather_correlated_heating_does_not_identify_pump_gain():
    model = AdaptiveThermalModel(initial_heat_loss=0.03, initial_heat_gain=0.8)
    for i in range(72):
        heat = (i % 10) / 10
        outdoor = 15 - 10 * heat
        model.add_interval(interval(i, 20, 20.1, outdoor, heat))
    assert not model.gain_identified
    assert model.heat_gain_coeff == 0.8


def test_direct_request_baseline_prevents_unnecessary_response_activation():
    model = PumpResponseModel()
    for i in range(180):
        request = float(i % 8 < 4)
        model.add_interval(interval(i, heat=request, request=request, hours=0.25))
    assert not model.ready
    assert model.reason == "no_validation_improvement"


def test_restore_and_legacy_seed_migration():
    original = AdaptiveThermalModel(initial_heat_loss=0.02, initial_heat_gain=0.6)
    train_house(original, lambda i: float(i % 7 < 3), 48)
    restored = AdaptiveThermalModel()
    payload = original.export_state()
    payload["history"] = [["diagnostic history must not overwrite observations"]]
    assert restored.restore(payload)
    assert restored.background_gain == original.background_gain
    assert len(restored.history) == len(original.history)
    for kind in ("ekf", "rls"):
        legacy = build_thermal_model_from_options(
            {"learning_model": kind, "initial_heat_gain_coefficient": 0.7}
        )
        migrated = AdaptiveThermalModel()
        assert migrated.restore(legacy.export_state())
        assert migrated.heat_gain_coeff == 0.7


def test_bounded_solver_freezes_gain_and_solves_other_terms():
    samples = [
        ((x, h, 1), 0.02 * x + 0.6 * h + 0.15) for x in (-20, -10, -5) for h in (0, 1)
    ]
    assert bounded_fit(
        samples, [(0.001, 0.25), (0.6, 0.6), (-0.5, 0.5)]
    ) == pytest.approx([0.02, 0.6, 0.15])


def pump_samples(count=180):
    rng = random.Random(42)
    requests = [rng.choice((0.0, 0.25, 0.75, 1.0)) for _ in range(count)]
    parameters = ResponseParameters(2, math.exp(-0.25 / 0.5), 0.8, 0.1)
    heat, queue = 0.0, (0.0, 0.0)
    for i, request in enumerate(requests):
        heat, queue = parameters.advance(request, heat, queue)
        yield interval(i, heat=heat, request=request, hours=0.25)


def test_delayed_response_requires_history_then_passes_future_validation():
    model = PumpResponseModel()
    for i, sample in enumerate(pump_samples()):
        model.add_interval(sample)
        if i < 111:
            assert not model.ready
    assert model.ready, model.diagnostics()
    assert model.parameters.delay_steps == 2
    assert model.parameters.retention == pytest.approx(math.exp(-0.5))
    assert model.validation_mae < 0.1 * model.baseline_mae
    state = model.initial_state(sample.end)
    assert len(state[1]) == 2
    assert model.initial_state(sample.end + 1201) is None
    restored = PumpResponseModel()
    restored.restore(model.export_state())
    assert not restored.ready
    assert restored.initial_state(sample.end) is None


def test_response_does_not_identify_gain_from_no_heating_or_no_request_variation():
    for heat, request, reason in [
        (0, 0.5, "insufficient_heating_variation"),
        (0.5, 0.5, "insufficient_request_variation"),
    ]:
        model = PumpResponseModel()
        for i in range(180):
            model.add_interval(interval(i, heat=heat, request=request, hours=0.25))
        assert not model.ready
        assert model.reason == reason


def test_pump_gap_resets_history():
    model = PumpResponseModel()
    for sample in pump_samples():
        model.add_interval(sample)
    model.add_interval(interval(200, heat=0, request=0, hours=0.25))
    assert not model.ready
    assert len(model.history) == 1


def test_virtual_request_uses_actual_value_and_settings_are_independent():
    assert virtual_request(5, -5, 10) == 1
    assert virtual_request(5, 15, 10) == 0
    assert virtual_request(5, 5, 10) == 0.5
    assert virtual_request(None, 5, 10) is None
    settings = build_runtime_settings({})
    assert settings.control_interval_minutes == 15
    assert settings.learning_interval_minutes == 60
    assert settings.learning_model == "adaptive"
    assert build_runtime_settings({"learning_model": "ekf"}).learning_model == "ekf"


def test_response_state_aligns_to_sensor_trigger_between_bin_boundaries():
    manager = LearningManager(AdaptiveThermalModel())
    # Request on during minutes 15..30; heating begins at minute 25.
    for minute in range(41):
        manager.observe(
            minute * 60, 20, 5, float(minute >= 25), float(15 <= minute < 30)
        )
    manager.pump.ready = True
    manager.pump.parameters = ResponseParameters(delay_steps=2)
    heat, queue = manager.response_state(40 * 60)
    # Origin minute 40: windows are 10..25 and 25..40, not 0..15 and 15..30.
    assert heat == 1
    assert queue == pytest.approx((10 / 15, 5 / 15))
    assert manager.response_state(46 * 60) is None
