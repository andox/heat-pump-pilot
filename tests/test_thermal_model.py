"""Unit tests for the thermal model estimator and price baseline handling."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Allow running the tests without Home Assistant installed by importing the modules
# directly from this integration directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from mpc_controller import MpcController  # noqa: E402
from price_history import PriceHistoryStorage  # noqa: E402
from performance_history import PerformanceHistoryStorage  # noqa: E402
from thermal_model import ThermalModelEstimator, ThermalModelRlsEstimator, ThermalModelStorage  # noqa: E402


def test_thermal_model_learns_gain_from_heating() -> None:
    """Heating pulses should increase the estimated gain coefficient."""
    estimator = ThermalModelEstimator(
        seed=0.8, initial_heat_loss=0.05, initial_heat_gain=0.2, initial_temp=20.0
    )
    initial_gain = estimator.heat_gain_coeff
    temp = 20.0
    outdoor = 0.0
    for _ in range(6):
        # Simulate the room warming a bit while heating is on.
        temp += 0.3
        estimator.step(temp, outdoor, heat_on=True, dt_hours=0.25)
    assert estimator.heat_gain_coeff > initial_gain
    assert estimator.heat_loss_coeff > 0  # Sanity check it remains positive.


def test_thermal_model_persistence_round_trip() -> None:
    """Saved estimator state should restore cleanly."""
    estimator = ThermalModelEstimator(seed=0.1, initial_heat_loss=0.1, initial_heat_gain=0.5, initial_temp=19.5)
    estimator.step(20.0, outdoor_temp=5.0, heat_on=True, dt_hours=0.5)
    state = estimator.export_state()

    with TemporaryDirectory() as tmp:
        storage = ThermalModelStorage(Path(tmp) / "state.json")
        storage.save(state)
        loaded = storage.load()

    restored = ThermalModelEstimator(seed=0.5)
    assert loaded is not None
    assert restored.restore(loaded)
    assert pytest.approx(restored.heat_loss_coeff, rel=0.1) == estimator.heat_loss_coeff
    assert pytest.approx(restored.heat_gain_coeff, rel=0.1) == estimator.heat_gain_coeff
    assert pytest.approx(restored.indoor_temp, rel=0.1) == estimator.indoor_temp


def test_rls_model_learns_gain_from_heating() -> None:
    """RLS should adjust gain coefficient from heating updates."""
    estimator = ThermalModelRlsEstimator(
        seed=0.8, initial_heat_loss=0.05, initial_heat_gain=0.2, initial_temp=20.0, forgetting_factor=0.98
    )
    initial_gain = estimator.heat_gain_coeff
    temp = 20.0
    outdoor = 0.0
    for _ in range(6):
        temp += 0.3
        estimator.step(temp, outdoor, heat_on=True, dt_hours=0.25)
    assert estimator.heat_gain_coeff > initial_gain
    assert estimator.heat_loss_coeff > 0


def test_rls_persistence_round_trip() -> None:
    """Saved RLS state should restore cleanly."""
    estimator = ThermalModelRlsEstimator(
        seed=0.2, initial_heat_loss=0.08, initial_heat_gain=0.3, initial_temp=19.0, forgetting_factor=0.97
    )
    estimator.step(19.6, outdoor_temp=5.0, heat_on=True, dt_hours=0.5)
    state = estimator.export_state()

    with TemporaryDirectory() as tmp:
        storage = ThermalModelStorage(Path(tmp) / "state.json")
        storage.save(state)
        loaded = storage.load()

    restored = ThermalModelRlsEstimator(seed=0.5, forgetting_factor=0.99)
    assert loaded is not None
    assert restored.restore(loaded)
    assert pytest.approx(restored.heat_loss_coeff, rel=0.1) == estimator.heat_loss_coeff
    assert pytest.approx(restored.heat_gain_coeff, rel=0.1) == estimator.heat_gain_coeff
    assert pytest.approx(restored.indoor_temp, rel=0.1) == estimator.indoor_temp


def test_thermal_model_storage_accepts_history_payload() -> None:
    """Storage should persist payloads that include learning history."""
    payload = {
        "version": 2,
        "model_type": "ekf",
        "state": [21.0, 0.05, 0.4],
        "covariance": [[0.5, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.05]],
        "history": [["2025-12-13T12:15:00+00:00", 0.05, 0.4]],
    }
    with TemporaryDirectory() as tmp:
        storage = ThermalModelStorage(Path(tmp) / "state.json")
        storage.save(payload)
        loaded = storage.load()

    assert loaded == payload


def test_price_history_persistence_round_trip() -> None:
    """Saved price history should restore cleanly."""
    payload = {
        "version": 1,
        "bucket_minutes": 15,
        "last_bucket_start": "2025-12-13T12:15:00+00:00",
        "history": [0.5, 1.0, 0.75],
    }
    with TemporaryDirectory() as tmp:
        storage = PriceHistoryStorage(Path(tmp) / "prices.json")
        storage.save(payload)
        loaded = storage.load()

    assert loaded == payload


def test_performance_history_persistence_round_trip() -> None:
    """Saved performance history should restore cleanly."""
    payload = {
        "version": 1,
        "history": [
            {
                "time": "2025-12-13T12:15:00+00:00",
                "indoor_temp": 21.0,
                "target_temp": 20.0,
                "heating_detected": True,
                "price": 0.2,
                "prediction_error": -0.1,
            }
        ],
    }
    with TemporaryDirectory() as tmp:
        storage = PerformanceHistoryStorage(Path(tmp) / "performance.json")
        storage.save(payload)
        loaded = storage.load()

    assert loaded == payload


def test_observe_temperature_updates_only_temp() -> None:
    """observe_temperature should not modify learned coefficients."""
    estimator = ThermalModelEstimator(seed=0.5, initial_heat_loss=0.08, initial_heat_gain=0.4, initial_temp=20.0)
    before_loss = estimator.heat_loss_coeff
    before_gain = estimator.heat_gain_coeff
    estimator.observe_temperature(21.5)
    assert estimator.indoor_temp == pytest.approx(21.5)
    assert estimator.heat_loss_coeff == pytest.approx(before_loss)
    assert estimator.heat_gain_coeff == pytest.approx(before_gain)


def test_price_history_baseline_reduces_peak_cost() -> None:
    """Past expensive prices should reduce the relative penalty of moderate upcoming prices."""
    controller = MpcController(
        target_temperature=21.0,
        price_comfort_weight=0.5,
        comfort_temperature_tolerance=0.25,
        prediction_horizon_hours=1,
        heat_loss_coeff=0.05,
        heat_gain_coeff=0.6,
    )
    prices = [10.0, 10.0, 10.0, 10.0]
    outdoor = [5.0] * 4
    past_expensive = [50.0] * 192

    _, result_without_history = controller.suggest_control(20.0, outdoor, prices, past_prices=None)
    _, result_with_history = controller.suggest_control(20.0, outdoor, prices, past_prices=past_expensive)

    assert result_with_history.price_baseline > result_without_history.price_baseline
    assert result_with_history.cost < result_without_history.cost
    assert sum(result_with_history.sequence) >= sum(result_without_history.sequence)


def test_solver_cost_nonzero_when_outside_comfort_band() -> None:
    """Slow drift should not produce a zero-cost artifact (avoid quantization lock)."""
    controller = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.9,
        comfort_temperature_tolerance=2.0,
        prediction_horizon_hours=24,
        heat_loss_coeff=0.006974286751628748,
        heat_gain_coeff=0.8,
    )
    outdoor = [6.5] * 96  # 24h at 15-min steps
    prices = [1.006] * 96
    decision, result = controller.suggest_control(19.9, outdoor, prices, past_prices=[1.0] * 16)
    assert decision in (True, False)
    assert result is not None
    # A "coast" sequence (no heating) falls outside the comfort band and should have non-zero cost.
    coast = controller._simulate_sequence(19.9, outdoor, [False] * 96)  # noqa: SLF001
    assert min(coast) < 18.5
    coast_cost = 0.0
    for temp in coast:
        comfort_penalty = max(0.0, abs(temp - controller.target_temperature) - controller.comfort_temperature_tolerance)
        coast_cost += (1 - controller.price_comfort_weight) * comfort_penalty * controller.time_step_hours
    assert coast_cost > 0.0
    # The solver should never return a zero-cost plan in this scenario.
    assert result.cost > 0.0
