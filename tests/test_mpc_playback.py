from __future__ import annotations

from mpc_controller import MpcController
from virtual_outdoor_utils import compute_planned_virtual_outdoor_temperatures

STEP_HOURS = 0.25


def _repeat(value: float, count: int) -> list[float]:
    return [value] * count


def _steps(hours: int) -> int:
    return int(hours / STEP_HOURS)


def test_preheat_when_prices_rise() -> None:
    """Playback scenario: heat early when prices rise later."""
    horizon_hours = 24
    steps = _steps(horizon_hours)
    controller = MpcController(
        target_temperature=20.0,
        price_comfort_weight=0.8,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.04,
        heat_gain_coeff=0.6,
    )

    indoor_temp = 19.0
    outdoor_forecast = _repeat(4.0, steps)
    price_forecast = _repeat(0.25, 32) + _repeat(0.4, 16) + _repeat(1.2, steps - 48)

    action, result = controller.suggest_control(indoor_temp, outdoor_forecast, price_forecast)

    assert result is not None
    assert action is True
    assert result.sequence[0] is True
    assert sum(result.sequence[:32]) >= 1
    assert result.predicted_temperatures[0] == indoor_temp


def test_above_target_prefers_idle() -> None:
    """Playback scenario: above target should stay off."""
    horizon_hours = 24
    steps = _steps(horizon_hours)
    controller = MpcController(
        target_temperature=20.0,
        price_comfort_weight=0.7,
        comfort_temperature_tolerance=0.25,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.05,
        heat_gain_coeff=0.7,
    )

    indoor_temp = 21.0
    outdoor_forecast = _repeat(22.0, steps)
    price_forecast = _repeat(0.9, steps)

    action, result = controller.suggest_control(indoor_temp, outdoor_forecast, price_forecast)

    assert result is not None
    assert action is False
    assert sum(result.sequence) == 0


def test_tolerance_changes_action() -> None:
    """Playback scenario: tighter tolerance should heat sooner."""
    horizon_hours = 12
    steps = _steps(horizon_hours)
    outdoor_forecast = _repeat(18.5, steps)
    price_forecast = _repeat(0.9, steps)

    tighter = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.2,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.03,
        heat_gain_coeff=0.7,
    )
    looser = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.2,
        comfort_temperature_tolerance=0.6,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.03,
        heat_gain_coeff=0.7,
    )

    indoor_temp = 20.0

    action_tight, tight_result = tighter.suggest_control(indoor_temp, outdoor_forecast, price_forecast)
    action_loose, loose_result = looser.suggest_control(indoor_temp, outdoor_forecast, price_forecast)

    assert action_tight is True
    assert action_loose in (True, False)
    assert tight_result is not None
    assert loose_result is not None
    assert sum(loose_result.sequence) <= sum(tight_result.sequence)


def test_planned_virtual_outdoor_warm_shift_with_prices() -> None:
    """Playback scenario: warm shift when idle and prices are high."""
    steps = _steps(24)
    sequence = [False] * steps
    outdoor_forecast = _repeat(0.5, steps)
    price_forecast = _repeat(1.4, steps)
    predicted_temperatures = _repeat(21.5, steps)

    planned = compute_planned_virtual_outdoor_temperatures(
        sequence,
        outdoor_forecast,
        price_forecast,
        predicted_temperatures=predicted_temperatures,
        base_outdoor_fallback=outdoor_forecast[0],
        virtual_heat_offset=5.0,
        price_comfort_weight=0.5,
        price_baseline=1.0,
        target_temperature=20.0,
        overshoot_warm_bias_enabled=True,
        overshoot_warm_bias_margin=0.3,
        overshoot_warm_bias_full=1.5,
        max_virtual_outdoor=25.0,
    )

    assert planned is not None
    assert len(planned) == len(sequence)
    assert any(p > o for p, o in zip(planned, outdoor_forecast))
    for planned_value, base in zip(planned, outdoor_forecast):
        assert planned_value >= base
        assert planned_value <= base + 5.0


def test_price_weight_reduces_heating_during_spike() -> None:
    """Playback scenario: higher price weight should cut heating during a spike."""
    horizon_hours = 24
    steps = _steps(horizon_hours)
    outdoor_forecast = _repeat(2.0, steps)
    price_forecast = _repeat(0.4, 32) + _repeat(1.2, steps - 32)
    indoor_temp = 20.0

    comfort_focused = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.2,
        comfort_temperature_tolerance=0.2,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.05,
        heat_gain_coeff=0.6,
    )
    price_focused = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.9,
        comfort_temperature_tolerance=0.2,
        prediction_horizon_hours=horizon_hours,
        time_step_hours=STEP_HOURS,
        heat_loss_coeff=0.05,
        heat_gain_coeff=0.6,
    )

    _, comfort_result = comfort_focused.suggest_control(indoor_temp, outdoor_forecast, price_forecast)
    _, price_result = price_focused.suggest_control(indoor_temp, outdoor_forecast, price_forecast)

    assert comfort_result is not None
    assert price_result is not None
    assert sum(price_result.sequence) <= sum(comfort_result.sequence)
