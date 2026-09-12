from __future__ import annotations

from itertools import product

import pytest

from const import PRICE_BASELINE_FLOOR
from mpc_controller import MpcController


def test_mpc_does_not_heat_when_comfort_already_satisfied() -> None:
    """Regression: coarse/bias quantization must not force unnecessary heating.

    Scenario based on a real HA snapshot where indoor temperature is above the
    target, and the all-off plan stays within the comfort band for the full
    horizon. The optimizer should not choose heating (which is always non-negative
    cost with this objective).
    """

    controller = MpcController(
        target_temperature=20.5,
        price_comfort_weight=0.9,
        comfort_temperature_tolerance=2.0,
        prediction_horizon_hours=24,
        time_step_hours=0.25,
        heat_loss_coeff=0.005,
        heat_gain_coeff=0.7682129956996044,
    )

    indoor_temp = 21.9
    outdoor_forecast = [
        7.3,
        7.4,
        7.5,
        7.7,
        7.5,
        7.4,
        7.4,
        7.4,
        7.4,
        6.8,
        6.6,
        6.6,
        6.5,
        6.2,
        6.1,
        6.0,
        6.0,
        5.7,
        5.9,
        6.0,
        6.2,
        6.1,
        6.1,
        6.1,
        6.3,
        6.6,
        6.9,
        7.2,
        7.3,
        6.7,
        6.0,
        5.8,
        5.6,
        5.9,
        6.2,
        6.5,
        6.8,
        7.0,
        7.0,
        7.1,
        7.1,
        7.0,
        7.0,
        6.8,
        6.7,
        6.7,
        6.4,
        6.2,
        6.2,
        6.6,
        7.0,
        7.1,
        6.5,
        5.6,
        4.9,
    ]
    price_forecast = [
        0.595,
        0.59,
        0.602,
        0.598,
        0.595,
        0.58,
        0.609,
        0.607,
        0.607,
        0.604,
        0.623,
        0.621,
        0.626,
        0.628,
        0.637,
        0.639,
        0.64,
        0.643,
        0.624,
        0.629,
        0.633,
        0.638,
        0.628,
        0.635,
        0.644,
        0.654,
        0.634,
        0.646,
        0.656,
        0.675,
        0.661,
        0.673,
        0.683,
        0.69,
        0.683,
        0.69,
        0.696,
        0.701,
        0.678,
        0.68,
        0.687,
        0.69,
        0.678,
        0.679,
        0.681,
        0.68,
        0.675,
        0.679,
        0.679,
        0.672,
        0.654,
        0.651,
        0.643,
        0.638,
        0.632,
        0.629,
        0.625,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
        0.622,
    ]

    action, result = controller.suggest_control(indoor_temp, outdoor_forecast, price_forecast)
    assert result is not None
    assert action is False
    assert result.sequence[0] is False
    assert result.cost == 0.0


def test_price_comfort_weight_changes_optimal_first_action() -> None:
    """The price/comfort weight must influence the optimization outcome."""
    indoor_temp = 19.0
    outdoor_forecast = [19.0, 19.0]  # No heat loss term when equal to indoor.
    price_forecast = [10.0, 1.0]  # Expensive now, cheap later.

    comfort_first = MpcController(
        target_temperature=20.0,
        price_comfort_weight=0.0,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=2,
        time_step_hours=1.0,
        heat_loss_coeff=0.0,
        heat_gain_coeff=1.0,
    )
    action_comfort, _ = comfort_first.suggest_control(indoor_temp, outdoor_forecast, price_forecast)
    assert action_comfort is True

    price_first = MpcController(
        target_temperature=20.0,
        price_comfort_weight=1.0,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=2,
        time_step_hours=1.0,
        heat_loss_coeff=0.0,
        heat_gain_coeff=1.0,
    )
    action_price, _ = price_first.suggest_control(indoor_temp, outdoor_forecast, price_forecast)
    assert action_price is False


def test_negative_prices_are_bounded_in_cost() -> None:
    """Negative prices should incentivize heating without unbounded cost."""
    controller = MpcController(
        target_temperature=20.0,
        price_comfort_weight=1.0,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=2,
        time_step_hours=1.0,
        heat_loss_coeff=0.0,
        heat_gain_coeff=0.0,
    )
    prices = [-1000.0, -1000.0]
    outdoor = [20.0, 20.0]
    decision, result = controller.suggest_control(20.0, outdoor, prices)
    assert result is not None
    assert decision is True
    assert result.price_baseline == PRICE_BASELINE_FLOOR
    assert result.cost == -2.0


def test_overshoot_bias_increases_above_target_penalty() -> None:
    """Above-target penalty should increase when overshoot warm bias is enabled."""
    with_bias = MpcController(
        target_temperature=20.0,
        price_comfort_weight=0.5,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=1,
        time_step_hours=1.0,
        heat_loss_coeff=0.0,
        heat_gain_coeff=1.0,
        virtual_heat_offset=5.0,
        overshoot_warm_bias_enabled=True,
        overshoot_warm_bias_curve="linear",
    )
    without_bias = MpcController(
        target_temperature=20.0,
        price_comfort_weight=0.5,
        comfort_temperature_tolerance=0.0,
        prediction_horizon_hours=1,
        time_step_hours=1.0,
        heat_loss_coeff=0.0,
        heat_gain_coeff=1.0,
        virtual_heat_offset=5.0,
        overshoot_warm_bias_enabled=False,
        overshoot_warm_bias_curve="linear",
    )

    assert with_bias._comfort_penalty(19.0) == without_bias._comfort_penalty(19.0)
    assert with_bias._comfort_penalty(21.0) > without_bias._comfort_penalty(21.0)


def _unrounded_cost(controller, indoor, outdoor, prices, sequence):
    """Independently replay the linear-price objective with full precision."""
    temp = indoor
    total = 0.0
    previous = None
    for ambient, price, action in zip(outdoor, prices, sequence):
        error = max(0.0, abs(temp - controller.target_temperature) - controller.comfort_temperature_tolerance)
        total += (1.0 - controller.price_comfort_weight) * error * controller.time_step_hours
        total += controller.price_comfort_weight * price * action * controller.time_step_hours
        if previous is not None and previous != action:
            total += 0.05
        temp += controller.time_step_hours * (
            controller.heat_loss_coeff * (ambient - temp) + controller.heat_gain_coeff * action
        )
        previous = action
    return total


def test_sub_bucket_cooling_accumulates_and_changes_plan() -> None:
    controller = MpcController(
        target_temperature=20.0, price_comfort_weight=0.5,
        comfort_temperature_tolerance=0.2, prediction_horizon_hours=24,
        heat_loss_coeff=0.001, heat_gain_coeff=0.4,
    )
    outdoor, prices = [-10.0] * 96, [1.0] * 96
    _, result = controller.suggest_control(20.0, outdoor, prices, price_baseline_override=1.0)
    coast_cost = _unrounded_cost(controller, 20.0, outdoor, prices, [False] * 96)
    assert any(result.sequence)
    assert 0.0 < result.cost < coast_cost
    assert result.cost == pytest.approx(_unrounded_cost(controller, 20.0, outdoor, prices, result.sequence))


@pytest.mark.parametrize('ambient', [-10.0, 50.0])
def test_sub_bucket_passive_drift_cost_matches_actual_forecast(ambient) -> None:
    controller = MpcController(
        target_temperature=20.0, price_comfort_weight=0.5,
        comfort_temperature_tolerance=0.2, prediction_horizon_hours=24,
        heat_loss_coeff=0.001, heat_gain_coeff=0.0,
    )
    outdoor, prices = [ambient] * 96, [1.0] * 96
    _, result = controller.suggest_control(20.007, outdoor, prices, price_baseline_override=1.0)
    assert not any(result.sequence)
    assert abs(result.predicted_temperatures[-1] - 20.007) > 0.7
    assert result.cost > 0.0
    assert result.cost == pytest.approx(_unrounded_cost(controller, 20.007, outdoor, prices, result.sequence))


def test_sub_bucket_heating_accumulates() -> None:
    controller = MpcController(
        target_temperature=20.0, price_comfort_weight=0.0,
        comfort_temperature_tolerance=0.0, prediction_horizon_hours=4,
        heat_loss_coeff=0.0, heat_gain_coeff=0.02,
    )
    outdoor, prices = [19.0] * 16, [1.0] * 16
    _, result = controller.suggest_control(19.0, outdoor, prices, price_baseline_override=1.0)
    assert all(result.sequence)
    assert result.predicted_temperatures[-1] == pytest.approx(19.08)
    assert result.cost < 4.0
    assert result.cost == pytest.approx(_unrounded_cost(controller, 19.0, outdoor, prices, result.sequence))


@pytest.mark.parametrize('loss,gain', [(0.001, 0.4), (0.02, 0.8), (0.0, 0.02)])
def test_short_horizon_matches_exhaustive_unrounded_search(loss, gain) -> None:
    controller = MpcController(
        target_temperature=20.0, price_comfort_weight=0.1,
        comfort_temperature_tolerance=0.05, prediction_horizon_hours=2,
        heat_loss_coeff=loss, heat_gain_coeff=gain,
    )
    outdoor = [-10.0, -9.0, -8.0, -10.0, -11.0, -10.0, -9.0, -10.0]
    prices = [0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 0.5, 0.5]
    _, result = controller.suggest_control(19.807, outdoor, prices, price_baseline_override=1.0)
    optimal_cost = min(
        _unrounded_cost(controller, 19.807, outdoor, prices, sequence)
        for sequence in product((False, True), repeat=8)
    )
    assert result.cost == pytest.approx(optimal_cost)
