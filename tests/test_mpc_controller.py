from __future__ import annotations

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
