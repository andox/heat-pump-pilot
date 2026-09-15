"""Delayed delivery and applied-command forecast consistency."""

import pytest

from mpc_controller import MpcController
from pump_response import ResponseParameters
from response_optimizer import VirtualActuator, replay, step_cost


def controller(**options):
    defaults = dict(
        target_temperature=21,
        price_comfort_weight=0.4,
        comfort_temperature_tolerance=0.3,
        prediction_horizon_hours=3,
        heat_loss_coeff=0.01,
        heat_gain_coeff=0.8,
        background_gain=0.1,
    )
    return MpcController(**{**defaults, **options})


def test_background_is_part_of_binary_and_continuous_predictions():
    mpc = controller(heat_loss_coeff=0)
    assert mpc._predict_temp(20, 5, 0) == pytest.approx(20.025)
    mpc.update_settings(background_gain=0.2)
    assert mpc._predict_temp(20, 5, 0) == pytest.approx(20.05)


def test_response_plan_cost_and_temperature_replay_match():
    mpc = controller()
    outdoor, prices = [5.0] * 12, [0.2] * 4 + [3.0] * 8
    context = (
        ResponseParameters(2, 0.5, 0.8, 0),
        (0.2, (0.0, 0.0)),
        VirtualActuator(10, -15, 0.5, 5, entity_step=0.5),
    )
    _, plan = mpc.suggest_control(
        20.5, outdoor, prices, price_baseline_override=1, response_context=context
    )
    assert len(plan.duty_sequence) == len(plan.predicted_heating) == 12
    assert len(plan.predicted_temperatures) == 13
    assert plan.predicted_heating[:2] == pytest.approx([0.1, 0.05])
    duties, temperatures, heating, virtuals, cost = replay(
        mpc, 20.5, outdoor, prices, 1, plan.duty_sequence, *context
    )
    assert temperatures == pytest.approx(plan.predicted_temperatures)
    assert cost == pytest.approx(plan.cost)
    # First command changed by output limiting: forecast must change accordingly.
    modified = replay(mpc, 20.5, outdoor, prices, 1, duties, *context, first_virtual=15)
    assert modified[3][0] == 15
    assert modified[2][:2] == heating[:2]
    assert modified[2][2] <= heating[2]


def test_residual_heating_still_has_price_cost_after_request_stops():
    mpc = controller()
    assert step_cost(mpc, 21, 0.5, 2, 1, 0, 0) > step_cost(mpc, 21, 0, 2, 1, 0, 0)


def test_number_step_rounding_matches_applied_service():
    actuator = VirtualActuator(10, -15, 1, 5, -20, 20, 0.5)
    assert actuator.clamp(-2.25) == -2.5
    assert actuator.clamp(25) == 20


def test_direct_controller_fallback_has_no_response_forecast():
    _, result = controller().suggest_control(20, [5], [1])
    assert result.duty_sequence is None
    assert result.predicted_heating is None
