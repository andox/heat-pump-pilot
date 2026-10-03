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


def test_first_change_penalty_starts_from_current_request():
    from response_optimizer import optimize
    # Flat thermal/electricity costs isolate the change penalty.
    mpc = controller(heat_loss_coeff=0, heat_gain_coeff=0, background_gain=0, price_comfort_weight=0)
    context = (ResponseParameters(0, 0, 0, 0), (0, ()),
               VirtualActuator(10, -15, 1, 5, initial_duty=0.75))
    plan = optimize(mpc, 21, [5], [1], 1, *context)
    assert plan[0] == [0.75]
    changed = replay(mpc, 21, [5], [1], 1, [0], *context)
    assert changed[-1] - plan[-1] == pytest.approx(0.05 * 0.75)


def test_response_plan_models_elapsed_ramp_and_smoothing():
    from response_optimizer import optimize
    mpc = controller()
    actuator = VirtualActuator(10, -15, 0.8, 15, initial_duty=0,
                              first_elapsed_seconds=60, anti_chatter=True)
    context = (ResponseParameters(0, 0, 0.8, 0), (0, ()), actuator)
    plan = optimize(mpc, 21, [5]*4, [1]*4, 1, *context)
    assert plan[0][0] <= 0.25 / 15
    assert replay(mpc, 21, [5]*4, [1]*4, 1, plan[0], *context) == plan
    # Hold expiry halfway through a control interval only allows half a ramp.
    assert actuator.request(mpc, 0, 1, 21, 900, 1350) == pytest.approx(0.875)
    expected = 15 + (1 - 0.2**(60/900)) * (5-15)
    assert actuator.value(mpc, 5, 1, 1, 21, 0.5, 15, 60) == pytest.approx(expected)


def test_replay_keeps_applied_first_request_then_limits_future_requests():
    mpc = controller()
    context = (ResponseParameters(0, 0, 0.8, 0), (0, ()),
               VirtualActuator(10, -15, 0.8, 15, initial_duty=0,
                               first_elapsed_seconds=60, anti_chatter=True))
    plan = replay(mpc, 21, [5]*3, [1]*3, 1, [1, 0, 0], *context, first_virtual=-5)
    assert plan[0] == pytest.approx([1, 1, 0.75])
    assert plan[3][0] == -5
