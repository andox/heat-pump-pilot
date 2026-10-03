"""Behavioral contracts for price-aware heat storage and coasting.

The plant rollout below is independent of the optimizer's temperature and
response methods. Tests compare delivered heat/cost, not just an ON first slot.
"""
from itertools import product
from dataclasses import replace

import pytest

from mpc_controller import MpcController
from pump_response import ResponseParameters
from response_optimizer import VirtualActuator, optimize, replay


def controller(weight=0.8, **changes):
    args = dict(target_temperature=20.5, price_comfort_weight=weight,
                comfort_temperature_tolerance=1.2, prediction_horizon_hours=12,
                heat_loss_coeff=0.04, heat_gain_coeff=1.0, background_gain=0.05)
    return MpcController(**(args | changes))


def plan(c, prices=None, indoor=20.5, outdoor=None, response=None, smoothing=1):
    count = int(c.prediction_horizon_hours * 4)
    prices = prices if prices is not None else [0.1]*16 + [2]*16 + [0.5]*(count-32)
    outdoor = outdoor if outdoor is not None else [10]*count
    response = response or ResponseParameters()
    actuator = VirtualActuator(10, -15, smoothing, min(25, outdoor[0]+10),
                               initial_duty=0, learned_response=False)
    _, result = c.suggest_control(indoor, outdoor, prices, price_baseline_override=1,
                                 response_context=(response, (0, (0,)*response.delay_steps), actuator))
    return result


def electrical_proxy(result, prices):
    return sum(h*p*0.25 for h,p in zip(result.predicted_heating, prices))


def test_preheat_from_target_then_coast_through_expensive_peak():
    c = controller()
    result = plan(c)
    assert result.comfort_status == "within_band"
    assert result.predicted_temperatures[16] > c.target_temperature + 0.6
    assert sum(result.predicted_heating[:16])*0.25 > 2
    assert sum(result.predicted_heating[16:32])*0.25 < 0.1
    assert result.predicted_temperatures[32] < result.predicted_temperatures[16] - 1
    assert min(result.predicted_temperatures) >= 19.3 - 1e-8
    assert max(result.predicted_temperatures) <= 21.7 + 1e-8
    # Actual warm backoff follows the plan while coasting.
    assert all(v >= 19.9 for v in result.planned_virtual_outdoor[16:32])


def test_price_weight_keeps_meaning_inside_comfort_boundaries():
    prices = [0.1]*16 + [2]*16 + [0.5]*16
    comfort, saving = plan(controller(0.2), prices), plan(controller(0.8), prices)
    assert saving.comfort_status == comfort.comfort_status == "within_band"
    assert electrical_proxy(saving, prices) < electrical_proxy(comfort, prices)*0.85
    assert sum(saving.predicted_heating[16:32]) < sum(comfort.predicted_heating[16:32])
    assert sum(abs(t-20.5) for t in comfort.predicted_temperatures) < sum(abs(t-20.5) for t in saving.predicted_temperatures)


def test_price_peak_timing_changes_when_heat_is_stored():
    c = controller()
    early_cheap = plan(c)
    late_cheap = plan(c, [2]*16 + [0.1]*16 + [0.5]*16)
    assert sum(early_cheap.predicted_heating[:16]) > sum(late_cheap.predicted_heating[:16])
    assert sum(early_cheap.predicted_heating[16:32]) < sum(late_cheap.predicted_heating[16:32])


@pytest.mark.parametrize("weight", [0, 0.5, 0.8, 1])
def test_even_extreme_prices_cannot_buy_avoidable_cold(weight):
    c = controller(weight)
    result = plan(c, [100]*48)
    assert result.comfort_status == "within_band"
    assert min(result.predicted_temperatures) >= 19.3 - 1e-8
    assert sum(result.predicted_heating) > 0


def test_negative_prices_do_not_buy_overheating():
    result = plan(controller(1), [-5]*48)
    assert result.comfort_status == "within_band"
    assert max(result.predicted_temperatures) <= 21.7 + 1e-8
    assert sum(result.predicted_heating) < 48


def test_sun_warmed_house_backs_off_without_a_separate_bias():
    c = controller(background_gain=0.32, heat_loss_coeff=0.01)
    result = plan(c, [0.1]*48, indoor=21.8, outdoor=[0]*48)
    # Passive sunshine already raises temperature; heating can only worsen it.
    assert not any(result.duty_sequence)
    assert result.comfort_status == "predicted_breach"
    assert all(v == 10 for v in result.planned_virtual_outdoor)


def test_insufficient_capacity_reports_breach_and_heats_despite_price():
    c = controller(1, heat_gain_coeff=0.1, heat_loss_coeff=0.08)
    result = plan(c, [100]*48, indoor=19.3, outdoor=[-10]*48)
    assert result.comfort_status == "predicted_breach"
    assert result.comfort_violation_degree_hours > 0
    # The -15 C output floor at -10 C outdoors caps delivered request at 75%.
    assert all(h == pytest.approx(0.75) for h in result.predicted_heating)


def test_delayed_pump_preheat_is_verified_in_independent_plant():
    c = controller()
    response = ResponseParameters(delay_steps=2, retention=0.6, slope=0.9, idle=0.05)
    result = plan(c, response=response, smoothing=0.8)
    temp, heat, queue = 20.5, 0, [0, 0]
    temps, delivered = [temp], []
    for virtual in result.planned_virtual_outdoor:
        request = max(0, min(1, (20-virtual)/20))
        delayed = queue.pop(0)
        queue.append(request)
        heat = 0.6*heat + 0.4*min(1, 0.05 + 0.9*delayed)
        temp += 0.25*(0.04*(10-temp) + heat + 0.05)
        temps.append(temp)
        delivered.append(heat)
    assert temps == pytest.approx(result.predicted_temperatures)
    assert delivered == pytest.approx(result.predicted_heating)
    assert min(temps) >= 19.3 - 1e-8
    assert max(temps) <= 21.7 + 1e-8
    assert temps[16] > 20.5
    assert sum(delivered[16:32]) < sum(delivered[:16])
    assert any(d == 0 and h > 0.1 for d,h in zip(result.duty_sequence, delivered))


def test_terminal_temperature_counts_even_on_final_step():
    c = controller(1, prediction_horizon_hours=0.25)
    result = plan(c, [100], indoor=19.31, outdoor=[10])
    assert result.duty_sequence[0] > 0
    assert result.predicted_temperatures[-1] >= 19.3


def test_small_response_search_matches_exhaustive_independent_objective():
    c = controller(prediction_horizon_hours=1)
    prices = [0.1, 0.1, 3, 3]
    params = ResponseParameters(delay_steps=1, retention=0.3, slope=0.8, idle=0.05)
    actuator = VirtualActuator(10, -15, 0.8, 20, initial_duty=0)
    actual = optimize(c, 19.5, [10]*4, prices, 1, params, (0, (0,)), actuator)
    def score(duties):
        temp, heat, queue, virtual, previous = 19.5, 0, [0], 20, 0
        violation = cost = 0
        for duty, price in zip(duties, prices):
            virtual += 0.8*(20-20*duty-virtual)
            delayed = queue.pop(0)
            queue.append((20-virtual)/20)
            heat = 0.3*heat + 0.7*(0.05+0.8*delayed)
            temp += 0.25*(0.04*(10-temp)+heat+0.05)
            violation += max(0, abs(temp-20.5)-1.2)*0.25
            cost += (0.2*((temp-20.5)/1.2)**2 + 0.8*price*heat)*0.25 + 0.05*abs(duty-previous)
            previous = duty
        return violation, cost
    expected = min(score(ds) for ds in product((0,0.25,0.5,0.75,1), repeat=4))
    assert score(actual[0]) == pytest.approx(expected)
    assert actual[-1] == pytest.approx(expected[1])


def test_replanning_executes_preheat_then_coast_in_independent_house():
    # Each step executes only the first new command, as HA does; this catches
    # beautiful open-loop plans that abandon preheating on the next replan.
    c = controller(prediction_horizon_hours=6)
    prices = [0.1]*8 + [3]*8 + [0.5]*32
    temp, virtual, previous = 20.5, 20, 0
    actual_temps, actual_heat = [temp], []
    for step in range(16):
        actuator = VirtualActuator(10, -15, 1, virtual, initial_duty=previous)
        _, result = c.suggest_control(temp, [10]*24, prices[step:step+24],
            price_baseline_override=1,
            response_context=(ResponseParameters(), (0,()), actuator))
        previous = result.duty_sequence[0]
        virtual = result.planned_virtual_outdoor[0]
        delivered = (20-virtual)/20
        temp += 0.25*(0.04*(10-temp) + delivered + 0.05)
        actual_temps.append(temp)
        actual_heat.append(delivered)
    assert actual_temps[8] > 20.5 + 0.2
    assert sum(actual_heat[8:]) < sum(actual_heat[:8])*0.25
    assert actual_temps[-1] < actual_temps[8]
    assert min(actual_temps) >= 19.3 - 1e-8
    assert max(actual_temps) <= 21.7 + 1e-8
