"""Bounded continuous-request MPC with learned actuator delay and heat delivery."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .pump_response import virtual_request
    from .virtual_outdoor_utils import compute_virtual_outdoor_from_mpc_step
except ImportError:
    from pump_response import virtual_request
    from virtual_outdoor_utils import compute_virtual_outdoor_from_mpc_step


@dataclass(frozen=True)
class VirtualActuator:
    offset: float
    minimum: float
    smoothing: float
    initial_virtual: float
    entity_min: float = -100.0
    entity_max: float = 100.0
    entity_step: float = 0.0

    def value(self, controller, outdoor, price, baseline, indoor, duty, previous):
        value = compute_virtual_outdoor_from_mpc_step(
            base_outdoor=outdoor,
            heat_on=duty > 0,
            duty_ratio=duty,
            virtual_heat_offset=self.offset,
            price=price,
            price_baseline=baseline,
            price_comfort_weight=controller.price_comfort_weight,
            price_penalty_curve=controller.price_penalty_curve,
            price_ratio_cap=controller.price_ratio_cap,
            predicted_temp=indoor,
            target_temperature=controller.target_temperature,
            comfort_temperature_tolerance=controller.comfort_temperature_tolerance,
            overshoot_warm_bias_enabled=controller.overshoot_warm_bias_enabled,
            overshoot_warm_bias_curve=controller.overshoot_warm_bias_curve,
        )
        if outdoor >= self.minimum:
            value = max(self.minimum, value)
        value = previous + self.smoothing * (value - previous)
        value = min(outdoor + self.offset, 25.0, max(outdoor - self.offset, value))
        if outdoor >= self.minimum:
            value = max(self.minimum, value)
        return self.clamp(value)

    def clamp(self, value):
        """Match number service bounds and step rounding."""
        value = min(self.entity_max, max(self.entity_min, value))
        if self.entity_step > 0:
            from decimal import Decimal, ROUND_HALF_UP

            step = Decimal(str(self.entity_step))
            value = float(
                (Decimal(str(value)) / step).to_integral_value(rounding=ROUND_HALF_UP)
                * step
            )
        return min(self.entity_max, max(self.entity_min, value))


@dataclass(slots=True)
class Node:
    temp: float
    heat: float
    queue: tuple
    virtual: float
    duty: float | None
    cost: float
    previous: Node | None


def step_cost(controller, temp, heat, price, baseline, duty, previous_duty):
    ratio = max(-1.0, price / max(baseline, 0.01))
    return (
        (1 - controller.price_comfort_weight) * controller._comfort_penalty(temp)
        + controller.price_comfort_weight
        * controller._apply_price_penalty_curve(ratio)
        * heat
    ) * controller.time_step_hours + (
        0.05 * abs(duty - previous_duty) if previous_duty is not None else 0.0
    )


def optimize(
    controller, indoor, outdoor, prices, baseline, parameters, initial_state, actuator
):
    """Keep full temperature/response state, pruning to 256 candidate plans per step."""
    heat, queue = initial_state
    nodes = [Node(indoor, heat, queue, actuator.initial_virtual, None, 0.0, None)]
    for ambient, price in zip(outdoor, prices):
        candidates = {}
        for node in nodes:
            for duty in (0.0, 0.25, 0.5, 0.75, 1.0):
                virtual = actuator.value(
                    controller, ambient, price, baseline, node.temp, duty, node.virtual
                )
                request = virtual_request(ambient, virtual, actuator.offset)
                heat, queue = parameters.advance(request, node.heat, node.queue, ambient)
                temp = controller._predict_temp(node.temp, ambient, heat)
                cost = node.cost + step_cost(
                    controller, node.temp, heat, price, baseline, duty, node.duty
                )
                key = (
                    controller._quantize_temp(temp),
                    round(heat * 10),
                    tuple(round(v * 4) for v in queue),
                    round(request * 10),
                    duty,
                )
                if key not in candidates or cost < candidates[key].cost:
                    candidates[key] = Node(temp, heat, queue, virtual, duty, cost, node)
        nodes = sorted(candidates.values(), key=lambda node: node.cost)[:256]
    best = min(nodes, key=lambda node: node.cost)
    duties = []
    while best.previous is not None:
        duties.append(best.duty)
        best = best.previous
    duties.reverse()
    return replay(
        controller,
        indoor,
        outdoor,
        prices,
        baseline,
        duties,
        parameters,
        initial_state,
        actuator,
    )


def replay(
    controller,
    indoor,
    outdoor,
    prices,
    baseline,
    duties,
    parameters,
    initial_state,
    actuator,
    first_virtual=None,
):
    heat, queue = initial_state
    temperature, virtual, previous, cost = indoor, actuator.initial_virtual, None, 0.0
    temperatures, heating, virtuals = [indoor], [], []
    for i, (ambient, price, duty) in enumerate(zip(outdoor, prices, duties)):
        virtual = (
            first_virtual
            if i == 0 and first_virtual is not None
            else actuator.value(
                controller, ambient, price, baseline, temperature, duty, virtual
            )
        )
        heat, queue = parameters.advance(
            virtual_request(ambient, virtual, actuator.offset), heat, queue, ambient
        )
        cost += step_cost(
            controller, temperature, heat, price, baseline, duty, previous
        )
        temperature = controller._predict_temp(temperature, ambient, heat)
        temperatures.append(temperature)
        heating.append(heat)
        virtuals.append(virtual)
        previous = duty
    return list(duties), temperatures, heating, virtuals, cost
