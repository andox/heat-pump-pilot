"""Bounded continuous-request MPC with learned actuator delay and heat delivery."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .control_request_utils import elapsed_smoothing_alpha, resolve_effective_heat_request
    from .pump_response import virtual_request
    from .virtual_outdoor_utils import compute_virtual_outdoor_from_mpc_step
except ImportError:
    from control_request_utils import elapsed_smoothing_alpha, resolve_effective_heat_request
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
    initial_duty: float | None = None
    first_elapsed_seconds: float | None = None
    initial_increase_age: float = float("inf")
    anti_chatter: bool = False
    reference_seconds: float = 900.0
    learned_response: bool = True

    def elapsed(self, controller, first):
        if first and self.first_elapsed_seconds is not None:
            return self.first_elapsed_seconds
        return controller.time_step_hours * 3600

    def request(self, controller, desired, previous, indoor, elapsed, increase_age):
        return resolve_effective_heat_request(
            raw_requested_duty_ratio=desired, previous_effective_duty_ratio=previous,
            elapsed_seconds=elapsed, seconds_since_increase=increase_age,
            predicted_temp=indoor, target_temperature=controller.target_temperature,
            comfort_tolerance=controller.comfort_temperature_tolerance,
            anti_chatter_enabled=self.anti_chatter,
        ).effective_requested_duty_ratio

    def value(self, controller, outdoor, price, baseline, indoor, duty, previous, elapsed=None):
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
        alpha = elapsed_smoothing_alpha(
            self.smoothing, controller.time_step_hours * 3600 if elapsed is None else elapsed,
            self.reference_seconds,
        )
        value = previous + alpha * (value - previous)
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
    increase_age: float
    violation: float = 0.0


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
    """Search command/delivery states, prioritizing the band before weighted cost."""
    heat, queue = initial_state
    nodes = [Node(indoor, heat, queue, actuator.initial_virtual, actuator.initial_duty, 0.0, None, actuator.initial_increase_age)]
    for index, (ambient, price) in enumerate(zip(outdoor, prices)):
        candidates = {}
        for node in nodes:
            elapsed = actuator.elapsed(controller, index == 0)
            age = node.increase_age + (elapsed if index else 0.0)
            for desired in (0.0, 0.25, 0.5, 0.75, 1.0):
                duty = actuator.request(controller, desired, node.duty, node.temp, elapsed, age)
                next_age = 0.0 if node.duty is None or duty > node.duty + 1e-9 else age
                virtual = actuator.value(
                    controller, ambient, price, baseline, node.temp, duty, node.virtual, elapsed
                )
                request = virtual_request(ambient, virtual, actuator.offset)
                heat, queue = parameters.advance(request, node.heat, node.queue, ambient)
                temp = controller._predict_temp(node.temp, ambient, heat)
                cost = node.cost + step_cost(
                    controller, temp, heat, price, baseline, duty, node.duty
                )
                violation = node.violation + controller._band_violation(temp) * controller.time_step_hours
                key = (
                    controller._quantize_temp(temp),
                    round(heat * 10),
                    tuple(round(v * 4) for v in queue),
                    round(request * 10),
                    duty,
                    min(900.0, next_age),
                )
                if key not in candidates or (violation, cost) < (candidates[key].violation, candidates[key].cost):
                    candidates[key] = Node(temp, heat, queue, virtual, duty, cost, node, next_age, violation)
        nodes = _prune(candidates.values())
    best = min(nodes, key=lambda node: (node.violation, node.cost))
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
    temperature, virtual, previous, cost = indoor, actuator.initial_virtual, actuator.initial_duty, 0.0
    age = actuator.initial_increase_age
    applied_duties = []
    temperatures, heating, virtuals = [indoor], [], []
    for i, (ambient, price, duty) in enumerate(zip(outdoor, prices, duties)):
        elapsed = actuator.elapsed(controller, i == 0)
        if i:
            age += elapsed
        # An externally applied first command has already passed live limiting.
        if not (i == 0 and first_virtual is not None):
            duty = actuator.request(controller, duty, previous, temperature, elapsed, age)
        if previous is None or duty > previous + 1e-9:
            age = 0.0
        applied_duties.append(duty)
        virtual = (
            first_virtual
            if i == 0 and first_virtual is not None
            else actuator.value(
                controller, ambient, price, baseline, temperature, duty, virtual, elapsed
            )
        )
        heat, queue = parameters.advance(
            virtual_request(ambient, virtual, actuator.offset), heat, queue, ambient
        )
        temperature = controller._predict_temp(temperature, ambient, heat)
        cost += step_cost(
            controller, temperature, heat, price, baseline, duty, previous
        )
        temperatures.append(temperature)
        heating.append(heat)
        virtuals.append(virtual)
        previous = duty
    return applied_duties, temperatures, heating, virtuals, cost


def _prune(candidates, limit=512):
    """Retain thermal diversity so cheap-now preheating is not pruned away.

    A cost-only beam can discard every warmer (initially more expensive) plan
    before the later price peak makes that stored heat valuable.
    """
    ordered = sorted(candidates, key=lambda node: (node.violation, node.cost))
    if len(ordered) <= limit:
        return ordered
    representatives = {}
    for node in ordered:
        key = (round(node.temp * 20), round(node.heat * 4))
        representatives.setdefault(key, node)
    reserved = list(representatives.values())[:limit]
    chosen = {id(node) for node in reserved}
    reserved.extend(node for node in ordered if id(node) not in chosen)
    return reserved[:limit]
