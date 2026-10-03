"""Binary on/off MPC optimizer for the heat pump without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from statistics import median
from typing import Iterable, Sequence

try:
    from .const import (
        DEFAULT_PRICE_PENALTY_CURVE,
        DEFAULT_PRICE_RATIO_CAP,
        PRICE_BASELINE_FLOOR,
        PRICE_PENALTY_CURVES,
        PRICE_RATIO_MIN,
    )
except ImportError:  # pragma: no cover - allow direct module imports in tests
    from const import (  # type: ignore
        DEFAULT_PRICE_PENALTY_CURVE,
        DEFAULT_PRICE_RATIO_CAP,
        PRICE_BASELINE_FLOOR,
        PRICE_PENALTY_CURVES,
        PRICE_RATIO_MIN,
    )

_LOGGER = logging.getLogger(__name__)

# Thermal model coefficients for a 1 hour step.
HEAT_LOSS_COEFF = 0.05
HEAT_GAIN_COEFF = 0.6
# Use 15-minute steps to enable finer preheat/coast behaviour.
TIME_STEP_HOURS = 0.25
# Temperature buckets limit the number of candidate plans retained at each step.
# Actual simulated temperatures are never rounded to these buckets.
TEMP_RESOLUTION = 0.02
# Discourage 15-minute binary chatter; continuous mode still refines the final request.
TOGGLE_PENALTY = 0.05


@dataclass
class ControlResult:
    """Result of an optimized control sequence."""

    sequence: list[bool]
    predicted_temperatures: list[float]
    cost: float
    price_baseline: float
    duty_sequence: list[float] | None = None
    predicted_heating: list[float] | None = None
    planned_virtual_outdoor: list[float] | None = None
    response_model_source: str | None = None
    comfort_violation_degree_hours: float = 0.0
    comfort_status: str = "within_band"


@dataclass(slots=True)
class _PlanNode:
    """One candidate with its exact temperature and a link to its preceding step."""

    temperature: float
    cost: float
    action: bool | None
    previous: _PlanNode | None
    violation: float = 0.0


class MpcController:
    """Binary on/off MPC optimizer using dynamic programming."""

    def __init__(
        self,
        target_temperature: float,
        price_comfort_weight: float,
        comfort_temperature_tolerance: float,
        prediction_horizon_hours: int,
        price_penalty_curve: str = DEFAULT_PRICE_PENALTY_CURVE,
        price_ratio_cap: float = DEFAULT_PRICE_RATIO_CAP,
        time_step_hours: float = TIME_STEP_HOURS,
        heat_loss_coeff: float = HEAT_LOSS_COEFF,
        heat_gain_coeff: float = HEAT_GAIN_COEFF,
        background_gain: float = 0.0,
        virtual_heat_offset: float = 0.0,
        overshoot_warm_bias_enabled: bool = False,
        overshoot_warm_bias_curve: str = "linear",
    ) -> None:
        self.target_temperature = target_temperature
        self.price_comfort_weight = price_comfort_weight
        self.price_penalty_curve = (
            price_penalty_curve if price_penalty_curve in PRICE_PENALTY_CURVES else DEFAULT_PRICE_PENALTY_CURVE
        )
        self.price_ratio_cap = max(float(price_ratio_cap), 1.0)
        self.comfort_temperature_tolerance = comfort_temperature_tolerance
        self.prediction_horizon_hours = prediction_horizon_hours
        self.time_step_hours = time_step_hours
        self.heat_loss_coeff = heat_loss_coeff
        self.heat_gain_coeff = heat_gain_coeff
        self.background_gain = background_gain
        self.virtual_heat_offset = virtual_heat_offset
        self.overshoot_warm_bias_enabled = overshoot_warm_bias_enabled
        self.overshoot_warm_bias_curve = overshoot_warm_bias_curve
        self._temp_resolution = TEMP_RESOLUTION

    def update_settings(
        self,
        *,
        target_temperature: float | None = None,
        price_comfort_weight: float | None = None,
        price_penalty_curve: str | None = None,
        price_ratio_cap: float | None = None,
        comfort_temperature_tolerance: float | None = None,
        prediction_horizon_hours: int | None = None,
        heat_loss_coeff: float | None = None,
        heat_gain_coeff: float | None = None,
        background_gain: float | None = None,
        virtual_heat_offset: float | None = None,
        overshoot_warm_bias_enabled: bool | None = None,
        overshoot_warm_bias_curve: str | None = None,
    ) -> None:
        """Update controller parameters."""
        if target_temperature is not None:
            self.target_temperature = target_temperature
        if price_comfort_weight is not None:
            self.price_comfort_weight = price_comfort_weight
        if price_penalty_curve is not None:
            self.price_penalty_curve = (
                price_penalty_curve if price_penalty_curve in PRICE_PENALTY_CURVES else DEFAULT_PRICE_PENALTY_CURVE
            )
        if price_ratio_cap is not None:
            self.price_ratio_cap = max(float(price_ratio_cap), 1.0)
        if comfort_temperature_tolerance is not None:
            self.comfort_temperature_tolerance = comfort_temperature_tolerance
        if prediction_horizon_hours is not None:
            self.prediction_horizon_hours = prediction_horizon_hours
        if heat_loss_coeff is not None:
            self.heat_loss_coeff = heat_loss_coeff
        if heat_gain_coeff is not None:
            self.heat_gain_coeff = heat_gain_coeff
        if background_gain is not None:
            self.background_gain = background_gain
        if virtual_heat_offset is not None:
            self.virtual_heat_offset = virtual_heat_offset
        if overshoot_warm_bias_enabled is not None:
            self.overshoot_warm_bias_enabled = overshoot_warm_bias_enabled
        if overshoot_warm_bias_curve is not None:
            self.overshoot_warm_bias_curve = overshoot_warm_bias_curve

    def suggest_control(
        self,
        indoor_temp: float,
        outdoor_forecast: Sequence[float],
        price_forecast: Sequence[float],
        past_prices: Sequence[float] | None = None,
        price_baseline_override: float | None = None,
        response_context: tuple | None = None,
    ) -> tuple[bool, ControlResult | None]:
        """Return the recommended control action and the best simulation result."""
        steps = max(1, int(self.prediction_horizon_hours / self.time_step_hours))
        outdoor = self._normalize_series(outdoor_forecast, steps, indoor_temp)
        prices = self._normalize_series(price_forecast, steps, 1.0)

        max_price = max(prices) if prices else 1.0
        price_baseline = None
        if price_baseline_override is not None:
            try:
                price_baseline = float(price_baseline_override)
            except (TypeError, ValueError):
                price_baseline = None
        if price_baseline is None or not math.isfinite(price_baseline) or price_baseline <= 0:
            baseline_pool = [p for p in prices if p is not None and math.isfinite(p)]
            if past_prices:
                baseline_pool.extend(p for p in past_prices if p is not None and math.isfinite(p))
            median_price = median(baseline_pool) if baseline_pool else max_price
            if median_price <= 0:
                median_price = PRICE_BASELINE_FLOOR
            price_baseline = max(median_price, PRICE_BASELINE_FLOOR)

        if response_context is not None and self.time_step_hours == 0.25:
            try:
                from .response_optimizer import optimize
            except ImportError:
                from response_optimizer import optimize
            duties, predicted, heating, virtuals, cost = optimize(
                self, indoor_temp, outdoor, prices, price_baseline, *response_context)
            result = ControlResult([d > 0 for d in duties], predicted, cost, price_baseline,
                                   duties, heating, virtuals,
                                   "learned" if response_context[2].learned_response else "fallback")
            self.update_comfort_diagnostics(result)
            return bool(duties and duties[0] > 0), result

        sequence, cost = self._optimize(indoor_temp, outdoor, prices, price_baseline, max_price)
        predicted = self._simulate_sequence(indoor_temp, outdoor, sequence)

        result = ControlResult(
            sequence=sequence,
            predicted_temperatures=predicted,
            cost=cost,
            price_baseline=price_baseline,
        )
        self.update_comfort_diagnostics(result)
        if not sequence:
            return False, result
        return bool(sequence[0]), result

    def _predict_temp(self, indoor_temp: float, outdoor_temp: float, heating_power: float) -> float:
        """Predict indoor temperature for the next step."""
        delta = self.heat_loss_coeff * (outdoor_temp - indoor_temp) * self.time_step_hours
        heating_effect = self.heat_gain_coeff * heating_power * self.time_step_hours
        return indoor_temp + delta + heating_effect + self.background_gain * self.time_step_hours

    def _quantize_temp(self, temp: float) -> int:
        """Group nearby candidate temperatures without changing their state."""
        scaled = temp / self._temp_resolution
        # Round to nearest bucket (symmetric for negative values).
        if scaled >= 0:
            return int(math.floor(scaled + 0.5))
        return int(math.ceil(scaled - 0.5))

    def _apply_price_penalty_curve(self, ratio: float) -> float:
        """Apply the configured price curve above the baseline ratio."""
        if ratio <= 1.0:
            return ratio
        capped_ratio = min(ratio, self.price_ratio_cap)
        x = max(0.0, capped_ratio - 1.0)
        if self.price_penalty_curve == "sqrt":
            adjusted = math.sqrt(x)
        elif self.price_penalty_curve == "quadratic":
            adjusted = x * x
        else:
            adjusted = x
        return 1.0 + adjusted

    def _optimize(
        self,
        indoor_temp: float,
        outdoor: Sequence[float],
        prices: Sequence[float],
        price_baseline: float,
        max_price: float,
    ) -> tuple[list[bool], float]:
        """Search forward, retaining the cheapest candidate per bucket/action.

        Buckets only prune similar plans; every surviving plan carries its exact
        simulated temperature. Feeding bucket centers back into the model would
        repeatedly erase small heat loss/gain increments. The search remains an
        approximation because nearby candidates are merged, but the returned
        cost and temperature trajectory describe the same unrounded plan.
        """
        initial = _PlanNode(indoor_temp, 0.0, None, None)
        candidates: dict[tuple[int, bool | None], _PlanNode] = {
            (self._quantize_temp(indoor_temp), None): initial
        }
        baseline_denom = max(price_baseline, PRICE_BASELINE_FLOOR)

        for idx, outdoor_temp in enumerate(outdoor):
            next_candidates: dict[tuple[int, bool | None], _PlanNode] = {}
            current_price = prices[idx] if prices else 0.0
            price_ratio = max(PRICE_RATIO_MIN, current_price / baseline_denom)
            heating_cost = (
                self.price_comfort_weight
                * self._apply_price_penalty_curve(price_ratio)
                * self.time_step_hours
            )
            for node in candidates.values():
                for action in (False, True):
                    next_temp = self._predict_temp(node.temperature, outdoor_temp, float(action))
                    comfort_cost = (
                        (1.0 - self.price_comfort_weight)
                        * self._comfort_penalty(next_temp) * self.time_step_hours
                    )
                    toggle_cost = TOGGLE_PENALTY if node.action is not None and node.action != action else 0.0
                    cost = node.cost + comfort_cost + (heating_cost if action else 0.0) + toggle_cost
                    violation = node.violation + self._band_violation(next_temp) * self.time_step_hours
                    key = (self._quantize_temp(next_temp), action)
                    incumbent = next_candidates.get(key)
                    if incumbent is None or (violation, cost) < (incumbent.violation, incumbent.cost):
                        next_candidates[key] = _PlanNode(next_temp, cost, action, node, violation)
            candidates = next_candidates

        best = min(candidates.values(), key=lambda node: (node.violation, node.cost))
        cost = best.cost
        path: list[bool] = []
        while best.previous is not None:
            path.append(bool(best.action))
            best = best.previous
        path.reverse()
        return path, cost

    def _band_violation(self, temp: float) -> float:
        """Temperature outside the allowed band, independent of price weight."""
        return max(0.0, abs(temp - self.target_temperature) - self.comfort_temperature_tolerance)

    def _comfort_penalty(self, temp: float) -> float:
        """Preference for the target within the band; no separate warm-side bias.

        One band-width from target costs one normalized comfort unit per hour.
        Price and comfort are preferences among equally feasible plans, not a
        mechanism for buying permission to breach the band.
        """
        scale = max(0.2, self.comfort_temperature_tolerance)
        return ((temp - self.target_temperature) / scale) ** 2

    def update_comfort_diagnostics(self, result: ControlResult) -> None:
        # Include the terminal temperature; the measured initial state cannot be
        # changed retroactively and is not scored as a planning choice.
        result.comfort_violation_degree_hours = sum(
            self._band_violation(t) * self.time_step_hours
            for t in result.predicted_temperatures[1:]
        )
        result.comfort_status = (
            "within_band" if result.comfort_violation_degree_hours < 1e-8
            else "predicted_breach"
        )

    def _simulate_sequence(self, indoor_temp: float, outdoor: Sequence[float], sequence: Sequence[bool]) -> list[float]:
        """Simulate temperatures for a chosen sequence.

        Returns one sample per step plus the terminal temperature after the last
        action so downstream consumers can plot the entire future horizon.
        """
        temp = indoor_temp
        predicted: list[float] = []
        for idx, action in enumerate(sequence):
            predicted.append(temp)
            heat_power = 1.0 if action else 0.0
            temp = self._predict_temp(temp, outdoor[idx], heat_power)
        predicted.append(temp)
        return predicted

    def _normalize_series(self, values: Iterable[float] | None, steps: int, fallback: float) -> list[float]:
        """Ensure we have a float series of the desired length."""
        normalized = [v for v in self._coerce_float_iterable(values) if v is not None]
        if not normalized:
            normalized = [fallback]

        if len(normalized) < steps:
            normalized.extend([normalized[-1]] * (steps - len(normalized)))
        else:
            normalized = normalized[:steps]
        return normalized

    @staticmethod
    def _coerce_float_iterable(values: Iterable[float] | None) -> list[float]:
        """Convert inputs to floats, dropping invalid entries."""
        if values is None:
            return []
        coerced: list[float] = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            coerced.append(numeric)
        return coerced
