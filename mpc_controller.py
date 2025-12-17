"""Binary on/off MPC optimizer for the heat pump without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from statistics import median
from typing import Iterable, Sequence

_LOGGER = logging.getLogger(__name__)

# Thermal model coefficients for a 1 hour step.
HEAT_LOSS_COEFF = 0.05
HEAT_GAIN_COEFF = 0.6
# Use 15-minute steps to enable finer preheat/coast behaviour.
TIME_STEP_HOURS = 0.25
# Temperature quantization for the DP cache.
# NOTE: This must be small enough to capture typical per-step heat loss at a 15 min
# cadence; otherwise quantization can cause the optimizer to think the temperature
# is "stuck" (too coarse) or drift unrealistically (biased quantization).
TEMP_RESOLUTION = 0.02
# Small penalty for actuator toggling.
TOGGLE_PENALTY = 0.01


@dataclass
class ControlResult:
    """Result of an optimized control sequence."""

    sequence: list[bool]
    predicted_temperatures: list[float]
    cost: float
    price_baseline: float


class MpcController:
    """Binary on/off MPC optimizer using dynamic programming."""

    def __init__(
        self,
        target_temperature: float,
        price_comfort_weight: float,
        comfort_temperature_tolerance: float,
        prediction_horizon_hours: int,
        time_step_hours: float = TIME_STEP_HOURS,
        heat_loss_coeff: float = HEAT_LOSS_COEFF,
        heat_gain_coeff: float = HEAT_GAIN_COEFF,
    ) -> None:
        self.target_temperature = target_temperature
        self.price_comfort_weight = price_comfort_weight
        self.comfort_temperature_tolerance = comfort_temperature_tolerance
        self.prediction_horizon_hours = prediction_horizon_hours
        self.time_step_hours = time_step_hours
        self.heat_loss_coeff = heat_loss_coeff
        self.heat_gain_coeff = heat_gain_coeff
        self._temp_resolution = TEMP_RESOLUTION

    def update_settings(
        self,
        *,
        target_temperature: float | None = None,
        price_comfort_weight: float | None = None,
        comfort_temperature_tolerance: float | None = None,
        prediction_horizon_hours: int | None = None,
        heat_loss_coeff: float | None = None,
        heat_gain_coeff: float | None = None,
    ) -> None:
        """Update controller parameters."""
        if target_temperature is not None:
            self.target_temperature = target_temperature
        if price_comfort_weight is not None:
            self.price_comfort_weight = price_comfort_weight
        if comfort_temperature_tolerance is not None:
            self.comfort_temperature_tolerance = comfort_temperature_tolerance
        if prediction_horizon_hours is not None:
            self.prediction_horizon_hours = prediction_horizon_hours
        if heat_loss_coeff is not None:
            self.heat_loss_coeff = heat_loss_coeff
        if heat_gain_coeff is not None:
            self.heat_gain_coeff = heat_gain_coeff

    def suggest_control(
        self,
        indoor_temp: float,
        outdoor_forecast: Sequence[float],
        price_forecast: Sequence[float],
        past_prices: Sequence[float] | None = None,
    ) -> tuple[bool, ControlResult | None]:
        """Return the recommended control action and the best simulation result."""
        steps = max(1, int(self.prediction_horizon_hours / self.time_step_hours))
        outdoor = self._normalize_series(outdoor_forecast, steps, indoor_temp)
        prices = self._normalize_series(price_forecast, steps, 1.0)

        max_price = max(prices) if prices else 1.0
        baseline_pool = list(prices)
        if past_prices:
            baseline_pool.extend(p for p in past_prices if p is not None)
        median_price = median(baseline_pool) if baseline_pool else max_price
        if median_price <= 0:
            median_price = max_price
        price_baseline = median_price

        sequence, cost = self._optimize(indoor_temp, outdoor, prices, price_baseline, max_price)
        predicted = self._simulate_sequence(indoor_temp, outdoor, sequence)

        result = ControlResult(
            sequence=sequence,
            predicted_temperatures=predicted,
            cost=cost,
            price_baseline=price_baseline,
        )
        return bool(sequence[0]), result

    def _predict_temp(self, indoor_temp: float, outdoor_temp: float, heating_power: float) -> float:
        """Predict indoor temperature for the next step."""
        delta = self.heat_loss_coeff * (outdoor_temp - indoor_temp) * self.time_step_hours
        heating_effect = self.heat_gain_coeff * heating_power * self.time_step_hours
        return indoor_temp + delta + heating_effect

    def _quantize_temp(self, temp: float) -> int:
        """Quantize temperature to an integer bucket for DP caching."""
        scaled = temp / self._temp_resolution
        # Round to nearest bucket (symmetric for negative values).
        if scaled >= 0:
            return int(math.floor(scaled + 0.5))
        return int(math.ceil(scaled - 0.5))

    def _dequantize_temp(self, bucket: int) -> float:
        """Convert a quantized temp bucket back to float."""
        return bucket * self._temp_resolution

    def _optimize(
        self,
        indoor_temp: float,
        outdoor: Sequence[float],
        prices: Sequence[float],
        price_baseline: float,
        max_price: float,
    ) -> tuple[list[bool], float]:
        """Dynamic programming solver for the optimal on/off sequence."""
        steps = len(outdoor)
        start_bucket = self._quantize_temp(indoor_temp)

        @lru_cache(maxsize=None)
        def solve(idx: int, temp_bucket: int, prev_action: int) -> tuple[float, tuple[bool, ...]]:
            if idx >= steps:
                return 0.0, ()

            temp = self._dequantize_temp(temp_bucket)
            best_cost = float("inf")
            best_path: tuple[bool, ...] = ()
            current_price = prices[idx] if prices else 0.0

            for action in (False, True):
                heat_power = 1.0 if action else 0.0
                comfort_penalty = max(0.0, abs(temp - self.target_temperature) - self.comfort_temperature_tolerance)
                # Penalize heating relative to the (median) baseline price.
                # Using max_price here would squash the relative differences we care about.
                baseline_denom = max(price_baseline, 1e-6)
                price_penalty = current_price / baseline_denom
                price_cost = self.price_comfort_weight * price_penalty * heat_power * self.time_step_hours
                comfort_cost = (1.0 - self.price_comfort_weight) * comfort_penalty * self.time_step_hours
                toggle_cost = TOGGLE_PENALTY if prev_action != -1 and bool(prev_action) != action else 0.0

                step_cost = price_cost + comfort_cost + toggle_cost
                next_temp = self._predict_temp(temp, outdoor[idx], heat_power)
                next_bucket = self._quantize_temp(next_temp)
                future_cost, future_path = solve(idx + 1, next_bucket, int(action))
                total_cost = step_cost + future_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = (action,) + future_path

            return best_cost, best_path

        cost, path = solve(0, start_bucket, -1)
        return list(path), cost

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
                coerced.append(float(value))
            except (TypeError, ValueError):
                continue
        return coerced
