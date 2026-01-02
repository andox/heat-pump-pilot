"""Thermal model estimator and persistence for the MPC heat pump controller."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Bounds for the learned parameters.
MIN_HEAT_LOSS = 0.005
MAX_HEAT_LOSS = 0.25
MIN_HEAT_GAIN = 0.1
MAX_HEAT_GAIN = 1.5

# Noise settings for the EKF.
PROCESS_NOISE_TEMP = 0.01
PROCESS_NOISE_LOSS = 0.0005
PROCESS_NOISE_GAIN = 0.002
MEASUREMENT_NOISE = 0.05

STATE_INDEX_TEMP = 0
STATE_INDEX_LOSS = 1
STATE_INDEX_GAIN = 2

RLS_INDEX_GAIN = 0
RLS_INDEX_LOSS = 1


@dataclass
class ThermalModelState:
    """Container for the EKF state."""

    state: list[float]
    covariance: list[list[float]]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float to bounds."""
    return max(minimum, min(maximum, value))


def _lerp(minimum: float, maximum: float, fraction: float) -> float:
    """Linear interpolation helper."""
    return minimum + (maximum - minimum) * fraction


def _defaults_from_seed(seed: float) -> tuple[float, float]:
    """Map a 0-1 seed to initial heat loss/gain values."""
    fraction = _clamp(seed, 0.0, 1.0)
    loss = _lerp(MAX_HEAT_LOSS, MIN_HEAT_LOSS, fraction)
    gain = _lerp(MAX_HEAT_GAIN, MIN_HEAT_GAIN, fraction)
    return loss, gain


class ThermalModelEstimator:
    """Tiny EKF that learns heat loss and gain coefficients."""

    def __init__(
        self,
        *,
        seed: float = 0.5,
        initial_heat_loss: float | None = None,
        initial_heat_gain: float | None = None,
        initial_temp: float | None = None,
    ) -> None:
        loss_seed, gain_seed = _defaults_from_seed(seed)
        loss = initial_heat_loss if initial_heat_loss is not None else loss_seed
        gain = initial_heat_gain if initial_heat_gain is not None else gain_seed
        loss = _clamp(loss, MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        gain = _clamp(gain, MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        start_temp = initial_temp if initial_temp is not None else 20.0
        self._state = [float(start_temp), loss, gain]
        self._covariance = self._initial_covariance()

    @property
    def heat_loss_coeff(self) -> float:
        """Return the current heat loss coefficient."""
        return self._state[STATE_INDEX_LOSS]

    @property
    def heat_gain_coeff(self) -> float:
        """Return the current heating gain coefficient."""
        return self._state[STATE_INDEX_GAIN]

    @property
    def indoor_temp(self) -> float:
        """Return the current indoor temperature estimate."""
        return self._state[STATE_INDEX_TEMP]

    def reseed(
        self,
        *,
        seed: float,
        initial_heat_loss: float | None = None,
        initial_heat_gain: float | None = None,
        initial_temp: float | None = None,
    ) -> None:
        """Reset the model using a new seed and optional overrides."""
        loss_seed, gain_seed = _defaults_from_seed(seed)
        loss = initial_heat_loss if initial_heat_loss is not None else loss_seed
        gain = initial_heat_gain if initial_heat_gain is not None else gain_seed
        loss = _clamp(loss, MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        gain = _clamp(gain, MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        start_temp = self._state[STATE_INDEX_TEMP] if initial_temp is None else initial_temp
        self._state = [float(start_temp), loss, gain]
        self._covariance = self._initial_covariance()

    def step(self, measured_temp: float, outdoor_temp: float, heat_on: float | bool, dt_hours: float) -> float:
        """Run one EKF step given a new measurement."""
        if dt_hours <= 0:
            dt_hours = 1 / 60  # Default to 1 minute if we somehow get zero.
        try:
            u = float(heat_on)
        except (TypeError, ValueError):
            u = 1.0 if heat_on else 0.0
        u = _clamp(u, 0.0, 1.0)
        temp, loss, gain = self._state

        # Predict.
        temp_pred = temp + dt_hours * (loss * (outdoor_temp - temp) + gain * u)
        state_pred = [temp_pred, loss, gain]
        F = [
            [1 + dt_hours * (-loss), dt_hours * (outdoor_temp - temp), dt_hours * u],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        P_pred = self._mat_mult(F, self._mat_mult(self._covariance, self._transpose(F)))
        P_pred = self._add_process_noise(P_pred)

        # Update.
        innovation = measured_temp - temp_pred
        S = P_pred[STATE_INDEX_TEMP][STATE_INDEX_TEMP] + MEASUREMENT_NOISE
        if S <= 0:
            S = MEASUREMENT_NOISE
        K = [
            P_pred[0][0] / S,
            P_pred[1][0] / S,
            P_pred[2][0] / S,
        ]
        for idx in range(3):
            state_pred[idx] += K[idx] * innovation

        I_KH = [
            [1 - K[0], 0.0, 0.0],
            [-K[1], 1.0, 0.0],
            [-K[2], 0.0, 1.0],
        ]
        P_new = self._mat_mult(I_KH, P_pred)

        state_pred[STATE_INDEX_LOSS] = _clamp(state_pred[STATE_INDEX_LOSS], MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        state_pred[STATE_INDEX_GAIN] = _clamp(state_pred[STATE_INDEX_GAIN], MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        self._state = state_pred
        self._covariance = self._enforce_covariance(P_new)
        return self._state[STATE_INDEX_TEMP]

    def observe_temperature(self, measured_temp: float) -> float:
        """Update only the indoor temperature estimate from a measurement.

        Used when there is no trustworthy heat-on signal (e.g. monitor-only mode).
        """
        try:
            self._state[STATE_INDEX_TEMP] = float(measured_temp)
        except (TypeError, ValueError):
            return self._state[STATE_INDEX_TEMP]
        self._covariance[STATE_INDEX_TEMP][STATE_INDEX_TEMP] = min(
            self._covariance[STATE_INDEX_TEMP][STATE_INDEX_TEMP],
            MEASUREMENT_NOISE,
        )
        return self._state[STATE_INDEX_TEMP]

    def export_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the model."""
        return {
            "model_type": "ekf",
            "state": list(self._state),
            "covariance": [list(row) for row in self._covariance],
        }

    def restore(self, payload: dict[str, Any]) -> bool:
        """Load state from a persisted snapshot."""
        if not payload:
            return False
        model_type = payload.get("model_type")
        if model_type is not None and model_type != "ekf":
            return False
        state = payload.get("state")
        covariance = payload.get("covariance")
        if (
            not isinstance(state, list)
            or len(state) != 3
            or not isinstance(covariance, list)
            or len(covariance) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in covariance)
        ):
            return False
        try:
            state = [float(v) for v in state]
            covariance = [[float(v) for v in row] for row in covariance]
        except (TypeError, ValueError):
            return False
        state[STATE_INDEX_LOSS] = _clamp(state[STATE_INDEX_LOSS], MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        state[STATE_INDEX_GAIN] = _clamp(state[STATE_INDEX_GAIN], MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        self._state = state
        self._covariance = self._enforce_covariance(covariance)
        return True

    @staticmethod
    def _initial_covariance() -> list[list[float]]:
        """Reasonable starting covariance."""
        return [
            [0.5, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.05],
        ]

    @staticmethod
    def _transpose(matrix: list[list[float]]) -> list[list[float]]:
        """Transpose a 3x3 matrix."""
        return [list(row) for row in zip(*matrix)]

    @staticmethod
    def _mat_mult(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        """Multiply 3x3 matrices."""
        result = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
        return result

    @staticmethod
    def _add_process_noise(covariance: list[list[float]]) -> list[list[float]]:
        """Add process noise to the covariance."""
        covariance[0][0] += PROCESS_NOISE_TEMP
        covariance[1][1] += PROCESS_NOISE_LOSS
        covariance[2][2] += PROCESS_NOISE_GAIN
        return covariance

    @staticmethod
    def _enforce_covariance(covariance: list[list[float]]) -> list[list[float]]:
        """Keep covariance symmetric and positive-ish on the diagonal."""
        for i in range(3):
            for j in range(3):
                avg = (covariance[i][j] + covariance[j][i]) / 2
                covariance[i][j] = covariance[j][i] = avg
            if covariance[i][i] < 1e-6:
                covariance[i][i] = 1e-6
        return covariance


class ThermalModelRlsEstimator:
    """RLS estimator that learns heat loss and gain coefficients."""

    def __init__(
        self,
        *,
        seed: float = 0.5,
        initial_heat_loss: float | None = None,
        initial_heat_gain: float | None = None,
        initial_temp: float | None = None,
        forgetting_factor: float = 0.99,
    ) -> None:
        loss_seed, gain_seed = _defaults_from_seed(seed)
        loss = initial_heat_loss if initial_heat_loss is not None else loss_seed
        gain = initial_heat_gain if initial_heat_gain is not None else gain_seed
        loss = _clamp(loss, MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        gain = _clamp(gain, MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        start_temp = initial_temp if initial_temp is not None else 20.0
        self._theta = [float(gain), float(loss)]
        self._covariance = self._initial_covariance()
        self._last_temp = float(start_temp)
        self._forgetting_factor = _clamp(forgetting_factor, 0.9, 1.0)

    @property
    def heat_loss_coeff(self) -> float:
        """Return the current heat loss coefficient."""
        return self._theta[RLS_INDEX_LOSS]

    @property
    def heat_gain_coeff(self) -> float:
        """Return the current heating gain coefficient."""
        return self._theta[RLS_INDEX_GAIN]

    @property
    def indoor_temp(self) -> float:
        """Return the last observed indoor temperature."""
        return self._last_temp

    def reseed(
        self,
        *,
        seed: float,
        initial_heat_loss: float | None = None,
        initial_heat_gain: float | None = None,
        initial_temp: float | None = None,
    ) -> None:
        """Reset the model using a new seed and optional overrides."""
        loss_seed, gain_seed = _defaults_from_seed(seed)
        loss = initial_heat_loss if initial_heat_loss is not None else loss_seed
        gain = initial_heat_gain if initial_heat_gain is not None else gain_seed
        loss = _clamp(loss, MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        gain = _clamp(gain, MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        start_temp = self._last_temp if initial_temp is None else initial_temp
        self._theta = [float(gain), float(loss)]
        self._covariance = self._initial_covariance()
        self._last_temp = float(start_temp)

    def step(self, measured_temp: float, outdoor_temp: float, heat_on: float | bool, dt_hours: float) -> float:
        """Run one RLS update given a new measurement."""
        if dt_hours <= 0:
            dt_hours = 1 / 60
        try:
            u = float(heat_on)
        except (TypeError, ValueError):
            u = 1.0 if heat_on else 0.0
        u = _clamp(u, 0.0, 1.0)
        last_temp = self._last_temp
        dtemp = measured_temp - last_temp
        phi0 = dt_hours * u
        phi1 = dt_hours * (outdoor_temp - last_temp)

        p00, p01 = self._covariance[0]
        p10, p11 = self._covariance[1]
        if u <= 0.0:
            # Avoid gain drift when heating is off by decoupling cross-covariance terms.
            p01 = 0.0
            p10 = 0.0
        p_phi0 = p00 * phi0 + p01 * phi1
        p_phi1 = p10 * phi0 + p11 * phi1
        denom = self._forgetting_factor + (phi0 * p_phi0 + phi1 * p_phi1)
        if denom <= 0:
            denom = self._forgetting_factor
        k0 = p_phi0 / denom
        k1 = p_phi1 / denom
        prediction = phi0 * self._theta[0] + phi1 * self._theta[1]
        innovation = dtemp - prediction
        self._theta[0] = _clamp(self._theta[0] + k0 * innovation, MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        self._theta[1] = _clamp(self._theta[1] + k1 * innovation, MIN_HEAT_LOSS, MAX_HEAT_LOSS)

        r0 = phi0 * p00 + phi1 * p10
        r1 = phi0 * p01 + phi1 * p11
        p00_new = (p00 - k0 * r0) / self._forgetting_factor
        p01_new = (p01 - k0 * r1) / self._forgetting_factor
        p10_new = (p10 - k1 * r0) / self._forgetting_factor
        p11_new = (p11 - k1 * r1) / self._forgetting_factor
        self._covariance = self._enforce_covariance([[p00_new, p01_new], [p10_new, p11_new]])

        self._last_temp = float(measured_temp)
        return self._last_temp

    def observe_temperature(self, measured_temp: float) -> float:
        """Update only the temperature estimate from a measurement."""
        try:
            self._last_temp = float(measured_temp)
        except (TypeError, ValueError):
            return self._last_temp
        return self._last_temp

    def export_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the model."""
        return {
            "model_type": "rls",
            "theta": list(self._theta),
            "covariance": [list(row) for row in self._covariance],
            "last_temp": self._last_temp,
        }

    def restore(self, payload: dict[str, Any]) -> bool:
        """Load state from a persisted snapshot."""
        if not payload:
            return False
        if payload.get("model_type") != "rls":
            return False
        theta = payload.get("theta")
        covariance = payload.get("covariance")
        last_temp = payload.get("last_temp")
        if (
            not isinstance(theta, list)
            or len(theta) != 2
            or not isinstance(covariance, list)
            or len(covariance) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in covariance)
        ):
            return False
        try:
            theta = [float(v) for v in theta]
            covariance = [[float(v) for v in row] for row in covariance]
            last_temp = float(last_temp)
        except (TypeError, ValueError):
            return False
        theta[RLS_INDEX_GAIN] = _clamp(theta[RLS_INDEX_GAIN], MIN_HEAT_GAIN, MAX_HEAT_GAIN)
        theta[RLS_INDEX_LOSS] = _clamp(theta[RLS_INDEX_LOSS], MIN_HEAT_LOSS, MAX_HEAT_LOSS)
        self._theta = theta
        self._covariance = self._enforce_covariance(covariance)
        self._last_temp = last_temp
        return True

    @staticmethod
    def _initial_covariance() -> list[list[float]]:
        """Reasonable starting covariance."""
        return [
            [0.25, 0.0],
            [0.0, 0.25],
        ]

    @staticmethod
    def _enforce_covariance(covariance: list[list[float]]) -> list[list[float]]:
        """Keep covariance symmetric and positive-ish on the diagonal."""
        avg = (covariance[0][1] + covariance[1][0]) / 2
        covariance[0][1] = covariance[1][0] = avg
        for idx in range(2):
            if covariance[idx][idx] < 1e-6:
                covariance[idx][idx] = 1e-6
        return covariance


class ThermalModelStorage:
    """Simple JSON file persistence for the estimator."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, Any] | None:
        """Load persisted state from disk."""
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, payload: ThermalModelState | dict[str, Any]) -> None:
        """Persist state atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, ThermalModelState):
            data = {"version": 1, "state": payload.state, "covariance": payload.covariance}
        elif isinstance(payload, dict):
            data = dict(payload)
        else:
            return
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                json.dump(data, file, ensure_ascii=True)
                file.flush()
                os.fsync(file.fileno())
            temp_path.replace(self._path)
        except FileNotFoundError as exc:
            LOGGER.warning("Thermal model temp file missing during save: %s", exc)
            self._path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Thermal model save failed (%s). Falling back to direct write.", exc)
            self._path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
