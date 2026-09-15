"""Learn a monotone delayed response from actual virtual outdoor values to heat."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

try:
    from .learning_math import bounded_fit, finite
except ImportError:
    from learning_math import bounded_fit, finite


def virtual_request(outdoor, virtual, offset):
    if any(finite(v) is None for v in (outdoor, virtual, offset)) or offset <= 0:
        return None
    return max(0.0, min(1.0, (outdoor + offset - virtual) / (2 * offset)))


@dataclass(frozen=True)
class ResponseParameters:
    delay_steps: int = 0
    retention: float = 0.0
    slope: float = 1.0
    idle: float = 0.0
    outdoor_gain: float = 0.0  # Extra heating fraction per 10 C colder.
    outdoor_reference: float = 0.0
    outdoor_min: float = 0.0
    outdoor_max: float = 0.0

    def target(self, request, outdoor=None):
        correction = 0.0
        if self.outdoor_gain and finite(outdoor) is not None:
            # Never extrapolate a mild-weather fit into an unseen winter.
            ambient = min(self.outdoor_max, max(self.outdoor_min, float(outdoor)))
            correction = self.outdoor_gain * (self.outdoor_reference - ambient) / 10
        return max(0.0, min(1.0, self.idle + self.slope * request + correction))

    def advance(self, request, heat, queue, outdoor=None):
        delayed = queue[0] if self.delay_steps else request
        next_queue = (*queue[1:], request) if self.delay_steps else ()
        target = self.target(delayed, outdoor)
        next_heat = self.retention * heat + (1 - self.retention) * target
        return next_heat, next_queue


class PumpResponseModel:
    """Fit hourly on complete 15-minute samples and validate on the last four hours."""

    def __init__(self):
        self.history = []
        self.parameters = ResponseParameters()
        self.ready = False
        self.reason = "insufficient_history"
        self.validation_mae = None
        self.baseline_mae = None
        self.last_fit = None
        self.outdoor_reason = "insufficient_history"
        self.simple_validation_mae = None
        self.outdoor_validation_mae = None

    def add_interval(self, interval):
        if interval.request is None or not 0.24 <= interval.hours <= 0.27:
            self.ready, self.reason = False, "unknown_request"
            return
        row = (interval.end, interval.request, interval.heat, finite(interval.outdoor))
        if self.history and abs(row[0] - self.history[-1][0] - 900) > 90:
            self.history = []
            self.ready, self.reason = False, "observation_gap"
        self.history.append(row)
        self.history = self.history[-288:]
        if self.last_fit is not None and row[0] - self.last_fit < 3600:
            return
        self.last_fit = row[0]
        self._fit()

    def _fit(self):
        rows = self.history
        self.ready = False
        self.reason = "insufficient_history"
        self.outdoor_reason = "insufficient_history"
        self.validation_mae = self.baseline_mae = None
        self.simple_validation_mae = self.outdoor_validation_mae = None
        if len(rows) < 112:
            return
        train, validation = rows[:-16], rows[-16:]
        heating = sum(r[2] for r in train) / 4
        if heating < 3 or len(train) / 4 - heating < 3:
            self.reason = "insufficient_heating_variation"
            return
        if statistics.pvariance(r[1] for r in train) < 0.02:
            self.reason = "insufficient_request_variation"
            return
        parameters = self._fit_candidate(train)
        if parameters is None:
            self.reason = "ill_conditioned"
            return
        self.simple_validation_mae = self._error(parameters, rows, len(train))
        self.validation_mae = self.simple_validation_mae
        self.outdoor_reason = self._outdoor_evidence(train, validation)
        if self.outdoor_reason == "eligible":
            candidate = self._fit_candidate(train, use_outdoor=True)
            if candidate is None:
                self.outdoor_reason = "ill_conditioned"
            else:
                self.outdoor_validation_mae = self._error(candidate, rows, len(train))
                if (candidate.outdoor_gain > 0.001
                        and self.outdoor_validation_mae <= self.simple_validation_mae * 0.9
                        and self.simple_validation_mae - self.outdoor_validation_mae >= 0.01):
                    parameters = candidate
                    self.validation_mae = self.outdoor_validation_mae
                    self.outdoor_reason = "validated"
                else:
                    self.outdoor_reason = "no_validation_improvement"
        baseline = min(
            statistics.mean(abs(r[1] - r[2]) for r in validation),
            statistics.mean(
                abs(statistics.mean(t[2] for t in train) - r[2]) for r in validation
            ),
        )
        self.baseline_mae = baseline
        if baseline < 0.02 or self.validation_mae > baseline * 0.9:
            self.reason = "no_validation_improvement"
            return
        self.parameters = parameters
        self.ready, self.reason = True, "ready"

    @staticmethod
    def _error(parameters, rows, start):
        heat = rows[start - 1][2]
        errors = []
        for i in range(start, len(rows)):
            target = parameters.target(rows[i - parameters.delay_steps][1], rows[i][3])
            heat = parameters.retention * heat + (1 - parameters.retention) * target
            errors.append(abs(heat - rows[i][2]))
        return statistics.mean(errors)

    @staticmethod
    def _outdoor_evidence(train, validation):
        if any(r[3] is None for r in (*train, *validation)):
            return "missing_outdoor_history"
        weather = [r[3] for r in train]
        if max(weather) - min(weather) < 4:
            return "insufficient_outdoor_variation"
        # Avoid attributing stronger requests in cold weather to weather alone.
        for delay in (0, 1, 2, 4, 8):
            pairs = [(train[i - delay][1], train[i][3] / 10) for i in range(8, len(train))]
            xmean = statistics.mean(x for x, _ in pairs)
            ymean = statistics.mean(y for _, y in pairs)
            variance = statistics.mean((x - xmean) ** 2 for x, _ in pairs)
            covariance = statistics.mean((x - xmean) * (y - ymean) for x, y in pairs)
            slope = covariance / variance if variance > 1e-9 else 0
            residual = statistics.mean((y - ymean - slope * (x - xmean)) ** 2 for x, y in pairs)
            if residual < 0.01:
                return "outdoor_request_confounded"
        return "eligible"

    def _fit_candidate(self, train, use_outdoor=False):
        reference = statistics.mean(r[3] for r in train) if use_outdoor else 0.0
        low = min(r[3] for r in train) if use_outdoor else 0.0
        high = max(r[3] for r in train) if use_outdoor else 0.0
        fitted = []
        for delay in (0, 1, 2, 4, 8):
            for tau in (0, 0.5, 1, 2):
                a = math.exp(-0.25 / tau) if tau else 0.0
                observations = [
                    (
                        (train[i - delay][1], 1.0)
                        + (((reference - train[i][3]) / 10,) if use_outdoor else ()),
                        (train[i][2] - a * train[i - 1][2]) / (1 - a),
                    )
                    for i in range(max(1, delay), len(train))
                ]
                bounds = [(0.05, 1.0), (0.0, 0.5)] + ([(0.0, 0.5)] if use_outdoor else [])
                result = bounded_fit(observations, bounds)
                if result is None:
                    continue
                slope, idle = result[:2]
                p = ResponseParameters(delay, a, min(slope, 1 - idle), idle,
                                       result[2] if use_outdoor else 0.0, reference, low, high)
                # Select lag on training error, then accept/reject exactly once on validation.
                fitted.append((self._error(p, train, 9), p))
        return min(fitted, key=lambda item: item[0])[1] if fitted else None

    def initial_state(self, now):
        if (
            not self.ready
            or not self.history
            or now - self.history[-1][0] > 1200
            or now < self.history[-1][0]
        ):
            return None
        n = self.parameters.delay_steps
        queue = tuple(r[1] for r in self.history[-n:]) if n else ()
        return self.history[-1][2], queue

    def export_state(self):
        return {
            "version": 2,
            "history": self.history,
            "parameters": asdict(self.parameters),
            "last_fit": self.last_fit,
        }

    def restore(self, payload):
        # Readiness is intentionally revalidated from fresh samples after restart.
        rows = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []
        self.history = []
        for r in rows[-288:]:
            if isinstance(r, (list, tuple)) and len(r) in (3, 4):
                core = tuple(finite(v) for v in r[:3])
                if all(v is not None for v in core) and 0 <= core[1] <= 1 and 0 <= core[2] <= 1:
                    self.history.append((*core, finite(r[3]) if len(r) == 4 else None))
        self.history.sort(key=lambda row: row[0])
        self.ready, self.reason = False, "waiting_for_fresh_validation"
        self.last_fit = None
        self.parameters = ResponseParameters()
        self.validation_mae = self.baseline_mae = None
        self.simple_validation_mae = self.outdoor_validation_mae = None
        self.outdoor_reason = "waiting_for_fresh_validation"

    def diagnostics(self):
        return {
            "ready": self.ready,
            "reason": self.reason,
            "samples": len(self.history),
            "delay_minutes": self.parameters.delay_steps * 15,
            "retention": self.parameters.retention,
            "request_gain": self.parameters.slope,
            "idle_heat": self.parameters.idle,
            "validation_mae": self.validation_mae,
            "baseline_mae": self.baseline_mae,
            "outdoor_active": self.ready and self.parameters.outdoor_gain > 0,
            "outdoor_reason": self.outdoor_reason,
            "outdoor_samples": sum(r[3] is not None for r in self.history),
            "outdoor_gain_per_10c": self.parameters.outdoor_gain,
            "outdoor_reference_c": self.parameters.outdoor_reference,
            "outdoor_range_c": [self.parameters.outdoor_min, self.parameters.outdoor_max],
            "simple_validation_mae": self.simple_validation_mae,
            "outdoor_validation_mae": self.outdoor_validation_mae,
        }
