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
        self.validated_at = None
        self.fresh_samples = 0
        self.outdoor_reason = "insufficient_history"
        self.simple_validation_mae = None
        self.outdoor_validation_mae = None

    def add_interval(self, interval):
        if interval.request is None or not 0.24 <= interval.hours <= 0.27:
            self.ready, self.reason = False, "unknown_request"
            return
        row = (interval.end, interval.request, interval.heat, finite(interval.outdoor))
        if self.history and row[0] <= self.history[-1][0]:
            return  # Duplicate/out-of-order observations are not new evidence.
        if self.history and abs(row[0] - self.history[-1][0] - 900) > 90:
            # Keep the evidence, but never carry dynamic state across a gap.
            self.fresh_samples = 0
            self.last_fit = None
            self.ready, self.reason = False, "observation_gap"
        self.history.append(row)
        self.fresh_samples += 1
        self.history = self.history[-288:]
        if self.last_fit is not None and row[0] - self.last_fit < 3600:
            return
        self.last_fit = row[0]
        self._fit()

    def _fit(self):
        # Training evidence can disappear during a long coast. Check a trusted
        # incumbent against recent observations before discarding it merely
        # because a replacement cannot yet be fitted.
        parameters, validated_at = self.parameters, self.validated_at
        self._fit_new_model()
        if not self.ready and validated_at is not None and len(self.history) >= 112:
            start = len(self.history) - 16
            if sum(i >= start for i in self._usable_indices(self.history)) >= 8:
                self.parameters, self.validated_at = parameters, validated_at
                self._retain_validated_response(self.history, start)

    def _fit_new_model(self):
        rows = self.history
        self.ready = False
        self.reason = "insufficient_history"
        self.outdoor_reason = "insufficient_history"
        self.validation_mae = self.baseline_mae = None
        self.simple_validation_mae = self.outdoor_validation_mae = None
        if len(rows) < 112:
            return
        train, validation = rows[:-16], rows[-16:]
        valid = self._usable_indices(rows)
        validation_indices = [i for i in valid if i >= len(train)]
        if len(validation_indices) < 8:
            self.reason = "insufficient_contiguous_evidence"
            return
        if len(self._usable_indices(train)) < 64:
            self.reason = "insufficient_contiguous_evidence"
            return
        validation = [rows[i] for i in validation_indices]
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
        self.validated_at = rows[-1][0]
        self.ready, self.reason = True, "ready"

    def _retain_validated_response(self, rows, start):
        if self.validated_at is None:
            return False
        validation = [rows[i] for i in self._usable_indices(rows) if i >= start]
        mean_heat = statistics.mean(r[2] for r in rows[:start])
        baseline = min(
            statistics.mean(abs(r[1] - r[2]) for r in validation),
            statistics.mean(abs(mean_heat - r[2]) for r in validation),
        )
        error = self._error(self.parameters, rows, start)
        quiet = baseline < 0.02
        if (quiet and error <= 0.02) or (not quiet and error <= baseline * 0.9):
            self.validation_mae = error
            self.baseline_mae = baseline
            self.ready = True
            self.reason = "ready_retained_low_variation" if quiet else "ready_retained_validation"
            if not quiet:
                self.validated_at = rows[-1][0]
            self.outdoor_reason = "retained_validation" if self.parameters.outdoor_gain else "not_in_use"
            return True
        # Informative observations must beat the same baseline as a new fit.
        # An incompatible
        # saved model must not become trusted again just because activity stops.
        self.validated_at = None
        return False

    @staticmethod
    def _usable_indices(rows):
        """Common scoring rows for all lags; exclude warm-up after every gap."""
        consecutive = 0
        indices = []
        for i in range(1, len(rows)):
            if abs(rows[i][0] - rows[i - 1][0] - 900) <= 90:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 8:
                indices.append(i)
        return indices

    @staticmethod
    def _error(parameters, rows, start):
        usable = set(PumpResponseModel._usable_indices(rows))
        heat = rows[start - 1][2]
        errors = []
        for i in range(start, len(rows)):
            if i not in usable:
                # Reseed from measured heat, never assume what happened offline.
                heat = rows[i][2]
                continue
            target = parameters.target(rows[i - parameters.delay_steps][1], rows[i][3])
            heat = parameters.retention * heat + (1 - parameters.retention) * target
            errors.append(abs(heat - rows[i][2]))
        return statistics.mean(errors) if errors else float("inf")

    @staticmethod
    def _outdoor_evidence(train, validation):
        if any(r[3] is None for r in (*train, *validation)):
            return "missing_outdoor_history"
        weather = [r[3] for r in train]
        if max(weather) - min(weather) < 4:
            return "insufficient_outdoor_variation"
        # Avoid attributing stronger requests in cold weather to weather alone.
        for delay in (0, 1, 2, 4, 8):
            pairs = [(train[i - delay][1], train[i][3] / 10) for i in PumpResponseModel._usable_indices(train)]
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
                    for i in self._usable_indices(train)
                ]
                bounds = [(0.05, 1.0), (0.0, 0.5)] + ([(0.0, 0.5)] if use_outdoor else [])
                result = bounded_fit(observations, bounds)
                if result is None:
                    continue
                slope, idle = result[:2]
                p = ResponseParameters(delay, a, min(slope, 1 - idle), idle,
                                       result[2] if use_outdoor else 0.0, reference, low, high)
                # Select lag on training error, then accept/reject exactly once on validation.
                fitted.append((self._error(p, train, 8), p))
        return min(fitted, key=lambda item: item[0])[1] if fitted else None

    def initial_state(self, now):
        if (
            not self.ready
            or not self.history
            or now - self.history[-1][0] > 1200
            or now < self.history[-1][0]
            or self.fresh_samples < max(1, self.parameters.delay_steps)
        ):
            return None
        n = self.parameters.delay_steps
        queue = tuple(r[1] for r in self.history[-n:]) if n else ()
        return self.history[-1][2], queue

    def export_state(self):
        return {
            "version": 4,
            "history": self.history,
            "parameters": asdict(self.parameters),
            "last_fit": self.last_fit,
            "validated_at": self.validated_at,
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
        self.history = sorted({row[0]: row for row in self.history}.values())
        self.fresh_samples = 0
        self.ready, self.reason = False, "waiting_for_fresh_validation"
        self.last_fit = None
        self.parameters = self._restore_parameters(payload.get("parameters") if isinstance(payload, dict) else None)
        stamp = finite(payload.get("validated_at")) if isinstance(payload, dict) else None
        self.validated_at = (
            stamp if stamp is not None and self.history and 0 <= stamp <= self.history[-1][0]
            and payload.get("parameters") == asdict(self.parameters) else None
        )
        self.validation_mae = self.baseline_mae = None
        self.simple_validation_mae = self.outdoor_validation_mae = None
        self.outdoor_reason = "waiting_for_fresh_validation"

    @staticmethod
    def _restore_parameters(values):
        """Keep saved coefficients, rejecting malformed or unsafe snapshots."""
        if not isinstance(values, dict):
            return ResponseParameters()
        defaults = asdict(ResponseParameters())
        parsed = {key: finite(values.get(key, default)) for key, default in defaults.items()}
        if any(value is None for value in parsed.values()):
            return ResponseParameters()
        if (parsed["delay_steps"] not in (0, 1, 2, 4, 8)
                or not 0 <= parsed["retention"] < 1
                or not 0 <= parsed["slope"] <= 1
                or not 0 <= parsed["idle"] <= 0.5
                or parsed["slope"] + parsed["idle"] > 1.000001
                or not 0 <= parsed["outdoor_gain"] <= 0.5
                or not parsed["outdoor_min"] <= parsed["outdoor_reference"] <= parsed["outdoor_max"]):
            return ResponseParameters()
        parsed["delay_steps"] = int(parsed["delay_steps"])
        return ResponseParameters(**parsed)

    def diagnostics(self):
        return {
            "ready": self.ready,
            "validated_at": self.validated_at,
            "reason": self.reason,
            "samples": len(self.history),
            "fresh_samples": self.fresh_samples,
            "fresh_samples_required": max(1, self.parameters.delay_steps),
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
