"""Time-aligned observations, independent of how often MPC is called."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .learning_math import finite
except ImportError:
    from learning_math import finite


@dataclass(frozen=True)
class LearningInterval:
    end: float
    hours: float
    start_temp: float | None
    end_temp: float | None
    outdoor: float
    heat: float
    request: float | None
    coverage: float


class IntervalCollector:
    """Integrate held signals before installing each new observation.

    A one-minute timer guarantees regular observations even with slow sensors.
    The accumulator is deliberately not persisted across restarts. Unknown
    time is not treated as zero heating or silently normalized to a full interval.
    """

    def __init__(self, minutes=60, *, require_indoor=True):
        self.seconds = minutes * 60
        self.require_indoor = require_indoor
        self.reset()
        self.last_status = "waiting_for_interval"
        self.last_coverage = 0.0

    def reset(self):
        self.start = self.last = None
        self.start_temp = None
        self.previous = None
        self.known = self.outdoor_sum = self.heat_sum = 0.0
        self.request_known = self.request_sum = 0.0

    def observe(self, now, indoor, outdoor, heat, request=None):
        values = tuple(finite(v) for v in (indoor, outdoor, heat, request))
        indoor, outdoor, heat, request = values
        if indoor is not None and not -10 < indoor < 50:
            indoor = None
        if heat is not None and not 0 <= heat <= 1:
            heat = None
        if request is not None and not 0 <= request <= 1:
            request = None
        if self.last is not None and (now < self.last or now - self.last > 300):
            self.reset()
            self.last_status = "observation_gap"
        if self.start is None:
            self.start, self.start_temp = now, indoor
        if self.last is not None and now > self.last:
            dt = now - self.last
            old_indoor, old_outdoor, old_heat, old_request = self.previous
            if (
                (not self.require_indoor or old_indoor is not None)
                and old_outdoor is not None
                and old_heat is not None
            ):
                self.known += dt
                self.outdoor_sum += dt * old_outdoor
                self.heat_sum += dt * old_heat
                if old_request is not None:
                    self.request_known += dt
                    self.request_sum += dt * old_request
        self.last, self.previous = now, (indoor, outdoor, heat, request)
        duration = now - self.start
        if duration < self.seconds:
            return None
        coverage = self.known / duration
        self.last_coverage = coverage
        result = None
        if (
            self.require_indoor
            and self.start_temp is not None
            and indoor is not None
            and abs(indoor - self.start_temp) > 2 * duration / 3600
        ):
            self.last_status = "temperature_jump"
        elif coverage >= 0.95 and (
            not self.require_indoor or (self.start_temp is not None and indoor is not None)
        ):
            result = LearningInterval(
                now,
                duration / 3600,
                self.start_temp,
                indoor,
                self.outdoor_sum / self.known,
                self.heat_sum / self.known,
                self.request_sum / self.request_known
                if self.request_known / duration >= 0.95
                else None,
                coverage,
            )
            self.last_status = "updated"
        else:
            self.last_status = "insufficient_coverage"
        self.reset()
        self.start = self.last = now
        self.start_temp, self.previous = indoor, (indoor, outdoor, heat, request)
        return result
