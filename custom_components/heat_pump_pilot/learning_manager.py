"""Collect measured evidence independently from control decisions."""

from __future__ import annotations

from collections import deque

try:
    from .learning_interval import IntervalCollector
    from .pump_response import PumpResponseModel
    from .learning_math import finite
except ImportError:
    from learning_interval import IntervalCollector
    from pump_response import PumpResponseModel
    from learning_math import finite


class LearningManager:
    def __init__(self, model, minutes=60):
        self.model = model
        self.house = IntervalCollector(minutes)
        self.pump_intervals = IntervalCollector(15)
        self.pump = PumpResponseModel()
        self.last_update = None
        self.recent = deque()

    def observe(self, now, indoor, outdoor, heat, request=None):
        """Return true only when a complete house interval was consumed."""
        # Keep actual observation edges to align delayed state at arbitrary
        # sensor-triggered control times, not only at a completed bin boundary.
        if self.recent and (now < self.recent[-1][0] or now - self.recent[-1][0] > 300):
            self.recent.clear()
        if self.recent and now == self.recent[-1][0]:
            self.recent.pop()
        self.recent.append((now, finite(request), finite(heat)))
        while len(self.recent) > 1 and self.recent[1][0] <= now - 8100:
            self.recent.popleft()
        pump_sample = self.pump_intervals.observe(now, indoor, outdoor, heat, request)
        if pump_sample is not None:
            self.pump.add_interval(pump_sample)
        elif self.pump_intervals.last_status in (
            "observation_gap",
            "insufficient_coverage",
            "temperature_jump",
        ):
            self.pump.ready = False
            self.pump.reason = self.pump_intervals.last_status
        sample = self.house.observe(now, indoor, outdoor, heat)
        if sample is None:
            # Keep the temperature estimate current, without adapting coefficients.
            if indoor is not None:
                self.model.observe_temperature(indoor)
            return False
        if hasattr(self.model, "add_interval"):
            self.model.add_interval(sample)
        else:
            # Restart at the measured beginning of this interval, never the last
            # controller invocation or a temperature restored from yesterday.
            self.model.observe_temperature(sample.start_temp)
            self.model.step(sample.end_temp, sample.outdoor, sample.heat, sample.hours)
        self.last_update = now
        return True

    def response_state(self, now):
        """Average the preceding bins relative to this exact planning origin."""
        if self.pump.initial_state(now) is None or not self.recent:
            return None
        if now - self.recent[-1][0] > 300:
            return None
        rows = list(self.recent) + [(now, None, None)]

        def mean(end, column):
            total = known = 0.0
            for left, right in zip(rows, rows[1:]):
                seconds = max(0.0, min(end, right[0]) - max(end - 900, left[0]))
                value = left[column]
                if value is not None and 0 <= value <= 1:
                    total += seconds * value
                    known += seconds
            return total / known if known >= 855 else None

        heat = mean(now, 2)
        queue = tuple(
            mean(now - i * 900, 1)
            for i in reversed(range(self.pump.parameters.delay_steps))
        )
        if heat is None or any(value is None for value in queue):
            return None
        return heat, queue

    def diagnostics(self):
        return {
            "interval_minutes": self.house.seconds / 60,
            "interval_status": self.house.last_status,
            "coverage": self.house.last_coverage,
            "last_update": self.last_update,
            "fit_status": getattr(self.model, "status", "legacy_estimator"),
            "gain_identified": getattr(self.model, "gain_identified", None),
            "background_gain": getattr(self.model, "background_gain", 0.0),
            "fit_mae_c_per_hour": getattr(self.model, "fit_error", None),
        }
