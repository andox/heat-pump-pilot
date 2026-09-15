"""Bounded hourly house learning with a background term and gain observability gate."""

from __future__ import annotations

from dataclasses import asdict
import statistics

try:
    from .learning_interval import LearningInterval
    from .learning_math import bounded_fit, finite
    from .thermal_model import ThermalModelEstimator
except ImportError:
    from learning_interval import LearningInterval
    from learning_math import bounded_fit, finite
    from thermal_model import ThermalModelEstimator


class AdaptiveThermalModel:
    """Fit all house terms together; never transplant a fitted background term alone."""

    def __init__(self, *, window_hours=72, **seeds):
        self.window_hours = window_hours
        self._base = ThermalModelEstimator(**seeds)
        self.background_gain = 0.0
        self.history = []
        self.status = "insufficient_history"
        self.gain_identified = False
        self.fit_error = None
        self.last_update = None

    @property
    def heat_loss_coeff(self):
        return self._base.heat_loss_coeff

    @property
    def heat_gain_coeff(self):
        return self._base.heat_gain_coeff

    @property
    def indoor_temp(self):
        return self._base.indoor_temp

    def observe_temperature(self, temperature):
        return self._base.observe_temperature(temperature)

    def reseed(self, **kwargs):
        self._base.reseed(**kwargs)
        self.background_gain = 0.0
        self.history = []
        self.last_update = None
        self.status = "insufficient_history"
        self.gain_identified = False

    def add_interval(self, sample):
        self.observe_temperature(sample.end_temp)
        self.history = [
            r
            for r in self.history
            if sample.end - self.window_hours * 3600 < r.end < sample.end
        ]
        self.history.append(sample)
        self.last_update = sample.end
        self.gain_identified = False
        hours = sum(r.hours for r in self.history)
        if hours < 24 or len(self.history) < 12:
            self.status = "insufficient_history"
            return
        differences = [r.outdoor - r.start_temp for r in self.history]
        if max(differences) - min(differences) < 2:
            self.status = "insufficient_weather_variation"
            return
        heating = sum(r.hours * r.heat for r in self.history)
        heat_values = [r.heat for r in self.history]
        heat_variance = statistics.pvariance(heat_values)
        weather_variance = statistics.pvariance(differences)
        covariance = statistics.mean(
            (d - statistics.mean(differences)) * (h - statistics.mean(heat_values))
            for d, h in zip(differences, heat_values)
        )
        # Heating that simply tracks outdoor temperature cannot identify loss
        # and pump gain separately, even when each input varies considerably.
        independent_heat_variance = heat_variance - covariance**2 / max(
            weather_variance, 1e-9
        )
        self.gain_identified = (
            heating >= 3
            and hours - heating >= 3
            and heat_variance >= 0.02
            and independent_heat_variance >= 0.01
        )
        loss, gain = self.heat_loss_coeff, self.heat_gain_coeff
        bounds = [
            (max(0.001, loss * 0.8**sample.hours), min(0.25, loss * 1.2**sample.hours)),
            (max(0.1, gain - 0.05 * sample.hours), min(1.5, gain + 0.05 * sample.hours))
            if self.gain_identified
            else (gain, gain),
            (
                max(-0.5, self.background_gain - 0.025 * sample.hours),
                min(0.5, self.background_gain + 0.025 * sample.hours),
            ),
        ]
        observations = [
            (
                (r.outdoor - r.start_temp, r.heat, 1.0),
                (r.end_temp - r.start_temp) / r.hours,
            )
            for r in self.history
        ]
        fit = bounded_fit(
            observations, bounds, (loss, gain, self.background_gain), (100.0, 1.0, 1.0)
        )
        if fit is None:
            self.status = "ill_conditioned"
            return
        state = self._base.export_state()
        state["state"] = [sample.end_temp, fit[0], fit[1]]
        self._base.restore(state)
        self.background_gain = fit[2]
        self.fit_error = statistics.mean(
            abs(y - sum(a * b for a, b in zip(x, fit))) for x, y in observations
        )
        self.status = "learning" if self.gain_identified else "gain_frozen"

    def export_state(self):
        return {
            "model_type": "adaptive",
            "base": self._base.export_state(),
            "background_gain": self.background_gain,
            "observations": [asdict(r) for r in self.history],
            "last_update": self.last_update,
        }

    def restore(self, payload):
        if not isinstance(payload, dict):
            return False
        if payload.get("model_type") != "adaptive":
            # Carry forward trusted legacy coefficients when explicitly switching models.
            if payload.get("model_type", "ekf") == "ekf":
                return self._base.restore(payload)
            if payload.get("model_type") == "rls":
                theta = payload.get("theta", [])
                if len(theta) == 2 and all(finite(v) is not None for v in theta):
                    self._base.reseed(
                        seed=0.5,
                        initial_heat_loss=theta[1],
                        initial_heat_gain=theta[0],
                        initial_temp=finite(payload.get("last_temp")),
                    )
                    return True
            return False
        background = finite(payload.get("background_gain"))
        if (
            background is None
            or not isinstance(payload.get("base"), dict)
            or not self._base.restore(payload["base"])
        ):
            return False
        self.background_gain = min(0.5, max(-0.5, background))
        self.history = []
        observations = payload.get("observations", [])
        for item in (observations if isinstance(observations, list) else [])[-336:]:
            try:
                r = LearningInterval(**item)
                if (
                    all(
                        finite(v) is not None
                        for v in (
                            r.end,
                            r.hours,
                            r.start_temp,
                            r.end_temp,
                            r.outdoor,
                            r.heat,
                            r.coverage,
                        )
                    )
                    and 0 < r.hours <= 3
                    and 0 <= r.heat <= 1
                    and r.coverage >= 0.95
                ):
                    self.history.append(r)
            except (TypeError, ValueError):
                continue
        self.last_update = finite(payload.get("last_update"))
        return True
