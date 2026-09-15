"""Deterministic circulation state machine; has no MPC or price inputs."""
from dataclasses import asdict, dataclass
import math

try:
    from .ufh_settings import ufh_options, validate_ufh
except ImportError:
    from ufh_settings import ufh_options, validate_ufh


@dataclass
class PumpState:
    state: str | None = None
    since: float | None = None
    exercise_until: float | None = None
    exercise_day: str | None = None


class UfhController:
    def __init__(self, options):
        self.options = ufh_options(options)
        self.pumps = {}
        self.hot_since = self.cold_since = None
        self.last_tick = None
        self.status = 'disabled'
        self.reasons = {}
        self.unavailable = set()

    def configure(self, options):
        new = ufh_options(options)
        if new != self.options:
            self.hot_since = self.cold_since = None
        if not new["ufh_exercise_enabled"]:
            for pump in self.pumps.values():
                pump.exercise_until = None
        self.options = new
        switches = new['ufh_switches'] if isinstance(new['ufh_switches'], list) else []
        self.pumps = {k: v for k, v in self.pumps.items() if k in switches}

    def evaluate(self, now, local_time, temperature, switch_states, *, sensor_fresh=True, monitor_only=False):
        c = self.options
        self.reasons = {}
        if error := validate_ufh(c):
            self.hot_since = self.cold_since = None
            self.status = error
            return {}
        # Never count downtime as continuously observed hot/cold water.
        if self.last_tick is not None and (now < self.last_tick or now - self.last_tick > 90):
            self.hot_since = self.cold_since = None
        self.last_tick = now
        for entity in c['ufh_switches']:
            p = self.pumps.setdefault(entity, PumpState())
            state = switch_states.get(entity)
            if state not in ('on', 'off'):
                self.unavailable.add(entity)
            if state in ('on', 'off') and (p.state != state or entity in self.unavailable):
                self.unavailable.discard(entity)
                p.state, p.since = state, now
                if state == 'off':
                    p.exercise_until = None
        error = validate_ufh(c)
        if error or not c['ufh_enabled'] or monitor_only:
            self.hot_since = self.cold_since = None
            for p in self.pumps.values():
                p.exercise_until = None
            self.status = error or ('monitor_only' if monitor_only else 'disabled')
            return {}
        valid = isinstance(temperature, (int, float)) and math.isfinite(temperature) and sensor_fresh
        if not valid:
            self.hot_since = self.cold_since = None
            self.status = 'sensor_unavailable_or_stale'
            self.reasons = {entity: 'hold_state_sensor_fault' for entity in c['ufh_switches']}
            return {}
        hot = temperature >= float(c['ufh_on_temperature'])
        cold = temperature <= float(c['ufh_off_temperature'])
        self.hot_since = (now if self.hot_since is None else self.hot_since) if hot else None
        self.cold_since = (now if self.cold_since is None else self.cold_since) if cold else None
        hot_ready = hot and now - self.hot_since >= float(c['ufh_on_hold_seconds'])
        cold_ready = cold and now - self.cold_since >= float(c['ufh_overrun_minutes']) * 60
        today = local_time.date().isoformat()
        exercise_minute = local_time.strftime('%H:%M') == c['ufh_exercise_time'][:5]
        commands = {}
        self.status = 'temperature_control'
        for entity in c['ufh_switches']:
            p = self.pumps[entity]
            state = switch_states.get(entity)
            if state not in ('on', 'off'):
                self.reasons[entity] = 'switch_unavailable'
                continue
            elapsed = max(0, now - (p.since if p.since is not None else now))
            if state == 'off':
                off_ready = elapsed >= float(c['ufh_min_off_minutes']) * 60
                if hot_ready and off_ready:
                    commands[entity] = True
                    self.reasons[entity] = 'warm_supply'
                elif p.exercise_until is not None and now < p.exercise_until and off_ready:
                    commands[entity] = True
                    self.reasons[entity] = 'exercise'
                elif c['ufh_exercise_enabled'] and exercise_minute and p.exercise_day != today and off_ready and elapsed >= float(c['ufh_exercise_idle_hours']) * 3600:
                    p.exercise_until = now + float(c['ufh_exercise_minutes']) * 60
                    p.exercise_day = today
                    commands[entity] = True
                    self.reasons[entity] = 'exercise'
                else:
                    self.reasons[entity] = 'minimum_off' if hot and not off_ready else 'waiting_for_warm_supply'
            else:
                # Hot water takes ownership from exercise; no blind timed shutdown.
                if p.exercise_until is not None and hot_ready:
                    p.exercise_until = None
                if p.exercise_until is not None:
                    if now < p.exercise_until:
                        self.reasons[entity] = 'exercise'
                        continue
                    if not cold:
                        p.exercise_until = None
                    elif cold:
                        commands[entity] = False
                        self.reasons[entity] = 'exercise_complete'
                        continue
                if cold_ready and elapsed >= float(c['ufh_min_on_minutes']) * 60:
                    commands[entity] = False
                    self.reasons[entity] = 'cold_supply'
                else:
                    self.reasons[entity] = 'minimum_on' if cold and elapsed < float(c['ufh_min_on_minutes']) * 60 else 'circulating_heat'
        return commands

    def export_state(self):
        return {'version': 1, 'pumps': {k: asdict(v) for k, v in self.pumps.items()}}

    def restore(self, payload, now):
        if not isinstance(payload, dict) or payload.get('version') != 1:
            return
        rows = payload.get('pumps', {})
        if not isinstance(rows, dict):
            return
        for entity, row in rows.items():
            if entity not in self.options['ufh_switches'] or not isinstance(row, dict):
                continue
            try:
                p = PumpState(**row)
                if p.state not in ('on', 'off') or not isinstance(p.since, (float, int)) or not math.isfinite(p.since) or not 0 <= p.since <= now:
                    continue
                if p.exercise_until is not None and (not isinstance(p.exercise_until, (int, float)) or not math.isfinite(p.exercise_until) or p.exercise_until > now + 3600):
                    p.exercise_until = None
                if p.exercise_day is not None and not isinstance(p.exercise_day, str):
                    p.exercise_day = None
                self.pumps[entity] = p
            except (TypeError, ValueError):
                continue
