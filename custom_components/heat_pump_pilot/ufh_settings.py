"""Options for temperature-driven circulation, independent of MPC settings."""
import math

UFH_DEFAULTS = {
    'ufh_enabled': False,
    'ufh_supply_entity': None,
    'ufh_switches': [],
    'ufh_on_temperature': 30.0,
    'ufh_off_temperature': 25.0,
    'ufh_on_hold_seconds': 30,
    'ufh_min_on_minutes': 60,
    'ufh_min_off_minutes': 5,
    'ufh_overrun_minutes': 30,
    'ufh_stale_minutes': 360,
    'ufh_exercise_enabled': False,
    'ufh_exercise_time': '13:00:00',
    'ufh_exercise_minutes': 15,
    'ufh_exercise_idle_hours': 24,
}
UFH_NUMBERS = {
    'ufh_on_temperature': (5, 80, 0.5, '°C'),
    'ufh_off_temperature': (0, 75, 0.5, '°C'),
    'ufh_on_hold_seconds': (0, 600, 1, 's'),
    'ufh_min_on_minutes': (0, 240, 1, 'min'),
    'ufh_min_off_minutes': (0, 120, 1, 'min'),
    'ufh_overrun_minutes': (0, 120, 1, 'min'),
    'ufh_stale_minutes': (1, 1440, 1, 'min'),
    'ufh_exercise_minutes': (1, 60, 1, 'min'),
    'ufh_exercise_idle_hours': (0, 168, 1, 'h'),
}


def ufh_options(options):
    """Retain invalid values for validation; never silently enable control."""
    return {k: options.get(k, list(v) if isinstance(v, list) else v) for k, v in UFH_DEFAULTS.items()}


def validate_ufh(options):
    c = ufh_options(options)
    if not isinstance(c['ufh_enabled'], bool) or not isinstance(c['ufh_exercise_enabled'], bool):
        return 'ufh_invalid_settings'
    for key, (low, high, _, _) in UFH_NUMBERS.items():
        try:
            value = float(c[key])
        except (TypeError, ValueError):
            return 'ufh_invalid_settings'
        if not math.isfinite(value) or not low <= value <= high:
            return 'ufh_invalid_settings'
    if float(c['ufh_off_temperature']) >= float(c['ufh_on_temperature']):
        return 'ufh_invalid_thresholds'
    try:
        parts = [int(p) for p in c['ufh_exercise_time'].split(':')]
        if len(parts) not in (2, 3) or not 0 <= parts[0] < 24 or not 0 <= parts[1] < 60 or (len(parts) == 3 and not 0 <= parts[2] < 60):
            return 'ufh_invalid_settings'
    except (ValueError, AttributeError, TypeError):
        return 'ufh_invalid_settings'
    switches = c['ufh_switches']
    if not isinstance(switches, list) or any(not isinstance(s, str) or not s.startswith('switch.') for s in switches):
        return 'ufh_invalid_settings'
    if c['ufh_enabled'] and (not switches or not isinstance(c['ufh_supply_entity'], str) or not c['ufh_supply_entity'].startswith('sensor.')):
        return 'ufh_missing_entities'
    return None
