"""Form grouping and backwards-compatible flat option storage."""

SECTIONS = {
    "connections": "Sensors and heat pump",
    "advanced_control": "Advanced: control and virtual outdoor temperature",
    "advanced_prices": "Advanced: electricity prices",
    "advanced_ufh": "Advanced: UFH circulation pumps",
    "advanced_ufh_exercise": "Advanced: UFH pump exercise",
    "advanced_summer": "Advanced: summer low-price heat window",
    "advanced_learning": "Advanced: learning",
    "advanced_detection": "Advanced: heating detection",
    "advanced_overshoot": "Advanced: temperature overshoot",
    "advanced_seeds": "Advanced: initial model estimates",
    "advanced_diagnostics": "Advanced: diagnostics",
}


def group_for(key):
    if key.startswith("ufh_exercise_"):
        return "advanced_ufh_exercise"
    if key.startswith("ufh_"):
        return "advanced_ufh"
    if key in (
        "target_temperature",
        "price_comfort_weight",
        "comfort_temperature_tolerance",
        "monitor_only",
    ):
        return None
    if key.endswith("_entity") or key in (
        "heating_detection_enabled",
        "heating_supply_temp_threshold",
    ):
        return "connections"
    if key in (
        "learning_window_hours",
        "performance_window_hours",
        "virtual_outdoor_trace_enabled",
    ):
        return "advanced_diagnostics"
    if key.startswith("summer_heat_window_"):
        return "advanced_summer"
    if key.startswith("price_"):
        return "advanced_prices"
    if key.startswith("overshoot_"):
        return "advanced_overshoot"
    if key.startswith("initial_") or key in (
        "heat_loss_coefficient",
        "thermal_response_seed",
    ):
        return "advanced_seeds"
    if key.startswith("heating_supply_") or key.startswith("learning_supply_"):
        return "advanced_detection"
    if key.startswith(("learning_", "rls_", "pump_response_")):
        return "advanced_learning"
    return "advanced_control"


def flatten_input(values, existing=None):
    """Keep saved advanced values when an entire section is omitted."""
    result = dict(existing or {})
    for key, value in values.items():
        if key in SECTIONS:
            if isinstance(value, dict):
                result.update(value)
                # Omitted optional fields in a submitted section mean cleared.
                for optional in (
                    "ufh_supply_entity",
                    "controlled_entity",
                    "heating_supply_temp_entity",
                    "initial_indoor_temp",
                    "initial_heat_loss_override",
                    "initial_heat_gain_coefficient",
                ):
                    if group_for(optional) == key and optional not in value:
                        result[optional] = None
        else:
            result[key] = value
    return result
