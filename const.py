"""Constants for the MPC Heat Pump Controller integration."""

from __future__ import annotations

DOMAIN = "mpc_heat_pump"

CONF_INDOOR_TEMP = "indoor_temp_entity"
CONF_OUTDOOR_TEMP = "outdoor_temp_entity"
CONF_PRICE_ENTITY = "price_entity"
CONF_WEATHER_FORECAST_ENTITY = "weather_forecast_entity"
CONF_CONTROLLED_ENTITY = "controlled_entity"

CONF_TARGET_TEMPERATURE = "target_temperature"
CONF_PRICE_COMFORT_WEIGHT = "price_comfort_weight"
CONF_CONTROL_INTERVAL_MINUTES = "control_interval_minutes"
CONF_PREDICTION_HORIZON_HOURS = "prediction_horizon_hours"
CONF_COMFORT_TEMPERATURE_TOLERANCE = "comfort_temperature_tolerance"
CONF_MONITOR_ONLY = "monitor_only"
CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET = "virtual_outdoor_heat_offset"
CONF_HEAT_LOSS_COEFFICIENT = "heat_loss_coefficient"
CONF_THERMAL_RESPONSE_SEED = "thermal_response_seed"
CONF_INITIAL_INDOOR_TEMP = "initial_indoor_temp"
CONF_INITIAL_HEAT_GAIN = "initial_heat_gain_coefficient"
CONF_INITIAL_HEAT_LOSS_OVERRIDE = "initial_heat_loss_override"
CONF_HEATING_SUPPLY_TEMP_ENTITY = "heating_supply_temp_entity"
CONF_HEATING_SUPPLY_TEMP_THRESHOLD = "heating_supply_temp_threshold"
CONF_HEATING_DETECTION_ENABLED = "heating_detection_enabled"
CONF_HEATING_SUPPLY_TEMP_HYSTERESIS = "heating_supply_temp_hysteresis"
CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS = "heating_supply_temp_debounce_seconds"

DEFAULT_TARGET_TEMPERATURE = 21.0
DEFAULT_PRICE_COMFORT_WEIGHT = 0.5
DEFAULT_CONTROL_INTERVAL_MINUTES = 15
DEFAULT_PREDICTION_HORIZON_HOURS = 24
DEFAULT_COMFORT_TEMPERATURE_TOLERANCE = 1.0
DEFAULT_MONITOR_ONLY = False
DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET = 5.0
DEFAULT_HEAT_LOSS_COEFFICIENT = 0.05
DEFAULT_THERMAL_RESPONSE_SEED = 0.5
DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD = 30.0
DEFAULT_HEATING_DETECTION_ENABLED = True
DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS = 1.0
DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS = 60

SIGNAL_OPTIONS_UPDATED = "mpc_heat_pump_options_updated"
SIGNAL_DECISION_UPDATED = "mpc_heat_pump_decision_updated"
