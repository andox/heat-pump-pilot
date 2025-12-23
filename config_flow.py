"""Config flow for the Heat Pump Pilot integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_COMFORT_TEMPERATURE_TOLERANCE,
    CONF_CONTROL_INTERVAL_MINUTES,
    CONF_CONTROLLED_ENTITY,
    CONF_HEATING_DETECTION_ENABLED,
    CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
    CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN,
    CONF_LEARNING_MODEL,
    CONF_INDOOR_TEMP,
    CONF_MONITOR_ONLY,
    CONF_INITIAL_HEAT_GAIN,
    CONF_INITIAL_HEAT_LOSS_OVERRIDE,
    CONF_INITIAL_INDOOR_TEMP,
    CONF_HEATING_SUPPLY_TEMP_ENTITY,
    CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
    CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
    CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
    CONF_OUTDOOR_TEMP,
    CONF_OVERSHOOT_WARM_BIAS_ENABLED,
    CONF_OVERSHOOT_WARM_BIAS_CURVE,
    CONF_PREDICTION_HORIZON_HOURS,
    CONF_PRICE_COMFORT_WEIGHT,
    CONF_PRICE_ENTITY,
    CONF_TARGET_TEMPERATURE,
    CONF_THERMAL_RESPONSE_SEED,
    CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET,
    CONF_WEATHER_FORECAST_ENTITY,
    CONF_HEAT_LOSS_COEFFICIENT,
    CONF_PERFORMANCE_WINDOW_HOURS,
    CONF_RLS_FORGETTING_FACTOR,
    DEFAULT_MONITOR_ONLY,
    DEFAULT_COMFORT_TEMPERATURE_TOLERANCE,
    DEFAULT_CONTROL_INTERVAL_MINUTES,
    DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
    DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN,
    DEFAULT_LEARNING_MODEL,
    DEFAULT_PERFORMANCE_WINDOW_HOURS,
    DEFAULT_PREDICTION_HORIZON_HOURS,
    DEFAULT_PRICE_COMFORT_WEIGHT,
    DEFAULT_RLS_FORGETTING_FACTOR,
    DEFAULT_TARGET_TEMPERATURE,
    DEFAULT_THERMAL_RESPONSE_SEED,
    DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_HEATING_DETECTION_ENABLED,
    DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS,
    DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
    DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD,
    DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED,
    DEFAULT_OVERSHOOT_WARM_BIAS_CURVE,
    DOMAIN,
    LEARNING_MODEL_EKF,
    LEARNING_MODEL_RLS,
    OVERSHOOT_WARM_BIAS_CURVES,
    PERFORMANCE_WINDOW_OPTIONS,
)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the Heat Pump Pilot."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> config_entries.FlowResult:
        """Handle the initial configuration step."""
        if user_input is not None:
            controlled_entity_raw = user_input.get(CONF_CONTROLLED_ENTITY)
            controlled_entity = controlled_entity_raw.strip() if isinstance(controlled_entity_raw, str) else None
            if not controlled_entity:
                controlled_entity = None
            unique_id = controlled_entity or f"{user_input[CONF_INDOOR_TEMP]}_{user_input[CONF_OUTDOOR_TEMP]}"

            # Allow multiple instances; key off the controlled entity when present, otherwise sensors.
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            options = {
                CONF_TARGET_TEMPERATURE: user_input[CONF_TARGET_TEMPERATURE],
                CONF_PRICE_COMFORT_WEIGHT: user_input[CONF_PRICE_COMFORT_WEIGHT],
                CONF_CONTROL_INTERVAL_MINUTES: DEFAULT_CONTROL_INTERVAL_MINUTES,
                CONF_PREDICTION_HORIZON_HOURS: DEFAULT_PREDICTION_HORIZON_HOURS,
                CONF_COMFORT_TEMPERATURE_TOLERANCE: DEFAULT_COMFORT_TEMPERATURE_TOLERANCE,
                CONF_MONITOR_ONLY: user_input.get(CONF_MONITOR_ONLY, DEFAULT_MONITOR_ONLY),
                CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET: DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET,
                CONF_OVERSHOOT_WARM_BIAS_ENABLED: DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED,
                CONF_OVERSHOOT_WARM_BIAS_CURVE: DEFAULT_OVERSHOOT_WARM_BIAS_CURVE,
                CONF_HEAT_LOSS_COEFFICIENT: DEFAULT_HEAT_LOSS_COEFFICIENT,
                CONF_THERMAL_RESPONSE_SEED: DEFAULT_THERMAL_RESPONSE_SEED,
                CONF_LEARNING_MODEL: user_input.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL),
                CONF_RLS_FORGETTING_FACTOR: user_input.get(
                    CONF_RLS_FORGETTING_FACTOR, DEFAULT_RLS_FORGETTING_FACTOR
                ),
                CONF_PERFORMANCE_WINDOW_HOURS: DEFAULT_PERFORMANCE_WINDOW_HOURS,
                CONF_HEATING_SUPPLY_TEMP_ENTITY: None,
                CONF_HEATING_SUPPLY_TEMP_THRESHOLD: DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD,
                CONF_HEATING_DETECTION_ENABLED: False,
                CONF_HEATING_SUPPLY_TEMP_HYSTERESIS: DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS,
                CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS: DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
                CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN: DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN,
                CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN: DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
                CONF_INITIAL_INDOOR_TEMP: None,
                CONF_INITIAL_HEAT_GAIN: None,
                CONF_INITIAL_HEAT_LOSS_OVERRIDE: None,
            }
            data = {
                CONF_INDOOR_TEMP: user_input[CONF_INDOOR_TEMP],
                CONF_OUTDOOR_TEMP: user_input[CONF_OUTDOOR_TEMP],
                CONF_PRICE_ENTITY: user_input[CONF_PRICE_ENTITY],
                CONF_WEATHER_FORECAST_ENTITY: user_input[CONF_WEATHER_FORECAST_ENTITY],
                CONF_CONTROLLED_ENTITY: controlled_entity,
            }
            return self.async_create_entry(title="Heat Pump Pilot", data=data, options=options)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_INDOOR_TEMP): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(CONF_OUTDOOR_TEMP): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(CONF_PRICE_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(CONF_WEATHER_FORECAST_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["weather", "sensor"])
                ),
                vol.Optional(
                    CONF_CONTROLLED_ENTITY,
                    default=None,
                ): vol.Any(None, str),
                vol.Required(CONF_MONITOR_ONLY, default=False): selector.BooleanSelector(),
                vol.Required(
                    CONF_TARGET_TEMPERATURE, default=DEFAULT_TARGET_TEMPERATURE
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=5,
                        max=28,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_PRICE_COMFORT_WEIGHT, default=DEFAULT_PRICE_COMFORT_WEIGHT
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(
                    CONF_LEARNING_MODEL, default=DEFAULT_LEARNING_MODEL
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[LEARNING_MODEL_EKF, LEARNING_MODEL_RLS],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_RLS_FORGETTING_FACTOR, default=DEFAULT_RLS_FORGETTING_FACTOR
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.90,
                        max=1.00,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="user", data_schema=data_schema)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        """Return the options flow handler."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options for the Heat Pump Pilot."""

    def __init__(self, entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> config_entries.FlowResult:
        """Manage the options."""
        current_data = self._config_entry.data
        if user_input is not None:
            controlled_entity_raw = user_input.get(CONF_CONTROLLED_ENTITY)
            controlled_entity = controlled_entity_raw.strip() if isinstance(controlled_entity_raw, str) else None
            if not controlled_entity:
                controlled_entity = None

            new_data = {
                CONF_INDOOR_TEMP: user_input[CONF_INDOOR_TEMP],
                CONF_OUTDOOR_TEMP: user_input[CONF_OUTDOOR_TEMP],
                CONF_PRICE_ENTITY: user_input[CONF_PRICE_ENTITY],
                CONF_WEATHER_FORECAST_ENTITY: user_input[CONF_WEATHER_FORECAST_ENTITY],
                CONF_CONTROLLED_ENTITY: controlled_entity,
            }
            heating_supply_entity = user_input.get(CONF_HEATING_SUPPLY_TEMP_ENTITY)
            heating_detection_enabled = bool(heating_supply_entity) and bool(
                user_input.get(CONF_HEATING_DETECTION_ENABLED, DEFAULT_HEATING_DETECTION_ENABLED)
            )
            new_options = {
                CONF_TARGET_TEMPERATURE: user_input[CONF_TARGET_TEMPERATURE],
                CONF_PRICE_COMFORT_WEIGHT: user_input[CONF_PRICE_COMFORT_WEIGHT],
                CONF_CONTROL_INTERVAL_MINUTES: user_input[CONF_CONTROL_INTERVAL_MINUTES],
                CONF_PREDICTION_HORIZON_HOURS: user_input[CONF_PREDICTION_HORIZON_HOURS],
                CONF_COMFORT_TEMPERATURE_TOLERANCE: user_input[CONF_COMFORT_TEMPERATURE_TOLERANCE],
                CONF_MONITOR_ONLY: user_input[CONF_MONITOR_ONLY],
                CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET: user_input[CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET],
                CONF_OVERSHOOT_WARM_BIAS_ENABLED: user_input.get(
                    CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED
                ),
                CONF_OVERSHOOT_WARM_BIAS_CURVE: user_input.get(
                    CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE
                ),
                CONF_HEAT_LOSS_COEFFICIENT: user_input[CONF_HEAT_LOSS_COEFFICIENT],
                CONF_THERMAL_RESPONSE_SEED: user_input[CONF_THERMAL_RESPONSE_SEED],
                CONF_LEARNING_MODEL: user_input.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL),
                CONF_RLS_FORGETTING_FACTOR: user_input.get(
                    CONF_RLS_FORGETTING_FACTOR, DEFAULT_RLS_FORGETTING_FACTOR
                ),
                CONF_PERFORMANCE_WINDOW_HOURS: user_input.get(
                    CONF_PERFORMANCE_WINDOW_HOURS, DEFAULT_PERFORMANCE_WINDOW_HOURS
                ),
                CONF_HEATING_SUPPLY_TEMP_ENTITY: heating_supply_entity,
                CONF_HEATING_SUPPLY_TEMP_THRESHOLD: user_input.get(
                    CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD
                ),
                CONF_HEATING_DETECTION_ENABLED: heating_detection_enabled,
                CONF_HEATING_SUPPLY_TEMP_HYSTERESIS: user_input.get(
                    CONF_HEATING_SUPPLY_TEMP_HYSTERESIS, DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS
                ),
                CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS: user_input.get(
                    CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS
                ),
                CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN: user_input.get(
                    CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN
                ),
                CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN: user_input.get(
                    CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN
                ),
                CONF_INITIAL_INDOOR_TEMP: user_input.get(CONF_INITIAL_INDOOR_TEMP),
                CONF_INITIAL_HEAT_GAIN: user_input.get(CONF_INITIAL_HEAT_GAIN),
                CONF_INITIAL_HEAT_LOSS_OVERRIDE: user_input.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE),
            }
            self.hass.config_entries.async_update_entry(self._config_entry, data=new_data)
            return self.async_create_entry(title="", data=new_options)

        return self.async_show_form(step_id="init", data_schema=self._build_options_schema())

    def _build_options_schema(self) -> vol.Schema:
        """Build the options schema (extracted to reuse on validation errors)."""
        current_data = self._config_entry.data
        options = self._config_entry.options
        return vol.Schema(
            {
                vol.Required(
                    CONF_INDOOR_TEMP,
                    default=current_data.get(CONF_INDOOR_TEMP),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(
                    CONF_OUTDOOR_TEMP,
                    default=current_data.get(CONF_OUTDOOR_TEMP),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(
                    CONF_PRICE_ENTITY,
                    default=current_data.get(CONF_PRICE_ENTITY),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor"])
                ),
                vol.Required(
                    CONF_WEATHER_FORECAST_ENTITY,
                    default=current_data.get(CONF_WEATHER_FORECAST_ENTITY),
                ): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["weather", "sensor"])
                ),
                vol.Optional(
                    CONF_CONTROLLED_ENTITY,
                    default=current_data.get(CONF_CONTROLLED_ENTITY),
                ): vol.Any(None, str),
                vol.Required(
                    CONF_TARGET_TEMPERATURE,
                    default=options.get(CONF_TARGET_TEMPERATURE, DEFAULT_TARGET_TEMPERATURE),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=5,
                        max=28,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_PRICE_COMFORT_WEIGHT,
                    default=options.get(CONF_PRICE_COMFORT_WEIGHT, DEFAULT_PRICE_COMFORT_WEIGHT),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(
                    CONF_CONTROL_INTERVAL_MINUTES,
                    default=options.get(CONF_CONTROL_INTERVAL_MINUTES, DEFAULT_CONTROL_INTERVAL_MINUTES),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=5,
                        max=120,
                        step=5,
                        unit_of_measurement="min",
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(
                    CONF_PREDICTION_HORIZON_HOURS,
                    default=options.get(CONF_PREDICTION_HORIZON_HOURS, DEFAULT_PREDICTION_HORIZON_HOURS),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=4,
                        max=48,
                        step=1,
                        unit_of_measurement="h",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_COMFORT_TEMPERATURE_TOLERANCE,
                    default=options.get(
                        CONF_COMFORT_TEMPERATURE_TOLERANCE, DEFAULT_COMFORT_TEMPERATURE_TOLERANCE
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.2,
                        max=3.0,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_PERFORMANCE_WINDOW_HOURS,
                    default=str(options.get(CONF_PERFORMANCE_WINDOW_HOURS, DEFAULT_PERFORMANCE_WINDOW_HOURS)),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[str(value) for value in PERFORMANCE_WINDOW_OPTIONS],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_MONITOR_ONLY,
                    default=options.get(CONF_MONITOR_ONLY, DEFAULT_MONITOR_ONLY),
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET,
                    default=options.get(
                        CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET, DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=15,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_OVERSHOOT_WARM_BIAS_ENABLED,
                    default=options.get(CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED),
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_OVERSHOOT_WARM_BIAS_CURVE,
                    default=options.get(CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=list(OVERSHOOT_WARM_BIAS_CURVES),
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_HEAT_LOSS_COEFFICIENT,
                    default=options.get(CONF_HEAT_LOSS_COEFFICIENT, DEFAULT_HEAT_LOSS_COEFFICIENT),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.005,
                        max=0.1,
                        step=0.005,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_THERMAL_RESPONSE_SEED,
                    default=options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(
                    CONF_LEARNING_MODEL,
                    default=options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[LEARNING_MODEL_EKF, LEARNING_MODEL_RLS],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_RLS_FORGETTING_FACTOR,
                    default=options.get(CONF_RLS_FORGETTING_FACTOR, DEFAULT_RLS_FORGETTING_FACTOR),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.90,
                        max=1.00,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_INITIAL_INDOOR_TEMP,
                    default=options.get(CONF_INITIAL_INDOOR_TEMP),
                ): vol.Any(None, vol.Coerce(float)),
                vol.Optional(
                    CONF_HEATING_SUPPLY_TEMP_ENTITY,
                    default=options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY),
                ): selector.EntitySelector(selector.EntitySelectorConfig(domain=["sensor"])),
                vol.Required(
                    CONF_HEATING_DETECTION_ENABLED,
                    default=options.get(
                        CONF_HEATING_DETECTION_ENABLED, bool(options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY))
                    ),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
                    default=options.get(CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=10,
                        max=80,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
                    default=options.get(CONF_HEATING_SUPPLY_TEMP_HYSTERESIS, DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=10.0,
                        step=0.1,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
                    default=options.get(
                        CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=600,
                        step=5,
                        unit_of_measurement="s",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN,
                    default=options.get(CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=10.0,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
                    default=options.get(CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=10.0,
                        step=0.5,
                        unit_of_measurement="°C",
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_INITIAL_HEAT_LOSS_OVERRIDE,
                    default=options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE),
                ): vol.Any(None, vol.Coerce(float)),
                vol.Optional(
                    CONF_INITIAL_HEAT_GAIN,
                    default=options.get(CONF_INITIAL_HEAT_GAIN),
                ): vol.Any(None, vol.Coerce(float)),
            }
        )
