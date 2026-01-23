"""Sensors exposing the MPC decision and health/status summaries."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory

from .const import (
    CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED,
    DEFAULT_VIRTUAL_OUTDOOR_TRACE_ENABLED,
    DOMAIN,
    SIGNAL_DECISION_UPDATED,
    VIRTUAL_OUTDOOR_TRACE_ATTRIBUTE_MAX_ENTRIES,
    VIRTUAL_OUTDOOR_TRACE_MAX_ENTRIES,
)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up sensors for a config entry."""
    entities: list[SensorEntity] = [
        MpcHeatPumpDecisionSensor(hass, entry),
        MpcHeatPumpHealthSensor(hass, entry),
        MpcHeatPumpControlStateSensor(hass, entry),
        MpcHeatPumpLearningStateSensor(hass, entry),
        MpcHeatPumpPriceStateSensor(hass, entry),
        MpcHeatPumpVirtualOutdoorSensor(hass, entry),
        MpcHeatPumpComfortScoreSensor(hass, entry),
        MpcHeatPumpPriceScoreSensor(hass, entry),
        MpcHeatPumpPredictionAccuracySensor(hass, entry),
    ]
    if bool(
        entry.options.get(CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED, DEFAULT_VIRTUAL_OUTDOOR_TRACE_ENABLED)
    ):
        entities.append(MpcHeatPumpVirtualOutdoorTraceSensor(hass, entry))
    async_add_entities(entities)


class _MpcHeatPumpBaseSensor(SensorEntity):
    """Common behaviour for sensors driven by the last decision payload."""

    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.config_entry = entry
        self._decision: dict[str, Any] | None = None
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        """Register for updates."""
        self._decision = self._get_entry_state()
        self._unsub = async_dispatcher_connect(
            self.hass,
            f"{SIGNAL_DECISION_UPDATED}_{self.config_entry.entry_id}",
            self._handle_update,
        )
        if self._unsub:
            self.async_on_remove(self._unsub)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up listeners."""
        if self._unsub:
            self._unsub()

    @callback
    def _handle_update(self) -> None:
        """Receive decision updates from the climate entity."""
        self._decision = self._get_entry_state()
        self.async_write_ha_state()

    def _get_entry_state(self) -> dict[str, Any] | None:
        """Fetch the last decision stored for this entry."""
        return self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id, {}).get("last_decision")


class _MpcHeatPumpPerformanceSensor(SensorEntity):
    """Common behaviour for sensors driven by performance summaries."""

    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.config_entry = entry
        self._performance: dict[str, Any] | None = None
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        """Register for updates."""
        self._performance = self._get_entry_state()
        self._unsub = async_dispatcher_connect(
            self.hass,
            f"{SIGNAL_DECISION_UPDATED}_{self.config_entry.entry_id}",
            self._handle_update,
        )
        if self._unsub:
            self.async_on_remove(self._unsub)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up listeners."""
        if self._unsub:
            self._unsub()

    @callback
    def _handle_update(self) -> None:
        """Receive performance updates from the climate entity."""
        self._performance = self._get_entry_state()
        self.async_write_ha_state()

    def _get_entry_state(self) -> dict[str, Any] | None:
        """Fetch the last performance summary stored for this entry."""
        return self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id, {}).get("performance")


class MpcHeatPumpDecisionSensor(_MpcHeatPumpBaseSensor):
    """Expose the latest MPC decision as a diagnostic sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:brain"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Decision"
        self._attr_unique_id = f"{entry.entry_id}_decision"

    @property
    def native_value(self) -> str | None:
        """Return the suggested action as sensor state."""
        if not self._decision:
            return None
        return "heat_on" if self._decision.get("suggested_heat_on") else "idle"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return decision details for diagnostics."""
        return self._decision or {}


class MpcHeatPumpHealthSensor(_MpcHeatPumpBaseSensor):
    """High-level system health for the controller."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:heart-pulse"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Health"
        self._attr_unique_id = f"{entry.entry_id}_health"

    @property
    def native_value(self) -> str | None:
        if not self._decision:
            return None
        return self._decision.get("health")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return {
            "reasons": self._decision.get("health_reasons"),
            "last_control_time": self._decision.get("last_control_time"),
            "control_state": self._decision.get("control_state"),
            "learning_state": self._decision.get("learning_state"),
            "curve_recommendation": self._decision.get("curve_recommendation"),
            "curve_recommendation_details": self._decision.get("curve_recommendation_details"),
        }


class MpcHeatPumpControlStateSensor(_MpcHeatPumpBaseSensor):
    """Expose whether the integration is monitoring or controlling."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:toggle-switch"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Control State"
        self._attr_unique_id = f"{entry.entry_id}_control_state"

    @property
    def native_value(self) -> str | None:
        if not self._decision:
            return None
        return self._decision.get("control_state")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return {
            "hvac_mode": self._decision.get("hvac_mode"),
            "hvac_action": self._decision.get("hvac_action"),
            "monitor_only": self._decision.get("monitor_only"),
            "heating_detected": self._decision.get("heating_detected"),
            "heating_detected_source": self._decision.get("heating_detected_source"),
        }


class MpcHeatPumpLearningStateSensor(_MpcHeatPumpBaseSensor):
    """Expose whether the estimator is learning or stable."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:school"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Learning State"
        self._attr_unique_id = f"{entry.entry_id}_learning_state"

    @property
    def native_value(self) -> str | None:
        if not self._decision:
            return None
        return self._decision.get("learning_state")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return self._decision.get("learning_details") or {}


class MpcHeatPumpPriceStateSensor(_MpcHeatPumpBaseSensor):
    """Expose the current electricity price classification."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:currency-usd"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Price State"
        self._attr_unique_id = f"{entry.entry_id}_price_state"

    @property
    def native_value(self) -> str | None:
        if not self._decision:
            return None
        return self._decision.get("price_classification")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return {
            "current_price": self._decision.get("current_price"),
            "price_baseline": self._decision.get("price_baseline"),
            "price_ratio": self._decision.get("price_ratio"),
            "price_baseline_kind": self._decision.get("price_baseline_kind"),
            "price_baseline_window_hours": self._decision.get("price_baseline_window_hours"),
            "price_baseline_history_samples": self._decision.get("price_baseline_history_samples"),
            "price_baseline_forecast_samples": self._decision.get("price_baseline_forecast_samples"),
        }


class MpcHeatPumpVirtualOutdoorSensor(_MpcHeatPumpBaseSensor):
    """Expose the recommended virtual outdoor temperature as a numeric sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_force_update = True
    _attr_icon = "mdi:thermometer"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Virtual Outdoor"
        self._attr_unique_id = f"{entry.entry_id}_virtual_outdoor"
        self._attr_native_unit_of_measurement = hass.config.units.temperature_unit

    @property
    def native_value(self) -> float | None:
        if not self._decision:
            return None
        value = self._decision.get("suggested_virtual_outdoor_temperature")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class MpcHeatPumpVirtualOutdoorTraceSensor(SensorEntity):
    """Expose recent virtual outdoor decisions for diagnostics."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_should_poll = False
    _attr_icon = "mdi:chart-timeline-variant"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.config_entry = entry
        self._trace: list[dict[str, Any]] = []
        self._unsub = None
        self._attr_name = "Heat Pump Pilot Virtual Outdoor Trace"
        self._attr_unique_id = f"{entry.entry_id}_virtual_outdoor_trace"

    async def async_added_to_hass(self) -> None:
        """Register for updates."""
        self._trace = self._get_entry_state()
        self._unsub = async_dispatcher_connect(
            self.hass,
            f"{SIGNAL_DECISION_UPDATED}_{self.config_entry.entry_id}",
            self._handle_update,
        )
        if self._unsub:
            self.async_on_remove(self._unsub)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up listeners."""
        if self._unsub:
            self._unsub()

    @callback
    def _handle_update(self) -> None:
        """Receive trace updates from the climate entity."""
        self._trace = self._get_entry_state()
        self.async_write_ha_state()

    def _get_entry_state(self) -> list[dict[str, Any]]:
        """Fetch the trace stored for this entry."""
        return (
            self.hass.data.get(DOMAIN, {})
            .get(self.config_entry.entry_id, {})
            .get("virtual_outdoor_trace", [])
        )

    @property
    def native_value(self) -> int | None:
        if self._trace is None:
            return None
        return len(self._trace)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        trace = self._trace or []
        if len(trace) > VIRTUAL_OUTDOOR_TRACE_ATTRIBUTE_MAX_ENTRIES:
            trace_view = trace[-VIRTUAL_OUTDOOR_TRACE_ATTRIBUTE_MAX_ENTRIES :]
            truncated = True
        else:
            trace_view = trace
            truncated = False
        return {
            "samples": len(trace),
            "max_samples": VIRTUAL_OUTDOOR_TRACE_MAX_ENTRIES,
            "trace": trace_view,
            "trace_truncated": truncated,
        }


class MpcHeatPumpComfortScoreSensor(_MpcHeatPumpPerformanceSensor):
    """Expose a comfort score based on time within tolerance."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:thermometer-check"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Comfort Score"
        self._attr_unique_id = f"{entry.entry_id}_comfort_score"
        self._attr_native_unit_of_measurement = "%"

    @property
    def native_value(self) -> float | None:
        if not self._performance:
            return None
        value = self._performance.get("comfort_score")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._performance:
            return {}
        return self._performance.get("comfort_details") or {}


class MpcHeatPumpPriceScoreSensor(_MpcHeatPumpPerformanceSensor):
    """Expose a price score based on heating during lower prices."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:cash-100"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Price Score"
        self._attr_unique_id = f"{entry.entry_id}_price_score"
        self._attr_native_unit_of_measurement = "%"

    @property
    def native_value(self) -> float | None:
        if not self._performance:
            return None
        value = self._performance.get("price_score")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._performance:
            return {}
        return self._performance.get("price_details") or {}


class MpcHeatPumpPredictionAccuracySensor(_MpcHeatPumpPerformanceSensor):
    """Expose prediction accuracy (MAE) for indoor temperature."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:target"
    _attr_suggested_display_precision = 2

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "Heat Pump Pilot Prediction Accuracy"
        self._attr_unique_id = f"{entry.entry_id}_prediction_accuracy"
        self._attr_native_unit_of_measurement = hass.config.units.temperature_unit

    @property
    def native_value(self) -> float | None:
        if not self._performance:
            return None
        value = self._performance.get("prediction_mae")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._performance:
            return {}
        return self._performance.get("prediction_details") or {}
