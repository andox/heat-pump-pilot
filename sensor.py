"""Sensors exposing the MPC decision and health/status summaries."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory

from .const import DOMAIN, SIGNAL_DECISION_UPDATED


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up sensors for a config entry."""
    async_add_entities(
        [
            MpcHeatPumpDecisionSensor(hass, entry),
            MpcHeatPumpHealthSensor(hass, entry),
            MpcHeatPumpControlStateSensor(hass, entry),
            MpcHeatPumpLearningStateSensor(hass, entry),
            MpcHeatPumpPriceStateSensor(hass, entry),
            MpcHeatPumpVirtualOutdoorSensor(hass, entry),
        ]
    )


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


class MpcHeatPumpDecisionSensor(_MpcHeatPumpBaseSensor):
    """Expose the latest MPC decision as a diagnostic sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Decision"
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

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Health"
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
        }


class MpcHeatPumpControlStateSensor(_MpcHeatPumpBaseSensor):
    """Expose whether the integration is monitoring or controlling."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Control State"
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

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Learning State"
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

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Price State"
        self._attr_unique_id = f"{entry.entry_id}_price_state"

    @property
    def native_value(self) -> str | None:
        if not self._decision:
            return None
        return self._decision.get("price_classification_history_24h") or self._decision.get("price_classification")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return {
            "current_price": self._decision.get("current_price"),
            "price_baseline": self._decision.get("price_baseline"),
            "price_ratio": self._decision.get("price_ratio"),
            "price_baseline_kind": self._decision.get("price_baseline_kind"),
            "price_baseline_history_24h": self._decision.get("price_baseline_history_24h"),
            "price_baseline_history_24h_samples": self._decision.get("price_baseline_history_24h_samples"),
            "price_ratio_history_24h": self._decision.get("price_ratio_history_24h"),
            "price_classification_history_24h": self._decision.get("price_classification_history_24h"),
        }


class MpcHeatPumpVirtualOutdoorSensor(_MpcHeatPumpBaseSensor):
    """Expose the recommended virtual outdoor temperature as a numeric sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_force_update = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_name = "MPC Heat Pump Virtual Outdoor"
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
