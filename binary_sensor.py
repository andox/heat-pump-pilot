"""Binary sensors for mpc_heat_pump."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory

from .const import DOMAIN, SIGNAL_DECISION_UPDATED


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up binary sensors for a config entry."""
    async_add_entities([MpcHeatPumpHeatingDetectedBinarySensor(hass, entry)])


class MpcHeatPumpHeatingDetectedBinarySensor(BinarySensorEntity):
    """Expose whether heating is currently detected."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.config_entry = entry
        self._decision: dict[str, Any] | None = None
        self._unsub = None
        self._attr_name = "MPC Heat Pump Heating Detected"
        self._attr_unique_id = f"{entry.entry_id}_heating_detected"

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

    @property
    def is_on(self) -> bool | None:
        if not self._decision:
            return None
        return self._decision.get("heating_detected")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self._decision:
            return {}
        return {
            "source": self._decision.get("heating_detected_source"),
            "heating_detection_enabled": self._decision.get("heating_detection_enabled"),
            "heating_supply_temp_entity": self._decision.get("heating_supply_temp_entity"),
            "heating_supply_temp_threshold": self._decision.get("heating_supply_temp_threshold"),
            "heating_supply_temp_hysteresis": self._decision.get("heating_supply_temp_hysteresis"),
            "heating_supply_temp_debounce_seconds": self._decision.get("heating_supply_temp_debounce_seconds"),
            "heating_supply_temperature": self._decision.get("heating_supply_temperature"),
        }

    @callback
    def _handle_update(self) -> None:
        self._decision = self._get_entry_state()
        self.async_write_ha_state()

    def _get_entry_state(self) -> dict[str, Any] | None:
        return self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id, {}).get("last_decision")
