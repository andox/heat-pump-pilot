"""Setup for the MPC Heat Pump Controller integration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

try:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.const import Platform
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.dispatcher import async_dispatcher_send
except ModuleNotFoundError:  # pragma: no cover - allows unit tests without HA installed
    class _DummyConfigEntry(Protocol):
        """Minimal Protocol to appease type-checkers when HA isn't available."""

        entry_id: str
        data: dict[str, Any]
        options: dict[str, Any]

        async def add_update_listener(self, *_args: Any, **_kwargs: Any):
            ...

    class ConfigEntry(_DummyConfigEntry):  # type: ignore[misc]
        """Fallback for tests; real class provided by Home Assistant."""

        entry_id = "test"
        data = {}
        options = {}

        async def add_update_listener(self, *_args: Any, **_kwargs: Any):
            """Return a dummy unsubscribe callable."""
            return lambda: None

    class HomeAssistant:  # type: ignore[misc]
        """Fallback for tests; real class provided by Home Assistant."""

        def __init__(self) -> None:
            self.data: dict[str, Any] = {}

        class config_entries:
            """Namespace shim for tests."""

            @staticmethod
            async def async_forward_entry_setups(entry, platforms):
                return None

            @staticmethod
            async def async_unload_platforms(entry, platforms):
                return True

    class Platform(str, Enum):  # type: ignore[misc]
        """Minimal enum fallback mirroring the HA constant."""

        CLIMATE = "climate"
        SENSOR = "sensor"
        BINARY_SENSOR = "binary_sensor"

    def async_dispatcher_send(*_args: Any, **_kwargs: Any) -> None:
        """Fallback dispatcher shim for running tests outside HA."""
        return None

from .const import DOMAIN, SIGNAL_OPTIONS_UPDATED

PLATFORMS: list[Platform] = [Platform.CLIMATE, Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up MPC Heat Pump Controller from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {}

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    update_unsub = entry.add_update_listener(async_update_listener)
    hass.data[DOMAIN][entry.entry_id]["update_unsub"] = update_unsub
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok and DOMAIN in hass.data:
        update_unsub = hass.data[DOMAIN].get(entry.entry_id, {}).pop("update_unsub", None)
        if update_unsub:
            update_unsub()
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok


async def async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update by notifying entities."""
    async_dispatcher_send(hass, f"{SIGNAL_OPTIONS_UPDATED}_{entry.entry_id}")
