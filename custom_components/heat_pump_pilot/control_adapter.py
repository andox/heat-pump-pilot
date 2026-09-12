"""Apply control decisions to target entities."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import logging
from typing import Callable

try:
    from homeassistant.components.climate import HVACMode
    from homeassistant.const import ATTR_TEMPERATURE
except ModuleNotFoundError:  # pragma: no cover - local tests without HA
    class HVACMode(str, Enum):
        HEAT = "heat"
        OFF = "off"

    ATTR_TEMPERATURE = "temperature"

_LOGGER = logging.getLogger(__name__)


class ControlAdapter:
    """Encapsulate entity actuation details for the climate controller."""

    def __init__(
        self,
        hass,
        *,
        get_state_as_float: Callable[[str], float | None],
    ) -> None:
        self.hass = hass
        self._get_state_as_float = get_state_as_float

    async def apply(
        self,
        *,
        monitor_only: bool,
        hvac_mode: HVACMode,
        controlled_entity: str | None,
        heat_on: bool,
        target_temperature: float,
        virtual_outdoor: float | None,
    ) -> None:
        """Apply a control decision to the configured entity."""
        if monitor_only:
            return

        if hvac_mode == HVACMode.OFF:
            heat_on = False

        entity_id = controlled_entity
        if entity_id is None:
            return

        domain = entity_id.split(".")[0]

        if domain == "number":
            if virtual_outdoor is None:
                return
            desired = float(virtual_outdoor)

            state_obj = self.hass.states.get(entity_id)
            step = None
            if state_obj is not None:
                min_attr = state_obj.attributes.get("min")
                max_attr = state_obj.attributes.get("max")
                step_attr = state_obj.attributes.get("step")
                try:
                    minimum = float(min_attr) if min_attr is not None else None
                except (TypeError, ValueError):
                    minimum = None
                try:
                    maximum = float(max_attr) if max_attr is not None else None
                except (TypeError, ValueError):
                    maximum = None
                if minimum is not None:
                    desired = max(minimum, desired)
                if maximum is not None:
                    desired = min(maximum, desired)
                try:
                    step = float(step_attr) if step_attr is not None else None
                except (TypeError, ValueError):
                    step = None

            desired_float = desired
            if step is not None and step > 0:
                step_dec = Decimal(str(step))
                desired_dec = Decimal(str(desired_float))
                rounded_dec = (desired_dec / step_dec).to_integral_value(rounding=ROUND_HALF_UP) * step_dec
                decimals = max(0, -step_dec.as_tuple().exponent)
                desired_float = float(rounded_dec)
                desired_value = f"{desired_float:.{decimals}f}"
            else:
                desired_value = str(desired_float)

            current_value = self._get_state_as_float(entity_id)
            if current_value is not None and abs(current_value - desired_float) < 0.01:
                return
            await self.hass.services.async_call(
                "number",
                "set_value",
                {"entity_id": entity_id, "value": desired_value},
                blocking=False,
            )
            return

        if domain == "switch":
            current_on = None
            state_obj = self.hass.states.get(entity_id)
            if state_obj is not None:
                current_on = state_obj.state.lower() == "on"
            if current_on is not None and current_on == heat_on:
                return

            service = "turn_on" if heat_on else "turn_off"
            await self.hass.services.async_call("switch", service, {"entity_id": entity_id}, blocking=False)
            return

        if domain == "climate":
            if heat_on:
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    {"entity_id": entity_id, ATTR_TEMPERATURE: target_temperature},
                    blocking=False,
                )
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": entity_id, "hvac_mode": HVACMode.HEAT},
                    blocking=False,
                )
            else:
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": entity_id, "hvac_mode": HVACMode.OFF},
                    blocking=False,
                )
            return

        if domain == "ohmonwifiplus":
            await self._call_turn_on(domain, entity_id)
            await self._call_temperature_service(domain, entity_id, value=virtual_outdoor)
            return

        if heat_on:
            await self._call_turn_on(domain, entity_id)
            if await self._call_temperature_service(domain, entity_id, value=target_temperature):
                return
        else:
            handled = await self._call_turn_off(domain, entity_id)
            if handled:
                return

        _LOGGER.debug("Unsupported controlled entity domain for %s", entity_id)

    async def _call_temperature_service(self, domain: str, entity_id: str, value: float | None) -> bool:
        """Try to call a set_temperature-like service for non-climate entities."""
        if not self.hass.services.has_service(domain, "set_temperature"):
            return False
        if value is None:
            return False
        await self.hass.services.async_call(
            domain,
            "set_temperature",
            {"entity_id": entity_id, ATTR_TEMPERATURE: value},
            blocking=False,
        )
        return True

    async def _call_turn_off(self, domain: str, entity_id: str) -> bool:
        """Try to call a turn_off service if available."""
        if not self.hass.services.has_service(domain, "turn_off"):
            return False
        await self.hass.services.async_call(domain, "turn_off", {"entity_id": entity_id}, blocking=False)
        return True

    async def _call_turn_on(self, domain: str, entity_id: str) -> bool:
        """Try to call a turn_on service if available."""
        if not self.hass.services.has_service(domain, "turn_on"):
            return False
        await self.hass.services.async_call(domain, "turn_on", {"entity_id": entity_id}, blocking=False)
        return True
