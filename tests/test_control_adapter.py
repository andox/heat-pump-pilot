"""Unit tests for control adapter behavior."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from control_adapter import ControlAdapter, HVACMode  # noqa: E402


@dataclass
class _State:
    state: str
    attributes: dict[str, object] = field(default_factory=dict)


class _StateStore:
    def __init__(self, states: dict[str, _State]) -> None:
        self._states = states

    def get(self, entity_id: str):
        return self._states.get(entity_id)


class _Services:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object], bool]] = []
        self.available = {
            ("number", "set_value"),
            ("switch", "turn_on"),
            ("switch", "turn_off"),
            ("climate", "set_temperature"),
            ("climate", "set_hvac_mode"),
        }

    def has_service(self, domain: str, service: str) -> bool:
        return (domain, service) in self.available

    async def async_call(self, domain: str, service: str, data: dict[str, object], blocking: bool = False):
        self.calls.append((domain, service, data, blocking))


class _Hass:
    def __init__(self, states: dict[str, _State]) -> None:
        self.states = _StateStore(states)
        self.services = _Services()


def _get_state_as_float_factory(hass: _Hass):
    def _get_state_as_float(entity_id: str) -> float | None:
        state = hass.states.get(entity_id)
        if state is None:
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            return None

    return _get_state_as_float


def test_number_control_rounds_and_skips_noop() -> None:
    hass = _Hass({"number.virtual": _State("12.5", {"min": 0, "max": 20, "step": 0.5})})
    adapter = ControlAdapter(hass, get_state_as_float=_get_state_as_float_factory(hass))

    asyncio.run(
        adapter.apply(
            monitor_only=False,
            hvac_mode=HVACMode.HEAT,
            controlled_entity="number.virtual",
            heat_on=True,
            target_temperature=21.0,
            virtual_outdoor=12.49,
        )
    )
    assert hass.services.calls == []

    asyncio.run(
        adapter.apply(
            monitor_only=False,
            hvac_mode=HVACMode.HEAT,
            controlled_entity="number.virtual",
            heat_on=True,
            target_temperature=21.0,
            virtual_outdoor=12.76,
        )
    )
    assert hass.services.calls[-1][:3] == (
        "number",
        "set_value",
        {"entity_id": "number.virtual", "value": "13.0"},
    )


def test_switch_and_climate_control_paths() -> None:
    hass = _Hass(
        {
            "switch.heater": _State("off"),
            "climate.room": _State("heat"),
        }
    )
    adapter = ControlAdapter(hass, get_state_as_float=_get_state_as_float_factory(hass))

    asyncio.run(
        adapter.apply(
            monitor_only=False,
            hvac_mode=HVACMode.HEAT,
            controlled_entity="switch.heater",
            heat_on=True,
            target_temperature=21.0,
            virtual_outdoor=None,
        )
    )
    asyncio.run(
        adapter.apply(
            monitor_only=False,
            hvac_mode=HVACMode.HEAT,
            controlled_entity="climate.room",
            heat_on=False,
            target_temperature=21.0,
            virtual_outdoor=None,
        )
    )

    assert hass.services.calls[0][:2] == ("switch", "turn_on")
    assert hass.services.calls[1][:2] == ("climate", "set_hvac_mode")
    assert hass.services.calls[1][2]["hvac_mode"] == HVACMode.OFF


def test_hvac_off_forces_control_off() -> None:
    hass = _Hass({"switch.heater": _State("on")})
    adapter = ControlAdapter(hass, get_state_as_float=_get_state_as_float_factory(hass))

    asyncio.run(
        adapter.apply(
            monitor_only=False,
            hvac_mode=HVACMode.OFF,
            controlled_entity="switch.heater",
            heat_on=True,
            target_temperature=21.0,
            virtual_outdoor=None,
        )
    )
    assert hass.services.calls[-1][:2] == ("switch", "turn_off")
