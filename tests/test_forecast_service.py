"""Unit tests for forecast service fallbacks and parsing."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from forecast_service import ForecastService  # noqa: E402


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
        self.responses: dict[tuple[str, str], object] = {}
        self.available: set[tuple[str, str]] = set()

    def has_service(self, domain: str, service: str) -> bool:
        return (domain, service) in self.available

    async def async_call(self, domain: str, service: str, data, blocking=False, return_response=False):
        return self.responses.get((domain, service))


class _Hass:
    def __init__(self, states: dict[str, _State]) -> None:
        self.states = _StateStore(states)
        self.services = _Services()


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def test_price_forecast_falls_back_to_current_price() -> None:
    hass = _Hass({"sensor.price": _State("1.23", {})})
    service = ForecastService(
        hass,
        price_entity="sensor.price",
        weather_entity="weather.home",
        outdoor_temp_entity="sensor.outdoor",
        prediction_horizon=24,
        weather_forecast_cache_seconds=1800,
        weather_forecast_service_type="hourly",
        state_to_float=_to_float,
    )
    forecast = service.extract_price_forecast(datetime.now(timezone.utc))
    assert forecast == [1.23]
    assert service.last_price_forecast_source == "current_price_only"


def test_build_outdoor_forecast_uses_weather_attribute_then_sensor_fallback() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    hass = _Hass(
        {
            "weather.home": _State(
                "sunny",
                {
                    "forecast": [
                        {"datetime": "2026-01-01T12:00:00+00:00", "temperature": 2.0},
                        {"datetime": "2026-01-01T13:00:00+00:00", "temperature": 3.0},
                    ]
                },
            ),
            "sensor.outdoor": _State("5.0"),
        }
    )
    service = ForecastService(
        hass,
        price_entity="sensor.price",
        weather_entity="weather.home",
        outdoor_temp_entity="sensor.outdoor",
        prediction_horizon=3,
        weather_forecast_cache_seconds=1800,
        weather_forecast_service_type="hourly",
        state_to_float=_to_float,
    )
    forecast = asyncio.run(service.build_outdoor_forecast(now, outdoor_temp=None))
    assert forecast == [2.0, 3.0]
    assert service.last_outdoor_forecast_source == "weather_attribute"

    hass = _Hass({"sensor.outdoor": _State("5.0")})
    service = ForecastService(
        hass,
        price_entity="sensor.price",
        weather_entity="weather.home",
        outdoor_temp_entity="sensor.outdoor",
        prediction_horizon=3,
        weather_forecast_cache_seconds=1800,
        weather_forecast_service_type="hourly",
        state_to_float=_to_float,
    )
    forecast = asyncio.run(service.build_outdoor_forecast(now, outdoor_temp=None))
    assert forecast == [5.0, 5.0, 5.0]
    assert service.last_outdoor_forecast_source == "outdoor_sensor_flat_fallback"
