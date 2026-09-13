"""Forecast and attribute extraction helpers for the climate entity."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

try:
    from homeassistant.util import dt as dt_util
except ModuleNotFoundError:  # pragma: no cover - local tests without HA
    class _DtUtilShim:
        @staticmethod
        def utcnow() -> datetime:
            return datetime.now(timezone.utc)

        @staticmethod
        def as_utc(value: datetime) -> datetime:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

    dt_util = _DtUtilShim()

try:
    from .forecast_utils import align_forecast_to_now, extract_timed_temperatures, extract_timed_values
except ImportError:  # pragma: no cover - direct test imports
    from forecast_utils import align_forecast_to_now, extract_timed_temperatures, extract_timed_values  # type: ignore

_LOGGER = logging.getLogger(__name__)


class ForecastService:
    """Encapsulate price/weather forecast extraction and caching."""

    def __init__(
        self,
        hass,
        *,
        price_entity: str,
        weather_entity: str,
        outdoor_temp_entity: str,
        prediction_horizon: int,
        weather_forecast_cache_seconds: int,
        weather_forecast_service_type: str,
        state_to_float: Callable[[Any], float | None],
    ) -> None:
        self.hass = hass
        self._price_entity = price_entity
        self._weather_entity = weather_entity
        self._outdoor_temp_entity = outdoor_temp_entity
        self._prediction_horizon = prediction_horizon
        self._weather_forecast_cache_seconds = weather_forecast_cache_seconds
        self._weather_forecast_service_type = weather_forecast_service_type
        self._state_to_float = state_to_float
        self._weather_forecast_cache_raw: list[Any] | None = None
        self._weather_forecast_cache_time = None
        self.last_price_forecast_source = "unavailable"
        self.last_outdoor_forecast_source = "unavailable"

    def update_entities(
        self,
        *,
        price_entity: str | None = None,
        weather_entity: str | None = None,
        outdoor_temp_entity: str | None = None,
        prediction_horizon: int | None = None,
    ) -> None:
        """Update tracked entity ids and dependent settings."""
        if price_entity is not None:
            self._price_entity = price_entity
        if weather_entity is not None and weather_entity != self._weather_entity:
            self._weather_entity = weather_entity
            self._weather_forecast_cache_raw = None
            self._weather_forecast_cache_time = None
        if outdoor_temp_entity is not None:
            self._outdoor_temp_entity = outdoor_temp_entity
        if prediction_horizon is not None:
            self._prediction_horizon = prediction_horizon

    def extract_price_forecast(self, now) -> list[float]:
        """Extract price forecast from the configured entity."""
        self.last_price_timed_values = []
        state = self.hass.states.get(self._price_entity)
        if not state:
            self.last_price_forecast_source = "unavailable"
            return []

        attrs = state.attributes
        forecast: list[float] = []
        source = "attribute_forecast"

        raw_today = attrs.get("raw_today")
        raw_tomorrow = attrs.get("raw_tomorrow")
        if raw_today or raw_tomorrow:
            source = "nordpool_raw"
            timed = extract_timed_values(raw_today)
            timed.extend(extract_timed_values(raw_tomorrow))
            self.last_price_timed_values = timed
            forecast = align_forecast_to_now(timed, now)
        else:
            if "prices" in attrs:
                forecast.extend(self._extract_price_list(attrs.get("prices")))
            if "forecast" in attrs and not forecast:
                forecast.extend(self._extract_price_list(attrs.get("forecast")))

        current_price = self._state_to_float(state.state)
        if current_price is not None and not forecast:
            forecast.append(current_price)
            source = "current_price_only"

        if not forecast:
            source = "empty"
        self.last_price_forecast_source = source
        return forecast

    async def build_outdoor_forecast(self, now, *, outdoor_temp: float | None) -> list[float]:
        """Collect outdoor temperature forecast."""
        now = dt_util.as_utc(now)

        weather_state = self.hass.states.get(self._weather_entity)
        if weather_state:
            raw = weather_state.attributes.get("forecast")
            forecast = self._extract_outdoor_forecast_from_raw(raw, now)
            if forecast:
                self.last_outdoor_forecast_source = "weather_attribute"
                if isinstance(raw, list):
                    self._weather_forecast_cache_raw = list(raw)
                    self._weather_forecast_cache_time = dt_util.utcnow()
                return forecast

        if self._weather_entity.startswith("weather."):
            cached = self._extract_outdoor_forecast_from_cache(now)
            if cached:
                self.last_outdoor_forecast_source = "weather_cache"
                return cached

            raw = await self._async_fetch_weather_forecast_service()
            forecast = self._extract_outdoor_forecast_from_raw(raw, now)
            if forecast:
                self.last_outdoor_forecast_source = "weather_service"
                return forecast

        base = outdoor_temp
        if base is None:
            state = self.hass.states.get(self._outdoor_temp_entity)
            base = self._state_to_float(state.state) if state else None
        if base is None:
            self.last_outdoor_forecast_source = "unavailable"
            return []
        horizon = max(1, int(self._prediction_horizon))
        self.last_outdoor_forecast_source = "outdoor_sensor_flat_fallback"
        return [base] * horizon

    @staticmethod
    def normalize_series(values: Sequence[float] | None, steps: int, fallback: float) -> list[float]:
        """Normalize a list of floats to a specific length."""
        normalized: list[float] = []
        if values:
            for val in values:
                try:
                    numeric = float(val)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                normalized.append(numeric)
        if not normalized:
            normalized = [fallback]
        if len(normalized) < steps:
            normalized.extend([normalized[-1]] * (steps - len(normalized)))
        else:
            normalized = normalized[:steps]
        return normalized

    @staticmethod
    def trim_series(values: Sequence[Any] | None, max_entries: int) -> list[Any] | None:
        """Return a bounded list for recorder-safe attributes."""
        if values is None:
            return None
        try:
            series = list(values)
        except TypeError:
            return None
        if max_entries <= 0:
            return []
        if len(series) > max_entries:
            return series[-max_entries:]
        return series

    def _extract_outdoor_forecast_from_cache(self, now) -> list[float]:
        """Return a cached weather forecast if it's still fresh."""
        raw = self._weather_forecast_cache_raw
        cached_at = self._weather_forecast_cache_time
        if not raw or cached_at is None:
            return []
        try:
            cached_at = dt_util.as_utc(cached_at)
        except (TypeError, ValueError):
            return []
        if (now - cached_at).total_seconds() >= self._weather_forecast_cache_seconds:
            return []
        return self._extract_outdoor_forecast_from_raw(raw, now)

    def _extract_outdoor_forecast_from_raw(self, raw: Any, now) -> list[float]:
        """Extract an aligned temperature forecast from raw weather forecast data."""
        if not raw:
            return []
        timed = extract_timed_temperatures(raw)
        aligned = align_forecast_to_now(timed, now)
        if aligned:
            return aligned
        return self._extract_temperatures(raw)

    async def _async_fetch_weather_forecast_service(self) -> list[Any]:
        """Fetch an hourly forecast via ``weather.get_forecasts`` when available."""
        if not self._weather_entity.startswith("weather."):
            return []
        if not self.hass.services.has_service("weather", "get_forecasts"):
            return []

        data = {"entity_id": self._weather_entity, "type": self._weather_forecast_service_type}
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                data,
                blocking=True,
                return_response=True,
            )
        except TypeError:
            return []
        except Exception as err:
            _LOGGER.debug("Weather forecast fetch failed for %s: %s", self._weather_entity, err)
            return []

        raw: Any = None
        if isinstance(response, dict):
            candidate = response.get(self._weather_entity)
            if isinstance(candidate, dict):
                raw = candidate.get("forecast")
            elif isinstance(candidate, list):
                raw = candidate
            elif "forecast" in response:
                raw = response.get("forecast")
            elif len(response) == 1:
                only = next(iter(response.values()))
                if isinstance(only, dict):
                    raw = only.get("forecast")
                elif isinstance(only, list):
                    raw = only

        if not isinstance(raw, list):
            return []

        self._weather_forecast_cache_raw = list(raw)
        self._weather_forecast_cache_time = dt_util.utcnow()
        return list(raw)

    def _extract_temperatures(self, forecast_data: Iterable[Any] | None) -> list[float]:
        """Extract temperature values from forecast data."""
        temperatures: list[float] = []
        if not forecast_data:
            return temperatures
        for item in forecast_data:
            if not isinstance(item, dict):
                continue
            temp = item.get("temperature") or item.get("temp")
            if temp is None:
                continue
            value = self._state_to_float(temp)
            if value is not None:
                temperatures.append(value)
        return temperatures

    def _extract_price_list(self, raw: Iterable[Any] | None) -> list[float]:
        """Extract a list of price values from raw attributes."""
        prices: list[float] = []
        if not raw:
            return prices

        for item in raw:
            if isinstance(item, (int, float, str)):
                val = self._state_to_float(item)
                if val is not None:
                    prices.append(val)
                continue

            if not isinstance(item, dict):
                continue

            value = item.get("value") or item.get("price") or item.get("average") or item.get("total")
            numeric = self._state_to_float(value)
            if numeric is not None:
                prices.append(numeric)
        return prices
