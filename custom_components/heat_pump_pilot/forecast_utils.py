"""Forecast parsing utilities for heat_pump_pilot.

These helpers are intentionally free of Home Assistant imports so they can be
unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class TimedValue:
    """A single forecast value with optional time bounds."""

    start: datetime | None
    end: datetime | None
    value: float


def _ensure_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (assume UTC when naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO datetime string (or datetime) into an aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _ensure_aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def extract_timed_values(raw: Iterable[Any] | None) -> list[TimedValue]:
    """Extract (start/end/value) tuples from Nordpool-style raw lists."""
    values: list[TimedValue] = []
    if not raw:
        return values

    for item in raw:
        if isinstance(item, dict):
            value_raw = item.get("value")
            if value_raw is None:
                value_raw = item.get("price") or item.get("average") or item.get("total")
            try:
                numeric = float(value_raw)
            except (TypeError, ValueError):
                continue

            start = parse_datetime(item.get("start") or item.get("from") or item.get("datetime"))
            end = parse_datetime(item.get("end") or item.get("to"))
            values.append(TimedValue(start=start, end=end, value=numeric))
            continue

        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        values.append(TimedValue(start=None, end=None, value=numeric))

    return values


def extract_timed_temperatures(
    raw: Iterable[Any] | None, *, default_step: timedelta | None = timedelta(hours=1)
) -> list[TimedValue]:
    """Extract (start/end/value) tuples from weather-style forecasts.

    Supports Home Assistant weather forecast dicts (typically with ``datetime`` and
    ``temperature`` keys) and applies a default interval when no explicit end
    timestamp is provided (hourly by default).
    """
    values: list[TimedValue] = []
    if not raw:
        return values

    for item in raw:
        if not isinstance(item, dict):
            continue

        temp_raw = item.get("temperature")
        if temp_raw is None:
            temp_raw = item.get("temp")
        if temp_raw is None:
            temp_raw = item.get("native_temperature")
        try:
            numeric = float(temp_raw)
        except (TypeError, ValueError):
            continue

        start = parse_datetime(item.get("datetime") or item.get("start") or item.get("from") or item.get("time"))
        end = parse_datetime(item.get("end") or item.get("to"))
        if end is None and start is not None and default_step is not None:
            end = start + default_step

        values.append(TimedValue(start=start, end=end, value=numeric))

    return values


def align_forecast_to_now(values: Iterable[TimedValue], now: datetime) -> list[float]:
    """Return values aligned so index 0 corresponds to 'now'.

    Only timed entries (with a start timestamp) can be aligned. Untimed values
    are ignored by this helper.
    """
    now = _ensure_aware(now)
    timed = [value for value in values if value.start is not None]
    if not timed:
        return []

    timed.sort(key=lambda entry: entry.start)

    current: float | None = None
    future: list[float] = []

    for entry in timed:
        if entry.end is not None and entry.start <= now < entry.end:
            current = entry.value
        if entry.start > now:
            future.append(entry.value)

    if current is None:
        first_start = timed[0].start
        if first_start is not None and now <= first_start:
            current = timed[0].value
            if future and future[0] == current:
                future = future[1:]

    forecast: list[float] = []
    if current is not None:
        forecast.append(current)
    forecast.extend(future)
    return forecast


def expand_to_steps(values: Iterable[float] | None, steps: int, step_hours: float) -> list[float]:
    """Expand an hourly-ish forecast series to match the MPC controller step count.

    Home Assistant price and weather forecasts are commonly provided as hourly values,
    while the MPC controller runs at a finer time step (e.g. 15 minutes). This helper
    repeats each value to fill the finer step grid and truncates/pads to ``steps``.

    If the input already contains at least ``steps`` values, it is simply truncated.
    """
    if steps <= 0:
        return []
    if not values:
        return []

    coerced: list[float] = []
    for value in values:
        try:
            coerced.append(float(value))
        except (TypeError, ValueError):
            continue
    if not coerced:
        return []

    if len(coerced) >= steps:
        return coerced[:steps]

    steps_per_hour = 1
    if step_hours > 0:
        steps_per_hour = int(round(1.0 / step_hours)) or 1
    steps_per_hour = max(1, steps_per_hour)

    expanded: list[float] = []
    for value in coerced:
        expanded.extend([value] * steps_per_hour)
        if len(expanded) >= steps:
            break

    if len(expanded) < steps:
        expanded.extend([expanded[-1]] * (steps - len(expanded)))
    return expanded[:steps]
