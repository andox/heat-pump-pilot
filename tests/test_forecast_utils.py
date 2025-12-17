"""Unit tests for price forecast alignment utilities."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow importing the integration modules directly from this directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from forecast_utils import (  # noqa: E402
    align_forecast_to_now,
    expand_to_steps,
    extract_timed_temperatures,
    extract_timed_values,
)


def test_align_forecast_drops_past_entries() -> None:
    """Alignment should start at the interval covering 'now'."""
    tz = timezone(timedelta(hours=1))
    raw = [
        {"start": "2025-12-12T00:00:00+01:00", "end": "2025-12-12T00:15:00+01:00", "value": 0.5},
        {"start": "2025-12-12T00:15:00+01:00", "end": "2025-12-12T00:30:00+01:00", "value": 0.6},
        {"start": "2025-12-12T00:30:00+01:00", "end": "2025-12-12T00:45:00+01:00", "value": 0.7},
    ]
    now = datetime(2025, 12, 12, 0, 20, tzinfo=tz)
    forecast = align_forecast_to_now(extract_timed_values(raw), now)
    assert forecast == [0.6, 0.7]


def test_align_forecast_at_interval_boundary() -> None:
    """If 'now' is exactly at a start boundary, that slot is current."""
    tz = timezone(timedelta(hours=1))
    raw = [
        {"start": "2025-12-12T01:00:00+01:00", "end": "2025-12-12T01:15:00+01:00", "value": 1.0},
        {"start": "2025-12-12T01:15:00+01:00", "end": "2025-12-12T01:30:00+01:00", "value": 2.0},
    ]
    now = datetime(2025, 12, 12, 1, 0, tzinfo=tz)
    forecast = align_forecast_to_now(extract_timed_values(raw), now)
    assert forecast == [1.0, 2.0]


def test_align_forecast_before_first_entry() -> None:
    """If 'now' is before the first entry, first entry becomes current."""
    tz = timezone.utc
    raw = [
        {"start": "2025-12-12T10:00:00+00:00", "end": "2025-12-12T10:15:00+00:00", "value": 1.1},
        {"start": "2025-12-12T10:15:00+00:00", "end": "2025-12-12T10:30:00+00:00", "value": 1.2},
    ]
    now = datetime(2025, 12, 12, 9, 55, tzinfo=tz)
    forecast = align_forecast_to_now(extract_timed_values(raw), now)
    assert forecast == [1.1, 1.2]


def test_align_weather_hourly_forecast_inside_interval() -> None:
    """Weather-style hourly forecasts should align using the implicit 1h step."""
    tz = timezone.utc
    raw = [
        {"datetime": "2025-12-12T00:00:00+00:00", "temperature": 5.0},
        {"datetime": "2025-12-12T01:00:00+00:00", "temperature": 6.0},
        {"datetime": "2025-12-12T02:00:00+00:00", "temperature": 7.0},
    ]
    now = datetime(2025, 12, 12, 0, 30, tzinfo=tz)
    forecast = align_forecast_to_now(extract_timed_temperatures(raw), now)
    assert forecast == [5.0, 6.0, 7.0]


def test_align_weather_hourly_forecast_before_first_entry() -> None:
    """If 'now' is before the first entry, the first becomes current."""
    tz = timezone.utc
    raw = [
        {"datetime": "2025-12-12T10:00:00+00:00", "native_temperature": 1.1},
        {"datetime": "2025-12-12T11:00:00+00:00", "native_temperature": 1.2},
    ]
    now = datetime(2025, 12, 12, 9, 55, tzinfo=tz)
    forecast = align_forecast_to_now(extract_timed_temperatures(raw), now)
    assert forecast == [1.1, 1.2]


def test_expand_to_steps_repeats_hourly_values_to_controller_steps() -> None:
    expanded = expand_to_steps([1.0, 2.0, 3.0], steps=8, step_hours=0.25)
    assert expanded == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_expand_to_steps_truncates_when_input_long_enough() -> None:
    expanded = expand_to_steps([1.0, 2.0, 3.0, 4.0], steps=2, step_hours=0.25)
    assert expanded == [1.0, 2.0]


def test_expand_to_steps_pads_when_forecast_too_short() -> None:
    expanded = expand_to_steps([1.0], steps=6, step_hours=0.25)
    assert expanded == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
