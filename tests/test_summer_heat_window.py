"""Unit tests for summer low-price heat window helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from summer_heat_window import (  # noqa: E402
    compute_heat_demand_ratio,
    find_summer_heat_window,
    find_summer_heat_window_from_timed_values,
)


@dataclass(frozen=True)
class Sample:
    when: datetime
    suggested_heat_on: bool | None


@dataclass(frozen=True)
class TimedPrice:
    start: datetime
    end: datetime | None
    value: float


def test_find_summer_heat_window_single_qualifying_block() -> None:
    now = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    window = find_summer_heat_window(
        now=now,
        prices=[0.5, 0.2, 0.2, 0.2, 0.2, 0.6],
        step_hours=0.25,
        duration_minutes=60,
        max_price=0.3,
    )

    assert window is not None
    assert window.start == now + timedelta(minutes=15)
    assert window.end == now + timedelta(minutes=75)


def test_find_summer_heat_window_skips_when_continuous_duration_missing() -> None:
    now = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    window = find_summer_heat_window(
        now=now,
        prices=[0.2, 0.2, 0.5, 0.2, 0.2],
        step_hours=0.25,
        duration_minutes=60,
        max_price=0.3,
    )

    assert window is None


def test_find_summer_heat_window_picks_lowest_average_block() -> None:
    now = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    window = find_summer_heat_window(
        now=now,
        prices=[0.2, 0.25, 0.25, 0.25, 0.4, 0.1, 0.1, 0.1, 0.1],
        step_hours=0.25,
        duration_minutes=60,
        max_price=0.3,
    )

    assert window is not None
    assert window.start == now + timedelta(minutes=75)
    assert window.average_price == 0.1


def test_find_summer_heat_window_handles_hourly_steps() -> None:
    now = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    window = find_summer_heat_window(
        now=now,
        prices=[0.4, 0.2, 0.2, 0.5],
        step_hours=1.0,
        duration_minutes=120,
        max_price=0.3,
    )

    assert window is not None
    assert window.start == now + timedelta(hours=1)
    assert window.end == now + timedelta(hours=3)


def test_find_summer_heat_window_does_not_cross_local_day() -> None:
    now = datetime(2026, 6, 1, 23, 30, tzinfo=timezone.utc)
    window = find_summer_heat_window(
        now=now,
        prices=[0.1, 0.1, 0.1, 0.1],
        step_hours=0.25,
        duration_minutes=60,
        max_price=0.3,
    )

    assert window is None


def test_find_summer_heat_window_from_timed_values_uses_price_slot_boundaries() -> None:
    tz = timezone(timedelta(hours=1))
    now = datetime(2026, 6, 3, 20, 46, tzinfo=tz)
    prices = [
        TimedPrice(datetime(2026, 6, 3, 20, 0, tzinfo=tz), None, 1.8),
        TimedPrice(datetime(2026, 6, 3, 21, 0, tzinfo=tz), None, 0.63),
        TimedPrice(datetime(2026, 6, 3, 22, 0, tzinfo=tz), None, 0.63),
    ]

    window = find_summer_heat_window_from_timed_values(
        now=now,
        values=prices,
        duration_minutes=60,
        max_price=0.7,
    )

    assert window is not None
    assert window.start == datetime(2026, 6, 3, 21, 0, tzinfo=tz)
    assert window.end == datetime(2026, 6, 3, 22, 0, tzinfo=tz)
    assert window.average_price == 0.63


def test_find_summer_heat_window_from_timed_values_can_start_now_when_current_slot_qualifies() -> None:
    tz = timezone(timedelta(hours=1))
    now = datetime(2026, 6, 3, 20, 46, tzinfo=tz)
    prices = [
        TimedPrice(datetime(2026, 6, 3, 20, 0, tzinfo=tz), None, 0.63),
        TimedPrice(datetime(2026, 6, 3, 21, 0, tzinfo=tz), None, 0.63),
    ]

    window = find_summer_heat_window_from_timed_values(
        now=now,
        values=prices,
        duration_minutes=60,
        max_price=0.7,
    )

    assert window is not None
    assert window.start == now
    assert window.end == now + timedelta(minutes=60)


def test_heat_demand_ratio_eligible_at_threshold() -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    samples = [Sample(now - timedelta(hours=i), False) for i in range(9)]
    samples.append(Sample(now - timedelta(hours=10), True))

    ratio, count = compute_heat_demand_ratio(samples, now=now, window_hours=48)

    assert count == 10
    assert ratio == 0.1


def test_heat_demand_ratio_ineligible_when_insufficient_history() -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    samples = [Sample(now, False), Sample(now - timedelta(minutes=15), False)]

    ratio, count = compute_heat_demand_ratio(samples, now=now, window_hours=48)

    assert count == 2
    assert ratio is None
