"""Summer low-price heat window helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Iterable


STATE_DISABLED = "disabled"
STATE_INELIGIBLE = "ineligible"
STATE_SCHEDULED = "scheduled"
STATE_ACTIVE = "active"
STATE_USED = "used"
STATE_SKIPPED = "skipped"


@dataclass(frozen=True)
class SummerHeatWindow:
    """A selected one-shot heat window."""

    start: datetime
    end: datetime
    average_price: float
    max_price: float


@dataclass(frozen=True)
class PriceInterval:
    """A price value with concrete time bounds."""

    start: datetime
    end: datetime
    price: float


def _coerce_prices(values: Iterable[object] | None) -> list[float]:
    if values is None:
        return []
    prices: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            prices.append(numeric)
    return prices


def find_summer_heat_window(
    *,
    now: datetime,
    prices: Iterable[object] | None,
    step_hours: float,
    duration_minutes: int,
    max_price: float,
) -> SummerHeatWindow | None:
    """Pick the cheapest continuous qualifying window in the current local day."""
    price_values = _coerce_prices(prices)
    if not price_values:
        return None
    if step_hours <= 0:
        return None
    duration = max(1, int(duration_minutes))
    required_steps = max(1, int(math.ceil(duration / (step_hours * 60))))
    if len(price_values) < required_steps:
        return None

    best: SummerHeatWindow | None = None
    local_day = now.date()
    step = timedelta(hours=step_hours)

    for start_idx in range(0, len(price_values) - required_steps + 1):
        start = now + step * start_idx
        end = start + step * required_steps
        if start.date() != local_day or (end - timedelta(microseconds=1)).date() != local_day:
            continue
        block = price_values[start_idx : start_idx + required_steps]
        if any(price > max_price for price in block):
            continue
        average_price = sum(block) / len(block)
        window = SummerHeatWindow(
            start=start,
            end=end,
            average_price=average_price,
            max_price=max(block),
        )
        if best is None or (window.average_price, window.max_price, window.start) < (
            best.average_price,
            best.max_price,
            best.start,
        ):
            best = window

    return best


def _coerce_price_intervals(values: Iterable[object] | None, now: datetime) -> list[PriceInterval]:
    if values is None:
        return []

    raw: list[tuple[datetime, datetime | None, float]] = []
    for value in values:
        start = getattr(value, "start", None)
        if start is None:
            continue
        end = getattr(value, "end", None)
        price_raw = getattr(value, "value", None)
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue

        if start.tzinfo is None:
            start = start.replace(tzinfo=now.tzinfo)
        else:
            start = start.astimezone(now.tzinfo)
        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=now.tzinfo)
            else:
                end = end.astimezone(now.tzinfo)
        raw.append((start, end, price))

    if not raw:
        return []

    raw.sort(key=lambda item: item[0])
    intervals: list[PriceInterval] = []
    for index, (start, end, price) in enumerate(raw):
        next_start = raw[index + 1][0] if index + 1 < len(raw) else None
        previous_start = raw[index - 1][0] if index > 0 else None
        effective_end = end
        if effective_end is None and next_start is not None and start < next_start:
            effective_end = next_start
        elif effective_end is None and previous_start is not None and previous_start < start:
            effective_end = start + (start - previous_start)
        if effective_end is None or effective_end <= start:
            continue
        intervals.append(PriceInterval(start=start, end=effective_end, price=price))
    return intervals


def _weighted_window_stats(
    intervals: list[PriceInterval], start: datetime, end: datetime
) -> tuple[float, float] | None:
    total_seconds = 0.0
    weighted_price = 0.0
    max_price = -math.inf

    for interval in intervals:
        overlap_start = max(start, interval.start)
        overlap_end = min(end, interval.end)
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        if overlap_seconds <= 0:
            continue
        total_seconds += overlap_seconds
        weighted_price += interval.price * overlap_seconds
        max_price = max(max_price, interval.price)

    duration_seconds = (end - start).total_seconds()
    if duration_seconds <= 0 or total_seconds < duration_seconds - 1e-6:
        return None
    return weighted_price / total_seconds, max_price


def _select_best_window_from_run(
    run: list[PriceInterval],
    duration: timedelta,
    best: SummerHeatWindow | None,
) -> SummerHeatWindow | None:
    if not run:
        return best

    run_end = run[-1].end
    candidate_starts = [interval.start for interval in run if interval.start + duration <= run_end]
    for start in candidate_starts:
        end = start + duration
        stats = _weighted_window_stats(run, start, end)
        if stats is None:
            continue
        average_price, max_price = stats
        window = SummerHeatWindow(
            start=start,
            end=end,
            average_price=average_price,
            max_price=max_price,
        )
        if best is None or (window.average_price, window.max_price, window.start) < (
            best.average_price,
            best.max_price,
            best.start,
        ):
            best = window
    return best


def find_summer_heat_window_from_timed_values(
    *,
    now: datetime,
    values: Iterable[object] | None,
    duration_minutes: int,
    max_price: float,
) -> SummerHeatWindow | None:
    """Pick the cheapest qualifying window using actual price timestamps."""
    intervals = _coerce_price_intervals(values, now)
    if not intervals:
        return None

    local_day = now.date()
    next_day = datetime.combine(local_day + timedelta(days=1), datetime.min.time(), tzinfo=now.tzinfo)
    duration = timedelta(minutes=max(1, int(duration_minutes)))
    best: SummerHeatWindow | None = None
    run: list[PriceInterval] = []

    for interval in intervals:
        start = max(interval.start, now)
        end = min(interval.end, next_day)
        if start.date() != local_day or end <= start:
            continue

        if interval.price > max_price:
            best = _select_best_window_from_run(run, duration, best)
            run = []
            continue

        clipped = PriceInterval(start=start, end=end, price=interval.price)
        if run and clipped.start > run[-1].end:
            best = _select_best_window_from_run(run, duration, best)
            run = []
        run.append(clipped)

    best = _select_best_window_from_run(run, duration, best)
    return best


def compute_heat_demand_ratio(
    samples: Iterable[Any],
    *,
    now: datetime,
    window_hours: float,
    min_samples: int = 4,
) -> tuple[float | None, int]:
    """Compute recent normal MPC heat-request ratio from performance samples."""
    cutoff = now - timedelta(hours=max(0.0, float(window_hours)))
    total = 0
    heat_on = 0
    for sample in samples:
        when = getattr(sample, "when", None)
        suggested = getattr(sample, "suggested_heat_on", None)
        if when is None or suggested not in (True, False):
            continue
        try:
            if when < cutoff:
                continue
        except TypeError:
            continue
        total += 1
        if suggested:
            heat_on += 1

    if total < min_samples:
        return None, total
    return heat_on / total, total


class SummerHeatWindowStorage:
    """Simple JSON file persistence for the selected daily summer heat window."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> dict[str, Any] | None:
        """Load persisted state from disk."""
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, payload: dict[str, Any]) -> None:
        """Persist state atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True)
        temp_path.replace(self._path)
