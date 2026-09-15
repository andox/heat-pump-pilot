"""Timestamped price buckets; missing history never acquires invented dates."""

from datetime import datetime, timedelta, timezone
from math import isfinite


class PriceObservations:
    def __init__(self, minutes=15):
        self.minutes = minutes
        self.buckets = {}

    def add(self, when, value):
        try:
            price = float(value)
        except (TypeError, ValueError):
            return
        if not isfinite(price):
            return
        when = when.astimezone(timezone.utc)
        bucket = when.replace(minute=when.minute // self.minutes * self.minutes, second=0, microsecond=0)
        self.buckets[bucket] = price

    def prune(self, now):
        cutoff = now - timedelta(days=30)
        self.buckets = {t: p for t, p in self.buckets.items() if cutoff < t <= now}

    def values(self, now, hours):
        cutoff = now - timedelta(hours=hours)
        end = now.replace(minute=now.minute // self.minutes * self.minutes, second=0, microsecond=0)
        return [p for t, p in sorted(self.buckets.items()) if cutoff <= t < end]

    def dump(self):
        return [{"time": t.isoformat(), "price": p} for t, p in sorted(self.buckets.items())]

    def restore(self, rows, now):
        for row in rows if isinstance(rows, list) else []:
            try:
                when = datetime.fromisoformat(row['time'])
                if when.tzinfo is None or when > now or when <= now - timedelta(days=30):
                    continue
                self.add(when, row['price'])
            except (KeyError, TypeError, ValueError):
                continue
        self.prune(now)

    def backfill(self, events, now):
        """Resample recorder changes, holding at most one hour across silence.

        An unavailable event ends the preceding observation. Existing live
        buckets win if recorder loading races with a control update.
        """
        ordered = sorted(events, key=lambda row: row[0])
        recovered = PriceObservations(self.minutes)
        for i, (start, value) in enumerate(ordered):
            end = min(now, start + timedelta(hours=1))
            if i + 1 < len(ordered):
                end = min(end, ordered[i + 1][0])
            when = start.replace(minute=start.minute // self.minutes * self.minutes, second=0, microsecond=0)
            if when < start:
                when += timedelta(minutes=self.minutes)
            while when < end:
                recovered.add(when, value)
                when += timedelta(minutes=self.minutes)
        recovered.buckets.update(self.buckets)
        self.buckets = recovered.buckets
        self.prune(now)
