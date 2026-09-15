"""Compare score definitions using an exported Pilot attribute history JSON.

Reconstructs control observations, not exact persisted performance history.
Uses only climate snapshots emitted within 60 seconds after their control time;
repeated snapshots of the same control are ignored. No network access required.
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'custom_components/heat_pump_pilot'))
from performance_utils import PerformanceSample, compute_comfort_score, compute_price_score


def compare(path, hours, tolerance):
    history = json.loads(path.read_text(encoding='utf-8-sig'))
    samples = {}
    for row in sorted(history['climate.heat_pump_pilot'], key=lambda r: r.get('lu', r.get('lc', 0))):
        a = row.get('a', {})
        try:
            when = datetime.fromisoformat(a['last_control_time'].replace('Z', '+00:00'))
            lag = float(row.get('lu', row.get('lc'))) - when.timestamp()
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if when.tzinfo is None or not 0 <= lag <= 60 or when in samples:
            continue
        samples[when] = PerformanceSample(
            when, a.get('current_temperature'), a.get('temperature'),
            a.get('heating_detected'), a.get('current_price'), None)
    if not samples:
        raise ValueError('No contemporaneous climate control snapshots found')
    end = max(samples)
    start = end - timedelta(hours=hours)
    rows = list(samples.values())
    selected = [s for s in rows if start <= s.when <= end]
    old_comfort, _ = compute_comfort_score(selected, tolerance)
    prices = [float(s.price) for s in selected if s.price is not None]
    heated = [float(s.price) for s in selected if s.price is not None and s.heating_detected is True]
    old_price = (100 * (1 - (mean(heated) - min(prices)) / max(max(prices) - min(prices), 1e-6))
                 if heated else None)
    kwargs = dict(now=end, window_start=start, max_gap_minutes=15)
    comfort, cd = compute_comfort_score(rows, tolerance, **kwargs)
    price, pd = compute_price_score(rows, **kwargs)
    return dict(source=str(path), reconstructed_samples=len(selected),
                start=start.isoformat(), end=end.isoformat(),
                limitation='Reconstructed control snapshots; 15-minute observation hold; not metered energy.',
                old_comfort=old_comfort, old_price=old_price,
                comfort=comfort, comfort_details=cd, price=price, price_details=pd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--tolerance', type=float, default=0.2)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('--hours must be positive')
    print(json.dumps(compare(args.history, args.hours, args.tolerance), indent=2, allow_nan=False))
