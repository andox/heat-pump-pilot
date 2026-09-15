"""Offline thermal-learning experiments on exported Home Assistant history.

No Home Assistant installation or third-party packages are required. No live
integration files or source data are modified. See --help and the generated report.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import collections
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "heat_pump_pilot"))
from thermal_model import ThermalModelEstimator, ThermalModelRlsEstimator  # noqa: E402
from adaptive_model import AdaptiveThermalModel  # noqa: E402
from learning_manager import LearningManager  # noqa: E402

INDOOR = "sensor.sonoff_snzb_02d_temperature"
OUTDOOR = "sensor.outdoor_temperature"
SUPPLY = "sensor.ground_source_heat_pump"
CLIMATE = "climate.heat_pump_pilot"
DECISION = "sensor.heat_pump_pilot_decision"
HEATING = "binary_sensor.heat_pump_pilot_heating_detected"
STEP = 900


def timestamp(value):
    return float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class Series:
    """Causal sample-and-hold series; unknowns and expired observations stay missing."""

    def __init__(self, pairs, max_age=21600):
        pairs = sorted(dict(pairs).items())
        self.times = [p[0] for p in pairs]
        self.values = [p[1] for p in pairs]
        self.max_age = max_age

    def at(self, t):
        i = bisect_right(self.times, t) - 1
        if i < 0 or t - self.times[i] > self.max_age:
            return None
        return self.values[i]

    def mean(self, start, end):
        """Time-weighted average, requiring complete known coverage."""
        cuts = [start, end]
        a, b = bisect_right(self.times, start), bisect_right(self.times, end)
        cuts.extend(self.times[a:b])
        cuts.extend(t + self.max_age for t in self.times[max(0, a - 1):b] if start < t + self.max_age < end)
        cuts = sorted(set(cuts))
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            value = self.at((left + right) / 2)
            if value is None:
                return None
            total += value * (right - left)
        return total / (end - start)


@dataclass
class Interval:
    t: float
    indoor: float
    next_indoor: float
    outdoor: float
    heat: float
    recorded: tuple[float, float] | None


def read_inputs(csv_path, json_path, max_age):
    groups = collections.defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["entity_id"]].append((timestamp(row["last_changed"]), numeric(row["state"])))
    history = json.loads(json_path.read_text(encoding="utf-8-sig"))
    audit = {}
    for entity, pairs in groups.items():
        pairs.sort()
        audit[entity] = {"records": len(pairs), "start": iso(pairs[0][0]), "end": iso(pairs[-1][0])}

    # The provided export switches from hourly statistics to raw events. Never
    # interpret the older hourly averages as instantaneous heating observations.
    raw = {}
    for entity in (INDOOR, OUTDOOR, SUPPLY):
        onset = min(t for t, _ in groups[entity] if abs(t % 3600) > 0.001)
        raw[entity] = Series([(t, v) for t, v in groups[entity] if t >= onset], max_age)
        audit[entity]["raw_start"] = iso(onset)
    heat_pairs = [(float(r.get("lu", r.get("lc"))), {"on": 1.0, "off": 0.0}.get(r["s"])) for r in history[HEATING]]
    heat = Series(heat_pairs, max_age)
    climate_pairs = []
    for row in history[CLIMATE]:
        attrs = row.get("a", {})
        loss = numeric(attrs.get("estimated_heat_loss_coefficient"))
        gain = numeric(attrs.get("estimated_heat_gain_coefficient"))
        climate_pairs.append((row.get("lu", row.get("lc")), (loss, gain) if loss is not None and gain is not None else None))
    coefficients = Series(climate_pairs, max_age)
    start = math.ceil(max(raw[INDOOR].times[0], raw[OUTDOOR].times[0], heat.times[0]) / 3600) * 3600
    end = math.floor(min(raw[INDOOR].times[-1], raw[OUTDOOR].times[-1], heat.times[-1]) / STEP) * STEP
    intervals = {}
    for t in range(int(start), int(end), STEP):
        indoor, next_indoor = raw[INDOOR].at(t), raw[INDOOR].at(t + STEP)
        outdoor, duty = raw[OUTDOOR].mean(t, t + STEP), heat.mean(t, t + STEP)
        if any(v is None for v in (indoor, next_indoor, outdoor, duty)):
            continue
        if not (10 <= indoor <= 35 and 10 <= next_indoor <= 35):
            continue
        intervals[t] = Interval(t, indoor, next_indoor, outdoor, duty, coefficients.at(t))
    controls = sorted(set(timestamp(r["a"]["last_control_time"]) for r in history[CLIMATE] if r.get("a", {}).get("last_control_time")))
    gaps = [(b - a) / 60 for a, b in zip(controls, controls[1:]) if b > a]
    audit["recent"] = {
        "start": iso(start), "end": iso(end), "valid_hours": len(intervals) / 4,
        "excluded_hours": (end - start) / 3600 - len(intervals) / 4,
        "detected_heating_hours": sum(r.heat for r in intervals.values()) / 4,
        "control_updates": len(controls), "median_control_interval_minutes": statistics.median(gaps),
        "updates_under_five_minutes": sum(g < 5 for g in gaps),
        "max_hold_hours": max_age / 3600,
    }
    return intervals, start, end, audit, history, groups


def solve_linear(matrix, rhs):
    """Small pivoted Gaussian elimination, used by the constrained regression."""
    n = len(rhs)
    a = [list(row) + [b] for row, b in zip(matrix, rhs)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-12:
            return None
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        a[i] = [v / scale for v in a[i]]
        for j in range(n):
            if j != i:
                scale = a[j][i]
                a[j] = [v - scale * w for v, w in zip(a[j], a[i])]
    return [row[-1] for row in a]


def fit_bounded(observations, bias=False):
    """Box-constrained least squares by enumerating active bounds (2-3 parameters)."""
    bounds = [(0.001, 0.25), (0.1, 1.5)] + ([(-0.5, 0.5)] if bias else [])
    n = len(bounds)
    xx = [[0.0] * n for _ in range(n)]
    xy = [0.0] * n
    yy = 0.0
    for feature, target in observations:
        feature = list(feature) + ([1.0] if bias else [])
        yy += target * target
        for i in range(n):
            xy[i] += feature[i] * target
            for j in range(n):
                xx[i][j] += feature[i] * feature[j]
    # Tiny numerical regularizer, not a substantive prior.
    for i in range(n):
        xx[i][i] += 1e-9
    best = None
    for active in product((-1, 0, 1), repeat=n):
        theta = [bounds[i][0 if a == -1 else 1] if a else 0.0 for i, a in enumerate(active)]
        free = [i for i, a in enumerate(active) if not a]
        if free:
            solution = solve_linear([[xx[i][j] for j in free] for i in free], [xy[i] - sum(xx[i][j] * theta[j] for j in range(n) if active[j]) for i in free])
            if solution is None:
                continue
            for i, value in zip(free, solution):
                theta[i] = value
        if any(v < lo - 1e-8 or v > hi + 1e-8 for v, (lo, hi) in zip(theta, bounds)):
            continue
        objective = yy - 2 * sum(v * b for v, b in zip(theta, xy)) + sum(theta[i] * xx[i][j] * theta[j] for i in range(n) for j in range(n))
        if best is None or objective < best[0]:
            best = objective, theta
    return tuple(best[1]) if bias else (*best[1], 0.0)


def filtered_heat(intervals, tau_hours):
    """Causal first-order heat-release proxy; reset across unknown intervals."""
    output = {}
    state, previous = 0.0, None
    for t, row in sorted(intervals.items()):
        if previous != t - STEP:
            state = row.heat
        if tau_hours:
            decay = math.exp(-0.25 / tau_hours)
            output[t] = row.heat + (state - row.heat) * tau_hours / 0.25 * (1 - decay)
            state = row.heat + (state - row.heat) * decay
        else:
            output[t] = row.heat
        previous = t
    return output


def hourly_observations(intervals, heat):
    output = []
    for end in sorted(t + STEP for t in intervals if (t + STEP) % 3600 == 0):
        keys = list(range(end - 3600, end, STEP))
        if not all(t in intervals for t in keys):
            continue
        rows = [intervals[t] for t in keys]
        features = (statistics.mean(r.outdoor - r.indoor for r in rows), statistics.mean(heat[t] for t in keys))
        output.append((end, features, rows[-1].next_indoor - rows[0].indoor))
    return output


class SlowEkf(ThermalModelEstimator):
    """Experimental: hourly updates, temperature noise per hour, 10x slower parameter diffusion."""

    @staticmethod
    def _add_process_noise(covariance):
        covariance[0][0] += 0.01 * 4
        covariance[1][1] += 0.0005 * 4 * 0.1
        covariance[2][2] += 0.002 * 4 * 0.1
        return covariance


def build_candidates(intervals, train_end):
    snapshots, inputs = {}, {}
    raw_heat = filtered_heat(intervals, 0)
    seed = intervals[min(intervals)].recorded or (0.05, 0.8)
    snapshots["recorded_model"] = {t: (*r.recorded, 0.0) for t, r in intervals.items() if r.recorded is not None}
    inputs["recorded_model"] = raw_heat
    snapshots["persistence"] = {t: (0.0, 0.0, 0.0) for t in intervals}
    inputs["persistence"] = raw_heat
    for name, cls, cadence in [("ekf_15m", ThermalModelEstimator, 1), ("ekf_60m", ThermalModelEstimator, 4), ("ekf_60m_slow_noise", SlowEkf, 4), ("rls_60m", ThermalModelRlsEstimator, 4)]:
        model, previous, pending = None, None, []
        states = {}
        for t, row in sorted(intervals.items()):
            if previous != t - STEP:
                loss, gain = (model.heat_loss_coeff, model.heat_gain_coeff) if model else seed
                model = cls(initial_temp=row.indoor, initial_heat_loss=loss, initial_heat_gain=gain)
                pending = []
            states[t] = (model.heat_loss_coeff, model.heat_gain_coeff, 0.0)
            pending.append(row)
            if len(pending) == cadence:
                model.step(row.next_indoor, statistics.mean(r.outdoor for r in pending), statistics.mean(r.heat for r in pending), cadence / 4)
                pending = []
            previous = t
        snapshots[name], inputs[name] = states, raw_heat
    for name, window, tau, bias in [("frozen_regression", None, 0, False), ("rolling_72h", 72, 0, False), ("rolling_72h_bias", 72, 0, True), ("rolling_72h_lag1h", 72, 1, False), ("rolling_72h_lag3h", 72, 3, False)]:
        heat = filtered_heat(intervals, tau)
        observations = hourly_observations(intervals, heat)
        states, fitted, last_hour = {}, (*seed, 0.0), None
        for t in sorted(intervals):
            cutoff = min(t, train_end) if window is None else t
            hour = int(cutoff // 3600)
            if last_hour != hour:
                eligible = [(x, y) for end, x, y in observations if end <= cutoff and (window is None or end > cutoff - window * 3600)]
                if len(eligible) >= 24:
                    fitted = fit_bounded(eligible, bias)
                last_hour = hour
            states[t] = fitted
        snapshots[name], inputs[name] = states, heat
    # Exercise the production collector and adaptive model on the same prepared
    # intervals. Minute ticks hold each interval's measured means; they create no
    # extra coefficient updates or synthetic temperature interpolation.
    model = AdaptiveThermalModel(initial_heat_loss=seed[0], initial_heat_gain=seed[1])
    manager = LearningManager(model)
    states = {}
    for t, row in sorted(intervals.items()):
        manager.observe(t, row.indoor, row.outdoor, row.heat)
        states[t] = (model.heat_loss_coeff, model.heat_gain_coeff, model.background_gain)
        for elapsed in range(60, STEP, 60):
            manager.observe(t + elapsed, row.indoor, row.outdoor, row.heat)
        manager.observe(t + STEP, row.next_indoor, None, None)
    snapshots["adaptive_hourly"], inputs["adaptive_hourly"] = states, raw_heat
    return snapshots, inputs


def evaluate(intervals, snapshots, inputs, train_end, test_start, end):
    predictions = []
    for t in sorted(intervals):
        if t % 3600 or t < train_end:
            continue
        split = "validation" if t < test_start else "test"
        boundary = test_start if split == "validation" else end
        for hours in (1, 6, 12):
            if t + hours * 3600 > boundary:
                continue
            keys = list(range(t, t + hours * 3600, STEP))
            if not all(k in intervals for k in keys) or not all(t in v for v in snapshots.values()):
                continue
            actual = intervals[keys[-1]].next_indoor
            heating = sum(intervals[k].heat for k in keys) / 4
            for name, states in snapshots.items():
                loss, gain, bias = states[t]
                temp = intervals[t].indoor
                for k in keys:
                    temp += 0.25 * (loss * (intervals[k].outdoor - temp) + gain * inputs[name][k] + bias)
                predictions.append({"origin": iso(t), "split": split, "horizon_hours": hours, "model": name, "actual": actual, "predicted": temp, "error": temp - actual, "future_detected_heat_hours": heating})
    return predictions


def metrics(predictions):
    grouped = collections.defaultdict(list)
    for p in predictions:
        grouped[(p["split"], p["horizon_hours"], p["model"])].append(p["error"])
    return [{"split": s, "horizon_hours": h, "model": name, "n": len(errors), "mae": statistics.mean(abs(v) for v in errors), "rmse": math.sqrt(statistics.mean(v * v for v in errors)), "bias": statistics.mean(errors)} for (s, h, name), errors in sorted(grouped.items())]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_audit(groups):
    """Older hourly averages are audited separately, never labeled as measured duty."""
    maps = {key: {t: v for t, v in groups[key] if t % 3600 == 0 and v is not None} for key in (INDOOR, OUTDOOR, SUPPLY)}
    common = sorted(set.intersection(*(set(v) for v in maps.values())))
    monthly = collections.defaultdict(list)
    for t in common:
        monthly[iso(t)[:7]].append(t)
    return [{"month": month, "matched_hours": len(ts), "indoor_min": min(maps[INDOOR][t] for t in ts), "indoor_max": max(maps[INDOOR][t] for t in ts), "outdoor_min": min(maps[OUTDOOR][t] for t in ts), "outdoor_max": max(maps[OUTDOOR][t] for t in ts), "mean_supply": statistics.mean(maps[SUPPLY][t] for t in ts)} for month, ts in sorted(monthly.items())]


def diagnostic_tables(intervals, snapshots, predictions, train_end, test_start):
    coverage = []
    for name, lo, hi in [("training", -math.inf, train_end), ("validation", train_end, test_start), ("test", test_start, math.inf)]:
        rows = [r for t, r in intervals.items() if lo <= t < hi]
        coverage.append({"split": name, "valid_hours": len(rows) / 4, "detected_heating_hours": sum(r.heat for r in rows) / 4})
    daily = collections.defaultdict(list)
    for p in predictions:
        if p["split"] == "test" and p["horizon_hours"] == 6:
            daily[(p["origin"][:10], p["model"])].append(abs(p["error"]))
    daily_rows = [{"origin_day_utc": d, "model": m, "n": len(v), "mae": statistics.mean(v)} for (d, m), v in sorted(daily.items())]
    params = []
    for name, states in snapshots.items():
        if name == "persistence":
            continue
        values = [theta for t, theta in states.items() if t >= test_start and t % 3600 == 0]
        if values:
            params.append({"model": name, "hourly_snapshots": len(values), "loss_median": statistics.median(v[0] for v in values), "gain_median": statistics.median(v[1] for v in values), "gain_at_bound_fraction": statistics.mean(abs(v[1] - 0.1) < 1e-7 or abs(v[1] - 1.5) < 1e-7 for v in values), "background_median_c_per_hour": statistics.median(v[2] for v in values)})
    return coverage, daily_rows, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/history.csv")
    parser.add_argument("--attributes", type=Path, default=ROOT / "data/pilot_attribute_history.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/learning_analysis")
    parser.add_argument("--train-days", type=float, default=4)
    parser.add_argument("--validation-days", type=float, default=2)
    parser.add_argument("--max-hold-hours", type=float, default=6, help="Maximum causal hold for state-change records; sensitivity parameter, not proof of sensor freshness.")
    args = parser.parse_args()
    if min(args.train_days, args.validation_days, args.max_hold_hours) <= 0:
        parser.error("Durations must be positive")
    intervals, start, end, audit, history, groups = read_inputs(args.history, args.attributes, args.max_hold_hours * 3600)
    train_end = start + args.train_days * 86400
    test_start = train_end + args.validation_days * 86400
    if end - test_start < 12 * 3600:
        parser.error("Need at least 12 hours of test data after training and validation")
    snapshots, inputs = build_candidates(intervals, train_end)
    predictions = evaluate(intervals, snapshots, inputs, train_end, test_start, end)
    scores = metrics(predictions)
    selection = [s for s in scores if s["split"] == "validation" and s["horizon_hours"] == 6 and s["model"] not in ("recorded_model", "persistence", "adaptive_hourly")]
    if not selection:
        parser.error("No complete six-hour validation windows")
    selected = min(selection, key=lambda row: row["mae"])["model"]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metrics.csv", scores)
    write_csv(args.output / "predictions.csv", predictions)
    write_csv(args.output / "hourly_coverage.csv", historical_audit(groups))
    coefficient_rows = [{"time": iso(t), "model": name, "loss": theta[0], "gain": theta[1], "bias_c_per_hour": theta[2]} for name, states in snapshots.items() for t, theta in states.items() if t % 3600 == 0]
    write_csv(args.output / "coefficients.csv", coefficient_rows)
    write_csv(args.output / "intervals.csv", [{"time": iso(t), "indoor": r.indoor, "next_indoor": r.next_indoor, "outdoor_mean": r.outdoor, "detected_duty": r.heat} for t, r in sorted(intervals.items())])
    coverage, daily, parameter_summary = diagnostic_tables(intervals, snapshots, predictions, train_end, test_start)
    write_csv(args.output / "split_coverage.csv", coverage)
    write_csv(args.output / "daily_test_metrics.csv", daily)
    write_csv(args.output / "parameter_summary.csv", parameter_summary)
    audit["split"] = {"training_start": iso(start), "validation_start": iso(train_end), "test_start": iso(test_start), "end": iso(end), "selected_on_validation_6h_mae": selected}
    audit["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.history, args.attributes)}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Historical learning experiment", "", f"Sources: `{args.history.name}` and `{args.attributes.name}`. All dates UTC.", "", f"Training: {iso(start)} to {iso(train_end)}. Validation: to {iso(test_start)}. Test: to {iso(end)}.", "", f"Selected on validation six-hour MAE: **{selected}**.", "", "## Data", "", f"Valid recent coverage: {audit['recent']['valid_hours']:.2f} hours; excluded: {audit['recent']['excluded_hours']:.2f} hours. Detected heating: {audit['recent']['detected_heating_hours']:.2f} hours.", f"Recorded control intervals: median {audit['recent']['median_control_interval_minutes']:.2f} minutes; {audit['recent']['updates_under_five_minutes']} intervals below five minutes.", "", "The adaptive_hourly model was added after inspecting this dataset. Its results are retrospective and are excluded from the original candidate-selection procedure; new independent heating-season history is needed.", "", "## Prediction errors", "", "Errors in degrees C. The test period is later than the selection period.", "", "| Split | Horizon | Model | Origins | MAE | RMSE | Bias |", "|---|---:|---|---:|---:|---:|---:|"]
    for s in scores:
        lines.append(f"| {s['split']} | {s['horizon_hours']}h | {s['model']} | {s['n']} | {s['mae']:.3f} | {s['rmse']:.3f} | {s['bias']:.3f} |")
    lines += ["", "## Heating information by split", "", "| Split | Known hours | Detected heating hours |", "|---|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['split']} | {row['valid_hours']:.2f} | {row['detected_heating_hours']:.2f} |")
    lines += ["", "## Test-period parameter diagnostics", "", "| Model | Median loss | Median gain | Gain at bounds | Median background C/h |", "|---|---:|---:|---:|---:|"]
    for row in parameter_summary:
        lines.append(f"| {row['model']} | {row['loss_median']:.4f} | {row['gain_median']:.3f} | {100 * row['gain_at_bound_fraction']:.0f}% | {row['background_median_c_per_hour']:.3f} |")
    lines += ["", "## Interpretation and limits", "", "- This is conditional temperature-model validation: all models receive the same observed future outdoor temperatures and detected heating. It does not measure weather forecasting, actuator prediction, savings, or performance of an alternative closed-loop controller.", "- Parameters are frozen at each forecast origin. Online learners may incorporate test observations only after those observations occur. Frozen regression fits only the training period. Candidate selection uses validation data only, not test scores.", "- Origins are hourly and overlap; counts are not independent experiments. Windows crossing missing intervals or split boundaries are excluded. All compared models use matched origins at each horizon.", "- Raw sensor values are held causally, never interpolated from future readings. Unchanged readings may be normal state-change exports, but a hold is not proof of freshness. Repeat with --max-hold-hours to assess this assumption. Unknown intervals are excluded completely.", "- The recent heating signal is the integration's supply-threshold detector, not measured delivered heat. Supply-pipe location and domestic-hot-water effects remain unconfirmed. The older hourly supply averages cannot reconstruct heating duty; older history is audited in hourly_coverage.csv rather than mislabeled as measured on/off data.", "- recorded_model uses coefficients as recorded at each origin, including deployed-version behavior and resets. ekf_15m/60m and rls_60m call the repository estimators on aligned observations, seeded from the first interval's recorded coefficients (or defaults if absent). They are controlled offline comparisons, not an exact reconstruction of the deployed event loop or covariance.", "- Hourly EKF/RLS aggregate four known 15-minute intervals. Slow EKF scales temperature process noise to an hour and reduces parameter diffusion relative to elapsed-time-scaled defaults. It is an experiment, not a tuned production setting.", "- Regression fits hourly temperature changes under the existing loss/gain bounds. rolling_72h refits on the preceding 72 hours; bias adds an unmeasured constant gain/loss term; lag models filter measured heating over 1 or 3 hours. These terms are phenomenological and do not identify physical building properties without reliable heat measurements.", "- A strong persistence baseline (temperature stays unchanged) would indicate that low errors alone do not establish useful heating-response learning. Few heating hours especially limit conclusions about gain and winter performance.", "- Source files and the live integration are unchanged. Experiments are deliberately isolated in tools/analyze_learning.py.", "", "## Reproduce", "", "```powershell", ".venv/Scripts/python.exe -B tools/analyze_learning.py", "```", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": selected, "coverage": audit["recent"], "test_6h": [s for s in scores if s["split"] == "test" and s["horizon_hours"] == 6], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
