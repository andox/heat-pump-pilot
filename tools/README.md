# Offline learning analysis

Run from the repository root with Python 3.10 or later. Only the standard library
and the integration's existing estimator classes are used. No HA login is needed.

```powershell
.venv/Scripts/python.exe -B tools/analyze_learning.py
```

The defaults read `data/history.csv` and `data/pilot_attribute_history.json` and
write `data/learning_analysis/`. Use `--history`, `--attributes`, and `--output`
to change these paths. The raw files and integration are never modified.

The script expects the entity IDs defined near its top. The CSV is the Home
Assistant History export (`entity_id,state,last_changed,...`). The JSON is the
WebSocket history response, with entity IDs mapped to records containing `s`, `a`,
and `lu`/`lc`. Attribute-only changes use `lu` (last updated).

The input supplied for this experiment changes from hourly statistics to raw
state changes around September 2, 2026. The script infers the raw portion from
the first timestamp not exactly on an hour for each temperature entity. **Check
`audit.json` when using a different export**: this inference is specific to that
export convention, not a universal HA guarantee. Older hourly averages are
audited separately and are not converted into heating duty.

## Evaluation

- 15-minute intervals use causal held indoor readings, time-weighted outdoor
  readings, and time-weighted supply-based heating detection.
- Missing/unknown intervals and temperature readings outside 10–35 C are excluded.
  No interpolation uses future readings. Holds default to six hours because the
  export records changes rather than a regular heartbeat; this does not establish
  sensor freshness. `--max-hold-hours` controls this assumption.
- The first four days are training, the next two validation, and the remaining
  days test. Change these with `--train-days` and `--validation-days`.
- Candidate selection minimizes six-hour validation MAE. At each hourly forecast
  origin, parameters fitted using only available past observations are frozen for
  1/6/12-hour forecasts. Online models continue learning as observations arrive in
  the test period. Only the frozen-regression baseline stops learning entirely.
- Models receive actual future outdoor temperature and detected heating, equally.
  This evaluates the thermal model conditional on observed inputs, not a live MPC
  controller, future weather forecasts, or financial savings.
- All models share the same eligible origins at a given horizon. Horizons cannot
  cross gaps or validation/test boundaries. Overlapping origins are correlated.
- Eleven models include recorded coefficients, temperature persistence, current
  EKF at fixed 15/60-minute intervals, slower parameter diffusion, hourly RLS,
  frozen/rolling regression, a background term, and heating-release lags.
- Regression reuses the integration's loss/gain bounds. The extra background term
  is limited to -0.5 to +0.5 C/hour. It represents unexplained net heating/cooling,
  not an identified physical source. A parameter on a bound is not convergence.

To assess missing-data sensitivity without changing model settings:

```powershell
.venv/Scripts/python.exe -B tools/analyze_learning.py --max-hold-hours 3 --output data/learning_analysis_hold3h
.venv/Scripts/python.exe -B tools/analyze_learning.py --max-hold-hours 12 --output data/learning_analysis_hold12h
```

These runs use different eligible windows; compare models within each run. Do
not select a new winning model based on test scores. More independent history is
needed before promoting a candidate to live control.

## Outputs

`report.md` contains metrics and methodology. `audit.json` includes source hashes,
coverage and splits. `metrics.csv` and `predictions.csv` contain aggregate errors
and each forecast. `intervals.csv` contains prepared observations, and
`coefficients.csv` contains hourly coefficient snapshots. Split coverage,
daily test errors and parameter-bound diagnostics are provided in separate CSVs.

`data/learning_analysis/findings.md` is the interpretation of the supplied
September 2026 dataset, not an automatically updated result for future exports.

To add an experiment, register it in `build_candidates()`. Return parameter
snapshots `[loss, gain, background]` and its causal heating-input series. Any
parameter snapshot at time t must depend only on observations available by t.
The tests include a check that changing future data cannot affect past fits.

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_learning_analysis.py
```
