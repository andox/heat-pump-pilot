# Heat Pump Pilot

Heat Pump Pilot is a custom Home Assistant integration that runs a lightweight
MPC (model predictive control) loop to steer a heat pump through a "virtual
outdoor temperature" setpoint. It balances comfort vs electricity price and
continuously learns a simple thermal model of your home.

Instead of directly switching the compressor, Heat Pump Pilot writes a virtual
outdoor temperature to a `number.*` entity. For ground source heat pumps, this
virtual outdoor temperature shifts the heating curve (outdoor temperature to
supply temperature), and the pump's own integral-minutes logic decides when to
start or stop. Lower virtual outdoor temperatures request higher supply
temperatures (more heating), while higher virtual temperatures back off
heating.

Repository: https://github.com/andox/heat-pump-pilot

## Features
- MPC optimizer with price vs comfort weighting and optional learned pump delay.
- Price-aware preheating and coasting within a configurable comfort band.
- Optional independent UFH circulation control and daily pump exercise.
- Hourly adaptive house learning with heat loss, heating gain and background warmth; legacy EKF/RLS remain available.
- Optional heating detection via supply/flow temperature sensor.
- Diagnostic sensors for decisions, health, learning state, price state, and scores.
- Comfort score, price score, and prediction accuracy metrics.
- Learning, price history, and performance history persist across restarts.

## Installation

### HACS (recommended)
1. Add this repository (`https://github.com/andox/heat-pump-pilot`) as a custom HACS integration.
2. Install "Heat Pump Pilot".
3. Restart Home Assistant.

### Manual
1. Copy `custom_components/heat_pump_pilot` into your Home Assistant
   `config/custom_components/` directory.
2. Restart Home Assistant.

## Configuration (UI)
Add the integration via **Settings > Devices & Services > Add Integration**.

Required entities:
- Indoor temperature sensor (`sensor.*`).
- Outdoor temperature sensor (`sensor.*`).
- Price sensor (`sensor.*`).
- Weather forecast entity (`weather.*`); current outdoor readings are the fallback if its forecast is unavailable.

Optional entities:
- Controlled entity (the output target you want the integration to drive).
- Heating supply/flow temperature sensor for heating detection.

### Controlled entity types
Heat Pump Pilot will attempt to control different entity types:
- `number`: best for "virtual outdoor temperature" setpoints. Uses `number.set_value`.
- `switch`: on/off control.
- `climate`: sets target temperature and `hvac_mode`.
- Other domains: tries `set_temperature` or `turn_on/turn_off` if available.

### Monitor only mode
When **monitor_only** is enabled, Heat Pump Pilot will *not* call any control
services. It still computes decisions, forecasts, and diagnostics, but the heat
signal used for learning must come from a supply temperature sensor or a
controlled entity state (switch on / climate hvac_action == heating). If no
reliable heat signal exists, learning is disabled.

Everyday comfort settings appear first, followed by **Sensors and heat pump**.
Advanced settings are in collapsed groups at the bottom: control, prices,
learning, detection, initial estimates, diagnostics and optional UFH control/exercise.
Existing options retain their original storage keys.

## Configuration options and defaults
Below are the main options exposed in the config/option flows, with defaults and
recommended values when you’re unsure. Values are in the UI unless noted.

Core control:
- Target temperature (default: 21.0°C): your comfort setpoint; set to your normal desired indoor temp.
- Price priority (default: 0.5): lower values favor staying near target; higher values favor cheaper heating and coasting inside the comfort band. Band violations are ranked first, even at 1.0.
- Price penalty curve (default: linear): shapes the excess price ratio above baseline; see the formulas below for linear, sqrt and quadratic.
- Price baseline window (default: 24 h): how much timestamped observed history
  is used alongside known future prices when scaling prices (24/48/72 h).
  The median includes zero and negative prices; the normalization baseline has
  a positive floor of 0.01 in the configured price units.
- Absolute low-price threshold (default: auto): cap classification at `normal` when the current price is below the threshold (`auto` = median of recent history, `off` disables the cap).
- Absolute low-price auto window (default: 30 d): window used for the `auto` threshold (7/14/30 days).

Price forecasts with timestamps are sampled at the actual MPC timestamps,
including 15-minute, hourly and daylight-saving transitions. Missing periods
stay missing in baseline calculations. For planning only, Pilot holds the last
known price (or the current sensor price before the first known interval).
The decision sensor exposes known/total forecast steps and the fallback policy.
Untimed legacy `prices`/`forecast` lists use an hourly convention; timestamped
data is preferred because its interval is unambiguous.

Price history stores dated 15-minute buckets and excludes data outside the
requested time window. Recorder backfill holds each valid state for at most
one hour, ending earlier at the next state change, including unavailable states.
Older saved history without timestamps is rebuilt from Recorder after an update;
until then, the baseline uses available forecast/new observations. This does not
reset learned house coefficients.

Further control options:
- Continuous control enabled (default: true): plans request strength and actual virtual-temperature commands directly for number outputs at the normal 15-minute interval.
- Continuous control window (legacy binary modes only, default: 2 h): duty averaging horizon (1–4 h). Hidden when the command-aware planner is selected; learned and unlearned delivery models both choose requests directly.
- Summer low-price heat window (default: off): optionally schedules one daily
  continuous heat window during low-demand periods when every price sample in
  the window is at or below the configured absolute max price.
- Summer heat window max price (default: 0.30): absolute price cap for the
  whole summer heat window.
- Summer heat window duration (default: 60 min): required continuous duration;
  the run is never split across multiple low-price periods.
- Summer heat demand history window/max recent demand ratio (default: 48 h / 10%): automatic
  season gate based on recent normal MPC heat-request ratio, without calendar
  dates.
- Summer heat virtual outdoor offset (default: same as normal virtual
  outdoor heat offset): virtual outdoor reduction used only when the summer
  window overrides idle.
- Control interval (default: 15 min): how often MPC runs; keep 15-30 min unless you have slow sensors.
- Prediction horizon (default: 24 h): MPC planning horizon; 12-24 h is typical.
- Comfort tolerance (default: 1.0°C): predicted allowed range is target ± tolerance. A wider band gives MPC more room to preheat and coast; the target remains preferred.
- Monitor only (default: false): true to disable control actions.

Virtual outdoor control:
- Virtual outdoor heat offset (default: 10.0°C): max shift colder when heating and warmer when backing off; start at 6–12°C.
- Virtual outdoor minimum (default: -15.0°C): never send a lower virtual outdoor temperature unless the actual outdoor temperature is already below this.
- Separate overshoot warm-bias controls are retired. Saved values remain compatible, but do not alter the new planner or live output.
- Virtual outdoor smoothing enabled (default: true): apply EMA smoothing to the output temperature.
- Output responsiveness / smoothing alpha (default: 0.5): 0–1 per normal control interval; lower is smoother/slower, higher is more responsive. Existing alpha settings retain their meaning at the normal cadence.

Learning:
- Learning model (new-install default: adaptive): jointly learns loss, heating gain and background warmth. Existing explicit EKF/RLS selections are retained.
- Learning interval (default: 60 min, range: 30–120): independent of the control interval.
- House learning history (default: 72 h, range: 24–168): rolling adaptive fitting window.
- Learn delayed pump response (default: enabled): learns from actual number states and measured heating. Planning requires validation, identifiable house gain, continuous control and a 15-minute control interval.
- RLS forgetting factor (default: 0.99): lower = faster adaptation; 0.97–0.995 is typical.
- Learning window (default: 12 h): legacy EKF/RLS stability diagnostic, separate from the adaptive fitting window.
- Thermal response seed (default: 0.5): initial guess for loss/gain; leave default unless you know your system.
- Base heat loss coefficient (default: 0.05): used until learning refines it; leave default in most cases.
- Initial indoor temp (default: unset): optional override; leave empty to use sensor value.
- Initial heat gain coefficient (default: unset): optional; leave empty unless you have a known value.
- Initial heat loss override (default: unset): optional; leave empty unless you have a known value.

Heating detection:
- Heating supply/flow temp entity (default: unset): optional but strongly recommended for better learning.
- Heating detection enabled (default: true): active only when a supply/flow sensor is configured.
- Supply temp threshold (default: 30°C): set to your pump’s “heating on” supply threshold.
- Supply temp hysteresis (default: 1.0°C): helps avoid chatter.
- Supply temp debounce (default: 60 s): helps avoid false positives on short spikes.
- Learning supply temp on/off margins (default: 1.0°C): extra buffer around the threshold for learning.

Performance metrics:
- Performance window (default: 24 h): 6/12/24/48/72/96 hours; 24–48 h is a good balance.

## Planning comfort, preheating and coasting

The existing target and comfort tolerance define a symmetric planning band:
`lower = target - tolerance`, `upper = target + tolerance`. For target 20.5 C
and tolerance 1.2 C, the allowed predicted range is 19.3 to 21.7 C. A larger
range permits both deeper coasting and more preheating; it does not require
heating to the upper limit.

MPC first minimizes predicted temperature excursions outside that band, then
uses **Price vs Comfort** to choose between equally feasible plans. It still
runs at the existing control cadence. A high price weight favors cheaper heating
and coasting inside the band; a low weight prefers staying close to target.
0.5 weights the normalized terms equally, not equal real-world discomfort and
currency. High prices alone cannot buy a breach when the search finds a plan
that stays inside the band. This changes the meaning of tolerance from a soft
penalty deadband to a planning boundary; existing values are not changed.

Normal virtual-temperature control includes the actual output bounds, rounding,
EMA smoothing and, when validated, the learned delivery delay/residual heating.
Until learning is ready, the same planner uses an explicitly unlearned immediate
proportional delivery estimate. This fallback cannot know an old pump's integral;
its predicted comfort band is not a guarantee of actual room temperature.

Backoff and preheating are outcomes of the same optimization. There is no extra
warm-bias rule or request hold/ramp rewriting the command-aware planner's first
action. A small command-change cost discourages needless reversals, and elapsed-
time output smoothing remains. Legacy nonstandard binary modes retain their
request limiter. Independent summer heating remains an explicit override.

The diagnostic fields `planning_model` (`learned`, `fallback`, or `binary`),
`planned_comfort_status`, `planned_comfort_violation_degree_hours`, and
`comfort_lower_bound` / `comfort_upper_bound` expose the result. A
`predicted_breach` means the selected plan crosses a boundary, including passive
warming above it or insufficient predicted capacity. The bounded search does
not prove global physical infeasibility; it reports a forecast, not a guarantee.

See [the design and behavioral test contracts](docs/comfort_planning.md).

## Virtual outdoor temperature

For request strength `u` from 0 to 1, the mapping is
`virtual = outdoor + offset * (1 - 2*u)`: zero requests maximum warm backoff,
one requests maximum heating, and 0.5 is neutral. The virtual-temperature
minimum, 25 C maximum, and controlled entity bounds/rounding still apply.
The planner sends its exact first predicted command. It learns from measured
heating, never assuming that requested idle instantly stops the compressor.

Smoothing uses elapsed time:
`effective_alpha = 1 - (1 - alpha) ** (elapsed / control_interval)`.
Extra sensor updates do not accelerate smoothing. Alpha 1 disables output
smoothing; the enable switch is retained for existing configurations. No new
user settings are introduced, and Price vs Comfort remains available.

### Summer low-price heat window
The optional summer heat window is intended for periods where normal space
heating demand is already low, but you still want one low-price daily heat pulse to
put warmth into a hydronic floor. It does not use calendar dates. Instead, it
checks the recent normal MPC heat-request ratio; with the default settings it is
eligible only when the last 48 hours requested heat no more than 10% of the
time.

When eligible, the controller searches the remaining local day for one
continuous price block where every sample is at or below the configured max
price. The block must cover the configured duration, defaults to 60 minutes,
and if several blocks qualify the cheapest average block is selected. If no
qualifying continuous block exists in the currently available price forecast,
the feature reports `skipped` but re-checks on later control runs. This handles
Nordpool-style next-day prices arriving later in the day. It never splits one
daily run across multiple sessions, and once a window has started or completed
it will not schedule another run for the same local day.

During the selected window, normal MPC still runs first. If the MPC already
requests heat, the window is counted without adding another behaviour. If the
MPC is idle, the summer window overrides idle by lowering the virtual outdoor
temperature using the dedicated **Summer heat virtual outdoor offset**,
which requests more heat from the heat pump. This lets the summer heat pulse use
a stronger trigger than normal MPC heating if the pump needs a colder virtual
outdoor value to start floor heating. The result is still constrained by the
global virtual outdoor minimum. In monitor-only mode, the schedule and
diagnostics are still computed, but no control service is called.

### Creating an outdoor temperature sensor from a weather entity
If you only have a `weather.*` entity, create a template sensor:

```yaml
template:
  - sensor:
      - name: "Outdoor Temperature"
        unit_of_measurement: "°C"
        device_class: temperature
        state_class: measurement
        availability: "{{ is_number(state_attr('weather.home', 'temperature')) }}"
        state: "{{ state_attr('weather.home', 'temperature') }}"
```

Use this sensor as the outdoor temperature entity, and use the `weather.home`
entity for forecasts.

## Learning (thermal model)
The default control interval remains **15 minutes**, with existing sensor-triggered
runs. A separate one-minute observation timer and sensor events collect evidence;
they do not run MPC. Coefficients update only after a complete learning interval,
normally **60 minutes**.

Intervals pair indoor temperature change with time-weighted outdoor temperature
and heating during the same period. At least 95% coverage is required. Unknown
states, runtime gaps over five minutes and temperature jumps above 2°C/hour are
excluded. Partial intervals are discarded after restart. Unchanged readings are
held. The separate 24-hour sensor-input warning checks reporting timestamps,
not whether a value changes (see Sensors and diagnostics).

All models need measured heating: supply/flow temperature with threshold,
hysteresis, debounce and on/off margins, or the controlled switch/climate state.
A virtual outdoor request is **not** evidence of heating. With a `number.*`
output, configure a heating detection sensor. Without a usable signal, the
coefficients stay unchanged. Supply temperature remains a proxy affected by
sensor placement and domestic-hot-water cycles.

### Adaptive house model

```text
indoor change per hour = loss × (outdoor − indoor) + gain × measured heating + background
```

The three terms are fitted together. Background represents unexplained net
warmth/cooling, such as solar or household gains, and is included in MPC
forecasts. It is not a separately measured physical source.

Fitting needs 24 usable hours and changing indoor/outdoor temperature differences.
Gain remains frozen unless there are three equivalent heating hours, three
coasting hours, and heating variation independent of outdoor temperature. Summer
history therefore cannot reliably identify winter heating capacity. Existing
loss/gain bounds (0.001–0.25 and 0.1–1.5) remain; background is bounded at
±0.5°C/hour. Parameter changes are also limited per elapsed hour.

For an existing installation, select **Options → Advanced: learning → Learning
model → adaptive**. Learned loss/gain values carry over; changing an initial
estimate explicitly reseeds them. Background starts at zero when entering
adaptive mode. New installations default to adaptive. Sensor or detector changes
discard the associated observation history.

### Delayed pump response

A separate model learns request strength, idle heating, delay (0–120 minutes)
and response smoothing from complete 15-minute averages of the actual controlled
number state and the debounced measured heating detector. Unlike house learning,
it does not require indoor temperature or apply the extra supply-temperature
learning margins. This preserves measured start/stop transitions during short
heating cycles and prevents sunshine at the indoor sensor from interrupting pump
learning. Missing pump, outdoor or request observations still fail coverage
checks; gaps are never filled with invented heating.

Initial fitting requires at least 28 hours of retained observations, with enough
continuous stretches, heating, coasting and varying requests. It fits at most
hourly on up to 288 quarter-hour observations; gaps split those observations
into separate stretches.

The same learner also tests a small outdoor-temperature correction, without an
additional configuration setting. Its target heating fraction is
`clip(idle + request_gain * delayed_request + outdoor_gain * (reference - outdoor) / 10, 0, 1)`.
The reference is the training period's mean outdoor temperature. The additional
coefficient is constrained to 0–0.5 heating fraction per 10°C colder; it cannot
make colder weather reduce heating or reverse the effect of a stronger request.
This models the observed effect of the heat pump's curve, not its exact curve
settings or measured thermal output. Outdoor temperature acts on the current
interval's response target; request delay and response smoothing remain separate.

The outdoor correction requires complete outdoor history, at least 4°C variation
in training, and at least 1°C residual standard deviation after accounting for
request strength at every candidate delay. It must improve held-out heating MAE
by both 10% and 0.01 heating fraction compared with the existing delayed model,
as well as pass the baseline checks below. Otherwise the existing delayed model
remains the candidate. Forecast outdoor temperatures are clamped to the training
range for this correction, so mild-weather observations are not extrapolated
into unobserved winter conditions. The ordinary house heat-loss calculation
still uses the actual forecast temperature.

Saved pump observations now include their interval-average outdoor temperature.
Older observations are retained for the simpler fit with outdoor marked unknown;
they are not assigned invented temperatures. Restarts retain observations and
fitted coefficients. Gaps split the evidence into separate stretches: fitting,
lag selection and validation never bridge unknown time. A new completed interval
triggers revalidation against the retained evidence. MPC additionally waits for
fresh measured heating and request history covering the learned delay (at least
15 minutes, up to two hours); restarting does not require rebuilding the entire
28-hour training history. Recent operating state and partial intervals are not
restored as if the pump had been observed while Home Assistant was offline. The decision sensor's `pump_response` attributes expose
`outdoor_active`, `outdoor_reason`, `outdoor_gain_per_10c`, `outdoor_range_c` and
both candidate validation errors. `outdoor_active` means the learner accepted
the correction; `pump_response_planning_active` separately indicates whether
the latest MPC plan actually used response learning.

Delay is selected on training observations. The following 16 observations must show
at least 10% lower heating prediction error than both direct-request and
constant-duty baselines. When the baseline error is below 0.02 heating fraction,
the period cannot validate a new response. An already validated response is
retained unchanged if its error on those observations is also at most 0.02. When heating resumes, the saved response can remain usable if it beats the same recent-validation baselines by at least 10%, even before the training window has enough heating to fit a replacement. Contradictory observations revoke that trust.
This allows long coasting periods without discarding a useful delay model.
Incompatible observations suspend that trust. The validation marker is saved
with the coefficients; restarts still require fresh measured state, and older
snapshots without the marker must pass normal validation first.

Restarts and gaps suspend live state until enough fresh evidence is available.
The command-aware planner uses the immediate proportional fallback when response
planning is unavailable. When using the adaptive house model,
its heating gain must also be identifiable before response planning activates.

When eligible, MPC chooses 0/25/50/75/100% requested duty and simulates delayed
heating, output limits and smoothing. The price penalty follows predicted
heating, including heating that continues after a request stops. The first duty
is applied directly; forecasts are replayed after the current output is limited.
Future hysteresis and unexpected sensor-triggered decisions are approximations;
subsequent control runs replan using observations. Optimization runs outside
Home Assistant's event loop.

Heating runtime is a price-cost proxy. This does not measure electrical power,
COP or guaranteed electricity savings.

### Legacy EKF and RLS

Existing selections remain supported with the same complete learning intervals.
EKF updates temperature/loss/gain covariance; RLS estimates loss/gain with its
forgetting factor. Neither adds background warmth. Noise/forgetting settings
apply per coefficient update, so hourly updates adapt more slowly than repeated
updates on every sensor-triggered control run.

### Learning diagnostics

Climate and Decision attributes expose `learning_details`, `pump_response` and
`pump_response_planning_active`. Climate also exposes `estimated_background_gain`;
Decision exposes planned duty and predicted heating. Adaptive `fit_status`
distinguishes insufficient history, insufficient weather variation, frozen gain
and learning. Frozen coefficients are not evidence of convergence. Legacy models
retain their rolling 5% loss/gain stability indicator.

See [the offline replay tool](tools/README.md) for comparisons using exported
Home Assistant history.

## Price baseline and classification
MPC and price classification use the same normalization baseline:
`max(0.01, median(known forecast prices + timestamped observed history))`.
The history and forecast portions are bounded by the configured 24/48/72-hour
baseline window. Zero and negative prices participate in the median; missing
prices are excluded. The positive floor is in the configured price units.

The **Absolute low-price threshold** caps classification at `normal`; it does
not force heating. `auto` uses the median of available history within the
configured 7/14/30-day window, including zero and negative prices. `off` disables
the cap. It is unavailable until there is usable history, then uses what has
actually been observed without pretending the whole window is covered.

MPC prices below baseline retain their price ratio. Above baseline, set
`x = min(ratio, 3) - 1`; the shaped value is `1 + x` (linear), `1 + sqrt(x)`
or `1 + x²` (quadratic). The curves shape only the excess above baseline;
`sqrt` is stronger than linear for excesses below one and gentler above one,
while quadratic does the reverse. Price weight multiplies this term, after
predicted comfort-band violations have been ranked. The planning cost is not a
metered currency estimate.

Classification uses `ratio = current_price / baseline` with labels:
- `< 0.75` -> `very_low`
- `< 0.90` -> `low`
- `< 1.10` -> `normal`
- `< 1.30` -> `high`
- `< 1.60` -> `very_high`
- `>= 1.60` -> `extreme`

## Performance metrics

The configurable performance window is 6/12/24/48/72/96 hours. Observations
are weighted by elapsed time; long gaps remain uncovered.

- **Comfort score**: percentage of covered time within target ± tolerance,
  with separate too-cold/too-warm percentages and degree-hours outside the band.
- **Price score**: detected heating timing compared with the cheapest and most
  expensive allocations of the same heating duration. 50 means heating at the
  average available price; no heating or no meaningful price comparison gives
  an unknown score with a reason. It does not measure electricity saved.
- **Prediction accuracy**: indoor-temperature MAE, with RMSE, bias and maximum
  error. It compares observations against prior plans, not a completed 24-hour
  forecast backtest. Positive bias means actual temperature exceeded prediction.

Performance samples survive restarts. Inspect coverage before interpreting a
score; short heating cycles can be missed between control-loop observations.

## Early-run behavior (history builds)
Some features depend on stored history and will be less informative on day one:
- **Absolute low-price threshold (auto)**: uses the available portion of its configured 7/14/30-day history window.
- **Price/comfort/accuracy scores**: need enough performance samples to be meaningful.
- **House learning**: requires usable completed intervals and temperature variation; heating gain also needs measured heating and coasting.
- **Pump response / curve recommendation**: require relevant heating/request evidence.
Elapsed days alone do not prove convergence or qualify a learned model for MPC.

## Persistence and restart behavior
The integration stores its state in `.storage`:
- `heat_pump_pilot_<entry_id>_thermal.json` (learning model + history)
- `heat_pump_pilot_<entry_id>_prices.json` (price history, up to ~30 days)
- `heat_pump_pilot_<entry_id>_performance.json` (performance samples)
- `heat_pump_pilot_<entry_id>_summer_heat_window.json` (selected daily heat window)

This means learning, price baselines, performance scores, and the selected daily
summer heat window survive restarts.

## Sensors and diagnostics
The existing **Sensor inputs** persistent notification warns when indoor,
outdoor or heating-detection supply temperature has not reported for more than
24 hours. It uses Home Assistant's `last_reported` timestamp (falling back to
`last_updated` on older versions), so repeated reports of an unchanged temperature
remain fresh. Missing or unavailable inputs are also reported; issues must persist
for five minutes, after a two-minute startup grace, before a notification appears.
The warning clears when valid reports resume. This notification threshold does
not change the separate, configurable UFH pump safety freshness limit.

Key diagnostic sensors:
- Heat Pump Pilot Decision: last applied request and forecasts. Use
  `effective_requested_duty_ratio` and `effective_heat_request_state` for the
  request, `heating_detected` for measured heating, and `planning_model`,
  `planned_comfort_status`, `comfort_lower_bound` / `comfort_upper_bound` and
  `planned_comfort_violation_degree_hours` for the plan. `predicted_heating`
  differs from `planned_requested_duty` when delivery is delayed.
  `pump_response_planning_active` means the latest plan used validated response
  learning; collecting observations alone does not make it active.
  Legacy warm-bias/anti-chatter attributes remain for compatibility and are
  not useful overlays for normal command-aware planning.
  Price, outdoor and plan series are capped at 192 entries for Recorder.
- Heat Pump Pilot Health: overall health with reasons (missing sensors, stale control, etc).
- Curve recommendation (Health attribute): suggests when to raise/lower the heat pump curve
  based on heating detected during low requested heat vs active requested heat over the
  performance window. In continuous mode this uses requested duty ratio (not only on/off),
  so small non-zero preheating is not treated as true idle.
- Heat Pump Pilot Control State: whether the integration is controlling or monitoring.
- Heat Pump Pilot Learning State: adaptive fitting status, gain identification, interval coverage and fit error. Legacy EKF/RLS also expose change ratios; frozen gain is not convergence.
- Heat Pump Pilot Price State: current price classification and baseline details.
- Heat Pump Pilot Virtual Outdoor: the current virtual outdoor temperature sent to the pump.
- Heat Pump Pilot Virtual Outdoor Trace: rolling history of recent virtual outdoor decisions
  (see the `trace` attribute for detailed entries).
- Heat Pump Pilot Comfort Score: percent of covered time within target ± comfort
  tolerance. Attributes separate `too_cold_pct` and `too_warm_pct`, and quantify
  discomfort beyond the band in cold/warm degree-hours. A warm house can score
  poorly even with heating off; this measures comfort, not who caused the error.
- Heat Pump Pilot Price Score: heating timing on a 0–100 scale. 50 means heating
  at the average observed price; 100 means the cheapest possible allocation of
  the same heating duration, and 0 the most expensive. Unknown heating periods
  are excluded. No heating, no idle comparison, or constant prices produce an
  unknown score with an explanatory `reason` attribute.
- Heat Pump Pilot Prediction Accuracy: MAE plus RMSE/bias for predicted indoor temperature.
- Heat Pump Pilot Heating Detected (binary): debounced heating detection from supply sensor.

Comfort and price scores weight observations by elapsed time, capped at one
configured control interval per observation. Additional updates do not receive
extra weight, and long gaps are not filled. `covered_hours` and `coverage_pct`
show how much of the selected performance window was usable. They remain
estimates from control-loop observations, so short heating cycles can be missed.
Comfort uses each observation's target and the current configured tolerance.

The price score is a timing diagnostic, **not measured electricity or money
saved**. It assumes equal heating power and compares theoretical schedules that
may not satisfy the house's thermal constraints. The
`price_advantage_per_kwh_proxy` attribute is average available price minus
average price during detected heating; it is not a metered saving. Price scores
from this method (`equal_runtime_price_opportunity_v2`) are not directly
comparable to older scores based on the minimum/maximum price alone.

## Dashboard examples

The examples use placeholder entity IDs: replace indoor/outdoor, electricity
price, supply-temperature and optional UFH switch IDs with your own. Pilot
entity IDs may also differ when you have renamed entities or use several instances.

- [ApexCharts overview](docs/dashboard_overview.yaml): current temperatures,
  predicted indoor/virtual/outdoor values, historical price baseline and future
  published prices. It follows the refreshed screenshot, with thin strokes,
  tooltips, a line for now without a label, and small activity lanes.
  Heating/request, absolute-low-price and supply-temperature series start hidden;
  click their legends to inspect them. Optional UFH lanes and summer-window
  markers are included; remove the UFH/supply series if you do not use them.
- [Pilot status and learning](docs/dashboard_details.yaml): a compact built-in
  card stack for current scores, adaptive coefficients, evidence and plan status.
  It needs no custom frontend card.

Install [ApexCharts Card](https://github.com/RomRider/apexcharts-card) for the
first example, then paste its YAML into a manual dashboard card. The price
forecast series expects Nordpool-style `raw_today` / `raw_tomorrow` entries with
`start`, `end` and `value`; remove that series for other price providers.
The actual price and baseline history still work without those attributes.
Forecast arrays use Pilot's 15-minute MPC grid, not the sensor update interval.

Solid traces show observations; dashed traces show the latest plan/forecast.
The baseline traces its historical values instead of projecting today's value
back over yesterday. Requested heating is a request percentage, not measured
compressor power. Activity lanes use separate hidden axes and do not expand the
temperature scale. The simple details example replaces the old duplicate score,
curve-recommendation and legacy warm-bias panels; detailed curve advice remains
available on the Health sensor when needed.

## Screenshots
<table>
  <tr>
    <td align="center">
      <img src="screenshots/chart_example_1.png" alt="Pilot overview with forecast, prices and compact activity lanes" width="420">
    </td>
    <td align="center">
      <img src="screenshots/sensor_1.png" alt="Comfort, heating-price and prediction evidence" width="420">
    </td>
  </tr>
</table>

## Tests

Tests cover adaptive learning, pump-response fitting and restart restoration,
command smoothing, preheating/coasting, price forecasts, performance scores,
sensor freshness and optional UFH control. Run the full suite with Python 3.12
(the CI version) or later:

```bash
python -m pip install pytest voluptuous
python -m pytest -q tests
```

## Notes and limitations
- In monitor-only mode without a reliable heat signal, learning is disabled.
- Heating detection via supply/flow sensor improves learning quality and speed.
- A weather entity is required during setup. A usable forecast can improve planning;
  if unavailable, Pilot falls back to current outdoor temperature.


## Optional UFH circulation pump control

In **Configure → Advanced: UFH circulation pumps**, enable the feature, select
its own heat-pump supply temperature sensor, and select one or more pump switches.
It is disabled by default. Select individual switches to track each pump's runtime
and idle period separately. Disable the old UFH automation before enabling this
controller; do not let both own the same switches.

Circulation follows **measured supply temperature only**. Neither an MPC request,
Pilot's HVAC mode, electricity prices, nor Summer Heating authorizes or blocks
circulation. The feature continues when Pilot HVAC is off because the heat pump
can produce heat independently. The integration's **Monitor only** setting still
suppresses all UFH commands.

Defaults are 30°C to start, 25°C to stop, 30 seconds continuously hot before
starting, 60 minutes minimum heating runtime, 5 minutes minimum off time, and
30 minutes continuously cold before stopping. A rise above the stop threshold
restarts the cold timer; a drop below the start threshold restarts the hot timer.
These defaults are configurable and must suit the installed circulation system.

**Advanced: UFH pump exercise** has a separate enable switch (off by default),
local daily time (13:00), duration (15 minutes), and minimum idle period (24 hours).
It does not depend on Summer Heating. Each eligible pump is exercised at most once
per local day; a missed exercise time is not replayed after downtime. Exercise
can end before the normal minimum heating runtime if the supply is cold. If the
supply warms, the pump continues under temperature control instead of blindly
switching off when the exercise timer expires.

Pump transition times and exercise state are persisted in `.storage` and restored
after restart. An unavailable switch returning to the same state retains its
exercise idle history, including during startup. A confirmed state change resets
that history; unobserved activity during downtime cannot be inferred. Minimum
on/off safeguards restart after switch unavailability independently of the
exercise idle timer. UFH diagnostics expose persisted `exercise_idle_since`,
`state_since`, `exercise_day` and `exercise_until` for dashboard use.
Short temperature qualification timers restart: downtime is not
proof that the supply stayed hot or cold. An unknown switch is not commanded;
unknown, restored, invalid or stale temperature holds current pump states. The
configurable stale limit uses the sensor's last report, not its last value change.
Inspect **Heat Pump Pilot UFH Control** for per-pump reasons, sensor faults and
switch errors. Failed commands are retried on subsequent evaluations. Decisions
react to entity changes and a 10-second timer, independently of the MPC interval.

Disabling/removing UFH control leaves the switches in their current states and
releases control. It does not issue an unconditional off command. This optional
controller is for suitable external UFH circulation pumps, not the heat pump's
primary circulation or a substitute for its built-in protections.
