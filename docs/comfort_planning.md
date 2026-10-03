# Unified comfort and cost planning

Status: local implementation and tests; not deployed to Home Assistant.

## User controls

Keep target, tolerance and Price vs Comfort. No new preheating or warm-side
margin setting. Tolerance is symmetric: target +/- tolerance. It defines what
is allowed; the target defines what is preferred. Existing stored values are
not changed. For example 20.5 +/- 1.2 permits 19.3..21.7 C, while 20.5 +/- 0.2
provides very little thermal storage and may require more expensive heating.

The four old warm-bias/hysteresis controls are hidden and inactive. Their saved
values remain readable for compatibility. The old duty averaging window is
hidden for normal number-output planning (continuous mode, 15-minute control).
Nonstandard binary modes retain their legacy output limiter and averaging.

## Objective

For each candidate, simulate the actual command, heating delivery and next
indoor temperature at every 15-minute step. Include the terminal temperature.
Rank complete candidates lexicographically by:

1. Degree-hours outside [target - tolerance, target + tolerance].
2. Sum of dt * ((1-w) * ((T-target)/max(tolerance,0.2))^2
   + w * shaped_price_ratio * predicted_heating_fraction), plus a small
   request-change cost (0.05 times the change in request fraction).

This is not a large arbitrary penalty added to price: no price magnitude can
outweigh a smaller band violation. Among zero-violation plans, weight controls
target tracking versus cheaper heating. Price shaping and baseline settings
remain supported. The score is dimensionless; it is not a currency estimate.
Heating fraction is an energy proxy, not measured electricity or a COP model.

If all retained candidates breach the band, choose the smallest accumulated
violation, with cost as the tie-breaker. This also works if the initial house is
already too cold/hot, sunshine makes warming unavoidable, or capacity is weak.
Report predicted_breach rather than claiming physical infeasibility. Approximate
state bucketing and bounded beam search cannot certify a global optimum.

## One decision path

For ordinary number outputs, use the existing response-aware search with either
validated response parameters or an explicit immediate proportional fallback.
The fallback is not represented as learned. Response learning remains optional
and retains its readiness, fresh-state and restart gates. House learning,
UFH circulation based on measured supply temperature, and exercise are unchanged.

Simulate request -> bounded/rounded/smoothed virtual temperature -> actual
request fraction -> delayed/residual heat -> indoor temperature. Send the exact
first simulated virtual command. The planner pays a cost for changes from the
current request, including its first action. Do not apply a separate warm-bias,
minimum hold or request ramp afterward. EMA is included in the optimization.
The explicit summer override still overrides ordinary planning and is diagnosed
separately; it is not an automatic comfort-preserving MPC decision.

Keep thermal diversity when pruning: retain warmer candidates even when their
upfront heating cost is higher. Otherwise a cheap-cost beam can lose every
preheat path before reaching the expensive part of the horizon.

The model accounts for expected cooling and delayed heat, so the lower bound is
not a thermostat trigger. It may start earlier to avoid crossing it. Predictions
are conditional on learned parameters and weather. A door near the indoor sensor,
unobserved sunlight, and unknown pump integral can invalidate a forecast.

## Executable behavioral tests

`tests/test_preheat_coast_planning.py` checks:

- Start at target; preheat above target during cheap hours; coast through a
  four-hour price peak with temperatures inside the band.
- Execute only the first action and replan repeatedly in an independent house
  model: preheating must actually happen, not be perpetually postponed.
- Shift the price peak: heat storage shifts to the newly cheaper period.
- Higher price weight lowers an electricity-cost proxy and peak-period heating;
  lower weight reduces deviation from the target. Both respect boundaries.
- All weights, including 1.0, protect a feasible band at very high prices.
- Negative prices do not justify overheating above the allowed upper bound.
- Passive sunny warming causes backoff without a separate temperature rule.
- Weak heating capacity reports a breach and uses maximum achievable output,
  including the virtual-temperature floor rather than an impossible command.
- Independently roll out delayed/residual heat and smoothing: requested idle
  may still deliver heat, and coasting must remain within the modeled band.
- The last forecast step counts; draining below the bound at the horizon edge
  is not free. No claims are made about temperatures beyond the forecast horizon.
- Compare a short continuous-action search against exhaustive independent scoring.

`tests/test_climate_learning_wiring.py` checks unlearned versus learned gating,
that the old request limiter cannot rewrite a planned command, and that live
output sends the exact planned command without extra smoothing/backoff.
`tests/test_config_sections.py` checks preservation of Price vs Comfort and
stored legacy values while redundant controls are hidden.

## Local historical replay

Run `python tools/replay_comfort_planner.py --snapshot PATH --price-curve sqrt`
on a saved climate/decision snapshot. The tool reads no credentials, contacts no
server and sends no commands. It reports predicted ranges and modeled heating
for existing and wider tolerances. This is a counterfactual model replay, not
proof of real electricity savings. Heating-cycle observation is still required
before drawing conclusions about the live house or retuning learned coefficients.
