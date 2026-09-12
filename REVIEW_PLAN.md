# Heat Pump Pilot Review Plan

Date: 2026-03-28

## Review summary

### High priority findings

1. HVAC mode is not persisted or restored.
   - File: `custom_components/heat_pump_pilot/climate.py`
   - Relevant lines: 201-228, 854-865
   - Impact: The entity always starts in `HVACMode.HEAT` after reload/restart because `_hvac_mode` is initialized in memory only and never restored. A user can turn the integration off, restart Home Assistant, or trigger an integration reload and unexpectedly resume active control.
   - Fix: Make the climate entity a `RestoreEntity` (or persist the mode in entry options/state) and restore the last HVAC mode before the first control run.

2. Config identity is not updated when core entities are changed in the options flow.
   - File: `custom_components/heat_pump_pilot/config_flow.py`
   - Relevant lines: 106-117, 246-351
   - Impact: The initial flow prevents duplicate setups through `unique_id`, but the options flow lets the user change the controlled entity or sensor pair without updating that identity. This can allow duplicate integrations to target the same entity or sensor combination.
   - Fix: Recompute and update the unique ID when the options flow changes the controlled entity or fallback sensor pair, and reject duplicates there as well.

3. Price baseline windowing does not actually constrain the forecast portion.
   - File: `custom_components/heat_pump_pilot/price_utils.py`
   - Relevant lines: 33-75
   - Impact: `window_hours` only trims history. Forecast data is expanded and used in full, so a “24 h baseline window” can still be dominated by 48 h or longer forecasts. That makes the option misleading and can shift both optimization and price classification.
   - Fix: Trim forecast samples to the same effective window before computing the median, and add tests for mixed 24 h / 48 h inputs.

### Structural risks

1. The main runtime logic is too concentrated in one file.
   - File: `custom_components/heat_pump_pilot/climate.py`
   - Impact: Control loop execution, storage, notifications, learning, forecast fetching, price classification, diagnostics, and actuation all live in one class. That raises regression risk and makes it hard to reason about option changes.

2. Integration-level coverage is thin around the highest-risk code.
   - Current tests cover helpers well, but there are no focused tests for:
     - climate entity lifecycle and restore behavior
     - options flow identity changes
     - control actuation dispatch
     - notification behavior from the full entity

## Remediation plan

### Phase 1: Fix correctness issues

1. Persist and restore HVAC mode before the first control loop run.
2. Update options-flow identity handling so controlled-entity/sensor changes cannot create duplicates.
3. Correct price baseline window semantics and document the exact behavior.
4. Add regression tests for all three fixes.

### Phase 2: Reduce climate entity complexity

1. Extract option normalization from `MpcHeatPumpClimate` into a dedicated module or dataclass.
2. Extract forecast/price acquisition and caching into a small service object.
3. Extract actuation logic (`number`, `switch`, `climate`, generic domains) into a controller adapter module.
4. Extract diagnostics/notification publishing into a dedicated helper.
5. Keep `climate.py` focused on orchestration and Home Assistant entity lifecycle only.

### Phase 3: Tighten testing

1. Add unit tests for config-flow duplicate protection after options edits.
2. Add entity-level tests for:
   - restart with HVAC off
   - trace-enabled reload preserving state
   - number/switch/climate actuation behavior
   - weather/price fallback paths
3. Add tests for price-baseline trimming across different forecast lengths and time steps.

### Phase 4: Cleanup and simplification

1. Replace repeated option-to-attribute assignments with a typed settings object.
2. Consolidate duplicated float/int coercion into shared helpers.
3. Consolidate repeated storage patterns in `thermal_model.py`, `price_history.py`, and `performance_history.py`.
4. Revisit `manifest.json` metadata, especially `integration_type`, to align it with the integration’s actual behavior.

## Suggested execution order

1. Restore/persist HVAC mode.
2. Fix options-flow identity updates.
3. Fix price baseline window semantics.
4. Add missing integration-level tests.
5. Split `climate.py` into smaller runtime modules.

## Status update

- Phase 1: completed
- Phase 2: completed
- Phase 3: completed locally with helper-level/unit coverage for duplicate identity checks, actuation behavior, fallback forecast behavior, diagnostics trace publication, storage round-trips, and price-baseline trimming.
- Phase 4: completed for the targeted cleanup items:
  - runtime settings normalization was centralized and exposed as a typed settings object
  - duplicated JSON storage logic was consolidated into a shared atomic storage helper
  - `manifest.json` was reviewed against current Home Assistant developer docs and `integration_type: "service"` was kept unchanged

## Remaining follow-up

1. `climate.py` is still the dominant orchestration file, especially around notifications, health evaluation, and learning/performance reporting.
2. There is still no true Home Assistant runtime/integration test coverage in this workspace because Home Assistant is not installed locally.
