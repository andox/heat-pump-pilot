"""Runtime settings normalization for the climate entity.

This module is intentionally free of Home Assistant imports so it can be unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from .config_helpers import normalize_hvac_mode
    from .const import (
        CONF_COMFORT_TEMPERATURE_TOLERANCE,
        CONF_CONTINUOUS_CONTROL_ENABLED,
        CONF_CONTINUOUS_CONTROL_WINDOW_HOURS,
        CONF_CONTROL_INTERVAL_MINUTES,
        CONF_HEATING_DETECTION_ENABLED,
        CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
        CONF_HEATING_SUPPLY_TEMP_ENTITY,
        CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
        CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
        CONF_HEAT_LOSS_COEFFICIENT,
        CONF_HVAC_MODE,
        CONF_INITIAL_HEAT_GAIN,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE,
        CONF_INITIAL_INDOOR_TEMP,
        CONF_LEARNING_MODEL,
        CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
        CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN,
        CONF_LEARNING_WINDOW_HOURS,
        CONF_MONITOR_ONLY,
        CONF_OVERSHOOT_WARM_BIAS_CURVE,
        CONF_OVERSHOOT_WARM_BIAS_ENABLED,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
        CONF_PERFORMANCE_WINDOW_HOURS,
        CONF_PREDICTION_HORIZON_HOURS,
        CONF_PRICE_ABSOLUTE_LOW_THRESHOLD,
        CONF_PRICE_ABSOLUTE_LOW_WINDOW_DAYS,
        CONF_PRICE_BASELINE_WINDOW_HOURS,
        CONF_PRICE_COMFORT_WEIGHT,
        CONF_PRICE_PENALTY_CURVE,
        CONF_RLS_FORGETTING_FACTOR,
        CONF_TARGET_TEMPERATURE,
        CONF_THERMAL_RESPONSE_SEED,
        CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET,
        CONF_VIRTUAL_OUTDOOR_MIN_TEMP,
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA,
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
        CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED,
        CONTINUOUS_CONTROL_WINDOW_OPTIONS,
        DEFAULT_COMFORT_TEMPERATURE_TOLERANCE,
        DEFAULT_CONTINUOUS_CONTROL_ENABLED,
        DEFAULT_CONTINUOUS_CONTROL_WINDOW_HOURS,
        DEFAULT_CONTROL_INTERVAL_MINUTES,
        DEFAULT_HEATING_DETECTION_ENABLED,
        DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
        DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS,
        DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD,
        DEFAULT_HEAT_LOSS_COEFFICIENT,
        DEFAULT_HVAC_MODE,
        DEFAULT_LEARNING_MODEL,
        DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
        DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN,
        DEFAULT_LEARNING_WINDOW_HOURS,
        DEFAULT_MONITOR_ONLY,
        DEFAULT_OVERSHOOT_WARM_BIAS_CURVE,
        DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED,
        DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS,
        DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
        DEFAULT_PERFORMANCE_WINDOW_HOURS,
        DEFAULT_PREDICTION_HORIZON_HOURS,
        DEFAULT_PRICE_ABSOLUTE_LOW_THRESHOLD,
        DEFAULT_PRICE_ABSOLUTE_LOW_WINDOW_DAYS,
        DEFAULT_PRICE_BASELINE_WINDOW_HOURS,
        DEFAULT_PRICE_COMFORT_WEIGHT,
        DEFAULT_PRICE_PENALTY_CURVE,
        DEFAULT_RLS_FORGETTING_FACTOR,
        DEFAULT_TARGET_TEMPERATURE,
        DEFAULT_THERMAL_RESPONSE_SEED,
        DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET,
        DEFAULT_VIRTUAL_OUTDOOR_MIN_TEMP,
        DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA,
        DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
        DEFAULT_VIRTUAL_OUTDOOR_TRACE_ENABLED,
        LEARNING_MODEL_EKF,
        LEARNING_MODEL_RLS,
        LEARNING_WINDOW_OPTIONS,
        OVERSHOOT_WARM_BIAS_CURVES,
        PERFORMANCE_WINDOW_OPTIONS,
        PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO,
        PRICE_ABSOLUTE_LOW_THRESHOLD_OFF,
        PRICE_ABSOLUTE_LOW_WINDOW_DAYS_OPTIONS,
        PRICE_BASELINE_WINDOW_OPTIONS,
        PRICE_PENALTY_CURVES,
    )
    from .thermal_model import ThermalModelEstimator, ThermalModelRlsEstimator
except ImportError:  # pragma: no cover - direct test imports
    from config_helpers import normalize_hvac_mode  # type: ignore
    from const import (  # type: ignore
        CONF_COMFORT_TEMPERATURE_TOLERANCE,
        CONF_CONTINUOUS_CONTROL_ENABLED,
        CONF_CONTINUOUS_CONTROL_WINDOW_HOURS,
        CONF_CONTROL_INTERVAL_MINUTES,
        CONF_HEATING_DETECTION_ENABLED,
        CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
        CONF_HEATING_SUPPLY_TEMP_ENTITY,
        CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
        CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
        CONF_HEAT_LOSS_COEFFICIENT,
        CONF_HVAC_MODE,
        CONF_INITIAL_HEAT_GAIN,
        CONF_INITIAL_HEAT_LOSS_OVERRIDE,
        CONF_INITIAL_INDOOR_TEMP,
        CONF_LEARNING_MODEL,
        CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
        CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN,
        CONF_LEARNING_WINDOW_HOURS,
        CONF_MONITOR_ONLY,
        CONF_OVERSHOOT_WARM_BIAS_CURVE,
        CONF_OVERSHOOT_WARM_BIAS_ENABLED,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
        CONF_PERFORMANCE_WINDOW_HOURS,
        CONF_PREDICTION_HORIZON_HOURS,
        CONF_PRICE_ABSOLUTE_LOW_THRESHOLD,
        CONF_PRICE_ABSOLUTE_LOW_WINDOW_DAYS,
        CONF_PRICE_BASELINE_WINDOW_HOURS,
        CONF_PRICE_COMFORT_WEIGHT,
        CONF_PRICE_PENALTY_CURVE,
        CONF_RLS_FORGETTING_FACTOR,
        CONF_TARGET_TEMPERATURE,
        CONF_THERMAL_RESPONSE_SEED,
        CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET,
        CONF_VIRTUAL_OUTDOOR_MIN_TEMP,
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA,
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
        CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED,
        CONTINUOUS_CONTROL_WINDOW_OPTIONS,
        DEFAULT_COMFORT_TEMPERATURE_TOLERANCE,
        DEFAULT_CONTINUOUS_CONTROL_ENABLED,
        DEFAULT_CONTINUOUS_CONTROL_WINDOW_HOURS,
        DEFAULT_CONTROL_INTERVAL_MINUTES,
        DEFAULT_HEATING_DETECTION_ENABLED,
        DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
        DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS,
        DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD,
        DEFAULT_HEAT_LOSS_COEFFICIENT,
        DEFAULT_HVAC_MODE,
        DEFAULT_LEARNING_MODEL,
        DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
        DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN,
        DEFAULT_LEARNING_WINDOW_HOURS,
        DEFAULT_MONITOR_ONLY,
        DEFAULT_OVERSHOOT_WARM_BIAS_CURVE,
        DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED,
        DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS,
        DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
        DEFAULT_PERFORMANCE_WINDOW_HOURS,
        DEFAULT_PREDICTION_HORIZON_HOURS,
        DEFAULT_PRICE_ABSOLUTE_LOW_THRESHOLD,
        DEFAULT_PRICE_ABSOLUTE_LOW_WINDOW_DAYS,
        DEFAULT_PRICE_BASELINE_WINDOW_HOURS,
        DEFAULT_PRICE_COMFORT_WEIGHT,
        DEFAULT_PRICE_PENALTY_CURVE,
        DEFAULT_RLS_FORGETTING_FACTOR,
        DEFAULT_TARGET_TEMPERATURE,
        DEFAULT_THERMAL_RESPONSE_SEED,
        DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET,
        DEFAULT_VIRTUAL_OUTDOOR_MIN_TEMP,
        DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA,
        DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
        DEFAULT_VIRTUAL_OUTDOOR_TRACE_ENABLED,
        LEARNING_MODEL_EKF,
        LEARNING_MODEL_RLS,
        LEARNING_WINDOW_OPTIONS,
        OVERSHOOT_WARM_BIAS_CURVES,
        PERFORMANCE_WINDOW_OPTIONS,
        PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO,
        PRICE_ABSOLUTE_LOW_THRESHOLD_OFF,
        PRICE_ABSOLUTE_LOW_WINDOW_DAYS_OPTIONS,
        PRICE_BASELINE_WINDOW_OPTIONS,
        PRICE_PENALTY_CURVES,
    )
    from thermal_model import ThermalModelEstimator, ThermalModelRlsEstimator  # type: ignore


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = default
    if minimum is not None:
        numeric = max(minimum, numeric)
    return numeric


@dataclass(frozen=True)
class ClimateRuntimeSettings:
    """Typed runtime settings used by the climate entity."""

    target_temperature: float
    price_comfort_weight: float
    price_penalty_curve: str
    price_baseline_window_hours: int
    price_absolute_low_threshold: float | str | None
    price_absolute_low_window_days: int
    continuous_control_enabled: bool
    continuous_control_window_hours: float
    control_interval_minutes: int
    prediction_horizon_hours: int
    comfort_temperature_tolerance: float
    monitor_only: bool
    hvac_mode: str
    virtual_outdoor_heat_offset: Any
    virtual_outdoor_min_temp: Any
    overshoot_warm_bias_enabled: bool
    overshoot_warm_bias_curve: str
    overshoot_warm_bias_hysteresis_enabled: bool
    overshoot_warm_bias_hysteresis: float
    heat_loss_coefficient: Any
    thermal_response_seed: Any
    learning_model: str
    rls_forgetting_factor: float
    learning_window_hours: int
    performance_window_hours: int
    heating_supply_temp_entity: Any
    heating_supply_temp_threshold: Any
    heating_detection_enabled: bool
    heating_supply_temp_hysteresis: float
    heating_supply_temp_debounce_seconds: int
    learning_supply_temp_on_margin: float
    learning_supply_temp_off_margin: float
    initial_indoor_temp: Any
    initial_heat_gain: Any
    initial_heat_loss_override: Any
    virtual_outdoor_trace_enabled: bool
    virtual_outdoor_smoothing_enabled: bool
    virtual_outdoor_smoothing_alpha: float


def merge_climate_options(options: dict[str, Any]) -> dict[str, Any]:
    """Merge entry options with runtime defaults."""
    control_interval = _coerce_int(
        options.get(CONF_CONTROL_INTERVAL_MINUTES, DEFAULT_CONTROL_INTERVAL_MINUTES),
        DEFAULT_CONTROL_INTERVAL_MINUTES,
        minimum=1,
    )
    prediction_horizon = _coerce_int(
        options.get(CONF_PREDICTION_HORIZON_HOURS, DEFAULT_PREDICTION_HORIZON_HOURS),
        DEFAULT_PREDICTION_HORIZON_HOURS,
        minimum=1,
    )
    heating_hysteresis = _coerce_float(options.get(CONF_HEATING_SUPPLY_TEMP_HYSTERESIS))
    if heating_hysteresis is None:
        heating_hysteresis = DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS

    target_temperature = _coerce_float(options.get(CONF_TARGET_TEMPERATURE))
    if target_temperature is None:
        target_temperature = DEFAULT_TARGET_TEMPERATURE

    price_comfort_weight = _coerce_float(options.get(CONF_PRICE_COMFORT_WEIGHT))
    if price_comfort_weight is None:
        price_comfort_weight = DEFAULT_PRICE_COMFORT_WEIGHT
    price_comfort_weight = min(1.0, max(0.0, float(price_comfort_weight)))

    price_penalty_curve = options.get(CONF_PRICE_PENALTY_CURVE, DEFAULT_PRICE_PENALTY_CURVE)
    if price_penalty_curve not in PRICE_PENALTY_CURVES:
        price_penalty_curve = DEFAULT_PRICE_PENALTY_CURVE

    price_baseline_window = options.get(
        CONF_PRICE_BASELINE_WINDOW_HOURS, DEFAULT_PRICE_BASELINE_WINDOW_HOURS
    )
    try:
        price_baseline_window = int(price_baseline_window)
    except (TypeError, ValueError):
        price_baseline_window = DEFAULT_PRICE_BASELINE_WINDOW_HOURS
    if price_baseline_window not in PRICE_BASELINE_WINDOW_OPTIONS:
        price_baseline_window = DEFAULT_PRICE_BASELINE_WINDOW_HOURS

    absolute_low_window_raw = options.get(
        CONF_PRICE_ABSOLUTE_LOW_WINDOW_DAYS, DEFAULT_PRICE_ABSOLUTE_LOW_WINDOW_DAYS
    )
    absolute_low_window_days = _coerce_int(
        absolute_low_window_raw, DEFAULT_PRICE_ABSOLUTE_LOW_WINDOW_DAYS, minimum=1
    )
    if absolute_low_window_days not in PRICE_ABSOLUTE_LOW_WINDOW_DAYS_OPTIONS:
        absolute_low_window_days = DEFAULT_PRICE_ABSOLUTE_LOW_WINDOW_DAYS

    absolute_low_raw = options.get(
        CONF_PRICE_ABSOLUTE_LOW_THRESHOLD, DEFAULT_PRICE_ABSOLUTE_LOW_THRESHOLD
    )
    absolute_low_threshold: float | str | None
    if absolute_low_raw is None:
        absolute_low_threshold = PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
    elif isinstance(absolute_low_raw, str):
        absolute_low_str = absolute_low_raw.strip().lower()
        if absolute_low_str == PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO:
            absolute_low_threshold = PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
        elif absolute_low_str == PRICE_ABSOLUTE_LOW_THRESHOLD_OFF:
            absolute_low_threshold = PRICE_ABSOLUTE_LOW_THRESHOLD_OFF
        else:
            try:
                absolute_low_value = float(absolute_low_str)
            except (TypeError, ValueError):
                absolute_low_threshold = PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
            else:
                absolute_low_threshold = (
                    absolute_low_value if absolute_low_value > 0 else PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
                )
    else:
        try:
            absolute_low_value = float(absolute_low_raw)
        except (TypeError, ValueError):
            absolute_low_threshold = PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
        else:
            absolute_low_threshold = (
                absolute_low_value if absolute_low_value > 0 else PRICE_ABSOLUTE_LOW_THRESHOLD_AUTO
            )

    continuous_enabled = bool(
        options.get(CONF_CONTINUOUS_CONTROL_ENABLED, DEFAULT_CONTINUOUS_CONTROL_ENABLED)
    )
    continuous_window_raw = options.get(
        CONF_CONTINUOUS_CONTROL_WINDOW_HOURS, DEFAULT_CONTINUOUS_CONTROL_WINDOW_HOURS
    )
    try:
        continuous_window = float(continuous_window_raw)
    except (TypeError, ValueError):
        continuous_window = DEFAULT_CONTINUOUS_CONTROL_WINDOW_HOURS
    if int(continuous_window) not in CONTINUOUS_CONTROL_WINDOW_OPTIONS:
        continuous_window = float(DEFAULT_CONTINUOUS_CONTROL_WINDOW_HOURS)

    comfort_tolerance = _coerce_float(options.get(CONF_COMFORT_TEMPERATURE_TOLERANCE))
    if comfort_tolerance is None:
        comfort_tolerance = DEFAULT_COMFORT_TEMPERATURE_TOLERANCE

    overshoot_enabled = bool(options.get(CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED))
    overshoot_curve = options.get(CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE)
    if overshoot_curve not in OVERSHOOT_WARM_BIAS_CURVES:
        overshoot_curve = DEFAULT_OVERSHOOT_WARM_BIAS_CURVE
    overshoot_hysteresis_enabled = bool(
        options.get(
            CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
            DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED,
        )
    )
    overshoot_hysteresis = _coerce_float(options.get(CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS))
    if overshoot_hysteresis is None:
        overshoot_hysteresis = DEFAULT_OVERSHOOT_WARM_BIAS_HYSTERESIS
    overshoot_hysteresis = max(0.0, float(overshoot_hysteresis))

    smoothing_enabled = bool(
        options.get(
            CONF_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
            DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED,
        )
    )
    smoothing_alpha = _coerce_float(options.get(CONF_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA))
    if smoothing_alpha is None:
        smoothing_alpha = DEFAULT_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA
    smoothing_alpha = min(1.0, max(0.0, float(smoothing_alpha)))

    learning_on_margin = _coerce_float(options.get(CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN))
    if learning_on_margin is None:
        learning_on_margin = DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN
    learning_on_margin = max(0.0, float(learning_on_margin))

    learning_off_margin = _coerce_float(options.get(CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN))
    if learning_off_margin is None:
        learning_off_margin = DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN
    learning_off_margin = max(0.0, float(learning_off_margin))

    performance_window_raw = options.get(CONF_PERFORMANCE_WINDOW_HOURS, DEFAULT_PERFORMANCE_WINDOW_HOURS)
    try:
        performance_window = int(performance_window_raw)
    except (TypeError, ValueError):
        performance_window = DEFAULT_PERFORMANCE_WINDOW_HOURS
    if performance_window not in PERFORMANCE_WINDOW_OPTIONS:
        performance_window = DEFAULT_PERFORMANCE_WINDOW_HOURS

    learning_window_raw = options.get(CONF_LEARNING_WINDOW_HOURS, DEFAULT_LEARNING_WINDOW_HOURS)
    try:
        learning_window = int(learning_window_raw)
    except (TypeError, ValueError):
        learning_window = DEFAULT_LEARNING_WINDOW_HOURS
    if learning_window not in LEARNING_WINDOW_OPTIONS:
        learning_window = DEFAULT_LEARNING_WINDOW_HOURS

    learning_model = options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL)
    if learning_model not in (LEARNING_MODEL_EKF, LEARNING_MODEL_RLS):
        learning_model = DEFAULT_LEARNING_MODEL

    rls_factor = _coerce_float(options.get(CONF_RLS_FORGETTING_FACTOR))
    if rls_factor is None:
        rls_factor = DEFAULT_RLS_FORGETTING_FACTOR
    rls_factor = min(1.0, max(0.9, float(rls_factor)))

    return {
        CONF_TARGET_TEMPERATURE: target_temperature,
        CONF_PRICE_COMFORT_WEIGHT: price_comfort_weight,
        CONF_PRICE_PENALTY_CURVE: price_penalty_curve,
        CONF_PRICE_BASELINE_WINDOW_HOURS: price_baseline_window,
        CONF_PRICE_ABSOLUTE_LOW_THRESHOLD: absolute_low_threshold,
        CONF_PRICE_ABSOLUTE_LOW_WINDOW_DAYS: absolute_low_window_days,
        CONF_CONTINUOUS_CONTROL_ENABLED: continuous_enabled,
        CONF_CONTINUOUS_CONTROL_WINDOW_HOURS: continuous_window,
        CONF_CONTROL_INTERVAL_MINUTES: control_interval,
        CONF_PREDICTION_HORIZON_HOURS: prediction_horizon,
        CONF_COMFORT_TEMPERATURE_TOLERANCE: comfort_tolerance,
        CONF_MONITOR_ONLY: options.get(CONF_MONITOR_ONLY, DEFAULT_MONITOR_ONLY),
        CONF_HVAC_MODE: normalize_hvac_mode(options.get(CONF_HVAC_MODE, DEFAULT_HVAC_MODE)),
        CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET: options.get(
            CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET, DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET
        ),
        CONF_VIRTUAL_OUTDOOR_MIN_TEMP: options.get(
            CONF_VIRTUAL_OUTDOOR_MIN_TEMP, DEFAULT_VIRTUAL_OUTDOOR_MIN_TEMP
        ),
        CONF_OVERSHOOT_WARM_BIAS_ENABLED: overshoot_enabled,
        CONF_OVERSHOOT_WARM_BIAS_CURVE: overshoot_curve,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED: overshoot_hysteresis_enabled,
        CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS: overshoot_hysteresis,
        CONF_HEAT_LOSS_COEFFICIENT: options.get(CONF_HEAT_LOSS_COEFFICIENT, DEFAULT_HEAT_LOSS_COEFFICIENT),
        CONF_THERMAL_RESPONSE_SEED: options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
        CONF_LEARNING_MODEL: learning_model,
        CONF_RLS_FORGETTING_FACTOR: rls_factor,
        CONF_LEARNING_WINDOW_HOURS: learning_window,
        CONF_PERFORMANCE_WINDOW_HOURS: performance_window,
        CONF_HEATING_SUPPLY_TEMP_ENTITY: options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY),
        CONF_HEATING_SUPPLY_TEMP_THRESHOLD: options.get(
            CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD
        ),
        CONF_HEATING_DETECTION_ENABLED: bool(
            options.get(CONF_HEATING_DETECTION_ENABLED, DEFAULT_HEATING_DETECTION_ENABLED)
        ),
        CONF_HEATING_SUPPLY_TEMP_HYSTERESIS: heating_hysteresis,
        CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS: _coerce_int(
            options.get(CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS),
            DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
            minimum=0,
        ),
        CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN: learning_on_margin,
        CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN: learning_off_margin,
        CONF_INITIAL_INDOOR_TEMP: options.get(CONF_INITIAL_INDOOR_TEMP),
        CONF_INITIAL_HEAT_GAIN: options.get(CONF_INITIAL_HEAT_GAIN),
        CONF_INITIAL_HEAT_LOSS_OVERRIDE: options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE),
        CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED: bool(
            options.get(CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED, DEFAULT_VIRTUAL_OUTDOOR_TRACE_ENABLED)
        ),
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED: smoothing_enabled,
        CONF_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA: smoothing_alpha,
    }


def build_runtime_settings(options: dict[str, Any]) -> ClimateRuntimeSettings:
    """Build a typed runtime settings object from normalized options."""
    normalized = merge_climate_options(options)
    return ClimateRuntimeSettings(
        target_temperature=normalized[CONF_TARGET_TEMPERATURE],
        price_comfort_weight=normalized[CONF_PRICE_COMFORT_WEIGHT],
        price_penalty_curve=normalized[CONF_PRICE_PENALTY_CURVE],
        price_baseline_window_hours=normalized[CONF_PRICE_BASELINE_WINDOW_HOURS],
        price_absolute_low_threshold=normalized[CONF_PRICE_ABSOLUTE_LOW_THRESHOLD],
        price_absolute_low_window_days=normalized[CONF_PRICE_ABSOLUTE_LOW_WINDOW_DAYS],
        continuous_control_enabled=normalized[CONF_CONTINUOUS_CONTROL_ENABLED],
        continuous_control_window_hours=normalized[CONF_CONTINUOUS_CONTROL_WINDOW_HOURS],
        control_interval_minutes=normalized[CONF_CONTROL_INTERVAL_MINUTES],
        prediction_horizon_hours=normalized[CONF_PREDICTION_HORIZON_HOURS],
        comfort_temperature_tolerance=normalized[CONF_COMFORT_TEMPERATURE_TOLERANCE],
        monitor_only=normalized[CONF_MONITOR_ONLY],
        hvac_mode=normalized[CONF_HVAC_MODE],
        virtual_outdoor_heat_offset=normalized[CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET],
        virtual_outdoor_min_temp=normalized[CONF_VIRTUAL_OUTDOOR_MIN_TEMP],
        overshoot_warm_bias_enabled=normalized[CONF_OVERSHOOT_WARM_BIAS_ENABLED],
        overshoot_warm_bias_curve=normalized[CONF_OVERSHOOT_WARM_BIAS_CURVE],
        overshoot_warm_bias_hysteresis_enabled=normalized[CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS_ENABLED],
        overshoot_warm_bias_hysteresis=normalized[CONF_OVERSHOOT_WARM_BIAS_HYSTERESIS],
        heat_loss_coefficient=normalized[CONF_HEAT_LOSS_COEFFICIENT],
        thermal_response_seed=normalized[CONF_THERMAL_RESPONSE_SEED],
        learning_model=normalized[CONF_LEARNING_MODEL],
        rls_forgetting_factor=normalized[CONF_RLS_FORGETTING_FACTOR],
        learning_window_hours=normalized[CONF_LEARNING_WINDOW_HOURS],
        performance_window_hours=normalized[CONF_PERFORMANCE_WINDOW_HOURS],
        heating_supply_temp_entity=normalized[CONF_HEATING_SUPPLY_TEMP_ENTITY],
        heating_supply_temp_threshold=normalized[CONF_HEATING_SUPPLY_TEMP_THRESHOLD],
        heating_detection_enabled=normalized[CONF_HEATING_DETECTION_ENABLED],
        heating_supply_temp_hysteresis=normalized[CONF_HEATING_SUPPLY_TEMP_HYSTERESIS],
        heating_supply_temp_debounce_seconds=normalized[CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS],
        learning_supply_temp_on_margin=normalized[CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN],
        learning_supply_temp_off_margin=normalized[CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN],
        initial_indoor_temp=normalized[CONF_INITIAL_INDOOR_TEMP],
        initial_heat_gain=normalized[CONF_INITIAL_HEAT_GAIN],
        initial_heat_loss_override=normalized[CONF_INITIAL_HEAT_LOSS_OVERRIDE],
        virtual_outdoor_trace_enabled=normalized[CONF_VIRTUAL_OUTDOOR_TRACE_ENABLED],
        virtual_outdoor_smoothing_enabled=normalized[CONF_VIRTUAL_OUTDOOR_SMOOTHING_ENABLED],
        virtual_outdoor_smoothing_alpha=normalized[CONF_VIRTUAL_OUTDOOR_SMOOTHING_ALPHA],
    )


def build_thermal_model_from_options(
    options: dict[str, Any],
) -> ThermalModelEstimator | ThermalModelRlsEstimator:
    """Create a thermal model estimator based on normalized options."""
    base_loss = options.get(CONF_HEAT_LOSS_COEFFICIENT, DEFAULT_HEAT_LOSS_COEFFICIENT)
    initial_heat_loss = options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE, base_loss)
    if initial_heat_loss is None:
        initial_heat_loss = base_loss
    learning_model = options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL)
    if learning_model not in (LEARNING_MODEL_EKF, LEARNING_MODEL_RLS):
        learning_model = DEFAULT_LEARNING_MODEL
    rls_factor = _coerce_float(options.get(CONF_RLS_FORGETTING_FACTOR))
    if rls_factor is None:
        rls_factor = DEFAULT_RLS_FORGETTING_FACTOR
    rls_factor = min(1.0, max(0.9, float(rls_factor)))
    if learning_model == LEARNING_MODEL_RLS:
        return ThermalModelRlsEstimator(
            seed=options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
            initial_heat_loss=initial_heat_loss,
            initial_heat_gain=options.get(CONF_INITIAL_HEAT_GAIN),
            initial_temp=options.get(CONF_INITIAL_INDOOR_TEMP),
            forgetting_factor=rls_factor,
        )
    return ThermalModelEstimator(
        seed=options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
        initial_heat_loss=initial_heat_loss,
        initial_heat_gain=options.get(CONF_INITIAL_HEAT_GAIN),
        initial_temp=options.get(CONF_INITIAL_INDOOR_TEMP),
    )
