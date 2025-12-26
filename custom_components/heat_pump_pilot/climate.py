"""Climate entity exposing the MPC-based heat pump controller."""

from __future__ import annotations

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from datetime import timedelta
import logging
import inspect
from functools import partial
from statistics import median
from typing import Any, Iterable, Sequence

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.components import persistent_notification
from homeassistant.const import ATTR_TEMPERATURE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util

from .const import (
    CONF_COMFORT_TEMPERATURE_TOLERANCE,
    CONF_CONTROL_INTERVAL_MINUTES,
    CONF_CONTROLLED_ENTITY,
    CONF_HEATING_DETECTION_ENABLED,
    CONF_HEATING_SUPPLY_TEMP_ENTITY,
    CONF_HEATING_SUPPLY_TEMP_HYSTERESIS,
    CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
    CONF_HEATING_SUPPLY_TEMP_THRESHOLD,
    CONF_INDOOR_TEMP,
    CONF_INITIAL_HEAT_GAIN,
    CONF_INITIAL_HEAT_LOSS_OVERRIDE,
    CONF_INITIAL_INDOOR_TEMP,
    CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
    CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN,
    CONF_LEARNING_MODEL,
    CONF_MONITOR_ONLY,
    CONF_OUTDOOR_TEMP,
    CONF_OVERSHOOT_WARM_BIAS_ENABLED,
    CONF_OVERSHOOT_WARM_BIAS_CURVE,
    CONF_PREDICTION_HORIZON_HOURS,
    CONF_PRICE_COMFORT_WEIGHT,
    CONF_PRICE_ENTITY,
    CONF_PRICE_PENALTY_CURVE,
    CONF_TARGET_TEMPERATURE,
    CONF_THERMAL_RESPONSE_SEED,
    CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET,
    CONF_WEATHER_FORECAST_ENTITY,
    CONF_HEAT_LOSS_COEFFICIENT,
    CONF_PERFORMANCE_WINDOW_HOURS,
    CONF_RLS_FORGETTING_FACTOR,
    DEFAULT_MONITOR_ONLY,
    DEFAULT_COMFORT_TEMPERATURE_TOLERANCE,
    DEFAULT_CONTROL_INTERVAL_MINUTES,
    DEFAULT_HEATING_DETECTION_ENABLED,
    DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS,
    DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
    DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD,
    DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN,
    DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN,
    DEFAULT_LEARNING_MODEL,
    DEFAULT_PERFORMANCE_WINDOW_HOURS,
    DEFAULT_PREDICTION_HORIZON_HOURS,
    DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED,
    DEFAULT_OVERSHOOT_WARM_BIAS_CURVE,
    DEFAULT_PRICE_COMFORT_WEIGHT,
    DEFAULT_PRICE_PENALTY_CURVE,
    DEFAULT_RLS_FORGETTING_FACTOR,
    DEFAULT_TARGET_TEMPERATURE,
    DEFAULT_THERMAL_RESPONSE_SEED,
    DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DOMAIN,
    LEARNING_MODEL_EKF,
    LEARNING_MODEL_RLS,
    OVERSHOOT_WARM_BIAS_CURVES,
    PRICE_PENALTY_CURVES,
    PERFORMANCE_WINDOW_OPTIONS,
    SIGNAL_DECISION_UPDATED,
    SIGNAL_OPTIONS_UPDATED,
)
from .forecast_utils import align_forecast_to_now, expand_to_steps, extract_timed_temperatures, extract_timed_values
from .learning_utils import should_reseed_thermal_model
from .mpc_controller import MpcController, ControlResult
from .notification_utils import NotificationTracker, NotificationUpdate
from .performance_utils import PerformanceSample, compute_comfort_score, compute_prediction_accuracy, compute_price_score
from .performance_history import PerformanceHistoryStorage
from .price_history import PriceHistoryStorage
from .thermal_model import ThermalModelEstimator, ThermalModelRlsEstimator, ThermalModelStorage
from .virtual_outdoor_utils import compute_overshoot_warm_bias, compute_planned_virtual_outdoor_temperatures

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 1

# Amount to make the virtual outdoor colder when heat is requested.
VIRTUAL_OUTDOOR_IDLE_FALLBACK = 10.0
MAX_VIRTUAL_OUTDOOR = 25.0
PRICE_HISTORY_MAX_ENTRIES = 192  # ~48h at 15-minute sampling.
THERMAL_PERSIST_INTERVAL = timedelta(minutes=15)

NOTIFY_HEALTH_COOLDOWN = timedelta(hours=1)
NOTIFY_SENSORS_COOLDOWN = timedelta(hours=1)
NOTIFY_FALLBACK_TRIGGER_AFTER = timedelta(minutes=30)
NOTIFY_FALLBACK_COOLDOWN = timedelta(hours=6)
NOTIFY_EXCEPTION_COOLDOWN = timedelta(minutes=30)
MODEL_HISTORY_MAX_ENTRIES = 192  # ~48h at 15-minute sampling.
NOMINAL_HEAT_POWER_KW = 3.0
LEARNING_STABLE_WINDOW = timedelta(hours=6)
LEARNING_STABLE_RELATIVE_DELTA = 0.05  # 5% relative change threshold.
PRICE_HISTORY_BACKFILL_MIN_SAMPLES = 16  # Avoid noisy baseline after restart.
PRICE_HISTORY_BACKFILL_MAX_ATTEMPTS = 5
# Suppress sensor notifications right after startup to allow entities to populate.
STARTUP_SENSOR_GRACE = timedelta(minutes=2)
STARTUP_HEALTH_GRACE = timedelta(minutes=2)
# Throttle sensor-driven control runs to avoid excessive solver executions.
SENSOR_CONTROL_DEBOUNCE_SECONDS = 10
SENSOR_CONTROL_MIN_INTERVAL_SECONDS = 60
INDOOR_TEMP_CONTROL_THRESHOLD = 0.1
OUTDOOR_TEMP_CONTROL_THRESHOLD = 0.2
WEATHER_FORECAST_CACHE_SECONDS = 30 * 60
WEATHER_FORECAST_SERVICE_TYPE = "hourly"
# Keep enough samples for the largest window even at the smallest control interval (5 min).
PERFORMANCE_HISTORY_MAX_ENTRIES = 96 * 12


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up the MPC climate entity."""
    async_add_entities([MpcHeatPumpClimate(hass, entry)])


class MpcHeatPumpClimate(ClimateEntity):
    """A thermostat entity that optimizes heating using a simple MPC strategy."""

    _attr_should_poll = False
    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.hass = hass
        self.config_entry = entry
        self._indoor_temp_entity: str = entry.data[CONF_INDOOR_TEMP]
        self._outdoor_temp_entity: str = entry.data[CONF_OUTDOOR_TEMP]
        self._price_entity: str = entry.data[CONF_PRICE_ENTITY]
        self._weather_entity: str = entry.data[CONF_WEATHER_FORECAST_ENTITY]
        self._controlled_entity: str | None = entry.data.get(CONF_CONTROLLED_ENTITY)

        self._options = self._merge_options(entry.options)
        self._target_temperature: float = self._options[CONF_TARGET_TEMPERATURE]
        self._price_comfort_weight: float = self._options[CONF_PRICE_COMFORT_WEIGHT]
        self._price_penalty_curve: str = self._options[CONF_PRICE_PENALTY_CURVE]
        self._control_interval: int = self._options[CONF_CONTROL_INTERVAL_MINUTES]
        self._prediction_horizon: int = self._options[CONF_PREDICTION_HORIZON_HOURS]
        self._comfort_tolerance: float = self._options[CONF_COMFORT_TEMPERATURE_TOLERANCE]
        self._monitor_only: bool = self._options[CONF_MONITOR_ONLY]
        self._virtual_heat_offset: float = self._options[CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET]
        self._heat_loss_coeff: float = self._options[CONF_HEAT_LOSS_COEFFICIENT]
        self._learning_model: str = self._options[CONF_LEARNING_MODEL]
        self._rls_forgetting_factor: float = self._options[CONF_RLS_FORGETTING_FACTOR]
        self._performance_window_hours: int = self._options[CONF_PERFORMANCE_WINDOW_HOURS]
        self._heating_supply_temp_entity: str | None = self._options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY)
        self._heating_supply_temp_threshold: float = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD)
        )
        self._heating_detection_enabled: bool = bool(
            self._options.get(CONF_HEATING_DETECTION_ENABLED, DEFAULT_HEATING_DETECTION_ENABLED)
        )
        self._heating_supply_temp_hysteresis: float = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_HYSTERESIS, DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS)
        )
        self._heating_supply_temp_debounce_seconds: float = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS)
        )
        self._overshoot_warm_bias_enabled: bool = bool(
            self._options.get(CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED)
        )
        self._overshoot_warm_bias_curve: str = str(
            self._options.get(CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE)
        )
        self._learning_supply_temp_on_margin: float = float(
            self._options.get(CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN)
        )
        self._learning_supply_temp_off_margin: float = float(
            self._options.get(CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN)
        )
        self._thermal_model = self._build_thermal_model(self._options)
        self._thermal_store = ThermalModelStorage(
            hass.config.path(".storage", f"{DOMAIN}_{entry.entry_id}_thermal.json")
        )
        self._price_store = PriceHistoryStorage(
            hass.config.path(".storage", f"{DOMAIN}_{entry.entry_id}_prices.json")
        )
        self._performance_store = PerformanceHistoryStorage(
            hass.config.path(".storage", f"{DOMAIN}_{entry.entry_id}_performance.json")
        )

        self._controller = MpcController(
            target_temperature=self._target_temperature,
            price_comfort_weight=self._price_comfort_weight,
            price_penalty_curve=self._price_penalty_curve,
            comfort_temperature_tolerance=self._comfort_tolerance,
            prediction_horizon_hours=self._prediction_horizon,
            heat_loss_coeff=self._heat_loss_coeff,
            heat_gain_coeff=self._thermal_model.heat_gain_coeff,
            overshoot_warm_bias_enabled=self._overshoot_warm_bias_enabled,
            virtual_heat_offset=self._virtual_heat_offset,
            overshoot_warm_bias_curve=self._overshoot_warm_bias_curve,
        )

        self._indoor_temp: float | None = None
        self._outdoor_temp: float | None = None
        self._hvac_mode = HVACMode.HEAT
        self._last_control_on = False
        self._last_result: ControlResult | None = None
        self._last_price_forecast: list[float] = []
        self._last_outdoor_forecast: list[float] = []
        self._last_price_forecast_source: str = "unavailable"
        self._last_outdoor_forecast_source: str = "unavailable"
        self._last_virtual_outdoor: float | None = None
        self._price_history: list[float] = []
        self._last_price_bucket_start = None
        self._price_history_source = "live"
        self._price_backfill_attempts = 0
        self._price_backfill_done = False
        self._price_backfill_unsub = None
        self._weather_forecast_cache_raw: list[Any] | None = None
        self._weather_forecast_cache_time = None
        self._control_unsub = None
        self._sensor_unsub = None
        self._options_unsub = None
        self._control_lock = asyncio.Lock()
        self._control_task: asyncio.Task | None = None
        self._control_rerun_requested = False
        self._sensor_control_debounce_unsub = None
        self._last_control_time = None
        self._last_persist_time = None
        self._model_history: list[tuple[Any, float, float]] = []
        self._heating_detected: bool | None = None
        self._heating_detected_candidate: bool | None = None
        self._heating_detected_candidate_since = None
        self._heating_duty_cycle_on_seconds = 0.0
        self._heating_duty_cycle_known_seconds = 0.0
        self._heating_duty_cycle_last_time = None
        self._heating_duty_cycle_last_state: bool | None = None
        self._notification_tracker = NotificationTracker()
        self._startup_time = dt_util.utcnow()
        self._performance_history: list[PerformanceSample] = []
        self._last_prediction: dict[str, Any] | None = None
        self._last_performance_persist_time = None

        self._attr_name = "Heat Pump Pilot"
        self._attr_unique_id = f"{entry.entry_id}_climate"
        self._attr_temperature_unit = hass.config.units.temperature_unit
        self._attr_hvac_action = HVACAction.IDLE

    async def async_added_to_hass(self) -> None:
        """Register listeners and kick off control loop."""
        self._resubscribe_sensors()
        self._options_unsub = async_dispatcher_connect(
            self.hass, f"{SIGNAL_OPTIONS_UPDATED}_{self.config_entry.entry_id}", self._handle_entry_update
        )

        await self._load_price_history()
        await self._load_thermal_state()
        await self._load_performance_history()
        self._schedule_control_loop()
        await self._async_run_control()
        self._schedule_price_history_backfill(30)

    async def async_will_remove_from_hass(self) -> None:
        """Clean up listeners."""
        if self._control_unsub:
            self._control_unsub()
        if self._sensor_unsub:
            self._sensor_unsub()
        if self._options_unsub:
            self._options_unsub()
        await self._persist_thermal_state(dt_util.utcnow())
        await self._persist_performance_history(dt_util.utcnow())
        await self._persist_price_history()

    async def _load_price_history(self) -> None:
        """Load persisted price history if available."""
        payload = await self.hass.async_add_executor_job(self._price_store.load)
        if not payload:
            return

        bucket_minutes = payload.get("bucket_minutes")
        expected_bucket = int(round(self._controller.time_step_hours * 60)) or 15
        if bucket_minutes not in (None, expected_bucket):
            return

        raw_history = payload.get("history")
        if isinstance(raw_history, list):
            history: list[float] = []
            for value in raw_history:
                try:
                    history.append(float(value))
                except (TypeError, ValueError):
                    continue
            if history:
                self._price_history = history[-PRICE_HISTORY_MAX_ENTRIES:]
                self._price_history_source = "storage"

        last_bucket_start = payload.get("last_bucket_start")
        if isinstance(last_bucket_start, str):
            parsed = dt_util.parse_datetime(last_bucket_start)
            if parsed is not None:
                self._last_price_bucket_start = dt_util.as_utc(parsed)

        if len(self._price_history) >= PRICE_HISTORY_BACKFILL_MIN_SAMPLES:
            self._price_backfill_done = True

    async def _persist_price_history(self) -> None:
        """Persist price history to disk."""
        bucket_minutes = int(round(self._controller.time_step_hours * 60)) or 15
        last_bucket = self._last_price_bucket_start
        payload = {
            "version": 1,
            "bucket_minutes": bucket_minutes,
            "last_bucket_start": last_bucket.isoformat() if last_bucket else None,
            "history": list(self._price_history),
        }
        await self.hass.async_add_executor_job(self._price_store.save, payload)

    @callback
    def _schedule_price_history_backfill(self, delay_seconds: int) -> None:
        """Schedule a one-shot backfill attempt after a delay."""
        if self._price_backfill_done:
            return
        if self._price_backfill_unsub:
            return
        if self._price_backfill_attempts >= PRICE_HISTORY_BACKFILL_MAX_ATTEMPTS:
            return

        try:
            from homeassistant.helpers.event import async_call_later  # type: ignore
        except Exception:  # pragma: no cover
            self._price_backfill_unsub = None
            self.hass.async_create_task(self._maybe_backfill_price_history())
            return

        @callback
        def _run_backfill(_now) -> None:
            self._price_backfill_unsub = None
            self.hass.async_create_task(self._maybe_backfill_price_history())

        self._price_backfill_unsub = async_call_later(self.hass, delay_seconds, _run_backfill)
        self.async_on_remove(self._price_backfill_unsub)

    async def _maybe_backfill_price_history(self) -> None:
        """Backfill price history from recorder if available and needed.

        Keeps complexity low by only backfilling when we have too few samples.
        """
        if self._price_backfill_done or len(self._price_history) >= PRICE_HISTORY_BACKFILL_MIN_SAMPLES:
            self._price_backfill_done = True
            return

        if "recorder" not in self.hass.config.components:
            self._price_backfill_attempts += 1
            self._schedule_price_history_backfill(60)
            return

        try:
            from homeassistant.components.recorder import history as recorder_history  # type: ignore
        except Exception:  # pragma: no cover
            self._price_backfill_attempts += 1
            self._schedule_price_history_backfill(60)
            return

        bucket_minutes = int(round(self._controller.time_step_hours * 60)) or 15
        end = dt_util.utcnow()
        start = end - timedelta(minutes=bucket_minutes * PRICE_HISTORY_MAX_ENTRIES)

        # Recorder history helpers are synchronous; run them in the executor.
        func = getattr(recorder_history, "state_changes_during_period", None)
        if func is None:  # pragma: no cover
            self._price_backfill_attempts += 1
            return

        self._price_backfill_attempts += 1

        try:
            sig = inspect.signature(func)
            params = sig.parameters
            kwargs: dict[str, Any] = {}
            if "entity_id" in params:
                kwargs["entity_id"] = self._price_entity
            elif "entity_ids" in params:
                kwargs["entity_ids"] = [self._price_entity]

            if "include_start_time_state" in params:
                kwargs["include_start_time_state"] = True
            if "significant_changes_only" in params:
                kwargs["significant_changes_only"] = False
            if "no_attributes" in params:
                kwargs["no_attributes"] = True

            job = partial(func, self.hass, start, end, **kwargs)
            states = await self.hass.async_add_executor_job(job)
        except Exception:
            if self._price_backfill_attempts < PRICE_HISTORY_BACKFILL_MAX_ATTEMPTS:
                self._schedule_price_history_backfill(60)
            return

        series = None
        if isinstance(states, dict):
            series = states.get(self._price_entity)
        if not isinstance(series, list) or not series:
            return

        buckets: dict[Any, float] = {}
        for state in series:
            value = self._state_to_float(getattr(state, "state", None))
            if value is None:
                continue
            updated = getattr(state, "last_updated", None)
            if updated is None:
                continue
            when = dt_util.as_utc(updated)
            bucket_start = when.replace(
                minute=(when.minute // bucket_minutes) * bucket_minutes,
                second=0,
                microsecond=0,
            )
            buckets[bucket_start] = float(value)

        if len(buckets) < PRICE_HISTORY_BACKFILL_MIN_SAMPLES:
            if self._price_backfill_attempts < PRICE_HISTORY_BACKFILL_MAX_ATTEMPTS:
                self._schedule_price_history_backfill(60)
            return

        ordered = [buckets[key] for key in sorted(buckets)]
        self._price_history = ordered[-PRICE_HISTORY_MAX_ENTRIES:]
        self._last_price_bucket_start = max(buckets)
        self._price_history_source = "recorder"
        self._price_backfill_done = True
        await self._persist_price_history()
        self._request_control_run()

    async def _load_thermal_state(self) -> None:
        """Load persisted thermal model state if available."""
        payload = await self.hass.async_add_executor_job(self._thermal_store.load)
        if payload and self._thermal_model.restore(payload):
            _LOGGER.debug("Restored thermal model state for %s", self.entity_id)
            self._heat_loss_coeff = self._thermal_model.heat_loss_coeff
            self._controller.update_settings(
                heat_loss_coeff=self._heat_loss_coeff, heat_gain_coeff=self._thermal_model.heat_gain_coeff
            )
            self._restore_model_history(payload)

    async def _persist_thermal_state(self, now) -> None:
        """Persist thermal model state to disk throttled to a safe cadence."""
        if self._last_persist_time and (now - self._last_persist_time) < THERMAL_PERSIST_INTERVAL:
            return
        snapshot = self._thermal_model.export_state()
        payload = {
            "version": 2,
            **snapshot,
            "history": self._serialize_model_history(),
        }
        await self.hass.async_add_executor_job(self._thermal_store.save, payload)
        self._last_persist_time = now

    async def _load_performance_history(self) -> None:
        """Load persisted performance history if available."""
        payload = await self.hass.async_add_executor_job(self._performance_store.load)
        if not payload:
            return
        raw_history = payload.get("history")
        if not isinstance(raw_history, list):
            return
        history: list[PerformanceSample] = []
        for entry in raw_history:
            if not isinstance(entry, dict):
                continue
            when_raw = entry.get("time")
            if not isinstance(when_raw, str):
                continue
            when = dt_util.parse_datetime(when_raw)
            if when is None:
                continue
            try:
                when = dt_util.as_utc(when)
            except (TypeError, ValueError):
                continue
            indoor_raw = entry.get("indoor_temp")
            target_raw = entry.get("target_temp")
            try:
                indoor = float(indoor_raw)
                target = float(target_raw)
            except (TypeError, ValueError):
                continue
            heating_detected = entry.get("heating_detected")
            if heating_detected not in (None, True, False):
                heating_detected = None
            price_raw = entry.get("price")
            prediction_raw = entry.get("prediction_error")
            try:
                price = float(price_raw) if price_raw is not None else None
            except (TypeError, ValueError):
                price = None
            try:
                prediction_error = float(prediction_raw) if prediction_raw is not None else None
            except (TypeError, ValueError):
                prediction_error = None
            history.append(
                PerformanceSample(
                    when=when,
                    indoor_temp=indoor,
                    target_temp=target,
                    heating_detected=heating_detected,
                    price=price,
                    prediction_error=prediction_error,
                )
            )
        if history:
            self._performance_history = history[-PERFORMANCE_HISTORY_MAX_ENTRIES:]

    async def _persist_performance_history(self, now) -> None:
        """Persist performance history to disk throttled to a safe cadence."""
        if self._last_performance_persist_time and (now - self._last_performance_persist_time) < THERMAL_PERSIST_INTERVAL:
            return
        serialized: list[dict[str, Any]] = []
        for sample in self._performance_history:
            when = sample.when
            try:
                when = dt_util.as_utc(when)
            except (TypeError, ValueError):
                continue
            serialized.append(
                {
                    "time": when.isoformat(),
                    "indoor_temp": sample.indoor_temp,
                    "target_temp": sample.target_temp,
                    "heating_detected": sample.heating_detected,
                    "price": sample.price,
                    "prediction_error": sample.prediction_error,
                }
            )
        payload = {"version": 1, "history": serialized}
        await self.hass.async_add_executor_job(self._performance_store.save, payload)
        self._last_performance_persist_time = now

    def _serialize_model_history(self) -> list[list[Any]]:
        """Return model history as JSON-friendly triples."""
        serialized: list[list[Any]] = []
        for when, loss, gain in self._model_history:
            if isinstance(when, str):
                parsed = dt_util.parse_datetime(when)
                when_dt = parsed if parsed is not None else None
            else:
                when_dt = when
            if when_dt is None:
                continue
            try:
                when_dt = dt_util.as_utc(when_dt)
                serialized.append([when_dt.isoformat(), float(loss), float(gain)])
            except (TypeError, ValueError):
                continue
        return serialized

    def _restore_model_history(self, payload: dict[str, Any]) -> None:
        """Restore model history from a persisted snapshot."""
        raw_history = payload.get("history")
        if not isinstance(raw_history, list):
            return
        history: list[tuple[Any, float, float]] = []
        for entry in raw_history:
            when_raw = loss_raw = gain_raw = None
            if isinstance(entry, dict):
                when_raw = entry.get("time")
                loss_raw = entry.get("loss")
                gain_raw = entry.get("gain")
            elif isinstance(entry, list) and len(entry) == 3:
                when_raw, loss_raw, gain_raw = entry
            else:
                continue
            if not isinstance(when_raw, str):
                continue
            when = dt_util.parse_datetime(when_raw)
            if when is None:
                continue
            try:
                history.append((dt_util.as_utc(when), float(loss_raw), float(gain_raw)))
            except (TypeError, ValueError):
                continue
        if history:
            self._model_history = history[-MODEL_HISTORY_MAX_ENTRIES:]

    @property
    def temperature_unit(self) -> str | None:
        """Return configured temperature unit."""
        return self.hass.config.units.temperature_unit

    @property
    def current_temperature(self) -> float | None:
        """Return the current indoor temperature."""
        if self._indoor_temp is not None:
            return self._indoor_temp
        return self._get_state_as_float(self._indoor_temp_entity)

    @property
    def target_temperature(self) -> float:
        """Return the active target temperature."""
        return self._target_temperature

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose last MPC decision and forecasts."""
        suggested_virtual_outdoor_temperature = self._last_virtual_outdoor
        last_control_time = (
            self._last_control_time.isoformat(timespec="milliseconds") if self._last_control_time else None
        )
        indoor_temp = self._indoor_temp if self._indoor_temp is not None else self._get_state_as_float(
            self._indoor_temp_entity
        )
        outdoor_temp = self._outdoor_temp if self._outdoor_temp is not None else self._get_state_as_float(
            self._outdoor_temp_entity
        )
        price_ratio_mpc, price_classification_mpc = self._classify_price()
        current_price_raw = self._last_price_forecast[0] if self._last_price_forecast else None
        try:
            current_price = float(current_price_raw) if current_price_raw is not None else None
        except (TypeError, ValueError):
            current_price = None
        history_baseline_24h, history_samples_24h = self._history_price_baseline(24)
        price_ratio_history_24h, price_classification_history_24h = self._classify_price_against_baseline(
            current_price, history_baseline_24h
        )
        heating_detected = self._get_heating_detected(dt_util.utcnow())
        nominal_heat_power_kw = NOMINAL_HEAT_POWER_KW
        time_step_hours = self._controller.time_step_hours
        cap_kwh_per_c = None
        res_c_per_kw = None
        try:
            if self._thermal_model.heat_gain_coeff > 0 and time_step_hours > 0:
                cap_kwh_per_c = (time_step_hours * nominal_heat_power_kw) / self._thermal_model.heat_gain_coeff
                if self._thermal_model.heat_loss_coeff > 0:
                    res_c_per_kw = time_step_hours / (self._thermal_model.heat_loss_coeff * cap_kwh_per_c)
        except (TypeError, ValueError, ZeroDivisionError):
            cap_kwh_per_c = None
            res_c_per_kw = None
        return {
            "suggested_heat_on": self._last_control_on,
            "suggested_virtual_outdoor_temperature": suggested_virtual_outdoor_temperature,
            "last_control_time": last_control_time,
            "control_interval_minutes": self._control_interval,
            "prediction_horizon_hours": self._prediction_horizon,
            "comfort_temperature_tolerance": self._comfort_tolerance,
            "price_comfort_weight": self._price_comfort_weight,
            "current_price": current_price,
            "price_ratio": price_ratio_mpc,
            "price_classification": price_classification_mpc,
            "price_baseline_kind": "median(price_forecast + price_history)",
            "price_baseline_history_24h": history_baseline_24h,
            "price_baseline_history_24h_samples": history_samples_24h,
            "price_ratio_history_24h": price_ratio_history_24h,
            "price_classification_history_24h": price_classification_history_24h,
            "monitor_only": self._monitor_only,
            "virtual_outdoor_heat_offset": self._virtual_heat_offset,
            "heat_loss_coefficient": self._heat_loss_coeff,
            "estimated_heat_loss_coefficient": self._thermal_model.heat_loss_coeff,
            "estimated_heat_gain_coefficient": self._thermal_model.heat_gain_coeff,
            "estimated_indoor_temperature": self._thermal_model.indoor_temp,
            "nominal_heat_power_kw": nominal_heat_power_kw,
            "estimated_thermal_capacitance_kwh_per_c": cap_kwh_per_c,
            "estimated_thermal_resistance_c_per_kw": res_c_per_kw,
            "thermal_model_heat_on_source": self._thermal_model_heat_on_source(),
            "heating_supply_temp_entity": self._heating_supply_temp_entity,
            "heating_supply_temp_threshold": self._heating_supply_temp_threshold,
            "heating_detection_enabled": self._heating_detection_active(),
            "heating_supply_temp_hysteresis": self._heating_supply_temp_hysteresis,
            "heating_supply_temp_debounce_seconds": self._heating_supply_temp_debounce_seconds,
            "learning_model": self._learning_model,
            "rls_forgetting_factor": self._rls_forgetting_factor,
            "heating_detected": heating_detected,
            "heating_supply_temperature": self._get_state_as_float(self._heating_supply_temp_entity)
            if self._heating_supply_temp_entity
            else None,
            "indoor_temperature": indoor_temp,
            "outdoor_temperature": outdoor_temp,
            "last_cost": self._last_result.cost if self._last_result else None,
            "price_baseline": self._last_result.price_baseline if self._last_result else None,
        }

    @property
    def hvac_mode(self) -> HVACMode:
        """Return current HVAC mode."""
        return self._hvac_mode

    @property
    def hvac_action(self) -> HVACAction:
        """Return current action."""
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        return HVACAction.HEATING if self._last_control_on else HVACAction.IDLE

    @property
    def available(self) -> bool:
        """Entity availability based on required sensors."""
        return self.hass.states.get(self._indoor_temp_entity) is not None and self.hass.states.get(
            self._outdoor_temp_entity
        ) is not None

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Handle target temperature changes from the UI."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return

        self._target_temperature = float(temperature)
        await self._persist_options({CONF_TARGET_TEMPERATURE: self._target_temperature})
        self._controller.update_settings(target_temperature=self._target_temperature)
        await self._async_run_control()
        self.async_write_ha_state()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set the HVAC mode."""
        if hvac_mode not in self.hvac_modes:
            _LOGGER.warning("Unsupported HVAC mode: %s", hvac_mode)
            return
        self._hvac_mode = hvac_mode
        if hvac_mode == HVACMode.OFF:
            await self._apply_control(False)
        else:
            await self._async_run_control()
        self.async_write_ha_state()

    async def _async_run_control(self) -> None:
        """Run one MPC iteration."""
        if self._hvac_mode == HVACMode.OFF:
            now = dt_util.utcnow()
            self._last_control_on = False
            self._last_virtual_outdoor = None
            self._publish_decision()
            await self._async_update_notifications(now)
            self.async_write_ha_state()
            return

        async with self._control_lock:
            now = dt_util.utcnow()
            indoor_temp = self._get_state_as_float(self._indoor_temp_entity)
            if indoor_temp is None:
                _LOGGER.warning("Indoor temperature unavailable: %s", self._indoor_temp_entity)
                self._publish_decision()
                await self._async_update_notifications(now)
                self.async_write_ha_state()
                return
            self._indoor_temp = indoor_temp

            outdoor_temp = self._get_state_as_float(self._outdoor_temp_entity)
            if outdoor_temp is not None:
                self._outdoor_temp = outdoor_temp

            prediction_error = self._compute_prediction_error(now, indoor_temp)

            self._update_thermal_model(now, indoor_temp, self._outdoor_temp)
            steps = max(1, int(self._prediction_horizon / self._controller.time_step_hours))
            price_forecast = self._extract_price_forecast(now)
            outdoor_forecast = await self._async_build_outdoor_forecast(now)

            price_expanded = expand_to_steps(price_forecast, steps, self._controller.time_step_hours)
            price_for_mpc = self._normalize_series(price_expanded, steps, 1.0)
            outdoor_for_mpc = expand_to_steps(outdoor_forecast, steps, self._controller.time_step_hours)
            self._last_price_forecast = price_for_mpc
            self._last_outdoor_forecast = outdoor_for_mpc
            self._update_price_history(now, price_for_mpc)

            decision, result = self._controller.suggest_control(
                indoor_temp=indoor_temp,
                outdoor_forecast=outdoor_for_mpc,
                price_forecast=price_for_mpc,
                past_prices=self._price_history,
            )
            self._last_result = result
            self._last_virtual_outdoor = self._compute_virtual_outdoor(decision, outdoor_for_mpc)
            await self._apply_control(decision)
            self._last_control_on = decision
            self._last_control_time = now
            if result is not None:
                self._last_prediction = {
                    "time": now,
                    "step_hours": self._controller.time_step_hours,
                    "temps": list(result.predicted_temperatures),
                }
            current_price = price_for_mpc[0] if price_for_mpc else None
            heating_detected = self._get_heating_detected(now)
            self._record_performance_sample(
                now,
                indoor_temp=indoor_temp,
                target_temp=self._target_temperature,
                heating_detected=heating_detected,
                price=current_price,
                prediction_error=prediction_error,
            )
            self._publish_decision()
            await self._async_update_notifications(now)
            await self._persist_thermal_state(now)
            await self._persist_performance_history(now)
            self.async_write_ha_state()

    def _schedule_control_loop(self) -> None:
        """Schedule the periodic MPC loop."""
        if self._control_unsub:
            self._control_unsub()
        interval = max(1, int(self._control_interval))
        self._control_unsub = async_track_time_interval(
            self.hass, self._handle_control_interval, timedelta(minutes=interval)
        )
        self.async_on_remove(self._control_unsub)

    def _update_thermal_model(self, now, indoor_temp: float, outdoor_temp: float | None) -> None:
        """Advance the thermal model using the last control decision."""
        dt_hours = self._compute_dt_hours(now)
        outdoor_for_model = outdoor_temp
        if outdoor_for_model is None:
            outdoor_for_model = self._get_state_as_float(self._outdoor_temp_entity)
        if outdoor_for_model is None:
            outdoor_for_model = indoor_temp
        heat_on = self._get_heat_on_for_model(now)
        if heat_on is None:
            # No trustworthy heat signal; keep parameters stable and only observe temperature.
            self._thermal_model.observe_temperature(indoor_temp)
            return

        self._thermal_model.step(indoor_temp, outdoor_for_model, heat_on, dt_hours)
        self._heat_loss_coeff = self._thermal_model.heat_loss_coeff
        self._record_model_history(now)
        self._controller.update_settings(
            heat_loss_coeff=self._heat_loss_coeff, heat_gain_coeff=self._thermal_model.heat_gain_coeff
        )
        if self._heating_detection_active():
            self._reset_heating_duty_cycle(now)

    def _record_model_history(self, now) -> None:
        """Record learned coefficients for learning/health status."""
        try:
            when = dt_util.as_utc(now)
        except (TypeError, ValueError):
            when = dt_util.utcnow()
        self._model_history.append((when, float(self._thermal_model.heat_loss_coeff), float(self._thermal_model.heat_gain_coeff)))
        if len(self._model_history) > MODEL_HISTORY_MAX_ENTRIES:
            self._model_history = self._model_history[-MODEL_HISTORY_MAX_ENTRIES:]

    def _heating_detection_active(self) -> bool:
        return bool(self._heating_supply_temp_entity) and bool(self._heating_detection_enabled)

    def _update_heating_detected_from_supply(self, now, supply_temp: float | None) -> bool:
        """Update debounced/hysteretic heating detection.

        Returns True if the committed detection state changed.
        """
        if not self._heating_detection_active() or supply_temp is None:
            if (
                self._heating_detected is None
                and self._heating_detected_candidate is None
                and self._heating_detected_candidate_since is None
            ):
                return False
            self._heating_detected = None
            self._heating_detected_candidate = None
            self._heating_detected_candidate_since = None
            return True

        hysteresis = max(0.0, float(self._heating_supply_temp_hysteresis))
        on_threshold = float(self._heating_supply_temp_threshold) + hysteresis / 2.0
        off_threshold = float(self._heating_supply_temp_threshold) - hysteresis / 2.0

        previous = self._heating_detected
        if previous is True:
            instantaneous = False if supply_temp <= off_threshold else True
        elif previous is False:
            instantaneous = True if supply_temp >= on_threshold else False
        else:
            if supply_temp >= on_threshold:
                instantaneous = True
            elif supply_temp <= off_threshold:
                instantaneous = False
            else:
                instantaneous = None

        # When initializing (e.g. after restart), if we're clearly outside the hysteresis
        # band we can safely commit immediately without waiting for debounce.
        if previous is None and instantaneous is not None:
            self._heating_detected = instantaneous
            self._heating_detected_candidate = None
            self._heating_detected_candidate_since = None
            return True

        if instantaneous == previous:
            self._heating_detected_candidate = None
            self._heating_detected_candidate_since = None
            return False

        debounce_seconds = max(0.0, float(self._heating_supply_temp_debounce_seconds))
        if self._heating_detected_candidate != instantaneous:
            self._heating_detected_candidate = instantaneous
            self._heating_detected_candidate_since = now
            if debounce_seconds == 0:
                self._heating_detected = instantaneous
                self._heating_detected_candidate = None
                self._heating_detected_candidate_since = None
                return True
            return False

        since = self._heating_detected_candidate_since or now
        if (now - since).total_seconds() >= debounce_seconds:
            self._heating_detected = instantaneous
            self._heating_detected_candidate = None
            self._heating_detected_candidate_since = None
            return True
        return False

    def _accumulate_heating_duty_cycle(self, now) -> None:
        """Accumulate heating duty-cycle time since the last update."""
        last_time = self._heating_duty_cycle_last_time
        if last_time is None:
            self._heating_duty_cycle_last_time = now
            self._heating_duty_cycle_last_state = self._heating_detected
            return
        try:
            delta = (now - last_time).total_seconds()
        except (TypeError, ValueError):
            delta = 0.0
        if delta <= 0:
            self._heating_duty_cycle_last_time = now
            self._heating_duty_cycle_last_state = self._heating_detected
            return
        if self._heating_duty_cycle_last_state is True:
            self._heating_duty_cycle_on_seconds += delta
            self._heating_duty_cycle_known_seconds += delta
        elif self._heating_duty_cycle_last_state is False:
            self._heating_duty_cycle_known_seconds += delta
        self._heating_duty_cycle_last_time = now
        self._heating_duty_cycle_last_state = self._heating_detected

    def _heating_duty_cycle_ratio(self) -> float | None:
        """Return the fraction of known time spent heating."""
        if self._heating_duty_cycle_known_seconds <= 0:
            return None
        ratio = self._heating_duty_cycle_on_seconds / self._heating_duty_cycle_known_seconds
        return max(0.0, min(1.0, ratio))

    def _reset_heating_duty_cycle(self, now) -> None:
        """Reset duty-cycle counters for the next control interval."""
        self._heating_duty_cycle_on_seconds = 0.0
        self._heating_duty_cycle_known_seconds = 0.0
        self._heating_duty_cycle_last_time = now
        self._heating_duty_cycle_last_state = self._heating_detected

    def _get_heating_detected(self, now=None) -> bool | None:
        """Return a best-effort 'heating is happening' signal."""
        if self._heating_detection_active():
            supply_temp = self._get_state_as_float(self._heating_supply_temp_entity)
            if now is None:
                now = dt_util.utcnow()
            self._accumulate_heating_duty_cycle(now)
            self._update_heating_detected_from_supply(now, supply_temp)
            self._heating_duty_cycle_last_time = now
            self._heating_duty_cycle_last_state = self._heating_detected
            return self._heating_detected

        entity_id = self._controlled_entity
        if not entity_id:
            return None

        state_obj = self.hass.states.get(entity_id)
        if state_obj is None:
            return None

        domain = entity_id.split(".")[0]
        if domain == "switch":
            return state_obj.state.lower() == "on"

        if domain == "climate":
            hvac_action = state_obj.attributes.get("hvac_action")
            if isinstance(hvac_action, str):
                return hvac_action.lower() == "heating"
            return None

        return None

    def _get_heat_on_for_model(self, now=None) -> float | None:
        """Return a best-effort heat-on signal for the estimator.

        - When not in monitor-only mode, we assume the last applied decision was executed.
        - In monitor-only mode, we only trust the actual controlled entity state when it
          clearly represents heating (e.g. switch on, climate hvac_action = heating).
        """
        if self._heating_detection_active():
            if now is None:
                now = dt_util.utcnow()
            self._accumulate_heating_duty_cycle(now)
            supply_temp = self._get_state_as_float(self._heating_supply_temp_entity)
            self._update_heating_detected_from_supply(now, supply_temp)
            self._heating_duty_cycle_last_time = now
            self._heating_duty_cycle_last_state = self._heating_detected

            duty_cycle = self._heating_duty_cycle_ratio()
            if duty_cycle is not None:
                return duty_cycle
            if supply_temp is None:
                return self._heating_detected

            threshold = float(self._heating_supply_temp_threshold)
            on_margin = max(0.0, float(self._learning_supply_temp_on_margin))
            off_margin = max(0.0, float(self._learning_supply_temp_off_margin))

            # Only return a signal when supply temperature is clearly on/off.
            # Everything in between is treated as ambiguous so the estimator doesn't
            # "learn" from low-power tails, DHW cycles, or other non-space-heating periods.
            if supply_temp >= threshold + on_margin:
                return 1.0
            if supply_temp <= threshold - off_margin:
                return 0.0
            return None

        if not self._monitor_only:
            return float(self._last_control_on)

        entity_id = self._controlled_entity
        if not entity_id:
            return None

        state_obj = self.hass.states.get(entity_id)
        if state_obj is None:
            return None

        domain = entity_id.split(".")[0]
        if domain == "switch":
            return 1.0 if state_obj.state.lower() == "on" else 0.0

        if domain == "climate":
            hvac_action = state_obj.attributes.get("hvac_action")
            if isinstance(hvac_action, str):
                return 1.0 if hvac_action.lower() == "heating" else 0.0
            return None

        return None

    def _compute_dt_hours(self, now) -> float:
        """Compute hours since the last control update."""
        if self._last_control_time is None:
            return max(1 / 60, self._control_interval / 60)
        delta = (now - self._last_control_time).total_seconds() / 3600
        return max(1 / 60, delta)

    async def _handle_control_interval(self, now) -> None:
        """Callback for periodic control loop."""
        self._request_control_run()

    @callback
    def _async_sensor_updated(self, event) -> None:
        """Handle updates from subscribed sensors."""
        entity_id = event.data.get("entity_id")
        new_state = event.data.get("new_state")
        value = self._state_to_float(new_state.state if new_state else None)
        now = dt_util.utcnow()
        if entity_id == self._indoor_temp_entity:
            previous = self._indoor_temp
            self._indoor_temp = value
            self.async_write_ha_state()
            if value is None:
                self._request_control_run()
                return
            if previous is None or abs(value - previous) >= INDOOR_TEMP_CONTROL_THRESHOLD:
                self._request_control_run_debounced()
            return
        if entity_id == self._outdoor_temp_entity:
            previous = self._outdoor_temp
            self._outdoor_temp = value
            self.async_write_ha_state()
            if value is None:
                self._request_control_run()
                return
            if previous is None or abs(value - previous) >= OUTDOOR_TEMP_CONTROL_THRESHOLD:
                self._request_control_run_debounced()
            return
        if entity_id == self._price_entity:
            # React quickly to price changes; the debounce/min-interval handles chattiness.
            old_state = event.data.get("old_state")
            old_value = self._state_to_float(old_state.state if old_state else None)
            if value is None:
                self._request_control_run()
                return
            # Ignore attribute-only updates where the numeric price doesn't change.
            if old_value is not None and abs(value - old_value) < 1e-6:
                return
            self._request_control_run_debounced()
            return
        if entity_id == self._heating_supply_temp_entity and self._heating_detection_active():
            self._accumulate_heating_duty_cycle(now)
            changed = self._update_heating_detected_from_supply(now, value)
            self._heating_duty_cycle_last_time = now
            self._heating_duty_cycle_last_state = self._heating_detected
            self.async_write_ha_state()
            if changed:
                self._request_control_run()
            return

    @callback
    def _resubscribe_sensors(self) -> None:
        """(Re)subscribe to relevant sensor state changes."""
        if self._sensor_unsub:
            self._sensor_unsub()
            self._sensor_unsub = None
        entity_ids: list[str] = [self._indoor_temp_entity, self._outdoor_temp_entity, self._price_entity]
        if self._heating_detection_active():
            entity_ids.append(self._heating_supply_temp_entity)
        self._sensor_unsub = async_track_state_change_event(self.hass, entity_ids, self._async_sensor_updated)

    @callback
    def _handle_entry_update(self) -> None:
        """Handle config entry option updates."""
        previous_options = self._options
        self._options = self._merge_options(self.config_entry.options)
        self._target_temperature = self._options[CONF_TARGET_TEMPERATURE]
        self._price_comfort_weight = self._options[CONF_PRICE_COMFORT_WEIGHT]
        self._price_penalty_curve = self._options[CONF_PRICE_PENALTY_CURVE]
        self._control_interval = self._options[CONF_CONTROL_INTERVAL_MINUTES]
        self._prediction_horizon = self._options[CONF_PREDICTION_HORIZON_HOURS]
        self._comfort_tolerance = self._options[CONF_COMFORT_TEMPERATURE_TOLERANCE]
        self._monitor_only = self._options[CONF_MONITOR_ONLY]
        self._virtual_heat_offset = self._options[CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET]
        self._learning_model = self._options[CONF_LEARNING_MODEL]
        self._rls_forgetting_factor = self._options[CONF_RLS_FORGETTING_FACTOR]
        self._heating_supply_temp_entity = self._options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY)
        self._heating_supply_temp_threshold = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD)
        )
        self._heating_detection_enabled = bool(
            self._options.get(CONF_HEATING_DETECTION_ENABLED, DEFAULT_HEATING_DETECTION_ENABLED)
        )
        self._heating_supply_temp_hysteresis = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_HYSTERESIS, DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS)
        )
        self._heating_supply_temp_debounce_seconds = float(
            self._options.get(CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS)
        )
        self._overshoot_warm_bias_enabled = bool(
            self._options.get(CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED)
        )
        self._overshoot_warm_bias_curve = str(
            self._options.get(CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE)
        )
        self._learning_supply_temp_on_margin = float(
            self._options.get(CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN)
        )
        self._learning_supply_temp_off_margin = float(
            self._options.get(CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN, DEFAULT_LEARNING_SUPPLY_TEMP_OFF_MARGIN)
        )
        self._performance_window_hours = self._options[CONF_PERFORMANCE_WINDOW_HOURS]
        now = dt_util.utcnow()
        if self._heating_detection_active():
            self._reset_heating_duty_cycle(now)
        else:
            self._heating_duty_cycle_on_seconds = 0.0
            self._heating_duty_cycle_known_seconds = 0.0
            self._heating_duty_cycle_last_time = None
            self._heating_duty_cycle_last_state = None

        previous_learning_model = previous_options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL)
        previous_rls_factor = self._state_to_float(previous_options.get(CONF_RLS_FORGETTING_FACTOR))
        if previous_rls_factor is None:
            previous_rls_factor = DEFAULT_RLS_FORGETTING_FACTOR
        model_changed = previous_learning_model != self._learning_model
        rls_changed = float(previous_rls_factor) != float(self._rls_forgetting_factor)

        # Only reseed (reset) the estimator when options that affect its initialization change.
        # This avoids wiping learning progress when a user tweaks unrelated settings like the
        # heating detection threshold/hysteresis.
        if model_changed or rls_changed:
            self._thermal_model = self._build_thermal_model(self._options)
            self._heat_loss_coeff = self._thermal_model.heat_loss_coeff
            self._model_history = []
        elif should_reseed_thermal_model(previous_options, self._options):
            self._heat_loss_coeff = self._options[CONF_HEAT_LOSS_COEFFICIENT]
            initial_heat_loss = self._options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE)
            if initial_heat_loss is None:
                initial_heat_loss = self._heat_loss_coeff
            initial_temp = self._options.get(CONF_INITIAL_INDOOR_TEMP)
            if initial_temp is None:
                initial_temp = self._indoor_temp
            self._thermal_model.reseed(
                seed=self._options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
                initial_heat_loss=initial_heat_loss,
                initial_heat_gain=self._options.get(CONF_INITIAL_HEAT_GAIN),
                initial_temp=initial_temp,
            )
            self._heat_loss_coeff = self._thermal_model.heat_loss_coeff
            self._model_history = []

        self._controller.update_settings(
            target_temperature=self._target_temperature,
            price_comfort_weight=self._price_comfort_weight,
            price_penalty_curve=self._price_penalty_curve,
            comfort_temperature_tolerance=self._comfort_tolerance,
            prediction_horizon_hours=self._prediction_horizon,
            heat_loss_coeff=self._heat_loss_coeff,
            heat_gain_coeff=self._thermal_model.heat_gain_coeff,
            overshoot_warm_bias_enabled=self._overshoot_warm_bias_enabled,
            virtual_heat_offset=self._virtual_heat_offset,
            overshoot_warm_bias_curve=self._overshoot_warm_bias_curve,
        )
        self._resubscribe_sensors()
        self._schedule_control_loop()
        self._request_control_run()
        self.hass.async_create_task(self._persist_thermal_state(dt_util.utcnow()))
        self.async_write_ha_state()

    async def _apply_control(self, heat_on: bool) -> None:
        """Apply the control decision to the underlying entity."""
        if self._monitor_only:
            return

        if self._hvac_mode == HVACMode.OFF:
            heat_on = False

        entity_id = self._controlled_entity
        if entity_id is None:
            return

        domain = entity_id.split(".")[0]

        if domain == "number":
            # Common for heat pumps controlled via an external "virtual outdoor" setpoint.
            if self._last_virtual_outdoor is None:
                return
            desired = float(self._last_virtual_outdoor)

            state_obj = self.hass.states.get(entity_id)
            step = None
            if state_obj is not None:
                min_attr = state_obj.attributes.get("min")
                max_attr = state_obj.attributes.get("max")
                step_attr = state_obj.attributes.get("step")
                try:
                    minimum = float(min_attr) if min_attr is not None else None
                except (TypeError, ValueError):
                    minimum = None
                try:
                    maximum = float(max_attr) if max_attr is not None else None
                except (TypeError, ValueError):
                    maximum = None
                if minimum is not None:
                    desired = max(minimum, desired)
                if maximum is not None:
                    desired = min(maximum, desired)
                try:
                    step = float(step_attr) if step_attr is not None else None
                except (TypeError, ValueError):
                    step = None

            desired_str = None
            desired_float = desired
            if step is not None and step > 0:
                step_dec = Decimal(str(step))
                desired_dec = Decimal(str(desired_float))
                rounded_dec = (desired_dec / step_dec).to_integral_value(rounding=ROUND_HALF_UP) * step_dec
                decimals = max(0, -step_dec.as_tuple().exponent)
                desired_float = float(rounded_dec)
                desired_str = f"{desired_float:.{decimals}f}"
            else:
                desired_str = str(desired_float)

            current_value = self._get_state_as_float(entity_id)
            if current_value is not None and abs(current_value - desired_float) < 0.01:
                return
            await self.hass.services.async_call(
                "number",
                "set_value",
                {"entity_id": entity_id, "value": desired_str},
                blocking=False,
            )
            return

        if domain == "switch":
            # Only skip if the current state already matches the desired action.
            current_on = None
            state_obj = self.hass.states.get(entity_id)
            if state_obj is not None:
                current_on = state_obj.state.lower() == "on"
            if current_on is not None and current_on == heat_on:
                return

            service = "turn_on" if heat_on else "turn_off"
            await self.hass.services.async_call("switch", service, {"entity_id": entity_id}, blocking=False)
            return

        if domain == "climate":
            if heat_on:
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    {"entity_id": entity_id, ATTR_TEMPERATURE: self._target_temperature},
                    blocking=False,
                )
                await self.hass.services.async_call(
                    "climate",
                    "set_hvac_mode",
                    {"entity_id": entity_id, "hvac_mode": HVACMode.HEAT},
                    blocking=False,
                )
            else:
                await self.hass.services.async_call(
                    "climate", "set_hvac_mode", {"entity_id": entity_id, "hvac_mode": HVACMode.OFF}, blocking=False
                )
            return

        if domain == "ohmonwifiplus":
            # Best-effort support for ohmonwifiplus custom entities.
            await self._call_turn_on(domain, entity_id)
            await self._call_temperature_service(domain, entity_id, value=self._last_virtual_outdoor)
            return

        # Generic handler for other domains that expose set_temperature/turn_off services.
        if heat_on:
            # Try to turn on first if available, then set a temperature.
            await self._call_turn_on(domain, entity_id)
            if await self._call_temperature_service(domain, entity_id, value=self._target_temperature):
                return
        else:
            handled = await self._call_turn_off(domain, entity_id)
            if handled:
                return

        _LOGGER.debug("Unsupported controlled entity domain for %s", entity_id)

    async def _call_temperature_service(self, domain: str, entity_id: str, value: float | None) -> bool:
        """Try to call a set_temperature-like service for non-climate entities."""
        if not self.hass.services.has_service(domain, "set_temperature"):
            return False
        if value is None:
            return False
        await self.hass.services.async_call(
            domain,
            "set_temperature",
            {"entity_id": entity_id, ATTR_TEMPERATURE: value},
            blocking=False,
        )
        return True

    async def _call_turn_off(self, domain: str, entity_id: str) -> bool:
        """Try to call a turn_off service if available."""
        if not self.hass.services.has_service(domain, "turn_off"):
            return False
        await self.hass.services.async_call(domain, "turn_off", {"entity_id": entity_id}, blocking=False)
        return True

    async def _call_turn_on(self, domain: str, entity_id: str) -> bool:
        """Try to call a turn_on service if available."""
        if not self.hass.services.has_service(domain, "turn_on"):
            return False
        await self.hass.services.async_call(domain, "turn_on", {"entity_id": entity_id}, blocking=False)
        return True

    def _compute_virtual_outdoor(self, heat_on: bool, outdoor_forecast: list[float]) -> float | None:
        """Compute the virtual outdoor temperature to send to the pump interface."""
        base = None
        if self._outdoor_temp is not None:
            base = self._outdoor_temp
        elif outdoor_forecast:
            base = outdoor_forecast[0]
        else:
            base = self._get_state_as_float(self._outdoor_temp_entity)

        if base is None:
            base = VIRTUAL_OUTDOOR_IDLE_FALLBACK

        if heat_on:
            value = base - self._virtual_heat_offset
        else:
            # When idle, let the pump see the actual outdoor temp (or a safe fallback).
            value = base
            offset = max(0.0, float(self._virtual_heat_offset))
            boost_total = 0.0

            # Price-aware suppression: push the virtual temp warmer during expensive slots.
            current_price = self._last_price_forecast[0] if self._last_price_forecast else None
            price_baseline = self._last_result.price_baseline if self._last_result else None
            if current_price is not None and price_baseline and price_baseline > 0 and offset > 0:
                ratio = current_price / price_baseline
                if ratio > 1.0:
                    suppression = self._price_comfort_weight  # 0-1 weight from options.
                    boost = suppression * (ratio - 1.0) * max(1.0, offset)
                    boost_total += max(0.0, boost)

            # Overshoot warm-bias: when indoor is above target, bias the curve warmer to back off heating.
            if self._overshoot_warm_bias_enabled and self._indoor_temp is not None and offset > 0:
                overshoot = float(self._indoor_temp) - float(self._target_temperature)
                if overshoot > float(self._comfort_tolerance):
                    bias, _, _, _ = compute_overshoot_warm_bias(
                        overshoot,
                        self._comfort_tolerance,
                        offset,
                        self._overshoot_warm_bias_curve,
                    )
                    boost_total += bias

            # Clamp the warm-shift so we never bias more than the configured offset.
            boost_total = min(boost_total, offset)
            value += boost_total

        # Never send a virtual outdoor warmer than 25C (summer/no heat).
        return min(value, MAX_VIRTUAL_OUTDOOR)

    async def _async_build_outdoor_forecast(self, now) -> list[float]:
        """Collect outdoor temperature forecast.

        - Supports sensor entities exposing ``forecast`` attributes.
        - Supports weather entities exposing ``forecast`` attributes.
        - Falls back to the ``weather.get_forecasts`` service when attributes are missing.
        - Falls back to the configured outdoor temperature sensor when no forecast is available.
        """
        now = dt_util.as_utc(now)

        weather_state = self.hass.states.get(self._weather_entity)
        if weather_state:
            raw = weather_state.attributes.get("forecast")
            forecast = self._extract_outdoor_forecast_from_raw(raw, now)
            if forecast:
                self._last_outdoor_forecast_source = "weather_attribute"
                if isinstance(raw, list):
                    self._weather_forecast_cache_raw = list(raw)
                    self._weather_forecast_cache_time = dt_util.utcnow()
                return forecast

        if self._weather_entity.startswith("weather."):
            cached = self._extract_outdoor_forecast_from_cache(now)
            if cached:
                self._last_outdoor_forecast_source = "weather_cache"
                return cached

            raw = await self._async_fetch_weather_forecast_service()
            forecast = self._extract_outdoor_forecast_from_raw(raw, now)
            if forecast:
                self._last_outdoor_forecast_source = "weather_service"
                return forecast

        base = self._outdoor_temp if self._outdoor_temp is not None else self._get_state_as_float(
            self._outdoor_temp_entity
        )
        if base is None:
            self._last_outdoor_forecast_source = "unavailable"
            return []
        horizon = max(1, int(self._prediction_horizon))
        self._last_outdoor_forecast_source = "outdoor_sensor_flat_fallback"
        return [base] * horizon

    def _extract_outdoor_forecast_from_cache(self, now) -> list[float]:
        """Return a cached weather forecast if it's still fresh."""
        raw = self._weather_forecast_cache_raw
        cached_at = self._weather_forecast_cache_time
        if not raw or cached_at is None:
            return []
        try:
            cached_at = dt_util.as_utc(cached_at)
        except (TypeError, ValueError):
            return []
        if (now - cached_at).total_seconds() >= WEATHER_FORECAST_CACHE_SECONDS:
            return []
        return self._extract_outdoor_forecast_from_raw(raw, now)

    def _extract_outdoor_forecast_from_raw(self, raw: Any, now) -> list[float]:
        """Extract an aligned temperature forecast from raw weather forecast data."""
        if not raw:
            return []
        timed = extract_timed_temperatures(raw)
        aligned = align_forecast_to_now(timed, now)
        if aligned:
            return aligned
        return self._extract_temperatures(raw)

    async def _async_fetch_weather_forecast_service(self) -> list[Any]:
        """Fetch an hourly forecast via ``weather.get_forecasts`` when available."""
        if not self._weather_entity.startswith("weather."):
            return []
        if not self.hass.services.has_service("weather", "get_forecasts"):
            return []

        data = {"entity_id": self._weather_entity, "type": WEATHER_FORECAST_SERVICE_TYPE}
        response = None
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                data,
                blocking=True,
                return_response=True,
            )
        except TypeError:
            # Older HA versions don't support service responses; fall back to sensor-only behaviour.
            return []
        except Exception as err:
            _LOGGER.debug("Weather forecast fetch failed for %s: %s", self._weather_entity, err)
            return []

        raw: Any = None
        if isinstance(response, dict):
            candidate = response.get(self._weather_entity)
            if isinstance(candidate, dict):
                raw = candidate.get("forecast")
            elif isinstance(candidate, list):
                raw = candidate
            elif "forecast" in response:
                raw = response.get("forecast")
            elif len(response) == 1:
                only = next(iter(response.values()))
                if isinstance(only, dict):
                    raw = only.get("forecast")
                elif isinstance(only, list):
                    raw = only

        if not isinstance(raw, list):
            return []

        self._weather_forecast_cache_raw = list(raw)
        self._weather_forecast_cache_time = dt_util.utcnow()
        return list(raw)

    def _request_control_run(self) -> None:
        """Coalesce control requests to avoid stacking many queued tasks."""
        if self._sensor_control_debounce_unsub:
            self._sensor_control_debounce_unsub()
            self._sensor_control_debounce_unsub = None

        if self._control_task and not self._control_task.done():
            self._control_rerun_requested = True
            return

        self._control_task = self.hass.async_create_task(self._async_run_control())
        self._control_task.add_done_callback(self._handle_control_task_done)

    @callback
    def _request_control_run_debounced(self) -> None:
        """Schedule a control run after a short debounce window."""
        if self._sensor_control_debounce_unsub:
            return
        try:
            from homeassistant.helpers.event import async_call_later  # type: ignore
        except Exception:  # pragma: no cover
            self._request_control_run()
            return

        delay = SENSOR_CONTROL_DEBOUNCE_SECONDS
        now = dt_util.utcnow()
        if self._last_control_time is not None:
            elapsed = (now - self._last_control_time).total_seconds()
            if elapsed < SENSOR_CONTROL_MIN_INTERVAL_SECONDS:
                delay = max(delay, SENSOR_CONTROL_MIN_INTERVAL_SECONDS - elapsed)

        @callback
        def _run(_now) -> None:
            self._sensor_control_debounce_unsub = None
            if self._last_control_time is not None:
                elapsed = (dt_util.utcnow() - self._last_control_time).total_seconds()
                if elapsed < SENSOR_CONTROL_MIN_INTERVAL_SECONDS:
                    self._request_control_run_debounced()
                    return
            self._request_control_run()

        self._sensor_control_debounce_unsub = async_call_later(
            self.hass, delay, _run
        )
        self.async_on_remove(self._sensor_control_debounce_unsub)

    @callback
    def _handle_control_task_done(self, task: asyncio.Task) -> None:
        """Handle completion of a scheduled control task."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc:
            _LOGGER.exception("Control loop task failed", exc_info=exc)

        now = dt_util.utcnow()
        signature = f"{type(exc).__name__}: {exc}" if exc else None
        updates = self._notification_tracker.update(
            event_id="control_exception",
            signature=signature,
            now=dt_util.as_utc(now),
            trigger_after=timedelta(0),
            cooldown=NOTIFY_EXCEPTION_COOLDOWN,
        )
        if updates:
            messages: dict[str, tuple[str, str]] = {}
            if signature:
                messages["control_exception"] = (
                    f"{self._attr_name}: Control loop error",
                    "\n".join(
                        [
                            "The MPC control task crashed with an exception:",
                            f"{signature}",
                            "",
                            "Check Home Assistant logs for the full traceback.",
                        ]
                    ),
                )
            self.hass.async_create_task(self._async_apply_notification_updates(updates, messages, now))

        if self._control_rerun_requested:
            self._control_rerun_requested = False
            self._control_task = self.hass.async_create_task(self._async_run_control())
            self._control_task.add_done_callback(self._handle_control_task_done)
            return
        self._control_task = None

    @staticmethod
    def _normalize_series(values: Sequence[float] | None, steps: int, fallback: float) -> list[float]:
        """Normalize a list of floats to a specific length."""
        normalized: list[float] = []
        if values:
            for val in values:
                try:
                    normalized.append(float(val))
                except (TypeError, ValueError):
                    continue
        if not normalized:
            normalized = [fallback]
        if len(normalized) < steps:
            normalized.extend([normalized[-1]] * (steps - len(normalized)))
        else:
            normalized = normalized[:steps]
        return normalized

    def _extract_price_forecast(self, now) -> list[float]:
        """Extract price forecast from the configured entity."""
        state = self.hass.states.get(self._price_entity)
        if not state:
            self._last_price_forecast_source = "unavailable"
            return []

        attrs = state.attributes
        forecast: list[float] = []
        source = "attribute_forecast"

        # Prefer Nordpool's raw lists so we can align the series to 'now'.
        raw_today = attrs.get("raw_today")
        raw_tomorrow = attrs.get("raw_tomorrow")
        if raw_today or raw_tomorrow:
            source = "nordpool_raw"
            timed = extract_timed_values(raw_today)
            timed.extend(extract_timed_values(raw_tomorrow))
            forecast = align_forecast_to_now(timed, now)
        else:
            if "prices" in attrs:
                forecast.extend(self._extract_price_list(attrs.get("prices")))
            if "forecast" in attrs and not forecast:
                forecast.extend(self._extract_price_list(attrs.get("forecast")))

        current_price = self._state_to_float(state.state)
        if current_price is not None and not forecast:
            forecast.append(current_price)
            source = "current_price_only"

        if not forecast:
            source = "empty"
        self._last_price_forecast_source = source
        return forecast

    def _extract_temperatures(self, forecast_data: Iterable[Any] | None) -> list[float]:
        """Extract temperature values from forecast data."""
        temperatures: list[float] = []
        if not forecast_data:
            return temperatures
        for item in forecast_data:
            if not isinstance(item, dict):
                continue
            temp = item.get("temperature") or item.get("temp")
            if temp is None:
                continue
            value = self._state_to_float(temp)
            if value is not None:
                temperatures.append(value)
        return temperatures

    def _extract_price_list(self, raw: Iterable[Any] | None) -> list[float]:
        """Extract a list of price values from raw attributes."""
        prices: list[float] = []
        if not raw:
            return prices

        for item in raw:
            if isinstance(item, (int, float, str)):
                val = self._state_to_float(item)
                if val is not None:
                    prices.append(val)
                continue

            if not isinstance(item, dict):
                continue

            value = item.get("value") or item.get("price") or item.get("average") or item.get("total")
            numeric = self._state_to_float(value)
            if numeric is not None:
                prices.append(numeric)
        return prices

    def _merge_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Merge entry options with defaults."""
        control_interval = self._coerce_int(
            options.get(CONF_CONTROL_INTERVAL_MINUTES, DEFAULT_CONTROL_INTERVAL_MINUTES),
            DEFAULT_CONTROL_INTERVAL_MINUTES,
            minimum=1,
        )
        prediction_horizon = self._coerce_int(
            options.get(CONF_PREDICTION_HORIZON_HOURS, DEFAULT_PREDICTION_HORIZON_HOURS),
            DEFAULT_PREDICTION_HORIZON_HOURS,
            minimum=1,
        )
        heating_hysteresis = self._state_to_float(options.get(CONF_HEATING_SUPPLY_TEMP_HYSTERESIS))
        if heating_hysteresis is None:
            heating_hysteresis = DEFAULT_HEATING_SUPPLY_TEMP_HYSTERESIS

        target_temperature = self._state_to_float(options.get(CONF_TARGET_TEMPERATURE))
        if target_temperature is None:
            target_temperature = DEFAULT_TARGET_TEMPERATURE

        price_comfort_weight = self._state_to_float(options.get(CONF_PRICE_COMFORT_WEIGHT))
        if price_comfort_weight is None:
            price_comfort_weight = DEFAULT_PRICE_COMFORT_WEIGHT
        price_comfort_weight = min(1.0, max(0.0, float(price_comfort_weight)))

        price_penalty_curve = options.get(CONF_PRICE_PENALTY_CURVE, DEFAULT_PRICE_PENALTY_CURVE)
        if price_penalty_curve not in PRICE_PENALTY_CURVES:
            price_penalty_curve = DEFAULT_PRICE_PENALTY_CURVE

        comfort_tolerance = self._state_to_float(options.get(CONF_COMFORT_TEMPERATURE_TOLERANCE))
        if comfort_tolerance is None:
            comfort_tolerance = DEFAULT_COMFORT_TEMPERATURE_TOLERANCE

        overshoot_enabled = bool(options.get(CONF_OVERSHOOT_WARM_BIAS_ENABLED, DEFAULT_OVERSHOOT_WARM_BIAS_ENABLED))
        overshoot_curve = options.get(CONF_OVERSHOOT_WARM_BIAS_CURVE, DEFAULT_OVERSHOOT_WARM_BIAS_CURVE)
        if overshoot_curve not in OVERSHOOT_WARM_BIAS_CURVES:
            overshoot_curve = DEFAULT_OVERSHOOT_WARM_BIAS_CURVE

        learning_on_margin = self._state_to_float(options.get(CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN))
        if learning_on_margin is None:
            learning_on_margin = DEFAULT_LEARNING_SUPPLY_TEMP_ON_MARGIN
        learning_on_margin = max(0.0, float(learning_on_margin))

        learning_off_margin = self._state_to_float(options.get(CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN))
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
        learning_model = options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL)
        if learning_model not in (LEARNING_MODEL_EKF, LEARNING_MODEL_RLS):
            learning_model = DEFAULT_LEARNING_MODEL
        rls_factor = self._state_to_float(options.get(CONF_RLS_FORGETTING_FACTOR))
        if rls_factor is None:
            rls_factor = DEFAULT_RLS_FORGETTING_FACTOR
        rls_factor = min(1.0, max(0.9, float(rls_factor)))
        merged = {
            CONF_TARGET_TEMPERATURE: target_temperature,
            CONF_PRICE_COMFORT_WEIGHT: price_comfort_weight,
            CONF_PRICE_PENALTY_CURVE: price_penalty_curve,
            CONF_CONTROL_INTERVAL_MINUTES: control_interval,
            CONF_PREDICTION_HORIZON_HOURS: prediction_horizon,
            CONF_COMFORT_TEMPERATURE_TOLERANCE: comfort_tolerance,
            CONF_MONITOR_ONLY: options.get(CONF_MONITOR_ONLY, DEFAULT_MONITOR_ONLY),
            CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET: options.get(
                CONF_VIRTUAL_OUTDOOR_HEAT_OFFSET, DEFAULT_VIRTUAL_OUTDOOR_HEAT_OFFSET
            ),
            CONF_OVERSHOOT_WARM_BIAS_ENABLED: overshoot_enabled,
            CONF_OVERSHOOT_WARM_BIAS_CURVE: overshoot_curve,
            CONF_HEAT_LOSS_COEFFICIENT: options.get(CONF_HEAT_LOSS_COEFFICIENT, DEFAULT_HEAT_LOSS_COEFFICIENT),
            CONF_THERMAL_RESPONSE_SEED: options.get(CONF_THERMAL_RESPONSE_SEED, DEFAULT_THERMAL_RESPONSE_SEED),
            CONF_LEARNING_MODEL: learning_model,
            CONF_RLS_FORGETTING_FACTOR: rls_factor,
            CONF_PERFORMANCE_WINDOW_HOURS: performance_window,
            CONF_HEATING_SUPPLY_TEMP_ENTITY: options.get(CONF_HEATING_SUPPLY_TEMP_ENTITY),
            CONF_HEATING_SUPPLY_TEMP_THRESHOLD: options.get(
                CONF_HEATING_SUPPLY_TEMP_THRESHOLD, DEFAULT_HEATING_SUPPLY_TEMP_THRESHOLD
            ),
            CONF_HEATING_DETECTION_ENABLED: bool(
                options.get(CONF_HEATING_DETECTION_ENABLED, DEFAULT_HEATING_DETECTION_ENABLED)
            ),
            CONF_HEATING_SUPPLY_TEMP_HYSTERESIS: heating_hysteresis,
            CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS: self._coerce_int(
                options.get(CONF_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS, DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS),
                DEFAULT_HEATING_SUPPLY_TEMP_DEBOUNCE_SECONDS,
                minimum=0,
            ),
            CONF_LEARNING_SUPPLY_TEMP_ON_MARGIN: learning_on_margin,
            CONF_LEARNING_SUPPLY_TEMP_OFF_MARGIN: learning_off_margin,
            CONF_INITIAL_INDOOR_TEMP: options.get(CONF_INITIAL_INDOOR_TEMP),
            CONF_INITIAL_HEAT_GAIN: options.get(CONF_INITIAL_HEAT_GAIN),
            CONF_INITIAL_HEAT_LOSS_OVERRIDE: options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE),
        }
        return merged

    def _build_thermal_model(self, options: dict[str, Any]) -> ThermalModelEstimator | ThermalModelRlsEstimator:
        """Create a thermal model estimator based on options."""
        base_loss = options.get(CONF_HEAT_LOSS_COEFFICIENT, DEFAULT_HEAT_LOSS_COEFFICIENT)
        initial_heat_loss = options.get(CONF_INITIAL_HEAT_LOSS_OVERRIDE, base_loss)
        if initial_heat_loss is None:
            initial_heat_loss = base_loss
        learning_model = options.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL)
        if learning_model not in (LEARNING_MODEL_EKF, LEARNING_MODEL_RLS):
            learning_model = DEFAULT_LEARNING_MODEL
        rls_factor = self._state_to_float(options.get(CONF_RLS_FORGETTING_FACTOR))
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

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int | None = None) -> int:
        """Convert value to an int with optional minimum clamp."""
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            numeric = default
        if minimum is not None:
            numeric = max(minimum, numeric)
        return numeric

    def _publish_decision(self) -> None:
        """Publish the latest decision for diagnostics sensors."""
        now = dt_util.utcnow()
        suggested_virtual_outdoor_temperature = self._last_virtual_outdoor
        price_ratio_mpc, price_classification_mpc = self._classify_price()
        health, health_reasons = self._compute_health(now)
        learning_state, learning_details = self._compute_learning_state(now)
        heating_detected = self._get_heating_detected(now)
        overshoot_bias, overshoot_warm_bias_multiplier, overshoot_min_bias, overshoot_max_bias = (
            self._comfort_overshoot_multiplier()
        )
        current_price_raw = self._last_price_forecast[0] if self._last_price_forecast else None
        try:
            current_price = float(current_price_raw) if current_price_raw is not None else None
        except (TypeError, ValueError):
            current_price = None
        history_baseline_24h, history_samples_24h = self._history_price_baseline(24)
        price_ratio_history_24h, price_classification_history_24h = self._classify_price_against_baseline(
            current_price, history_baseline_24h
        )
        heating_supply_temperature = (
            self._get_state_as_float(self._heating_supply_temp_entity) if self._heating_supply_temp_entity else None
        )
        heating_duty_cycle_ratio = None
        if self._heating_detection_active():
            heating_duty_cycle_ratio = self._heating_duty_cycle_ratio()
        heating_detected_source = (
            "heating_supply_temp_threshold"
            if self._heating_detection_active()
            else ("controlled_entity" if self._controlled_entity else "none")
        )
        base_outdoor_fallback = (
            self._outdoor_temp
            if self._outdoor_temp is not None
            else (self._last_outdoor_forecast[0] if self._last_outdoor_forecast else self._get_state_as_float(self._outdoor_temp_entity))
        )
        if base_outdoor_fallback is None:
            base_outdoor_fallback = VIRTUAL_OUTDOOR_IDLE_FALLBACK
        planned_virtual_outdoor = compute_planned_virtual_outdoor_temperatures(
            self._last_result.sequence if self._last_result else None,
            self._last_outdoor_forecast,
            self._last_price_forecast,
            predicted_temperatures=self._last_result.predicted_temperatures if self._last_result else None,
            base_outdoor_fallback=float(base_outdoor_fallback),
            virtual_heat_offset=self._virtual_heat_offset,
            price_comfort_weight=self._price_comfort_weight,
            price_baseline=self._last_result.price_baseline if self._last_result else None,
            target_temperature=self._target_temperature,
            overshoot_warm_bias_enabled=self._overshoot_warm_bias_enabled,
            comfort_temperature_tolerance=self._comfort_tolerance,
            overshoot_warm_bias_curve=self._overshoot_warm_bias_curve,
            max_virtual_outdoor=MAX_VIRTUAL_OUTDOOR,
        )
        payload = {
            "suggested_heat_on": self._last_control_on,
            "suggested_virtual_outdoor_temperature": suggested_virtual_outdoor_temperature,
            "planned_virtual_outdoor_temperatures": planned_virtual_outdoor,
            "target_temperature": self._target_temperature,
            "monitor_only": self._monitor_only,
            "control_state": "monitoring" if self._monitor_only else "controlling",
            "hvac_mode": self._hvac_mode,
            "hvac_action": self.hvac_action,
            "last_control_time": self._last_control_time.isoformat(timespec="milliseconds")
            if self._last_control_time
            else None,
            "health": health,
            "health_reasons": health_reasons,
            "learning_state": learning_state,
            "learning_details": learning_details,
            "heating_detected": heating_detected,
            "heating_detected_source": heating_detected_source,
            "heating_supply_temp_entity": self._heating_supply_temp_entity,
            "heating_supply_temp_threshold": self._heating_supply_temp_threshold,
            "heating_detection_enabled": self._heating_detection_active(),
            "heating_supply_temp_hysteresis": self._heating_supply_temp_hysteresis,
            "heating_supply_temp_debounce_seconds": self._heating_supply_temp_debounce_seconds,
            "heating_supply_temperature": heating_supply_temperature,
            "heating_duty_cycle_ratio": heating_duty_cycle_ratio,
            "virtual_outdoor_heat_offset": self._virtual_heat_offset,
            "overshoot_warm_bias_enabled": self._overshoot_warm_bias_enabled,
            "overshoot_warm_bias_curve": self._overshoot_warm_bias_curve,
            "overshoot_warm_bias_min_bias": overshoot_min_bias,
            "overshoot_warm_bias_max_bias": overshoot_max_bias,
            "overshoot_warm_bias_applied": overshoot_bias,
            "overshoot_warm_bias_multiplier": overshoot_warm_bias_multiplier,
            "learning_supply_temp_on_margin": self._learning_supply_temp_on_margin,
            "learning_supply_temp_off_margin": self._learning_supply_temp_off_margin,
            "learning_model": self._learning_model,
            "rls_forgetting_factor": self._rls_forgetting_factor,
            "heat_loss_coefficient": self._heat_loss_coeff,
            "price_baseline": self._last_result.price_baseline if self._last_result else None,
            "current_price": current_price,
            "price_ratio": price_ratio_mpc,
            "price_classification": price_classification_mpc,
            "price_baseline_kind": "median(price_forecast + price_history)",
            "price_baseline_history_24h": history_baseline_24h,
            "price_baseline_history_24h_samples": history_samples_24h,
            "price_ratio_history_24h": price_ratio_history_24h,
            "price_classification_history_24h": price_classification_history_24h,
            "cost": self._last_result.cost if self._last_result else None,
            "predicted_temperatures": self._last_result.predicted_temperatures if self._last_result else None,
            "price_forecast": self._last_price_forecast,
            "outdoor_forecast": self._last_outdoor_forecast,
            "price_history": self._price_history,
            "price_history_samples": len(self._price_history),
            "price_history_source": self._price_history_source,
        }
        entry_data = self.hass.data.setdefault(DOMAIN, {}).setdefault(self.config_entry.entry_id, {})
        entry_data["last_decision"] = payload
        entry_data["performance"] = self._compute_performance_summary(now)
        async_dispatcher_send(self.hass, f"{SIGNAL_DECISION_UPDATED}_{self.config_entry.entry_id}")

    def _comfort_overshoot_multiplier(self) -> tuple[float, float, float, float]:
        """Return (bias, multiplier, min_bias, max_bias) for above-target comfort penalty."""
        if self._indoor_temp is None:
            return 0.0, 1.0, 0.0, 0.0
        if not self._overshoot_warm_bias_enabled:
            return 0.0, 1.0, 0.0, 0.0
        delta = float(self._indoor_temp) - float(self._target_temperature)
        return compute_overshoot_warm_bias(
            delta,
            self._comfort_tolerance,
            self._virtual_heat_offset,
            self._overshoot_warm_bias_curve,
        )

    def _notification_id(self, event_id: str) -> str:
        return f"{DOMAIN}_{self.config_entry.entry_id}_{event_id}"

    async def _async_call_persistent_notification(self, func, *args, **kwargs) -> None:
        """Call a persistent_notification helper that might be sync or async."""
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _format_age(age_seconds: float | None) -> str | None:
        if age_seconds is None:
            return None
        minutes = int(max(0.0, age_seconds) // 60)
        if minutes < 1:
            return "<1m"
        hours, rem = divmod(minutes, 60)
        if hours:
            return f"{hours}h{rem}m" if rem else f"{hours}h"
        return f"{minutes}m"

    def _entity_problem(self, entity_id: str, now, *, stale_after: timedelta | None) -> tuple[str | None, float | None]:
        """Return (problem, age_seconds) for an entity, if any."""
        state_obj = self.hass.states.get(entity_id)
        if state_obj is None:
            return "missing", None
        if state_obj.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return "unavailable", None
        if stale_after is None:
            return None, None
        try:
            updated = dt_util.as_utc(state_obj.last_updated)
            now_utc = dt_util.as_utc(now)
        except (TypeError, ValueError, AttributeError):
            return None, None
        age = (now_utc - updated).total_seconds()
        if age > stale_after.total_seconds():
            return "stale", age
        return None, None

    def _collect_sensor_issues(self, now) -> list[tuple[str, str, float | None]]:
        """Collect issues for the sensors this integration depends on."""
        issues: list[tuple[str, str, float | None]] = []
        # Many temperature sensors/entities only emit state updates when the value changes,
        # so "last_updated" can be old even when the value is still reasonable to use.
        # Prefer a long staleness window to avoid noisy false positives.
        stale_required_temperature = timedelta(hours=6)
        # Supply temperature is only used for heating detection; if it stops updating, the
        # detection signal becomes unreliable relatively quickly.
        stale_required_supply = timedelta(minutes=max(3 * int(self._control_interval), 60))

        problem, age = self._entity_problem(self._indoor_temp_entity, now, stale_after=stale_required_temperature)
        if problem:
            issues.append((self._indoor_temp_entity, problem, age))

        problem, age = self._entity_problem(self._outdoor_temp_entity, now, stale_after=stale_required_temperature)
        if problem:
            issues.append((self._outdoor_temp_entity, problem, age))

        if self._heating_detection_active() and self._heating_supply_temp_entity:
            problem, age = self._entity_problem(self._heating_supply_temp_entity, now, stale_after=stale_required_supply)
            if problem:
                issues.append((self._heating_supply_temp_entity, problem, age))

        for entity_id in (self._price_entity, self._weather_entity):
            problem, age = self._entity_problem(entity_id, now, stale_after=None)
            if problem:
                issues.append((entity_id, problem, age))

        if not self._monitor_only and self._controlled_entity:
            problem, age = self._entity_problem(self._controlled_entity, now, stale_after=None)
            if problem:
                issues.append((self._controlled_entity, problem, age))

        return issues

    async def _async_apply_notification_updates(
        self,
        updates: list[NotificationUpdate],
        messages: dict[str, tuple[str, str]],
        now,
    ) -> None:
        """Apply NotificationTracker decisions via persistent notifications."""
        if not updates:
            return

        recovered_id = self._notification_id("recovery")
        for update in updates:
            notification_id = self._notification_id(update.event_id)
            if update.action == "create":
                title, message = messages.get(update.event_id, (self._attr_name, ""))
                await self._async_call_persistent_notification(
                    persistent_notification.async_create,
                    self.hass,
                    message,
                    title=title,
                    notification_id=notification_id,
                )
                continue

            if update.action != "dismiss":
                continue

            await self._async_call_persistent_notification(
                persistent_notification.async_dismiss, self.hass, notification_id
            )
            # Write/update a single recovery notification to avoid spamming.
            label = {
                "health": "health",
                "sensors": "sensor inputs",
                "fallback_price": "price data",
                "fallback_weather": "weather forecast",
                "control_exception": "control loop",
            }.get(update.event_id, update.event_id)
            recovery_title = f"{self._attr_name}: Recovered"
            recovery_message = f"Recovered from {label} issue at {dt_util.as_utc(now).isoformat()}."
            await self._async_call_persistent_notification(
                persistent_notification.async_create,
                self.hass,
                recovery_message,
                title=recovery_title,
                notification_id=recovered_id,
            )

    async def _async_update_notifications(self, now) -> None:
        """Update persistent notifications for health and data issues."""
        now = dt_util.as_utc(now)
        tracker = self._notification_tracker
        messages: dict[str, tuple[str, str]] = {}
        updates: list[NotificationUpdate] = []

        # 1) Controller health.
        health, reasons = self._compute_health(now)
        health_signature = None if health == "healthy" else health
        health_trigger_after = timedelta(0)
        if (now - self._startup_time) < STARTUP_HEALTH_GRACE:
            health_signature = None
        if health_signature and reasons == ["no_control_run_yet"]:
            health_trigger_after = timedelta(minutes=max(2 * int(self._control_interval), 15))
        if health_signature:
            title = f"{self._attr_name}: Health {health}"
            reason_text = ", ".join(reasons) if reasons else "unknown"
            messages["health"] = (
                title,
                "\n".join(
                    [
                        f"Health state: {health}",
                        f"Reasons: {reason_text}",
                        "",
                        "See the diagnostic sensors (e.g. 'Heat Pump Pilot Decision') for details.",
                    ]
                ),
            )
        updates.extend(
            tracker.update(
                event_id="health",
                signature=health_signature,
                now=now,
                trigger_after=health_trigger_after,
                cooldown=NOTIFY_HEALTH_COOLDOWN,
            )
        )

        # 2) Sensor staleness/unavailability.
        if (now - self._startup_time) < STARTUP_SENSOR_GRACE:
            sensor_issues = []
        else:
            sensor_issues = self._collect_sensor_issues(now)
        sensors_signature = None if not sensor_issues else "|".join(f"{eid}:{problem}" for (eid, problem, _age) in sensor_issues)
        if sensors_signature:
            lines = ["One or more input entities are unavailable or stale:"]
            for entity_id, problem, age in sensor_issues:
                line = f"- {entity_id}: {problem}"
                if problem == "stale":
                    age_text = self._format_age(age)
                    if age_text:
                        line += f" (last updated {age_text} ago)"
                lines.append(line)
            lines.append("")
            lines.append("The controller may fall back to less accurate data until this resolves.")
            messages["sensors"] = (f"{self._attr_name}: Sensor inputs", "\n".join(lines))
        updates.extend(
            tracker.update(
                event_id="sensors",
                signature=sensors_signature,
                now=now,
                trigger_after=timedelta(minutes=5),
                cooldown=NOTIFY_SENSORS_COOLDOWN,
            )
        )

        # 4) Data fallback warnings (delayed).
        price_fallback = self._last_price_forecast_source in {"unavailable", "empty", "current_price_only"}
        price_signature = self._last_price_forecast_source if price_fallback else None
        if price_signature:
            messages["fallback_price"] = (
                f"{self._attr_name}: Price data fallback",
                "\n".join(
                    [
                        f"Price source: {self._last_price_forecast_source}",
                        f"Entity: {self._price_entity}",
                        "",
                        "MPC is using fallback/flat price data; price-aware decisions may be less accurate.",
                    ]
                ),
            )
        updates.extend(
            tracker.update(
                event_id="fallback_price",
                signature=price_signature,
                now=now,
                trigger_after=NOTIFY_FALLBACK_TRIGGER_AFTER,
                cooldown=NOTIFY_FALLBACK_COOLDOWN,
            )
        )

        weather_fallback = self._last_outdoor_forecast_source in {"outdoor_sensor_flat_fallback", "unavailable"}
        weather_signature = self._last_outdoor_forecast_source if weather_fallback else None
        if weather_signature:
            messages["fallback_weather"] = (
                f"{self._attr_name}: Weather forecast fallback",
                "\n".join(
                    [
                        f"Outdoor forecast source: {self._last_outdoor_forecast_source}",
                        f"Entity: {self._weather_entity}",
                        "",
                        "MPC is using fallback outdoor temperatures; predictions may be less accurate.",
                    ]
                ),
            )
        updates.extend(
            tracker.update(
                event_id="fallback_weather",
                signature=weather_signature,
                now=now,
                trigger_after=NOTIFY_FALLBACK_TRIGGER_AFTER,
                cooldown=NOTIFY_FALLBACK_COOLDOWN,
            )
        )

        await self._async_apply_notification_updates(updates, messages, now)

    def _compute_learning_state(self, now) -> tuple[str, dict[str, Any]]:
        """Return a high-level learning state for the thermal model."""
        source = self._thermal_model_heat_on_source()
        if source == "none":
            return "disabled", {"reason": "no_heat_on_signal"}

        when = dt_util.as_utc(now)
        cutoff = when - LEARNING_STABLE_WINDOW
        samples = [(t, loss, gain) for (t, loss, gain) in self._model_history if t >= cutoff]
        if len(samples) < 4:
            return "learning", {"source": source, "samples": len(samples), "window_hours": LEARNING_STABLE_WINDOW.total_seconds() / 3600}

        first_t, first_loss, first_gain = samples[0]
        last_t, last_loss, last_gain = samples[-1]
        loss_delta = abs(last_loss - first_loss) / max(abs(first_loss), 1e-6)
        gain_delta = abs(last_gain - first_gain) / max(abs(first_gain), 1e-6)
        state = "stable" if (loss_delta < LEARNING_STABLE_RELATIVE_DELTA and gain_delta < LEARNING_STABLE_RELATIVE_DELTA) else "learning"
        return state, {
            "source": source,
            "window_hours": LEARNING_STABLE_WINDOW.total_seconds() / 3600,
            "samples": len(samples),
            "loss_change_ratio": loss_delta,
            "gain_change_ratio": gain_delta,
            "first_sample_time": first_t.isoformat(),
            "last_sample_time": last_t.isoformat(),
        }

    def _compute_health(self, now) -> tuple[str, list[str]]:
        """Compute a simple health classification and reasons."""
        reasons: list[str] = []

        indoor = self._get_state_as_float(self._indoor_temp_entity)
        if indoor is None:
            return "unavailable", ["indoor_temperature_unavailable"]

        outdoor = self._get_state_as_float(self._outdoor_temp_entity)
        if outdoor is None:
            return "unavailable", ["outdoor_temperature_unavailable"]

        if self._last_control_time is None:
            reasons.append("no_control_run_yet")
        else:
            age = (dt_util.as_utc(now) - dt_util.as_utc(self._last_control_time)).total_seconds()
            stale_after = max(2 * self._control_interval * 60, 15 * 60)
            if age > stale_after:
                reasons.append("control_loop_stale")

        if self.hass.states.get(self._price_entity) is None:
            reasons.append("price_entity_unavailable")
        if self.hass.states.get(self._weather_entity) is None:
            reasons.append("weather_entity_unavailable")
        if not self._last_price_forecast:
            reasons.append("price_forecast_empty")
        if self._last_result and (self._last_result.price_baseline is None or self._last_result.price_baseline <= 0):
            reasons.append("price_baseline_invalid")

        if self._heating_supply_temp_entity and self.hass.states.get(self._heating_supply_temp_entity) is None:
            reasons.append("heating_supply_temp_unavailable")

        if not self._monitor_only:
            if not self._controlled_entity:
                reasons.append("no_controlled_entity_configured")
            else:
                state_obj = self.hass.states.get(self._controlled_entity)
                if state_obj is None:
                    reasons.append("controlled_entity_unavailable")
                else:
                    domain = self._controlled_entity.split(".")[0]
                    if domain == "number" and not self.hass.services.has_service("number", "set_value"):
                        reasons.append("service_number_set_value_missing")
                    elif domain == "switch" and not self.hass.services.has_service("switch", "turn_on"):
                        reasons.append("service_switch_turn_on_missing")
                    elif domain == "climate" and not self.hass.services.has_service("climate", "set_temperature"):
                        reasons.append("service_climate_set_temperature_missing")

        # If we're missing only "nice-to-have" inputs, still report degraded.
        if "no_control_run_yet" in reasons:
            return "unavailable", reasons
        return ("healthy" if not reasons else "degraded"), reasons

    async def _persist_options(self, updates: dict[str, Any]) -> None:
        """Persist updated options on the config entry."""
        new_options = {**self.config_entry.options, **updates}
        self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)

    def _get_state_as_float(self, entity_id: str) -> float | None:
        """Get a state value as float."""
        state_obj = self.hass.states.get(entity_id)
        if not state_obj:
            return None
        return self._state_to_float(state_obj.state)

    @staticmethod
    def _state_to_float(value: StateType | str | None) -> float | None:
        """Convert a state to float if possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _update_price_history(self, now, price_forecast: list[float]) -> None:
        """Maintain a simple rolling history of observed prices for baseline comparisons."""
        if not price_forecast:
            return
        current_price = price_forecast[0]
        if current_price is None:
            return
        try:
            price_value = float(current_price)
        except (TypeError, ValueError):
            return
        now = dt_util.as_utc(now)

        # Only store one sample per 15-minute bucket, even if we run control more often.
        bucket_minutes = int(round(self._controller.time_step_hours * 60)) or 15
        bucket_start = now.replace(
            minute=(now.minute // bucket_minutes) * bucket_minutes,
            second=0,
            microsecond=0,
        )
        if self._last_price_bucket_start == bucket_start:
            return
        self._last_price_bucket_start = bucket_start

        self._price_history.append(price_value)
        if len(self._price_history) > PRICE_HISTORY_MAX_ENTRIES:
            self._price_history = self._price_history[-PRICE_HISTORY_MAX_ENTRIES:]
        self.hass.async_create_task(self._persist_price_history())

    @staticmethod
    def _price_label_from_ratio(ratio: float) -> str:
        """Map a price ratio to a human-friendly category."""
        if ratio < 0.75:
            return "very_low"
        if ratio < 0.9:
            return "low"
        if ratio < 1.1:
            return "normal"
        if ratio < 1.3:
            return "high"
        if ratio < 1.6:
            return "very_high"
        return "extreme"

    @classmethod
    def _classify_price_against_baseline(
        cls, current_price: float | None, baseline: float | None
    ) -> tuple[float | None, str | None]:
        """Classify a price against a given baseline."""
        if current_price is None or baseline is None:
            return None, None
        if baseline <= 0:
            return None, None
        ratio = current_price / baseline
        return ratio, cls._price_label_from_ratio(ratio)

    def _history_price_baseline(self, lookback_hours: int) -> tuple[float | None, int]:
        """Compute a median baseline from the stored observed price history."""
        if lookback_hours <= 0:
            return None, 0
        steps_per_hour = int(round(1 / self._controller.time_step_hours)) or 1
        max_samples = lookback_hours * steps_per_hour
        samples = self._price_history[-max_samples:] if max_samples > 0 else []
        if len(samples) < PRICE_HISTORY_BACKFILL_MIN_SAMPLES:
            return None, len(samples)
        baseline = median(samples)
        try:
            baseline_value = float(baseline)
        except (TypeError, ValueError):
            return None, len(samples)
        if baseline_value <= 0:
            return None, len(samples)
        return baseline_value, len(samples)

    def _classify_price(self) -> tuple[float | None, str | None]:
        """Classify current price relative to the MPC baseline used for optimization."""
        current_raw = self._last_price_forecast[0] if self._last_price_forecast else None
        baseline_raw = self._last_result.price_baseline if self._last_result else None
        try:
            current = float(current_raw) if current_raw is not None else None
        except (TypeError, ValueError):
            current = None
        try:
            baseline = float(baseline_raw) if baseline_raw is not None else None
        except (TypeError, ValueError):
            baseline = None
        return self._classify_price_against_baseline(current, baseline)

    def _thermal_model_heat_on_source(self) -> str:
        """Expose what heat-on signal (if any) drives the estimator."""
        if self._heating_detection_active():
            return "heating_supply_temp_threshold"
        if not self._monitor_only:
            return "mpc_applied_decision"
        entity_id = self._controlled_entity
        if not entity_id:
            return "none"
        domain = entity_id.split(".")[0]
        if domain == "switch":
            return "controlled_entity_switch_state"
        if domain == "climate":
            return "controlled_entity_climate_hvac_action"
        return "none"

    def _compute_prediction_error(self, now, indoor_temp: float) -> float | None:
        """Compare current temperature to last prediction, if available."""
        if not self._last_prediction:
            return None
        origin = self._last_prediction.get("time")
        step_hours = self._last_prediction.get("step_hours")
        temps = self._last_prediction.get("temps")
        if origin is None or step_hours is None or not temps:
            return None
        try:
            elapsed_hours = (dt_util.as_utc(now) - dt_util.as_utc(origin)).total_seconds() / 3600
        except (TypeError, ValueError):
            return None
        if elapsed_hours <= 0 or step_hours <= 0:
            return None
        index = int(round(elapsed_hours / step_hours))
        if index <= 0:
            return None
        if index >= len(temps):
            index = len(temps) - 1
        try:
            predicted = float(temps[index])
        except (TypeError, ValueError):
            return None
        return float(indoor_temp) - predicted

    def _record_performance_sample(
        self,
        now,
        *,
        indoor_temp: float,
        target_temp: float,
        heating_detected: bool | None,
        price: float | None,
        prediction_error: float | None,
    ) -> None:
        """Store a performance sample for scoring."""
        try:
            when = dt_util.as_utc(now)
        except (TypeError, ValueError):
            when = dt_util.utcnow()
        try:
            indoor = float(indoor_temp)
            target = float(target_temp)
        except (TypeError, ValueError):
            return
        sample = PerformanceSample(
            when=when,
            indoor_temp=indoor,
            target_temp=target,
            heating_detected=heating_detected,
            price=price,
            prediction_error=prediction_error,
        )
        self._performance_history.append(sample)
        if len(self._performance_history) > PERFORMANCE_HISTORY_MAX_ENTRIES:
            self._performance_history = self._performance_history[-PERFORMANCE_HISTORY_MAX_ENTRIES:]

    def _compute_performance_summary(self, now) -> dict[str, Any]:
        """Compute performance metrics over the configured window."""
        window_hours = self._performance_window_hours
        try:
            cutoff = dt_util.as_utc(now) - timedelta(hours=window_hours)
        except (TypeError, ValueError):
            cutoff = dt_util.utcnow() - timedelta(hours=window_hours)
        samples = [sample for sample in self._performance_history if sample.when >= cutoff]

        comfort_score, comfort_details = compute_comfort_score(samples, self._comfort_tolerance)
        price_score, price_details = compute_price_score(samples)
        prediction_mae, prediction_details = compute_prediction_accuracy(samples)

        comfort_details["window_hours"] = window_hours
        price_details["window_hours"] = window_hours
        price_details["sample_interval_minutes"] = self._control_interval
        prediction_details["window_hours"] = window_hours

        return {
            "window_hours": window_hours,
            "comfort_score": comfort_score,
            "comfort_details": comfort_details,
            "price_score": price_score,
            "price_details": price_details,
            "prediction_mae": prediction_mae,
            "prediction_details": prediction_details,
        }
