"""Helpers for effective heat-request diagnostics and anti-chatter limiting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANTI_CHATTER_REFERENCE_SECONDS = 15 * 60
ANTI_CHATTER_MAX_STEP_CHANGE = 0.25
ANTI_CHATTER_COMFORT_RECOVERY_DELTA = 0.4

HEAT_REQUEST_STATE_IDLE = "idle"
HEAT_REQUEST_STATE_LOW = "low"
HEAT_REQUEST_STATE_MEDIUM = "medium"
HEAT_REQUEST_STATE_HIGH = "high"
HEAT_REQUEST_STATE_FULL = "full"

HEAT_DETECTION_GAP_NONE = "none"
HEAT_DETECTION_GAP_IDLE_WHILE_HEATING = "idle_but_heating_detected"
HEAT_DETECTION_GAP_WEAK_NO_RESPONSE = "weak_request_no_response"
HEAT_DETECTION_GAP_STRONG_NO_RESPONSE = "strong_request_no_response"


@dataclass(frozen=True)
class EffectiveHeatRequest:
    """Normalized and possibly limited effective heat request."""

    raw_requested_duty_ratio: float
    effective_requested_duty_ratio: float
    effective_heat_request_state: str
    anti_chatter_limited: bool
    anti_chatter_reason: str | None


def resolve_effective_heat_request(
    *,
    raw_requested_duty_ratio: float | None,
    previous_effective_duty_ratio: float | None,
    elapsed_seconds: float,
    seconds_since_increase: float,
    predicted_temp: float | None,
    target_temperature: float,
    comfort_tolerance: float,
    anti_chatter_enabled: bool,
) -> EffectiveHeatRequest:
    """Return an effective duty ratio after internal anti-chatter limiting."""

    raw_ratio = normalize_requested_duty_ratio(raw_requested_duty_ratio, fallback=0.0)
    desired_ratio = raw_ratio
    previous_ratio = (
        normalize_requested_duty_ratio(previous_effective_duty_ratio, fallback=0.0)
        if previous_effective_duty_ratio is not None
        else None
    )
    if not anti_chatter_enabled or previous_ratio is None:
        return EffectiveHeatRequest(
            raw_requested_duty_ratio=raw_ratio,
            effective_requested_duty_ratio=desired_ratio,
            effective_heat_request_state=classify_effective_heat_request(desired_ratio),
            anti_chatter_limited=False,
            anti_chatter_reason=None,
        )

    comfort_recovery_active = _comfort_recovery_active(
        predicted_temp=predicted_temp,
        target_temperature=target_temperature,
        comfort_tolerance=comfort_tolerance,
    )
    if comfort_recovery_active and desired_ratio > previous_ratio:
        return EffectiveHeatRequest(
            raw_requested_duty_ratio=raw_ratio,
            effective_requested_duty_ratio=desired_ratio,
            effective_heat_request_state=classify_effective_heat_request(desired_ratio),
            anti_chatter_limited=False,
            anti_chatter_reason=None,
        )

    # A quarter of the request range per 15 minutes, independent of event count.
    elapsed = max(0.0, elapsed_seconds)
    step = ANTI_CHATTER_MAX_STEP_CHANGE * elapsed / ANTI_CHATTER_REFERENCE_SECONDS
    effective_ratio = desired_ratio
    limited = False
    reason = None
    too_warm = predicted_temp is not None and predicted_temp > target_temperature + comfort_tolerance
    if desired_ratio < previous_ratio and not too_warm:
        # Only the part of this interval after the hold expires is available
        # for ramping down. Repeated events cannot consume the hold early.
        available = max(0.0, seconds_since_increase - ANTI_CHATTER_REFERENCE_SECONDS)
        step = ANTI_CHATTER_MAX_STEP_CHANGE * min(elapsed, available) / ANTI_CHATTER_REFERENCE_SECONDS
        if step == 0:
            effective_ratio = previous_ratio
            limited = True
            reason = "minimum_persistence"
    if too_warm and desired_ratio < previous_ratio:
        step = 1.0  # Do not prolong a heat request above the comfort band.
    delta = desired_ratio - previous_ratio
    if not limited and abs(delta) > step:
        effective_ratio = previous_ratio + (step if delta > 0 else -step)
        limited = True
        reason = "ramp_up_limited" if delta > 0 else "ramp_down_limited"

    return EffectiveHeatRequest(
        raw_requested_duty_ratio=raw_ratio,
        effective_requested_duty_ratio=effective_ratio,
        effective_heat_request_state=classify_effective_heat_request(effective_ratio),
        anti_chatter_limited=limited,
        anti_chatter_reason=reason,
    )


def classify_effective_heat_request(requested_duty_ratio: float | None) -> str:
    """Return a stable label for the effective heat request."""

    ratio = normalize_requested_duty_ratio(requested_duty_ratio, fallback=0.0)
    if ratio <= 0.0:
        return HEAT_REQUEST_STATE_IDLE
    if ratio <= 0.25:
        return HEAT_REQUEST_STATE_LOW
    if ratio <= 0.5:
        return HEAT_REQUEST_STATE_MEDIUM
    if ratio < 1.0:
        return HEAT_REQUEST_STATE_HIGH
    return HEAT_REQUEST_STATE_FULL


def summarize_heating_detection_gap(
    *,
    effective_requested_duty_ratio: float | None,
    heating_detected: bool | None,
) -> dict[str, Any]:
    """Classify mismatch between requested heat and physical heating detection."""

    if effective_requested_duty_ratio is None:
        return {
            "heating_detection_gap": False,
            "heating_detection_gap_kind": None,
            "heating_detection_gap_expected": None,
        }
    ratio = normalize_requested_duty_ratio(effective_requested_duty_ratio, fallback=0.0)
    expected_heating = ratio > 0.1
    if heating_detected is None:
        return {
            "heating_detection_gap": False,
            "heating_detection_gap_kind": None,
            "heating_detection_gap_expected": expected_heating,
        }
    if expected_heating:
        if heating_detected:
            kind = HEAT_DETECTION_GAP_NONE
            gap = False
        elif ratio >= 0.5:
            kind = HEAT_DETECTION_GAP_STRONG_NO_RESPONSE
            gap = True
        else:
            kind = HEAT_DETECTION_GAP_WEAK_NO_RESPONSE
            gap = True
    else:
        if heating_detected:
            kind = HEAT_DETECTION_GAP_IDLE_WHILE_HEATING
            gap = True
        else:
            kind = HEAT_DETECTION_GAP_NONE
            gap = False
    return {
        "heating_detection_gap": gap,
        "heating_detection_gap_kind": kind,
        "heating_detection_gap_expected": expected_heating,
    }


def normalize_requested_duty_ratio(value: float | None, *, fallback: float) -> float:
    """Clamp a duty ratio to [0, 1], using a fallback when invalid."""

    try:
        ratio = float(value) if value is not None else float(fallback)
    except (TypeError, ValueError):
        ratio = float(fallback)
    return max(0.0, min(1.0, ratio))


def _comfort_recovery_active(
    *,
    predicted_temp: float | None,
    target_temperature: float,
    comfort_tolerance: float,
) -> bool:
    if predicted_temp is None:
        return False
    try:
        predicted = float(predicted_temp)
        target = float(target_temperature)
        tolerance = max(0.0, float(comfort_tolerance))
    except (TypeError, ValueError):
        return False
    shortfall = target - predicted
    return shortfall > max(ANTI_CHATTER_COMFORT_RECOVERY_DELTA, tolerance * 0.5)


def elapsed_smoothing_alpha(alpha: float, elapsed_seconds: float, reference_seconds: float) -> float:
    """Interpret alpha per normal control interval, not per sensor event."""
    alpha = max(0.0, min(1.0, alpha))
    if elapsed_seconds <= 0:
        return 0.0
    return 1.0 - (1.0 - alpha) ** (elapsed_seconds / max(1.0, reference_seconds))
