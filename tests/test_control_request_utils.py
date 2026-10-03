"""Unit tests for effective heat-request and anti-chatter helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from control_request_utils import (  # noqa: E402
    HEAT_DETECTION_GAP_NONE,
    HEAT_DETECTION_GAP_STRONG_NO_RESPONSE,
    HEAT_DETECTION_GAP_WEAK_NO_RESPONSE,
    HEAT_REQUEST_STATE_FULL,
    HEAT_REQUEST_STATE_IDLE,
    HEAT_REQUEST_STATE_LOW,
    resolve_effective_heat_request,
    summarize_heating_detection_gap,
)


def test_continuous_request_state_exposes_partial_request() -> None:
    result = resolve_effective_heat_request(
        raw_requested_duty_ratio=0.25,
        previous_effective_duty_ratio=None,
        elapsed_seconds=900, seconds_since_increase=900,
        predicted_temp=20.8,
        target_temperature=21.0,
        comfort_tolerance=1.0,
        anti_chatter_enabled=True,
    )
    assert result.raw_requested_duty_ratio == pytest.approx(0.25)
    assert result.effective_requested_duty_ratio == pytest.approx(0.25)
    assert result.effective_heat_request_state == HEAT_REQUEST_STATE_LOW


def request(desired, previous, elapsed, age, predicted=21):
    return resolve_effective_heat_request(
        raw_requested_duty_ratio=desired, previous_effective_duty_ratio=previous,
        elapsed_seconds=elapsed, seconds_since_increase=age,
        predicted_temp=predicted, target_temperature=21, comfort_tolerance=0.3,
        anti_chatter_enabled=True,
    )


def test_frequent_events_do_not_accelerate_ramp():
    previous = 0.0
    for _ in range(15):
        previous = request(1, previous, 60, 0).effective_requested_duty_ratio
    assert previous == pytest.approx(request(1, 0, 900, 0).effective_requested_duty_ratio)
    assert previous == pytest.approx(0.25)


def test_reversal_hold_uses_time_not_event_count():
    for age in range(0, 901, 30):
        result = request(0, 0.75, 30, age)
        assert result.effective_requested_duty_ratio == 0.75
        assert result.anti_chatter_reason == "minimum_persistence"
    assert request(0, 0.75, 60, 960).effective_requested_duty_ratio == pytest.approx(0.75 - 0.25 / 15)


def test_down_ramp_is_independent_of_event_count_after_hold():
    previous = 0.75
    for age in range(60, 1801, 60):
        previous = request(0, previous, 60, age).effective_requested_duty_ratio
    assert previous == pytest.approx(request(0, 0.75, 1800, 1800).effective_requested_duty_ratio)
    assert previous == pytest.approx(0.5)


def test_warm_house_can_back_off_without_waiting():
    assert request(0, 1, 10, 10, predicted=21.4).effective_requested_duty_ratio == 0


def test_no_time_does_not_advance_normal_ramp():
    assert request(1, 0, 0, 0).effective_requested_duty_ratio == 0


def test_small_fractional_requests_are_not_rounded_to_quarters():
    assert request(0.11, 0, 900, 900).effective_requested_duty_ratio == pytest.approx(0.11)


def test_ema_is_independent_of_event_count():
    from control_request_utils import elapsed_smoothing_alpha
    value = 5.0
    for _ in range(15):
        value += elapsed_smoothing_alpha(0.8, 60, 900) * (15 - value)
    assert value == pytest.approx(13)
    assert elapsed_smoothing_alpha(0.8, 0, 900) == 0
    assert elapsed_smoothing_alpha(0, 900, 900) == 0
    assert elapsed_smoothing_alpha(1, 60, 900) == 1


def test_anti_chatter_allows_fast_comfort_recovery() -> None:
    result = resolve_effective_heat_request(
        raw_requested_duty_ratio=1.0,
        previous_effective_duty_ratio=0.0,
        elapsed_seconds=900, seconds_since_increase=3600,
        predicted_temp=20.0,
        target_temperature=21.0,
        comfort_tolerance=0.5,
        anti_chatter_enabled=True,
    )
    assert result.effective_requested_duty_ratio == pytest.approx(1.0)
    assert result.effective_heat_request_state == HEAT_REQUEST_STATE_FULL
    assert result.anti_chatter_limited is False


def test_heating_detection_gap_classifies_weak_request_without_response() -> None:
    gap = summarize_heating_detection_gap(
        effective_requested_duty_ratio=0.25,
        heating_detected=False,
    )
    assert gap["heating_detection_gap"] is True
    assert gap["heating_detection_gap_kind"] == HEAT_DETECTION_GAP_WEAK_NO_RESPONSE


def test_heating_detection_gap_classifies_strong_request_without_response() -> None:
    gap = summarize_heating_detection_gap(
        effective_requested_duty_ratio=0.75,
        heating_detected=False,
    )
    assert gap["heating_detection_gap"] is True
    assert gap["heating_detection_gap_kind"] == HEAT_DETECTION_GAP_STRONG_NO_RESPONSE


def test_heating_detection_gap_classifies_aligned_response() -> None:
    gap = summarize_heating_detection_gap(
        effective_requested_duty_ratio=0.75,
        heating_detected=True,
    )
    assert gap["heating_detection_gap"] is False
    assert gap["heating_detection_gap_kind"] == HEAT_DETECTION_GAP_NONE


def test_heating_detection_gap_classifies_idle_without_gap() -> None:
    gap = summarize_heating_detection_gap(
        effective_requested_duty_ratio=0.0,
        heating_detected=False,
    )
    assert gap["heating_detection_gap"] is False
    assert gap["heating_detection_gap_kind"] == HEAT_DETECTION_GAP_NONE
    assert gap["heating_detection_gap_expected"] is False
    assert HEAT_REQUEST_STATE_IDLE == "idle"


def test_heating_detection_gap_classifies_idle_while_heating() -> None:
    gap = summarize_heating_detection_gap(
        effective_requested_duty_ratio=0.0,
        heating_detected=True,
    )
    assert gap["heating_detection_gap"] is True
    assert gap["heating_detection_gap_kind"] == "idle_but_heating_detected"
