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
        previous_same_ratio_runs=0,
        predicted_temp=20.8,
        target_temperature=21.0,
        comfort_tolerance=1.0,
        anti_chatter_enabled=True,
    )
    assert result.raw_requested_duty_ratio == pytest.approx(0.25)
    assert result.effective_requested_duty_ratio == pytest.approx(0.25)
    assert result.effective_heat_request_state == HEAT_REQUEST_STATE_LOW


def test_anti_chatter_limits_alternating_nighttime_requests() -> None:
    previous_ratio = None
    previous_runs = 0
    effective = []
    limited_flags = []
    for raw_ratio in (1.0, 0.0, 1.0, 0.0):
        result = resolve_effective_heat_request(
            raw_requested_duty_ratio=raw_ratio,
            previous_effective_duty_ratio=previous_ratio,
            previous_same_ratio_runs=previous_runs,
            predicted_temp=20.6,
            target_temperature=21.0,
            comfort_tolerance=1.0,
            anti_chatter_enabled=True,
        )
        effective.append(result.effective_requested_duty_ratio)
        limited_flags.append(result.anti_chatter_limited)
        if previous_ratio is not None and abs(previous_ratio - result.effective_requested_duty_ratio) < 1e-6:
            previous_runs += 1
        else:
            previous_runs = 1
        previous_ratio = result.effective_requested_duty_ratio

    assert effective == [1.0, 1.0, 1.0, 0.75]
    assert limited_flags == [False, True, False, True]


def test_anti_chatter_allows_fast_comfort_recovery() -> None:
    result = resolve_effective_heat_request(
        raw_requested_duty_ratio=1.0,
        previous_effective_duty_ratio=0.0,
        previous_same_ratio_runs=4,
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
