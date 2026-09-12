"""Unit tests for diagnostics publishing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from diagnostics_helpers import publish_entry_diagnostics  # noqa: E402


class _Hass:
    def __init__(self) -> None:
        self.data = {}


def test_publish_entry_diagnostics_stores_payload_and_deduplicates_trace() -> None:
    hass = _Hass()
    payload = {"value": 1}
    performance = {"score": 2}
    trace_entry = {"time": "2026-01-01T00:00:00+00:00", "virtual_outdoor": 5.0}

    publish_entry_diagnostics(
        hass,
        domain="heat_pump_pilot",
        entry_id="abc",
        signal="signal",
        payload=payload,
        performance=performance,
        trace_enabled=True,
        trace_entry=trace_entry,
        trace_max_entries=2,
    )
    publish_entry_diagnostics(
        hass,
        domain="heat_pump_pilot",
        entry_id="abc",
        signal="signal",
        payload=payload,
        performance=performance,
        trace_enabled=True,
        trace_entry=dict(trace_entry),
        trace_max_entries=2,
    )

    entry = hass.data["heat_pump_pilot"]["abc"]
    assert entry["last_decision"] == payload
    assert entry["performance"] == performance
    assert entry["virtual_outdoor_trace"] == [trace_entry]


def test_publish_entry_diagnostics_trims_trace() -> None:
    hass = _Hass()
    for idx in range(3):
        publish_entry_diagnostics(
            hass,
            domain="heat_pump_pilot",
            entry_id="abc",
            signal="signal",
            payload={"value": idx},
            performance={"score": idx},
            trace_enabled=True,
            trace_entry={"time": f"t{idx}", "virtual_outdoor": float(idx)},
            trace_max_entries=2,
        )
    trace = hass.data["heat_pump_pilot"]["abc"]["virtual_outdoor_trace"]
    assert [entry["time"] for entry in trace] == ["t1", "t2"]
