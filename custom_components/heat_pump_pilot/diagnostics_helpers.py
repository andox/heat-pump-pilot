"""Helpers for publishing decision and trace diagnostics."""

from __future__ import annotations

from typing import Any

try:
    from homeassistant.helpers.dispatcher import async_dispatcher_send
except ModuleNotFoundError:  # pragma: no cover - local tests without HA
    def async_dispatcher_send(*_args, **_kwargs) -> None:
        return None


def publish_entry_diagnostics(
    hass,
    *,
    domain: str,
    entry_id: str,
    signal: str,
    payload: dict[str, Any],
    performance: dict[str, Any],
    trace_enabled: bool,
    trace_entry: dict[str, Any] | None,
    trace_max_entries: int,
) -> None:
    """Store latest diagnostics in hass.data and notify listeners."""
    entry_data = hass.data.setdefault(domain, {}).setdefault(entry_id, {})
    entry_data["last_decision"] = payload
    entry_data["performance"] = performance
    if not trace_enabled:
        async_dispatcher_send(hass, signal)
        return

    if trace_entry is None:
        async_dispatcher_send(hass, signal)
        return

    trace = entry_data.setdefault("virtual_outdoor_trace", [])

    def _strip_time(entry: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in entry.items() if key != "time"}

    if trace and (
        trace[-1].get("time") == trace_entry.get("time")
        or _strip_time(trace[-1]) == _strip_time(trace_entry)
    ):
        trace[-1] = trace_entry
    else:
        trace.append(trace_entry)
        if len(trace) > trace_max_entries:
            del trace[: len(trace) - trace_max_entries]

    async_dispatcher_send(hass, signal)
