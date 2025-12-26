"""Helpers for throttled persistent notifications.

This module is intentionally Home Assistant free to keep unit tests simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class NotificationEventState:
    """State for one notification-worthy event."""

    signature: str
    first_seen: datetime
    active: bool = False
    last_notified: datetime | None = None
    pending_update: bool = False


@dataclass(frozen=True)
class NotificationUpdate:
    """An action for the HA layer to execute."""

    action: str  # "create" | "dismiss"
    event_id: str


class NotificationTracker:
    """Tracks events to avoid spamming notifications."""

    def __init__(self) -> None:
        self._events: dict[str, NotificationEventState] = {}

    def update(
        self,
        *,
        event_id: str,
        signature: str | None,
        now: datetime,
        trigger_after: timedelta,
        cooldown: timedelta,
    ) -> list[NotificationUpdate]:
        """Update event state and return actions to take.

        - When `signature` is not None, the event is considered "problem active".
        - A "create" is returned when the problem has been present for `trigger_after`
          and we have not notified within `cooldown`.
        - A "dismiss" is returned when the problem resolves after previously being active.
        """

        if signature is None:
            state = self._events.get(event_id)
            if not state:
                return []
            if state.active:
                self._events.pop(event_id, None)
                return [NotificationUpdate(action="dismiss", event_id=event_id)]
            # Problem resolved before it was ever notified.
            self._events.pop(event_id, None)
            return []

        # Problem active
        state = self._events.get(event_id)
        if not state:
            state = NotificationEventState(signature=signature, first_seen=now)
            self._events[event_id] = state
            if trigger_after <= timedelta(0):
                state.active = True
                state.last_notified = now
                return [NotificationUpdate(action="create", event_id=event_id)]
            return []

        signature_changed = False
        if state.signature != signature:
            signature_changed = True
            state.signature = signature
            if not state.active:
                state.first_seen = now
            else:
                state.pending_update = True

        age = now - state.first_seen
        if age < trigger_after:
            return []

        if state.last_notified is not None and (now - state.last_notified) < cooldown:
            return []

        if state.active and not signature_changed and not state.pending_update:
            return []

        state.active = True
        state.last_notified = now
        state.pending_update = False
        return [NotificationUpdate(action="create", event_id=event_id)]
