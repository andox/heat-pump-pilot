"""Unit tests for notification throttling helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow importing the integration modules directly from this directory.
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from notification_utils import NotificationTracker, NotificationUpdate  # noqa: E402


def test_immediate_create_when_trigger_after_zero() -> None:
    tracker = NotificationTracker()
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    actions = tracker.update(
        event_id="health",
        signature="degraded",
        now=now,
        trigger_after=timedelta(0),
        cooldown=timedelta(minutes=30),
    )
    assert actions == [NotificationUpdate(action="create", event_id="health")]


def test_create_only_after_trigger_after() -> None:
    tracker = NotificationTracker()
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    trigger = timedelta(minutes=5)

    actions = tracker.update(
        event_id="sensors",
        signature="indoor_missing",
        now=now,
        trigger_after=trigger,
        cooldown=timedelta(minutes=30),
    )
    assert actions == []

    actions = tracker.update(
        event_id="sensors",
        signature="indoor_missing",
        now=now + trigger,
        trigger_after=trigger,
        cooldown=timedelta(minutes=30),
    )
    assert actions == [NotificationUpdate(action="create", event_id="sensors")]


def test_dismiss_only_when_active() -> None:
    tracker = NotificationTracker()
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Problem appears but resolves before trigger_after; no dismiss should occur.
    trigger = timedelta(minutes=5)
    tracker.update(
        event_id="fallback",
        signature="price_flat",
        now=now,
        trigger_after=trigger,
        cooldown=timedelta(minutes=30),
    )
    actions = tracker.update(
        event_id="fallback",
        signature=None,
        now=now + timedelta(minutes=1),
        trigger_after=trigger,
        cooldown=timedelta(minutes=30),
    )
    assert actions == []


def test_dismiss_after_recovery_from_active() -> None:
    tracker = NotificationTracker()
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    tracker.update(
        event_id="health",
        signature="degraded",
        now=now,
        trigger_after=timedelta(0),
        cooldown=timedelta(minutes=30),
    )

    actions = tracker.update(
        event_id="health",
        signature=None,
        now=now + timedelta(minutes=1),
        trigger_after=timedelta(0),
        cooldown=timedelta(minutes=30),
    )
    assert actions == [NotificationUpdate(action="dismiss", event_id="health")]


def test_signature_change_updates_after_cooldown() -> None:
    tracker = NotificationTracker()
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    cooldown = timedelta(hours=1)

    tracker.update(
        event_id="exception",
        signature="ValueError",
        now=now,
        trigger_after=timedelta(0),
        cooldown=cooldown,
    )

    # Signature changed, but still within cooldown, so no update.
    actions = tracker.update(
        event_id="exception",
        signature="TimeoutError",
        now=now + timedelta(minutes=30),
        trigger_after=timedelta(0),
        cooldown=cooldown,
    )
    assert actions == []

    # After cooldown, allow an update.
    actions = tracker.update(
        event_id="exception",
        signature="TimeoutError",
        now=now + timedelta(hours=1, minutes=1),
        trigger_after=timedelta(0),
        cooldown=cooldown,
    )
    assert actions == [NotificationUpdate(action="create", event_id="exception")]

