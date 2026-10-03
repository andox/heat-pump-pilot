"""Expected startup sensor gaps stay quiet; persistent runtime gaps warn."""

import ast
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize("age_seconds,expected_level", [(6, logging.DEBUG), (120, logging.WARNING), (3600, logging.WARNING)])
def test_missing_indoor_temperature_log_respects_startup_grace(caplog, age_seconds, expected_level):
    path = Path(__file__).resolve().parents[1] / "custom_components/heat_pump_pilot/climate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entity_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in entity_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_async_run_control")
    now = datetime(2026, 10, 3, 8, 49, 25, tzinfo=timezone.utc)
    logger = logging.getLogger("test.pilot.startup")
    namespace = {
        "HVACMode": SimpleNamespace(OFF="off"),
        "dt_util": SimpleNamespace(utcnow=lambda: now),
        "STARTUP_SENSOR_GRACE": timedelta(minutes=2),
        "_LOGGER": logger,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    entity = SimpleNamespace(
        _hvac_mode="heat",
        _control_lock=asyncio.Lock(),
        _startup_time=now - timedelta(seconds=age_seconds),
        _indoor_temp_entity="sensor.indoor",
        _get_state_as_float=lambda _: None,
        _publish_decision=Mock(),
        _async_update_notifications=AsyncMock(),
        async_write_ha_state=Mock(),
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        asyncio.run(MethodType(namespace["_async_run_control"], entity)())
    records = [record for record in caplog.records if record.name == logger.name]
    assert len(records) == 1
    assert records[0].levelno == expected_level
    assert "Indoor temperature unavailable" in records[0].getMessage()
    entity._async_update_notifications.assert_awaited_once_with(now)
    entity._publish_decision.assert_called_once()
    entity.async_write_ha_state.assert_called_once()
