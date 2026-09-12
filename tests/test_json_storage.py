"""Unit tests for shared JSON storage helpers."""

from __future__ import annotations

import sys
from pathlib import Path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

from json_storage import AtomicJsonStorage  # noqa: E402
from performance_history import PerformanceHistoryStorage  # noqa: E402
from price_history import PriceHistoryStorage  # noqa: E402
from thermal_model import ThermalModelState, ThermalModelStorage  # noqa: E402


def test_atomic_json_storage_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    storage = AtomicJsonStorage(path)
    payload = {"hello": "world"}
    storage.save(payload)
    assert storage.load() == payload


def test_price_and_performance_storage_share_round_trip_behavior(tmp_path: Path) -> None:
    price_storage = PriceHistoryStorage(str(tmp_path / "prices.json"))
    performance_storage = PerformanceHistoryStorage(str(tmp_path / "performance.json"))
    payload = {"history": [1, 2, 3]}
    price_storage.save(payload)
    performance_storage.save(payload)
    assert price_storage.load() == payload
    assert performance_storage.load() == payload


def test_thermal_model_storage_serializes_dataclass_payload(tmp_path: Path) -> None:
    storage = ThermalModelStorage(str(tmp_path / "thermal.json"))
    payload = ThermalModelState(state=[1.0, 2.0, 3.0], covariance=[[1.0] * 3 for _ in range(3)])
    storage.save(payload)
    assert storage.load() == {
        "version": 1,
        "state": [1.0, 2.0, 3.0],
        "covariance": [[1.0] * 3 for _ in range(3)],
    }
