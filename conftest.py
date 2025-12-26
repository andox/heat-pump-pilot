"""Pytest collection configuration."""

from __future__ import annotations

import sys
from pathlib import Path

collect_ignore = ["__init__.py"]
collect_ignore_glob = ["__init__.py"]


def pytest_configure() -> None:
    """Add integration module path for local tests."""
    root = Path(__file__).resolve().parent
    integration_path = root / "custom_components" / "heat_pump_pilot"
    if integration_path.exists():
        sys.path.insert(0, str(integration_path))
