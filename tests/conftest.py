"""Test configuration for local runs."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
collect_ignore = [os.path.join(ROOT_DIR, "__init__.py")]


def pytest_configure() -> None:
    """Ensure the integration root is importable for tests."""
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
