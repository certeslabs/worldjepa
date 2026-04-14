"""
pytest configuration — WorldJEPA test suite
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


def pytest_configure(config):
    """Set deterministic seed for reproducible tests."""
    torch.manual_seed(42)


def pytest_collection_modifyitems(items):
    """Mark slow tests (those loading real V-JEPA 2 weights)."""
    for item in items:
        if "vjepa2" in item.name.lower() or "download" in item.name.lower():
            item.add_marker(pytest.mark.slow)
