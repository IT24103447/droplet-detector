"""Tests for evaluate script."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts directory to path to allow importing evaluate
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from evaluate import discover_fabrics


def test_discover_fabrics(tmp_path: Path) -> None:
    """Test that fabrics are discovered dynamically from the raw directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    
    # Create fake fabric directories
    (raw_dir / "test_fabric_b").mkdir()
    (raw_dir / "test_fabric_a").mkdir()
    
    # Should discover and sort the directories
    fabrics = discover_fabrics(tmp_path)
    assert fabrics == ["test_fabric_a", "test_fabric_b"]


def test_discover_fabrics_empty(tmp_path: Path) -> None:
    """Test that an empty list is returned if raw dir exists but is empty."""
    (tmp_path / "raw").mkdir(parents=True)
    assert discover_fabrics(tmp_path) == []


def test_discover_fabrics_missing_raw_dir(tmp_path: Path) -> None:
    """Test that an empty list is returned if raw dir does not exist."""
    assert discover_fabrics(tmp_path) == []
