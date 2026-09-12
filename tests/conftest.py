"""Synthetic test fixtures for the droplet-detector pipeline.

Until real fabric photos from the physical rig exist, these fixtures
generate synthetic images: a flat noisy 'fabric' background, a version
with drawn circles (bright highlight + darker ring) standing in for
droplets, and a long thin line standing in for a wrinkle.

These synthetic images have been validated to produce:
- 0 false positives on a pure-noise pair (two independently-seeded dry images)
- 0 false positives on a wrinkle-only pair
- Correct detections on a two-droplet pair
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


def _blank_fabric(
    width: int = 640, height: int = 480, texture_seed: int = 0
) -> np.ndarray:
    """Generate a flat noisy 'fabric' background image (BGR)."""
    rng = np.random.default_rng(texture_seed)
    base = np.full((height, width, 3), 200, dtype=np.uint8)
    noise = rng.integers(-8, 8, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def dry_reference_image() -> np.ndarray:
    """A synthetic dry fabric reference (seed=1)."""
    return _blank_fabric(texture_seed=1)


@pytest.fixture
def another_dry_image() -> np.ndarray:
    """A second independently-seeded dry image (seed=2) — no real droplets.

    Diffing this against dry_reference_image should yield near-zero signal.
    """
    return _blank_fabric(texture_seed=2)


@pytest.fixture
def wet_image_with_two_droplets(dry_reference_image: np.ndarray) -> np.ndarray:
    """Dry reference + two synthetic droplets drawn on top.

    Each droplet is a dark circle with a brighter highlight near the centre,
    mimicking the refracted-light appearance of a real water droplet.
    """
    img = dry_reference_image.copy()
    for cx, cy, r in [(150, 120, 9), (400, 300, 11)]:
        # Darker ring (the droplet body)
        cv2.circle(img, (cx, cy), r, (120, 120, 120), -1)
        # Brighter highlight near centre (refracted light)
        cv2.circle(
            img, (cx - r // 3, cy - r // 3), max(r // 3, 1), (230, 230, 230), -1
        )
    return img


@pytest.fixture
def wrinkle_image(dry_reference_image: np.ndarray) -> np.ndarray:
    """Dry reference + a long thin line (wrinkle) — should be rejected."""
    img = dry_reference_image.copy()
    cv2.line(img, (100, 50), (500, 55), (100, 100, 100), 2)
    return img


@pytest.fixture
def single_droplet_image(dry_reference_image: np.ndarray) -> np.ndarray:
    """Dry reference + one large synthetic droplet."""
    img = dry_reference_image.copy()
    cx, cy, r = 320, 240, 12
    cv2.circle(img, (cx, cy), r, (110, 110, 110), -1)
    cv2.circle(img, (cx - r // 3, cy - r // 3), max(r // 3, 1), (240, 240, 240), -1)
    return img


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory structure mimicking the project layout."""
    for subdir in ["results/video", "results/images", "results/reports"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def default_config(tmp_data_dir: Path):
    """A DropletDetectorConfig pointing at the temporary test directory."""
    from droplet_detector.config import DropletDetectorConfig

    return DropletDetectorConfig(
        video_save_path=tmp_data_dir / "results" / "video",
        image_save_path=tmp_data_dir / "results" / "images",
        results_save_path=tmp_data_dir / "results" / "reports",
        sensitivity=0.3,  # lower threshold for synthetic images
    )


def save_temp_image(img: np.ndarray, directory: Path, name: str) -> str:
    """Helper: save a BGR image to a temp directory and return its path."""
    path = directory / name
    cv2.imwrite(str(path), img)
    return str(path)
