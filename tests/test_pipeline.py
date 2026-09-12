"""End-to-end tests for the detection pipeline.

These tests exercise the full flow from image loading through to
DropletDetection output, using synthetic fixtures.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import save_temp_image
from droplet_detector.config import DropletDetectorConfig
from droplet_detector.pipeline import detect_droplets_in_image
from droplet_detector.models import DropletDetection


class TestPipelineEndToEnd:
    """Full pipeline integration tests."""

    def test_detects_synthetic_droplets(
        self,
        dry_reference_image: np.ndarray,
        wet_image_with_two_droplets: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """The pipeline should find droplets in the synthetic wet image."""
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        wet_path = save_temp_image(
            wet_image_with_two_droplets, tmp_data_dir, "wet.jpg"
        )
        detections = detect_droplets_in_image(wet_path, dry_path, default_config)
        assert len(detections) >= 1, "Should detect at least one droplet"
        for d in detections:
            assert isinstance(d, DropletDetection)
            assert d.confidence >= default_config.sensitivity
            assert d.droplet_number >= 1

    def test_zero_false_positives_on_pure_noise(
        self,
        dry_reference_image: np.ndarray,
        another_dry_image: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """Two independently-seeded noise images (no real droplets) should
        produce zero or near-zero detections.

        This test caught a real bug: the first version of preprocessing.py
        used a naive min-max contrast stretch on the diff and produced 750+
        false positives on pure noise.  The fixed version using
        suppress_noise_floor produces zero.
        """
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        noise_path = save_temp_image(another_dry_image, tmp_data_dir, "noise.jpg")
        detections = detect_droplets_in_image(noise_path, dry_path, default_config)
        assert len(detections) <= 1, (
            f"Expected 0-1 false positives on noise-only pair, got {len(detections)}"
        )

    def test_wrinkle_rejected(
        self,
        dry_reference_image: np.ndarray,
        wrinkle_image: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """A wrinkle (elongated line) should produce zero detections."""
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        wrinkle_path = save_temp_image(wrinkle_image, tmp_data_dir, "wrinkle.jpg")
        detections = detect_droplets_in_image(
            wrinkle_path, dry_path, default_config
        )
        assert len(detections) == 0, (
            f"Wrinkle should be rejected, got {len(detections)} detection(s)"
        )

    def test_missing_image_raises_error(
        self,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """Non-existent image paths should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_droplets_in_image(
                str(tmp_data_dir / "nonexistent.jpg"),
                str(tmp_data_dir / "also_nonexistent.jpg"),
                default_config,
            )

    def test_mm_fields_none_without_calibration(
        self,
        dry_reference_image: np.ndarray,
        wet_image_with_two_droplets: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """Without mm_per_pixel, x_mm and y_mm should be None."""
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        wet_path = save_temp_image(
            wet_image_with_two_droplets, tmp_data_dir, "wet.jpg"
        )
        detections = detect_droplets_in_image(wet_path, dry_path, default_config)
        for d in detections:
            assert d.x_mm is None
            assert d.y_mm is None

    def test_detections_serialise_to_json(
        self,
        dry_reference_image: np.ndarray,
        wet_image_with_two_droplets: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        """DropletDetection models should serialise to JSON cleanly."""
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        wet_path = save_temp_image(
            wet_image_with_two_droplets, tmp_data_dir, "wet.jpg"
        )
        detections = detect_droplets_in_image(wet_path, dry_path, default_config)
        for d in detections:
            json_data = d.model_dump(mode="json")
            assert "droplet_number" in json_data
            assert "x_px" in json_data
            assert "confidence" in json_data


class TestPipelineSingleDroplet:
    """Tests with a single large droplet."""

    def test_single_large_droplet_detected(
        self,
        dry_reference_image: np.ndarray,
        single_droplet_image: np.ndarray,
        tmp_data_dir: Path,
        default_config: DropletDetectorConfig,
    ) -> None:
        dry_path = save_temp_image(dry_reference_image, tmp_data_dir, "dry.jpg")
        wet_path = save_temp_image(
            single_droplet_image, tmp_data_dir, "single.jpg"
        )
        detections = detect_droplets_in_image(wet_path, dry_path, default_config)
        assert len(detections) >= 1, "Should detect the single large droplet"
