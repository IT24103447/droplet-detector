"""Tests for the preprocessing module.

Key scenarios:
- Illumination flattening produces a roughly uniform image from an
  unevenly-lit input.
- Noise-floor suppression zeroes out small values without rescaling.
- Dry-reference differencing of two noise-only images yields near-zero signal.
- Dry-reference differencing of a droplet image yields detectable signal.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from droplet_detector.preprocessing import (
    flatten_illumination,
    suppress_noise_floor,
    diff_against_dry_reference,
)


class TestFlattenIllumination:
    """Tests for flatten_illumination()."""

    def test_output_shape_matches_input(self) -> None:
        gray = np.full((100, 100), 128, dtype=np.uint8)
        result = flatten_illumination(gray)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_uniform_input_stays_near_128(self) -> None:
        gray = np.full((100, 100), 180, dtype=np.uint8)
        result = flatten_illumination(gray)
        # After dividing by itself, should normalise close to 128
        assert abs(float(np.mean(result)) - 128.0) < 10.0

    def test_gradient_is_flattened(self) -> None:
        """A left-to-right brightness gradient should become roughly uniform."""
        gray = np.zeros((100, 200), dtype=np.uint8)
        for col in range(200):
            gray[:, col] = int(80 + 120 * col / 199)
        result = flatten_illumination(gray)
        # Standard deviation should be much lower after flattening
        assert float(np.std(result)) < float(np.std(gray))

    def test_odd_kernel_size_enforced(self) -> None:
        """Even blur_ksize should be forced odd without error."""
        gray = np.full((50, 50), 150, dtype=np.uint8)
        result = flatten_illumination(gray, blur_ksize=50)  # even
        assert result.shape == gray.shape


class TestSuppressNoiseFloor:
    """Tests for suppress_noise_floor()."""

    def test_values_below_threshold_are_zeroed(self) -> None:
        # Use a 2D array (cv2.threshold expects at least 2D)
        diff = np.array([[5, 10, 14, 16, 20, 100]], dtype=np.uint8)
        result = suppress_noise_floor(diff, noise_floor=15)
        # THRESH_TOZERO: values <= threshold → 0, values > threshold → kept
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[0, 2] == 0
        assert result[0, 3] == 16  # above threshold → kept
        assert result[0, 4] == 20
        assert result[0, 5] == 100

    def test_is_not_contrast_stretch(self) -> None:
        """Critical: suppression must NOT rescale values to fill 0-255.
        A naive min-max stretch on near-empty diffs causes 750+ false
        positives (confirmed during prototyping)."""
        diff = np.array([0, 1, 2, 3, 5, 8, 12], dtype=np.uint8)
        result = suppress_noise_floor(diff, noise_floor=15)
        # All values < 15 → should be zeroed, NOT stretched to fill 0-255
        assert np.all(result == 0)

    def test_zero_noise_floor_keeps_everything(self) -> None:
        diff = np.array([[1, 50, 200]], dtype=np.uint8)
        result = suppress_noise_floor(diff, noise_floor=0)
        np.testing.assert_array_equal(result, diff)


class TestDiffAgainstDryReference:
    """Tests for diff_against_dry_reference()."""

    def test_identical_images_yield_zero_diff(
        self, dry_reference_image: np.ndarray
    ) -> None:
        diff = diff_against_dry_reference(
            dry_reference_image, dry_reference_image.copy()
        )
        assert diff.max() == 0

    def test_noise_only_pair_yields_near_zero(
        self, dry_reference_image: np.ndarray, another_dry_image: np.ndarray
    ) -> None:
        """Two independently-generated noisy images with no real droplet.
        The diff should be almost entirely zeros after noise-floor suppression."""
        diff = diff_against_dry_reference(dry_reference_image, another_dry_image)
        nonzero_fraction = np.count_nonzero(diff) / diff.size
        assert nonzero_fraction < 0.05, (
            f"Expected near-zero diff from noise-only pair, got "
            f"{nonzero_fraction:.1%} non-zero pixels"
        )

    def test_droplet_image_yields_signal(
        self,
        dry_reference_image: np.ndarray,
        wet_image_with_two_droplets: np.ndarray,
    ) -> None:
        """A wet image with droplets should produce detectable signal."""
        diff = diff_against_dry_reference(
            wet_image_with_two_droplets, dry_reference_image
        )
        assert diff.max() > 0, "Expected detectable signal from droplet image"

    def test_output_is_single_channel_uint8(
        self, dry_reference_image: np.ndarray
    ) -> None:
        diff = diff_against_dry_reference(
            dry_reference_image, dry_reference_image.copy()
        )
        assert diff.ndim == 2
        assert diff.dtype == np.uint8
