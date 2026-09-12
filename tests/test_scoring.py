"""Tests for the scoring module.

Covers:
- size_score linear interpolation and boundary conditions
- bright_center_score on synthetic patches
- confidence weighted combination
"""
from __future__ import annotations

import numpy as np
import pytest

from droplet_detector.candidate_detection import Candidate
from droplet_detector.scoring import (
    size_score,
    bright_center_score,
    confidence,
)


class TestSizeScore:
    """Tests for size_score()."""

    def test_perfect_match_scores_one(self) -> None:
        c = Candidate(x=0, y=0, radius_px=10, circularity=1.0, source="test")
        assert size_score(c, expected_radius_px=10, tolerance_px=5) == 1.0

    def test_at_tolerance_boundary_scores_zero(self) -> None:
        c = Candidate(x=0, y=0, radius_px=15, circularity=1.0, source="test")
        score = size_score(c, expected_radius_px=10, tolerance_px=5)
        assert abs(score) < 0.01

    def test_beyond_tolerance_scores_zero(self) -> None:
        c = Candidate(x=0, y=0, radius_px=20, circularity=1.0, source="test")
        score = size_score(c, expected_radius_px=10, tolerance_px=5)
        assert score == 0.0

    def test_halfway_scores_half(self) -> None:
        c = Candidate(x=0, y=0, radius_px=12.5, circularity=1.0, source="test")
        score = size_score(c, expected_radius_px=10, tolerance_px=5)
        assert abs(score - 0.5) < 0.01

    def test_symmetric_for_smaller_radius(self) -> None:
        c = Candidate(x=0, y=0, radius_px=7.5, circularity=1.0, source="test")
        score = size_score(c, expected_radius_px=10, tolerance_px=5)
        assert abs(score - 0.5) < 0.01


class TestBrightCenterScore:
    """Tests for bright_center_score()."""

    def test_bright_center_scores_high(self) -> None:
        """A patch with a bright centre and darker surround should score high."""
        patch = np.full((50, 50), 100, dtype=np.uint8)
        cv2.circle(patch, (25, 25), 5, 200, -1)  # bright centre
        c = Candidate(x=25, y=25, radius_px=10, circularity=1.0, source="test")
        score = bright_center_score(patch, c)
        assert score > 0.5

    def test_uniform_patch_scores_around_half(self) -> None:
        """A uniform patch has zero contrast → score should be ~0.5."""
        patch = np.full((50, 50), 150, dtype=np.uint8)
        c = Candidate(x=25, y=25, radius_px=10, circularity=1.0, source="test")
        score = bright_center_score(patch, c)
        assert 0.4 <= score <= 0.6

    def test_dark_center_scores_low(self) -> None:
        """A patch with a dark centre and brighter surround should score low."""
        patch = np.full((50, 50), 200, dtype=np.uint8)
        cv2.circle(patch, (25, 25), 5, 50, -1)  # dark centre
        c = Candidate(x=25, y=25, radius_px=10, circularity=1.0, source="test")
        score = bright_center_score(patch, c)
        assert score < 0.5

    def test_empty_patch_returns_zero(self) -> None:
        patch = np.zeros((10, 10), dtype=np.uint8)
        # Candidate way outside the image
        c = Candidate(x=500, y=500, radius_px=5, circularity=1.0, source="test")
        score = bright_center_score(patch, c)
        assert score == 0.0

    def test_score_is_clipped_to_0_1(self) -> None:
        patch = np.full((50, 50), 50, dtype=np.uint8)
        patch[25, 25] = 255  # extremely bright single pixel
        c = Candidate(x=25, y=25, radius_px=3, circularity=1.0, source="test")
        score = bright_center_score(patch, c)
        assert 0.0 <= score <= 1.0


class TestConfidence:
    """Tests for the composite confidence function."""

    def test_perfect_candidate_scores_high(self) -> None:
        c = Candidate(x=25, y=25, radius_px=10, circularity=1.0, source="test")
        # Create an image with bright centre
        gray = np.full((50, 50), 100, dtype=np.uint8)
        cv2.circle(gray, (25, 25), 5, 220, -1)
        conf = confidence(c, gray, expected_radius_px=10, tolerance_px=5)
        assert conf > 0.5

    def test_wrong_size_lowers_score(self) -> None:
        c = Candidate(x=25, y=25, radius_px=30, circularity=1.0, source="test")
        gray = np.full((100, 100), 150, dtype=np.uint8)
        conf = confidence(c, gray, expected_radius_px=10, tolerance_px=5)
        # Size score is 0, so overall confidence should be lower
        assert conf < 0.8

    def test_low_circularity_lowers_score(self) -> None:
        c = Candidate(x=25, y=25, radius_px=10, circularity=0.3, source="test")
        gray = np.full((50, 50), 150, dtype=np.uint8)
        conf = confidence(c, gray, expected_radius_px=10, tolerance_px=5)
        assert conf < 0.9  # circularity penalty

    def test_confidence_in_valid_range(self) -> None:
        c = Candidate(x=25, y=25, radius_px=10, circularity=0.8, source="test")
        gray = np.full((50, 50), 150, dtype=np.uint8)
        conf = confidence(c, gray, expected_radius_px=10, tolerance_px=5)
        assert 0.0 <= conf <= 1.0


# Need cv2 import for drawing in bright_center tests
import cv2
