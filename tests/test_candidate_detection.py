"""Tests for candidate detection (Hough circles + MSER blobs).

Key scenarios:
- Hough detects drawn circles in a diff image.
- MSER detects blob regions.
- Elongated features (wrinkles) are rejected by the circularity filter.
- merge_candidates collapses near-duplicates.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from droplet_detector.candidate_detection import (
    Candidate,
    detect_via_hough,
    detect_via_mser,
    merge_candidates,
)
from droplet_detector.preprocessing import diff_against_dry_reference


class TestDetectViaHough:
    """Tests for detect_via_hough()."""

    def test_finds_circle_in_synthetic_diff(self) -> None:
        """Draw a bright circle on a black background — Hough should find it."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 15, 200, -1)
        candidates = detect_via_hough(img, min_radius_px=10, max_radius_px=25)
        assert len(candidates) >= 1
        best = candidates[0]
        assert abs(best.x - 100) < 10
        assert abs(best.y - 100) < 10
        assert best.source == "hough"
        assert best.circularity == 1.0

    def test_empty_image_yields_no_candidates(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        candidates = detect_via_hough(img, min_radius_px=5, max_radius_px=30)
        assert len(candidates) == 0

    def test_returns_candidate_dataclass(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (80, 80), 12, 180, -1)
        candidates = detect_via_hough(img, min_radius_px=8, max_radius_px=20)
        if candidates:
            c = candidates[0]
            assert isinstance(c, Candidate)
            assert isinstance(c.x, float)
            assert isinstance(c.y, float)
            assert isinstance(c.radius_px, float)


class TestDetectViaMSER:
    """Tests for detect_via_mser()."""

    def test_finds_blob_in_synthetic_diff(self) -> None:
        """A bright blob on a dark background should be detected."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 15, 200, -1)
        min_area = int(3.14 * 10**2)
        max_area = int(3.14 * 25**2)
        candidates = detect_via_mser(img, min_area=min_area, max_area=max_area)
        assert len(candidates) >= 1
        assert candidates[0].source == "mser"

    def test_rejects_elongated_feature(self) -> None:
        """A long thin line (wrinkle) should be rejected by circularity < 0.7."""
        img = np.zeros((200, 500), dtype=np.uint8)
        cv2.line(img, (50, 100), (450, 105), 200, 3)
        min_area = 20
        max_area = 5000
        candidates = detect_via_mser(img, min_area=min_area, max_area=max_area)
        # Any candidates found should have high circularity (i.e., the line was rejected)
        for c in candidates:
            assert c.circularity >= 0.7, (
                f"Elongated feature should be rejected but got circularity={c.circularity:.2f}"
            )

    def test_empty_image_yields_no_candidates(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        candidates = detect_via_mser(img, min_area=50, max_area=5000)
        assert len(candidates) == 0


class TestMergeCandidates:
    """Tests for merge_candidates()."""

    def test_nearby_candidates_are_merged(self) -> None:
        c1 = Candidate(x=100, y=100, radius_px=10, circularity=0.9, source="hough")
        c2 = Candidate(x=103, y=101, radius_px=11, circularity=0.85, source="mser")
        merged = merge_candidates([c1, c2], merge_dist_px=10)
        assert len(merged) == 1
        assert merged[0].source == "merged"

    def test_distant_candidates_stay_separate(self) -> None:
        c1 = Candidate(x=50, y=50, radius_px=10, circularity=0.9, source="hough")
        c2 = Candidate(x=400, y=400, radius_px=10, circularity=0.85, source="mser")
        merged = merge_candidates([c1, c2], merge_dist_px=10)
        assert len(merged) == 2

    def test_merged_position_is_averaged(self) -> None:
        c1 = Candidate(x=100, y=100, radius_px=10, circularity=0.8, source="hough")
        c2 = Candidate(x=104, y=100, radius_px=12, circularity=0.9, source="mser")
        merged = merge_candidates([c1, c2], merge_dist_px=10)
        assert len(merged) == 1
        assert abs(merged[0].x - 102) < 0.01
        assert abs(merged[0].radius_px - 11) < 0.01

    def test_merged_circularity_takes_maximum(self) -> None:
        c1 = Candidate(x=100, y=100, radius_px=10, circularity=0.75, source="hough")
        c2 = Candidate(x=102, y=100, radius_px=10, circularity=0.95, source="mser")
        merged = merge_candidates([c1, c2], merge_dist_px=10)
        assert merged[0].circularity == 0.95

    def test_empty_input(self) -> None:
        merged = merge_candidates([], merge_dist_px=10)
        assert merged == []

    def test_single_candidate(self) -> None:
        c = Candidate(x=50, y=50, radius_px=8, circularity=0.9, source="hough")
        merged = merge_candidates([c], merge_dist_px=10)
        assert len(merged) == 1
        assert merged[0].x == 50


class TestWrinkleRejectionEndToEnd:
    """Integration-level test: wrinkle features should not produce detections."""

    def test_wrinkle_not_detected_as_droplet(
        self, dry_reference_image, wrinkle_image
    ) -> None:
        """A wrinkle (elongated line) should be rejected by the pipeline.

        This test caught a real bug during prototyping: without the
        circularity filter in MSER, wrinkles produced false positives.
        """
        diff = diff_against_dry_reference(wrinkle_image, dry_reference_image)
        min_r, max_r = 5, 25
        hough = detect_via_hough(diff, min_r, max_r)
        mser = detect_via_mser(diff, int(3.14 * min_r**2), int(3.14 * max_r**2))
        all_candidates = hough + mser
        # Either no candidates found, or all have high circularity (round)
        for c in all_candidates:
            assert c.circularity >= 0.7, (
                f"Wrinkle should be rejected: got candidate at ({c.x:.0f}, {c.y:.0f}) "
                f"with circularity={c.circularity:.2f}"
            )
