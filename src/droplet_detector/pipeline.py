"""Top-level detection pipeline: image in → droplet detections out.

Orchestrates preprocessing, candidate detection, merging, scoring, and
(optionally) consistency checking.  This is the primary entry point for
Step 1's single-image detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DropletDetectorConfig
from .preprocessing import diff_against_dry_reference
from .candidate_detection import (
    Candidate,
    detect_via_hough,
    detect_via_wet_image_hough,
    detect_via_mser,
    detect_via_contours,
    has_reference_change,
    merge_candidates,
)
from .scoring import confidence
from .models import DropletDetection


class ReferencePairMismatchError(ValueError):
    """The dry and wet photos differ too broadly for reliable comparison."""


def detect_droplets_in_image(
    current_image_path: str,
    dry_reference_path: str,
    config: DropletDetectorConfig,
) -> list[DropletDetection]:
    """Run the full detection pipeline on a single image.

    Parameters
    ----------
    current_image_path:
        Path to the current (possibly wet) image.
    dry_reference_path:
        Path to the dry reference image of the same fabric.
    config:
        Detection configuration (sensitivity, size range, noise floor, etc.).

    Returns
    -------
    list[DropletDetection]
        Detections whose confidence meets or exceeds ``config.sensitivity``.

    Raises
    ------
    FileNotFoundError
        If either image cannot be read by OpenCV.
    """
    current = cv2.imread(current_image_path)
    dry = cv2.imread(dry_reference_path)
    if current is None or dry is None:
        raise FileNotFoundError(
            f"Could not read current ({current_image_path}) "
            f"or dry-reference ({dry_reference_path}) image"
        )

    # Ensure dry reference matches current image dimensions
    if dry.shape[:2] != current.shape[:2]:
        if (dry.shape[0], dry.shape[1]) == (current.shape[1], current.shape[0]):
            dry = cv2.rotate(dry, cv2.ROTATE_90_CLOCKWISE)
        if dry.shape[:2] != current.shape[:2]:
            dry = cv2.resize(
                dry, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA
            )

    diff = diff_against_dry_reference(current, dry, noise_floor=config.diff_noise_floor)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    changed_fraction = float(np.count_nonzero(diff)) / diff.size
    if changed_fraction > config.max_reference_change_fraction:
        raise ReferencePairMismatchError(
            "The dry reference and test photo differ across "
            f"{changed_fraction:.0%} of the image. Retake both photos with "
            "the same fabric position, camera angle, distance, and lighting."
        )

    # Real mm/pixel comes from ArUco calibration in Step 2.
    # Step 1 accepts a rough manual estimate, or leaves output in pixels only.
    mm_per_px = config.mm_per_pixel or 0.05
    min_r_px = max(int((config.droplet_min_diameter_mm / 2) / mm_per_px), 2)
    # Before ArUco calibration there is no reliable pixel scale.  Keep the
    # search broad enough for small droplets in ordinary webcam photographs.
    if config.mm_per_pixel is None:
        min_r_px = min(min_r_px, 3)
    max_r_px = max(
        int((config.droplet_max_diameter_mm / 2) / mm_per_px), min_r_px + 1
    )

    hough = detect_via_hough(diff, min_r_px, max_r_px)
    # The wet-image fallback catches larger, reflective droplets whose rim is
    # clear to the eye but faint after dry-reference differencing.  Keep only
    # circles containing genuine dry-to-wet change to avoid fabric texture.
    wet_hough = [
        candidate
        for candidate in detect_via_wet_image_hough(gray_current, min_r_px, max_r_px)
        if has_reference_change(diff, candidate)
    ]
    mser = detect_via_mser(
        diff, int(3.14 * min_r_px**2), int(3.14 * max_r_px**2)
    )
    contours = detect_via_contours(
        diff, int(3.14 * min_r_px**2), int(3.14 * max_r_px**2)
    )
    merged = merge_candidates(
        hough + wet_hough + mser + contours, merge_dist_px=min_r_px
    )

    if config.mm_per_pixel is None:
        # Without calibration this is a broad pixel search, so scoring against
        # the midpoint would unfairly penalise the small droplets we explicitly
        # allowed above.  Use a small-droplet prior and the full search span as
        # tolerance; calibrated runs retain the precise midpoint scoring.
        expected_r = float(min_r_px * 2)
        tolerance = float(max_r_px - min_r_px)
    else:
        expected_r = (min_r_px + max_r_px) / 2
        tolerance = (max_r_px - min_r_px) / 2 + 1

    detections: list[DropletDetection] = []
    for i, cand in enumerate(merged, start=1):
        conf = confidence(cand, gray_current, expected_r, tolerance)
        if conf >= config.sensitivity:
            detections.append(
                DropletDetection(
                    droplet_number=i,
                    x_px=cand.x,
                    y_px=cand.y,
                    x_mm=cand.x * mm_per_px if config.mm_per_pixel else None,
                    y_mm=cand.y * mm_per_px if config.mm_per_pixel else None,
                    radius_px=cand.radius_px,
                    confidence=conf,
                )
            )
    return detections
