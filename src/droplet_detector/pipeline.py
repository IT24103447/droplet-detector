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
    detect_via_mser,
    merge_candidates,
)
from .scoring import confidence
from .models import DropletDetection


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

    diff = diff_against_dry_reference(current, dry, noise_floor=config.diff_noise_floor)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Real mm/pixel comes from ArUco calibration in Step 2.
    # Step 1 accepts a rough manual estimate, or leaves output in pixels only.
    mm_per_px = config.mm_per_pixel or 0.05
    min_r_px = max(int((config.droplet_min_diameter_mm / 2) / mm_per_px), 2)
    max_r_px = max(
        int((config.droplet_max_diameter_mm / 2) / mm_per_px), min_r_px + 1
    )

    hough = detect_via_hough(diff, min_r_px, max_r_px)
    mser = detect_via_mser(
        diff, int(3.14 * min_r_px**2), int(3.14 * max_r_px**2)
    )
    merged = merge_candidates(hough + mser, merge_dist_px=min_r_px)

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
