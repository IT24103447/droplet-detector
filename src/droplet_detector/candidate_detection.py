"""Candidate droplet detection via Hough circles and MSER blobs.

Two independent detectors run on the noise-suppressed diff image:

* **Hough circles** — good at finding round shapes of a known radius range.
* **MSER blobs** — good at finding stable extremal regions (water on fabric
  creates a locally darker / brighter patch).

Both detectors' outputs are merged (near-duplicates collapsed) before
scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    """A candidate droplet region before confidence scoring."""

    x: float
    y: float
    radius_px: float
    circularity: float
    source: str  # "hough", "mser", or "merged"


def detect_via_hough(
    diff_img: np.ndarray,
    min_radius_px: int,
    max_radius_px: int,
) -> list[Candidate]:
    """Detect circular candidates using the Hough gradient method.

    Parameters
    ----------
    diff_img:
        Single-channel uint8 diff image (noise-floor already suppressed).
    min_radius_px, max_radius_px:
        Allowed radius range in pixels.

    Returns
    -------
    list[Candidate]
        Candidates found via Hough, each with ``circularity=1.0`` by
        definition (Hough only returns circles).
    """
    blurred = cv2.medianBlur(diff_img, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius_px, 1),
        param1=80,
        param2=25,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    results: list[Candidate] = []
    if circles is not None:
        for x, y, r in circles[0]:
            results.append(
                Candidate(
                    x=float(x),
                    y=float(y),
                    radius_px=float(r),
                    circularity=1.0,
                    source="hough",
                )
            )
    return results


def detect_via_mser(
    diff_img: np.ndarray,
    min_area: int,
    max_area: int,
) -> list[Candidate]:
    """Detect blob candidates using Maximally Stable Extremal Regions.

    Only regions whose convex hull has circularity >= 0.7 are kept —
    this rejects wrinkles and threads (long and thin).

    Parameters
    ----------
    diff_img:
        Single-channel uint8 diff image.
    min_area, max_area:
        Allowed blob area range in pixels² (π·r²).

    Returns
    -------
    list[Candidate]
        Round-enough MSER blobs with measured circularity.
    """
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, _ = mser.detectRegions(diff_img)
    results: list[Candidate] = []
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.7:  # rules out wrinkles/threads (long & thin)
            continue
        (x, y), radius = cv2.minEnclosingCircle(hull)
        results.append(
            Candidate(
                x=x,
                y=y,
                radius_px=radius,
                circularity=circularity,
                source="mser",
            )
        )
    return results


def merge_candidates(
    candidates: list[Candidate],
    merge_dist_px: float,
) -> list[Candidate]:
    """Collapse near-duplicate detections from Hough and MSER.

    Candidates whose centres are within ``merge_dist_px`` pixels of each
    other are merged into a single candidate.  Position and radius are
    averaged; circularity takes the maximum (best shape evidence wins).

    Parameters
    ----------
    candidates:
        Combined list from both detectors.
    merge_dist_px:
        Maximum centre-to-centre distance to consider two candidates as
        the same droplet.

    Returns
    -------
    list[Candidate]
        De-duplicated candidates.
    """
    merged: list[Candidate] = []
    used = [False] * len(candidates)
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        cluster = [c]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            other = candidates[j]
            dist = ((c.x - other.x) ** 2 + (c.y - other.y) ** 2) ** 0.5
            if dist < merge_dist_px:
                cluster.append(other)
                used[j] = True
        avg_x = sum(m.x for m in cluster) / len(cluster)
        avg_y = sum(m.y for m in cluster) / len(cluster)
        avg_r = sum(m.radius_px for m in cluster) / len(cluster)
        best_circ = max(m.circularity for m in cluster)
        merged.append(
            Candidate(
                x=avg_x,
                y=avg_y,
                radius_px=avg_r,
                circularity=best_circ,
                source="merged",
            )
        )
    return merged
