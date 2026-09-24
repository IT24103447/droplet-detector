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


def detect_via_wet_image_hough(
    gray_img: np.ndarray,
    min_radius_px: int,
    max_radius_px: int,
) -> list[Candidate]:
    """Find droplet-shaped highlights directly in the wet fabric image.

    Dry-reference differencing is the primary detector, but a larger clear
    droplet can change only its rim and leave a faint difference map.  Local
    contrast enhancement makes that rim visible to Hough circles.  The
    pipeline confirms every resulting circle has changed pixels in the dry
    reference difference map before it can become a detection.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    circles = cv2.HoughCircles(
        cv2.medianBlur(enhanced, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius_px * 2, 12),
        param1=80,
        param2=15,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    if circles is None:
        return []
    return [
        Candidate(float(x), float(y), float(r), 1.0, "wet_hough")
        for x, y, r in circles[0]
    ]


def has_reference_change(diff_img: np.ndarray, candidate: Candidate) -> bool:
    """Require compact, visible dry-to-wet change inside a wet-image circle."""
    yy, xx = np.ogrid[: diff_img.shape[0], : diff_img.shape[1]]
    radius = max(candidate.radius_px, 1.0)
    inside = (xx - candidate.x) ** 2 + (yy - candidate.y) ** 2 <= radius**2
    changed_mask = inside & (diff_img > 0)
    changed = int(np.count_nonzero(changed_mask))
    required = max(12, int(np.pi * radius**2 * 0.015))
    if changed < required:
        return False

    # A wrinkle can form a circular-looking edge in the enhanced wet image.
    # Its changed pixels still form a long thin line, unlike a droplet's
    # compact rim/body change.
    ys, xs = np.where(changed_mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect_ratio = max(width, height) / max(1, min(width, height))
    return aspect_ratio <= 3.0


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


def detect_via_contours(
    diff_img: np.ndarray,
    min_area: int,
    max_area: int,
    min_circularity: float = 0.35,
) -> list[Candidate]:
    """Detect compact changed regions from the thresholded difference image.

    A real droplet often appears as a filled, soft patch rather than a clean
    circle edge.  Hough therefore has no edge to vote for and MSER can reject
    the patch when its intensity is nearly uniform.  External contours provide
    a small, robust fallback for that case.  A closing step reconnects a
    droplet's bright rim and dark body, which frequently appear as separate
    regions in a dry-reference difference image.
    """
    if diff_img.ndim != 2 or not np.any(diff_img):
        return []

    mask = np.where(diff_img > 0, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Small droplets can be slightly below the nominal minimum area because
    # thresholding clips their soft edge.  Allow a quarter of that area while
    # retaining the upper bound to avoid treating broad lighting changes as a
    # droplet.
    # A genuine small droplet can have a broken outline, so permit a relaxed
    # circularity score.  Requiring a meaningful connected area keeps isolated
    # sensor noise from becoming a candidate.
    lower_area = max(30, int(min_area * 0.25))
    results: list[Candidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < lower_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = float(4 * np.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        results.append(Candidate(float(x), float(y), float(radius), circularity, "contour"))
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
            # Two Hough circles can describe the inner highlight and outer
            # rim of one droplet.  Merge overlapping circles as well as
            # candidates whose centres are simply close together.
            overlap_distance = 0.75 * (c.radius_px + other.radius_px)
            if dist < max(merge_dist_px, overlap_distance):
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
