"""Confidence scoring for candidate droplets.

Each surviving candidate is scored on three axes:

1. **Size score** — how close the candidate's radius is to the expected
   droplet size range.  Perfect match → 1.0, outside tolerance → 0.0.
2. **Bright-centre score** — water droplets refract overhead light and
   tend to show a brighter highlight near the centre.  Flat reflections
   and wrinkle shadows do not.
3. **Circularity** — already computed during detection; droplets are round,
   wrinkles and threads are elongated.

The final confidence is a weighted combination:
``0.40 * size + 0.35 * bright_centre + 0.25 * circularity``.
"""
from __future__ import annotations

import numpy as np

from .candidate_detection import Candidate


def size_score(
    candidate: Candidate,
    expected_radius_px: float,
    tolerance_px: float,
) -> float:
    """Score how close the candidate's radius is to the expected value.

    Returns 1.0 for a perfect match and linearly drops to 0.0 at
    ±``tolerance_px`` away.

    Parameters
    ----------
    candidate:
        The candidate to score.
    expected_radius_px:
        Mid-point of the expected radius range.
    tolerance_px:
        Half-width of the acceptable range.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    diff = abs(candidate.radius_px - expected_radius_px)
    return max(0.0, 1.0 - diff / tolerance_px)


def bright_center_score(gray_img: np.ndarray, candidate: Candidate) -> float:
    """Score whether the candidate has a brighter centre than its rim.

    Water droplets refract overhead light and typically show a bright
    highlight near the centre.  Flat reflections or wrinkle shadows
    do not exhibit this concentric brightness gradient.

    Parameters
    ----------
    gray_img:
        Full grayscale image (uint8) of the *current* frame (not the diff).
    candidate:
        The candidate to score.

    Returns
    -------
    float
        Score in [0.0, 1.0].  Higher = stronger bright-centre evidence.
    """
    x, y, r = int(candidate.x), int(candidate.y), max(int(candidate.radius_px), 1)
    h, w = gray_img.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    patch = gray_img[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    center_val = float(patch[patch.shape[0] // 2, patch.shape[1] // 2])
    ring_mean = float(np.mean(patch))
    contrast = (center_val - ring_mean) / 255.0
    return float(np.clip(contrast + 0.5, 0.0, 1.0))


def confidence(
    candidate: Candidate,
    gray_img: np.ndarray,
    expected_radius_px: float,
    tolerance_px: float,
) -> float:
    """Composite confidence score for a single candidate.

    Weights: 40% size, 35% bright-centre, 25% circularity.

    Parameters
    ----------
    candidate:
        The candidate to score.
    gray_img:
        Full grayscale image of the current frame.
    expected_radius_px:
        Mid-point of the expected radius range.
    tolerance_px:
        Half-width of the acceptable radius range.

    Returns
    -------
    float
        Confidence in [0.0, 1.0].
    """
    s = size_score(candidate, expected_radius_px, tolerance_px)
    b = bright_center_score(gray_img, candidate)
    c = candidate.circularity
    return float(np.clip(0.4 * s + 0.35 * b + 0.25 * c, 0.0, 1.0))
