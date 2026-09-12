"""Image preprocessing: illumination flattening, noise-floor suppression,
and dry-reference differencing.

Key design decision: illumination flattening is applied to each raw image
*before* differencing — never to the diff itself.  Applying contrast
stretching to a near-empty diff image rescales sensor noise to full contrast
and produces hundreds of false-positive "edges" (confirmed the hard way).
"""
from __future__ import annotations

import cv2
import numpy as np


def flatten_illumination(gray: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
    """Remove broad, slow-varying lighting gradients.

    Divides the image by a heavily blurred copy of itself, normalising
    uneven illumination (e.g. soft angled light being brighter on one side)
    while preserving local detail.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale image.
    blur_ksize:
        Kernel size for the Gaussian blur that estimates the background
        illumination.  Must be large enough to span the broadest gradient
        but small enough not to smear real droplet features.  Forced odd.

    Returns
    -------
    np.ndarray
        Flattened uint8 grayscale image, mean normalised to ~128.
    """
    gray_f = gray.astype(np.float32)
    k = blur_ksize | 1  # must be odd
    background = cv2.GaussianBlur(gray_f, (k, k), 0)
    flattened = gray_f / (background + 1e-5) * 128.0
    return np.clip(flattened, 0, 255).astype(np.uint8)


def suppress_noise_floor(diff: np.ndarray, noise_floor: int = 15) -> np.ndarray:
    """Zero out small pixel differences (sensor noise, tiny lighting flicker).

    Uses a hard threshold — pixel values below ``noise_floor`` become 0,
    values at or above it are kept as-is.  Deliberately *not* a min-max
    contrast stretch: stretching a near-empty diff to fill 0-255 turns
    pure noise into fake full-contrast "edges".

    Parameters
    ----------
    diff:
        Single-channel uint8 absolute-difference image.
    noise_floor:
        Pixel-value threshold below which differences are considered noise.

    Returns
    -------
    np.ndarray
        Thresholded diff image (same dtype, same shape).
    """
    _, thresholded = cv2.threshold(diff, noise_floor, 255, cv2.THRESH_TOZERO)
    return thresholded


def diff_against_dry_reference(
    current_bgr: np.ndarray,
    dry_reference_bgr: np.ndarray,
    noise_floor: int = 15,
) -> np.ndarray:
    """Compute the structural difference between the current frame and a
    dry-reference image of the same fabric sample.

    Steps:
    1. Convert both images to grayscale.
    2. Flatten illumination gradients on each image independently.
    3. Light Gaussian blur to soften per-pixel noise.
    4. Absolute difference.
    5. Suppress noise floor.

    This cancels out anything that doesn't change between shots —
    the weave pattern, fixed reflections baked into the dry reference,
    and broad lighting gradients.

    Parameters
    ----------
    current_bgr:
        Current frame (BGR, uint8).
    dry_reference_bgr:
        Dry reference frame (BGR, uint8, same resolution).
    noise_floor:
        Pixel-value threshold for noise suppression.

    Returns
    -------
    np.ndarray
        Single-channel uint8 difference image with noise suppressed.
    """
    current_gray = flatten_illumination(cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY))
    dry_gray = flatten_illumination(cv2.cvtColor(dry_reference_bgr, cv2.COLOR_BGR2GRAY))

    current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
    dry_gray = cv2.GaussianBlur(dry_gray, (5, 5), 0)

    diff = cv2.absdiff(current_gray, dry_gray)
    return suppress_noise_floor(diff, noise_floor)
