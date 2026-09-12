"""Frame-to-frame consistency check (simplified tracking preview).

A real droplet stays roughly in the same spot across successive frames,
or grows slightly as more water appears.  Camera vibration and lighting
flicker rarely produce a bright spot in the exact same location twice.
This module filters transient noise by requiring spatial consistency.

This is a *simplified preview* of the full multi-frame tracking logic
that arrives in Step 3 (ByteTrack/BoT-SORT with droplet-count stopping
criteria).  For Step 1 it is exercised whenever more than one frame
of the same sample is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .candidate_detection import Candidate


@dataclass
class TrackedPoint:
    """A candidate that has been observed across one or more frames."""

    x: float
    y: float
    radius_px: float
    frames_seen: int = field(default=1)


def match_to_previous(
    candidates: list[Candidate],
    previous: list[TrackedPoint],
    max_move_px: float,
) -> tuple[list[TrackedPoint], list[Candidate]]:
    """Match current-frame candidates to previously tracked points.

    A candidate matches a previous point if:
    * Its centre is within ``max_move_px`` pixels, **and**
    * Its radius hasn't shrunk by more than 1 px (droplets grow or
      stay the same; they don't shrink).

    Returns
    -------
    tuple[list[TrackedPoint], list[Candidate]]
        - Updated tracked points (matched points have ``frames_seen``
          incremented; unmatched candidates become new tracks with
          ``frames_seen=1``).
        - The subset of candidates that did **not** match any previous
          point (newly appeared).
    """
    matched: list[TrackedPoint] = []
    remaining = list(candidates)
    for prev in previous:
        best: Candidate | None = None
        best_dist = max_move_px
        for cand in remaining:
            dist = ((cand.x - prev.x) ** 2 + (cand.y - prev.y) ** 2) ** 0.5
            if dist <= best_dist and cand.radius_px >= prev.radius_px - 1:
                best = cand
                best_dist = dist
        if best is not None:
            matched.append(
                TrackedPoint(
                    x=best.x,
                    y=best.y,
                    radius_px=best.radius_px,
                    frames_seen=prev.frames_seen + 1,
                )
            )
            remaining.remove(best)
    # Unmatched candidates become fresh tracks
    for cand in remaining:
        matched.append(
            TrackedPoint(
                x=cand.x, y=cand.y, radius_px=cand.radius_px, frames_seen=1
            )
        )
    return matched, remaining
