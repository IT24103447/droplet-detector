"""Associate frame detections with persistent leak positions."""
from dataclasses import dataclass
from math import hypot

from .models import DropletDetection


@dataclass
class _Track:
    first_seen: float
    last_seen: float
    detection: DropletDetection
    hits: int = 1
    confirmed: bool = False


class LeakTracker:
    def __init__(self, min_hits: int, match_px: float, max_gap_s: float):
        self.tracks: list[_Track] = []
        self.min_hits = min_hits
        self.match_px = match_px
        self.max_gap_s = max_gap_s

    def update(self, detections: list[DropletDetection], elapsed_s: float) -> list[_Track]:
        self.tracks = [
            t for t in self.tracks if t.confirmed or elapsed_s - t.last_seen <= self.max_gap_s
        ]
        pairs = []
        for ti, track in enumerate(self.tracks):
            previous = track.detection
            for di, candidate in enumerate(detections):
                distance = hypot(previous.x_px - candidate.x_px, previous.y_px - candidate.y_px)
                # Limit the effect of growth so a large box doesn't absorb nearby leaks.
                gate = max(self.match_px, min(previous.radius_px, candidate.radius_px))
                if distance <= gate:
                    pairs.append((distance, ti, di))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        touched = []
        for _, ti, di in sorted(pairs):
            if ti in matched_tracks or di in matched_detections:
                continue
            track = self.tracks[ti]
            track.last_seen = elapsed_s
            track.detection = detections[di]
            track.hits += 1
            matched_tracks.add(ti)
            matched_detections.add(di)
            touched.append(track)
        for di, detection in enumerate(detections):
            if di not in matched_detections:
                track = _Track(elapsed_s, elapsed_s, detection)
                self.tracks.append(track)
                touched.append(track)
        confirmed = sorted(
            (t for t in touched if not t.confirmed and t.hits >= self.min_hits),
            key=lambda t: (t.first_seen, -t.detection.confidence),
        )
        for track in confirmed:
            track.confirmed = True
        return confirmed
