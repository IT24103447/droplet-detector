"""Opening-frame evidence for spreading water, independent of bead proposals.

This is a conservative visual heuristic, not a trained water segmentation model.
Broad appearance changes alone are insufficient: a new curved boundary must
persist across several samples. Coordinates describe that observed boundary,
not the origin of a leak or the complete extent of the water.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SurfaceWaterObservation:
    detection_time_s: float
    confirmation_time_s: float
    boundary: list[tuple[int, int]]


class SurfaceWaterDetector:
    interval_s = .25
    max_gap_s = .8

    def __init__(self, opening: np.ndarray, roi: tuple[int, int, int, int], shape="ellipse"):
        self.roi = roi
        self.frame_shape = opening.shape
        self.shape = shape
        self.reference = self._gray(opening).astype(np.float32)
        self.height, self.width = self.reference.shape
        yy, xx = np.ogrid[:self.height, :self.width]
        self.inside = (((2*xx+1)/self.width-1)**2 + ((2*yy+1)/self.height-1)**2 < .97**2
                       if shape == "ellipse" else np.ones(self.reference.shape, dtype=bool))
        self.reference_edges = cv2.dilate(cv2.Canny(self.reference.astype(np.uint8), 30, 80),
                                          np.ones((5, 5), np.uint8))
        # Pressure can bend or shift the stitched cross without any leakage.
        # Exclude a band around long opening-frame seams, including their crossing.
        self.seams = np.zeros(self.reference.shape, np.uint8)
        minimum = min(self.width, self.height)
        lines = cv2.HoughLinesP(cv2.Canny(self.reference.astype(np.uint8), 30, 80),
                                1, np.pi/180, 60, minLineLength=minimum*.35,
                                maxLineGap=minimum*.03)
        for line in (() if lines is None else lines[:, 0]):
            x1, y1, x2, y2 = line
            direction = np.array([x2-x1, y2-y1], dtype=float)
            direction *= 2*max(self.width, self.height)/np.linalg.norm(direction)
            centre = np.array([(x1+x2)/2, (y1+y2)/2])
            cv2.line(self.seams, tuple(np.rint(centre-direction).astype(int)),
                     tuple(np.rint(centre+direction).astype(int)), 255, max(9, round(minimum*.08)))
        self.next_sample = 0.
        self.first_seen = None
        self.last_seen = None
        self.hits = 0
        self.previous_boundary = None
        self.confirmed = False

    def _gray(self, frame):
        x, y, w, h = self.roi
        if frame.shape != self.frame_shape or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError("Image dimensions changed during surface-water analysis")
        crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        scale = min(1., 512/max(w, h))
        return cv2.resize(crop, (max(8, round(w*scale)), max(8, round(h*scale))))

    def candidate(self, frame, current_roi=None):
        """Return a newly appearing curved edge; ignore light shifts and straight seams."""
        now = self._gray(frame)
        inside = self.inside
        if current_roi is not None:
            x, y, w, h = self.roi
            cx, cy, cw, ch = current_roi
            yy, xx = np.ogrid[:self.height, :self.width]
            nx, ny = x+(xx+.5)*w/self.width, y+(yy+.5)*h/self.height
            allowed = (((2*(nx-cx)/cw-1)**2+(2*(ny-cy)/ch-1)**2 < .97**2)
                       if self.shape == "ellipse" else
                       (nx >= cx) & (nx < cx+cw) & (ny >= cy) & (ny < cy+ch))
            inside = inside & allowed
        if inside.sum() < self.inside.sum()*.5:
            return None
        shift, response = cv2.phaseCorrelate(self.reference, now.astype(np.float32))
        if not np.isfinite(shift).all() or response < .1:
            return None
        dx, dy = shift
        if max(abs(dx), abs(dy)) > min(self.width, self.height)*.015:
            return None
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        before = cv2.warpAffine(self.reference, transform, (self.width, self.height),
                                borderMode=cv2.BORDER_REFLECT)
        old_edges = cv2.warpAffine(self.reference_edges, transform, (self.width, self.height),
                                   borderMode=cv2.BORDER_CONSTANT)
        seams = cv2.warpAffine(self.seams, transform, (self.width, self.height))
        change = now.astype(np.float32)-before
        change -= np.median(change[inside])
        changed = (np.abs(cv2.GaussianBlur(change, (5, 5), 1)) > 12) & inside
        if changed.sum() < inside.sum()*.10:
            return None
        edges = cv2.Canny(now, 30, 80)
        near_change = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), np.uint8))
        edges[(old_edges > 0) | (seams > 0) | ~inside | (near_change == 0)] = 0
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            _, (short, long), _ = cv2.minAreaRect(contour)
            short, long = sorted((short, long))
            # Hull area measures curvature of an open arc; contourArea alone
            # is near zero on a one-pixel-wide water front.
            if (min(w, h) >= min(self.width, self.height)*.12
                    and cv2.arcLength(contour, False) >= min(self.width, self.height)*.6
                    and short >= long*.12
                    and hull_area >= self.inside.sum()*.02):
                candidates.append((hull_area, contour))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def observe(self, frame, elapsed_s: float, current_roi=None) -> SurfaceWaterObservation | None:
        if self.confirmed or elapsed_s < self.next_sample:
            return None
        self.next_sample = elapsed_s+self.interval_s
        boundary = self.candidate(frame, current_roi)
        if boundary is None:
            self.hits = 0
            self.first_seen = self.last_seen = self.previous_boundary = None
            return None
        # Require spatial continuity as well as consecutive observations.
        box = cv2.boundingRect(boundary)
        continuous = False
        if self.previous_boundary is not None and self.last_seen is not None and elapsed_s-self.last_seen <= self.max_gap_s:
            a, b = box, self.previous_boundary
            overlap = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
                0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
            continuous = overlap >= .25*min(a[2]*a[3], b[2]*b[3])
        if not continuous:
            self.first_seen, self.hits = elapsed_s, 0
        self.hits += 1
        self.last_seen, self.previous_boundary = elapsed_s, box
        if self.hits < 3:
            return None
        self.confirmed = True
        x, y, w, h = self.roi
        points = cv2.approxPolyDP(boundary, 1.5, False).reshape(-1, 2)
        native = [(round(x+px*w/self.width), round(y+py*h/self.height)) for px, py in points]
        return SurfaceWaterObservation(self.first_seen, elapsed_s, native)
