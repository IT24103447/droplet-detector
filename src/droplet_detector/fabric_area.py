"""Locate the inner white specimen rim and reject unchanged opening-frame marks."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import DropletDetection


def find_fabric_area(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Find a central circular white rim; fail instead of scanning the whole rig."""
    height, width = frame.shape[:2]
    scale = min(1., 640 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    minimum = min(gray.shape)
    circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (7, 7), 1.5),
                              cv2.HOUGH_GRADIENT, 1.2, minimum * .2,
                              param1=100, param2=40, minRadius=round(minimum * .24),
                              maxRadius=round(minimum * .49))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    best = None
    for cx, cy, radius in (() if circles is None else circles[0]):
        if not (.25 * gray.shape[1] < cx < .75 * gray.shape[1]
                and .2 * gray.shape[0] < cy < .8 * gray.shape[0]):
            continue
        radii = np.arange(round(radius * .72), round(radius * 1.18) + 1)
        xs = np.rint(cx + radii[:, None] * np.cos(angles)).astype(int)
        ys = np.rint(cy + radii[:, None] * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < gray.shape[1]) & (ys >= 0) & (ys < gray.shape[0])
        pixels = hsv[np.clip(ys, 0, gray.shape[0]-1), np.clip(xs, 0, gray.shape[1]-1)]
        coverage = ((pixels[..., 2] > 205) & (pixels[..., 1] < 110) & valid).mean(axis=1)
        # Require a broad, nearly complete bright band, not a few clamp LEDs.
        indices = np.flatnonzero(coverage >= .65)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        for run in runs:
            if len(run) < max(4, minimum * .012):
                continue
            start = int(run[0])
            # Cloth inside the band must be distinguishable from a solid white disk.
            inner = coverage[max(0, start-18):max(1, start-5)]
            if not len(inner) or np.median(inner) > .5:
                continue
            inner_radius = radii[start] - max(3, minimum * .008)
            score = float(coverage[run].mean() * len(run))
            if best is None or score > best[0]:
                best = score, cx, cy, inner_radius
    if best is None:
        raise ValueError("Could not locate the inside of the white fabric rim. "
                         "Set the fabric area manually; the analyzer will not scan outside it.")
    _, cx, cy, radius = best
    x, y = round((cx-radius)/scale), round((cy-radius)/scale)
    diameter = round(2*radius/scale)
    if x < 0 or y < 0 or x+diameter > width or y+diameter > height:
        raise ValueError("The fabric opening is clipped. Set the fabric area manually.")
    return x, y, diameter, diameter


@dataclass
class OpeningFrameEvidence:
    """Conservative new-appearance check; the opening specimen must be dry."""
    opening: np.ndarray
    roi: tuple[int, int, int, int]
    motion_rejections: int = 0
    _reference: np.ndarray = field(init=False, repr=False)
    _small_reference: np.ndarray = field(init=False, repr=False)
    _inside: np.ndarray = field(init=False, repr=False)
    _kernel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        x, y, w, h = self.roi
        self._reference = cv2.cvtColor(self.opening[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        size = (max(8, round(w * min(1., 320/w))), max(8, round(h * min(1., 320/w))))
        self._small_reference = cv2.resize(self._reference, size)
        yy, xx = np.ogrid[:h, :w]
        self._inside = ((2*xx+1)/w-1)**2 + ((2*yy+1)/h-1)**2 <= 1
        kernel = max(3, round(min(w,h)*.012) | 1)
        self._kernel = np.ones((kernel,kernel), np.uint8)

    def filter(self, frame: np.ndarray, detections: list[DropletDetection]) -> list[DropletDetection]:
        if not detections:
            return []
        if frame.shape != self.opening.shape:
            raise ValueError("Image dimensions changed during the test; recalibrate the fabric area.")
        x, y, w, h = self.roi
        before = self._reference
        now = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        size = (self._small_reference.shape[1], self._small_reference.shape[0])
        shift, response = cv2.phaseCorrelate(self._small_reference, cv2.resize(now, size))
        dx, dy = shift[0]*w/size[0], shift[1]*h/size[1]
        if response > .1 and np.isfinite([dx,dy]).all():
            if max(abs(dx),abs(dy)) > min(w,h)*.015:
                self.motion_rejections += 1
                return []
            before = cv2.warpAffine(before, np.float32([[1,0,dx],[0,1,dy]]), (w,h),
                                    borderMode=cv2.BORDER_REFLECT)
        change = cv2.GaussianBlur(now-before, (3,3), .6)
        inside = self._inside
        change -= np.median(change[inside])
        # A small model box can sit on a much larger pool or moving reflection.
        # Reject candidates connected to broad changes, not just oversized boxes.
        changed = ((np.abs(change) >= 10) & inside).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, self._kernel)
        changed[~inside] = 0
        _, components, stats, _ = cv2.connectedComponentsWithStats(changed)
        large = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] > w*h*.20)
        large = large[large != 0]
        accepted = []
        for d in detections:
            cx, cy = round(d.x_px-x), round(d.y_px-y)
            radius = max(3, round(d.radius_px))
            outer = radius * 3
            local = change[max(0,cy-radius):min(h,cy+radius+1), max(0,cx-radius):min(w,cx+radius+1)]
            surround = change[max(0,cy-outer):min(h,cy+outer+1), max(0,cx-outer):min(w,cx+outer+1)]
            if not local.size or not surround.size:
                continue
            labels = components[max(0,cy-radius):min(h,cy+radius+1), max(0,cx-radius):min(w,cx+radius+1)]
            if large.size and np.isin(labels, large).mean() > .1:
                continue
            signal = np.percentile(np.abs(local), 90)
            background = np.percentile(np.abs(surround), 65)
            if signal >= max(10, background * 1.6 + 4):
                accepted.append(d)
        return accepted
