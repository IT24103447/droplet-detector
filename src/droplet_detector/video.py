"""Public API for replaying a recorded test or monitoring a USB camera."""
from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Literal
from functools import lru_cache

import cv2
import numpy as np
from pydantic import BaseModel, Field, computed_field

from .leak_tracking import LeakTracker
from .models import DropletDetection
from .single_image import FrameDetector
from .vision_language import REQUEST_TIMEOUT_S, VisionLanguageDetector
from .fabric_area import OpeningFrameEvidence, find_fabric_area
from .surface_water import SurfaceWaterDetector


VideoDetector = Callable[[np.ndarray], list[DropletDetection]]


@dataclass(frozen=True)
class VideoSettings:
    model_path: str | Path = "models/droplet-seg.pt"
    roi: tuple[int, int, int, int] | None = None  # x, y, width, height
    confidence_threshold: float = 0.25
    confirmation_hits: int = 3
    match_distance_px: float = 10
    max_candidate_gap_s: float = 0.5
    resolution: tuple[int, int] | None = None
    frame_rate: float | None = None
    detector_backend: Literal["yolo", "vision"] = "yolo"
    roi_shape: Literal["rectangle", "ellipse"] = "ellipse"
    vision_interval_s: float = 2.0
    auto_fabric_area: bool = True
    require_new_appearance: bool = True
    detect_surface_water: bool = True

    def __post_init__(self):
        if self.detector_backend not in ("yolo", "vision") or self.roi_shape not in ("rectangle", "ellipse"):
            raise ValueError("Unknown detector backend or fabric area shape")
        if not math.isfinite(self.vision_interval_s) or self.vision_interval_s <= 0:
            raise ValueError("The vision analysis interval must be finite and positive")
        if not 0 < self.confidence_threshold <= 1 or self.confirmation_hits < 1:
            raise ValueError("Set a confidence threshold in (0, 1] and at least one confirmation hit")
        if not all(math.isfinite(v) and v > 0 for v in (self.match_distance_px, self.max_candidate_gap_s)):
            raise ValueError("Tracking distance and candidate gap must be finite and positive")
        if self.frame_rate is not None and not (math.isfinite(self.frame_rate) and self.frame_rate > 0):
            raise ValueError("Frame rate must be finite and positive")
        if self.resolution is not None and min(self.resolution) <= 0:
            raise ValueError("Resolution must be positive")
        if self.roi is not None:
            x, y, w, h = self.roi
            if min(x, y) < 0 or min(w, h) <= 0:
                raise ValueError("ROI must have a non-negative origin and positive dimensions")


@dataclass(frozen=True)
class OutputPaths:
    video: str | Path
    images: str | Path
    results: str | Path
    final_image: str | Path

    @classmethod
    def in_directory(cls, directory: str | Path) -> OutputPaths:
        root = Path(directory)
        return cls(root / "recording.mp4", root, root / "results.json", root / "final_marked.jpg")


class LeakEvent(BaseModel):
    droplet_number: int
    x_px: float
    y_px: float
    confidence: float
    detection_time_s: float
    confirmation_time_s: float
    image_filename: str | None = None


class SurfaceWaterEvent(BaseModel):
    detection_time_s: float
    confirmation_time_s: float
    boundary: list[tuple[int, int]]
    image_filename: str | None = None


class TestResult(BaseModel):
    status: Literal["completed", "stopped", "error"]
    leaks: list[LeakEvent] = Field(default_factory=list)
    surface_water: SurfaceWaterEvent | None = None
    video_filename: str | None = None
    final_image_filename: str | None = None
    results_filename: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    fabric_roi: tuple[int, int, int, int] | None = None
    fabric_roi_shape: str | None = None

    @computed_field
    @property
    def leakage_detected(self) -> bool:
        """Presence indication includes water films without countable droplets."""
        return bool(self.leaks) or self.surface_water is not None


@dataclass(frozen=True)
class VideoPreview:
    jpeg: bytes
    elapsed_s: float
    leak_count: int
    water_detected: bool = False


def _output_targets(outputs: OutputPaths, source: str | int) -> tuple[Path, Path, Path, list[Path]]:
    video, result, final = (Path(p).resolve() for p in (outputs.video, outputs.results, outputs.final_image))
    images = Path(outputs.images).resolve()
    snapshots = [images / f"leak_{n:02}.jpg" for n in range(1, 4)]
    paths = [video, result, final, *snapshots, images / "surface_water.jpg"]
    if len(set(paths)) != len(paths) or (isinstance(source, str) and Path(source).resolve() in paths):
        raise ValueError("Output paths must be distinct and cannot overwrite the input video")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    return video, result, final, snapshots


def _save_image(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def _annotate(frame: np.ndarray, leaks: list[LeakEvent], roi=None, roi_shape="rectangle",
              surface_water=None, draw_boundary=False) -> np.ndarray:
    marked = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        if roi_shape == "ellipse":
            cv2.ellipse(marked, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (255,255,0), 2)
        else:
            cv2.rectangle(marked, (x,y), (x+w,y+h), (255,255,0), 2)
    for leak in leaks:
        centre = round(leak.x_px), round(leak.y_px)
        cv2.drawMarker(marked, centre, (0, 255, 0), cv2.MARKER_CROSS, 19, 2)
        cv2.putText(marked, f"#{leak.droplet_number} {leak.detection_time_s:.2f}s",
                    (centre[0] + 6, max(14, centre[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
    if surface_water is not None:
        if draw_boundary:
            cv2.polylines(marked, [np.array(surface_water.boundary, dtype=np.int32)],
                          False, (0, 180, 255), 2)
        cv2.putText(marked, f"Visible water leakage: {surface_water.detection_time_s:.2f}s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 180, 255), 2)
    return marked


def _preview_image(frame: np.ndarray, leaks: list[LeakEvent], roi, roi_shape, surface_water=None) -> np.ndarray:
    if frame.shape[1] <= 960:
        return _annotate(frame, leaks, roi, roi_shape, surface_water)
    scale = 960/frame.shape[1]
    small = cv2.resize(frame, (960,max(1,round(frame.shape[0]*scale))))
    markers = [d.model_copy(update={"x_px":d.x_px*scale,"y_px":d.y_px*scale}) for d in leaks]
    area = tuple(round(v*scale) for v in roi) if roi is not None else None
    return _annotate(small,markers,area,roi_shape,surface_water)


@lru_cache(maxsize=8)
def _ellipse_outside(height: int, width: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    mask = ((2*xx+1)/width-1)**2 + ((2*yy+1)/height-1)**2 > 1
    mask.setflags(write=False)
    return mask


def _detect_frame(detector: VideoDetector, frame: np.ndarray, roi: tuple[int, int, int, int] | None,
                  roi_shape: Literal["rectangle", "ellipse"] = "rectangle",
                  evidence: OpeningFrameEvidence | None = None) -> list[DropletDetection]:
    if roi is None:
        detections = detector(frame)
        return evidence.filter(frame, detections) if evidence else detections
    x, y, w, h = roi
    if x + w > frame.shape[1] or y + h > frame.shape[0]:
        raise ValueError("The fabric ROI extends outside the captured frame")
    crop = frame[y:y + h, x:x + w]
    if roi_shape == "ellipse":
        crop = crop.copy()
        crop[_ellipse_outside(h,w)] = 128

    detections = []
    for d in detector(crop):
        if not all(math.isfinite(v) for v in (d.x_px,d.y_px,d.radius_px)) or d.radius_px <= 0:
            continue
        if not (d.radius_px <= d.x_px <= w-d.radius_px and d.radius_px <= d.y_px <= h-d.radius_px):
            continue
        if roi_shape == "ellipse":
            if d.radius_px > .1 * min(w,h):
                continue  # A fixture-sized box or pooled region is not an individual bead.
            corners = ((-1,-1),(-1,1),(1,-1),(1,1))
            if any(((2*(d.x_px+d.radius_px*dx)/w-1)**2
                    + (2*(d.y_px+d.radius_px*dy)/h-1)**2) > 1 for dx,dy in corners):
                continue
        detections.append(d.model_copy(update={"x_px":d.x_px+x, "y_px":d.y_px+y}))
    return evidence.filter(frame, detections) if evidence else detections


def _run(source: str | int, outputs: OutputPaths, settings: VideoSettings,
         stop: Event, started_at: float, test_start_s: float,
         detector: VideoDetector | None = None,
         on_preview: Callable[[VideoPreview], None] | None = None) -> TestResult:
    result = TestResult(status="completed")
    capture = None
    writer = None
    executor = None
    result_path = None
    last_frame = None
    live = isinstance(source, int)
    wrote_video = False
    skipped = 0
    observations = 0
    area_failures = 0
    evidence = None
    water_detector = None
    vision = settings.detector_backend == "vision"
    roi, roi_shape = settings.roi, settings.roi_shape
    gap = settings.max_candidate_gap_s
    if vision:
        gap = max(gap, 2 * settings.vision_interval_s)
        if live:
            gap = max(gap, REQUEST_TIMEOUT_S + settings.vision_interval_s)
        result.warnings.append(
            f"Vision analysis samples at least {settings.vision_interval_s:g}s apart and uses the opening "
            "frame as its reference. Timestamps locate the first sampled observation."
        )
    tracker = LeakTracker(settings.confirmation_hits, settings.match_distance_px, gap)

    def process(observation, frame: np.ndarray, elapsed: float) -> None:
        detections, water = observation
        if water is not None and result.surface_water is None:
            event = SurfaceWaterEvent(detection_time_s=water.detection_time_s,
                                      confirmation_time_s=water.confirmation_time_s,
                                      boundary=water.boundary)
            path = Path(outputs.images).resolve() / "surface_water.jpg"
            _save_image(path, _annotate(frame, result.leaks, roi, roi_shape, event, True))
            event.image_filename = str(path)
            result.surface_water = event
        for track in tracker.update(detections, elapsed):
            if len(result.leaks) == 3:
                return
            d = track.detection
            event = LeakEvent(droplet_number=len(result.leaks) + 1, x_px=d.x_px, y_px=d.y_px,
                              confidence=d.confidence, detection_time_s=track.first_seen,
                              confirmation_time_s=elapsed)
            result.leaks.append(event)
            path = snapshots[event.droplet_number - 1]
            _save_image(path, _annotate(frame, result.leaks, roi, roi_shape))
            event.image_filename = str(path)

    def detect_live_frame(frame: np.ndarray, elapsed: float):
        # Keep rim detection with inference, off the live capture/recording loop.
        nonlocal roi, roi_shape, area_failures
        current_roi = roi
        if settings.auto_fabric_area and settings.roi is None:
            try:
                current_roi = find_fabric_area(frame)
            except ValueError:
                area_failures += 1
                return [], None
            roi, roi_shape = current_roi, "ellipse"
        detections = (_detect_frame(detector, frame, current_roi, roi_shape, evidence)
                      if len(result.leaks) < 3 else [])
        water = water_detector.observe(frame, elapsed, current_roi) if water_detector else None
        return detections, water

    try:
        video_path, result_path, final_path, snapshots = _output_targets(outputs, source)
        if detector is None:
            if vision:
                local_detector = VisionLanguageDetector(
                    settings.confidence_threshold, compare_to_opening_frame=True,
                )
                local_detector.check_available()
                detector = local_detector
            else:
                detector = FrameDetector(settings.model_path, settings.confidence_threshold)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise OSError(f"Could not open {'camera' if live else 'video'}: {source}")
        if live:
            if settings.resolution:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
            if settings.frame_rate:
                capture.set(cv2.CAP_PROP_FPS, settings.frame_rate)
            executor = ThreadPoolExecutor(max_workers=1)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            if not live:
                raise OSError("Recorded video did not report a usable frame rate")
            fps = settings.frame_rate or 30.0
            result.warnings.append(f"Camera frame rate unavailable. Recording uses {fps:g} FPS.")
        elif live and settings.frame_rate and abs(settings.frame_rate - fps) > 0.5:
            result.warnings.append(f"Camera reported {fps:g} FPS instead of {settings.frame_rate:g} FPS.")
        expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) if not live else 0
        frame_index = 0
        last_source_time = -1.0
        last_preview = -math.inf
        next_inference = 0.0
        inference: tuple[Future, np.ndarray, float] | None = None
        while not stop.is_set():
            ok, frame = capture.read()
            acquired_at = time.monotonic()
            if not ok:
                if stop.is_set():
                    break
                if live:
                    raise OSError("USB camera stopped returning frames")
                if expected_frames > frame_index + 1:
                    result.warnings.append("Video ended before its reported frame count.")
                break
            if live:
                elapsed = acquired_at - started_at
            else:
                source_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(source_time) or source_time < 0 or source_time <= last_source_time:
                    source_time = max(frame_index / fps, last_source_time + 1 / fps)
                    warning = "Video timestamps unavailable or non-monotonic. Estimated affected times from FPS."
                    if warning not in result.warnings:
                        result.warnings.append(warning)
                last_source_time = source_time
                elapsed = source_time - test_start_s
            frame_index += 1
            if elapsed < 0:
                continue
            area_valid = True
            if settings.auto_fabric_area and settings.roi is None and (not live or writer is None):
                try:
                    roi = find_fabric_area(frame)
                    roi_shape = "ellipse"
                except ValueError:
                    if writer is None:
                        raise
                    area_failures += 1
                    area_valid = False
            if writer is None:
                height, width = frame.shape[:2]
                if settings.require_new_appearance and not vision:
                    evidence = OpeningFrameEvidence(frame.copy(), roi or (0,0,width,height))
                if settings.detect_surface_water and roi is not None:
                    water_detector = SurfaceWaterDetector(frame, roi, roi_shape)
                    water_detector.max_gap_s = max(.8, 2*settings.vision_interval_s) if vision else .8
                if live and settings.resolution and (width, height) != settings.resolution:
                    result.warnings.append(f"Camera captured {width}x{height} instead of requested resolution.")
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not open output video: {video_path}")
            result.fabric_roi, result.fabric_roi_shape = roi, roi_shape if roi else None
            writer.write(frame)
            wrote_video = True
            last_frame = frame
            if inference is not None and inference[0].done():
                future, previous_frame, previous_elapsed = inference
                process(future.result(), previous_frame, previous_elapsed)
                inference = None
            needs_water = water_detector is not None and not water_detector.confirmed
            if area_valid and (len(result.leaks) < 3 or needs_water) and elapsed >= next_inference:
                if executor is None:
                    detections = (_detect_frame(detector, frame, roi, roi_shape, evidence)
                                  if len(result.leaks) < 3 else [])
                    water = water_detector.observe(frame, elapsed, roi) if water_detector else None
                    process((detections, water), frame, elapsed)
                    if vision:
                        observations += 1
                        next_inference = elapsed + settings.vision_interval_s
                elif inference is None:
                    inference = (executor.submit(detect_live_frame, frame, elapsed), frame, elapsed)
                    if vision:
                        observations += 1
                        next_inference = elapsed + settings.vision_interval_s
                else:
                    skipped += 1
            if on_preview is not None and acquired_at - last_preview >= 0.1:
                preview = _preview_image(frame, result.leaks, roi, roi_shape, result.surface_water)
                encoded, jpg = cv2.imencode(".jpg", preview)
                if encoded:
                    on_preview(VideoPreview(jpg.tobytes(), elapsed, len(result.leaks), result.surface_water is not None))
                last_preview = acquired_at
        if inference is not None:
            future, frame, elapsed = inference
            process(future.result(), frame, elapsed)
        if last_frame is None and not stop.is_set():
            raise OSError("No usable frames at or after test start")
        if stop.is_set():
            result.status = "stopped"
    except Exception as exc:
        result.status = "error"
        result.errors.append(str(exc))
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if executor is not None:
            executor.shutdown(wait=True)
    if skipped:
        result.warnings.append(f"Inference skipped {skipped} live frames while busy. Capture kept recording.")
    if area_failures:
        result.warnings.append(f"Detection paused on {area_failures} frames where the white fabric rim was not reliable.")
    if evidence is not None and evidence.motion_rejections:
        result.warnings.append(f"Rejected candidates on {evidence.motion_rejections} frames with camera/fabric movement.")
    if evidence is not None:
        result.warnings.append("New-leak verification used the opening frame as its dry reference; pre-existing droplets may be ignored.")
    if water_detector is not None:
        result.warnings.append("Spreading-water analysis compares with the opening frame, which should be dry. "
                               "Its timestamp is the first persistent curved-boundary observation, not the exact leak onset. "
                               "This visual check can miss water without a clear boundary.")
    if vision and wrote_video and observations < 2:
        result.warnings.append("The recording ended before a frame could be compared with the opening frame.")
    if wrote_video:
        if video_path.is_file() and video_path.stat().st_size:
            result.video_filename = str(video_path)
        else:
            result.errors.append(f"Recording could not be saved: {video_path}")
            result.status = "error"
    if last_frame is not None:
        try:
            _save_image(final_path, _annotate(last_frame, result.leaks, roi, roi_shape, result.surface_water))
            result.final_image_filename = str(final_path)
        except (OSError, cv2.error) as exc:
            result.errors.append(str(exc))
            result.status = "error"
    if result_path is not None:
        try:
            result.results_filename = str(result_path)
            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            result.results_filename = None
            result.errors.append(str(exc))
            result.status = "error"
    return result


def analyze_video(video_path: str | Path, *, outputs: OutputPaths,
                  settings: VideoSettings = VideoSettings(), test_start_s: float = 0.0,
                  on_preview: Callable[[VideoPreview], None] | None = None) -> TestResult:
    """Replay a video, reporting up to three beads and a separate water-film event."""
    if not math.isfinite(test_start_s) or test_start_s < 0:
        raise ValueError("Test start must be a finite, non-negative video timestamp")
    return _run(str(Path(video_path).resolve()), outputs, settings, Event(), time.monotonic(),
                test_start_s, on_preview=on_preview)


class LiveTest:
    """A running capture session. The caller owns its Start and Stop signals."""

    def __init__(self, camera_id: int, outputs: OutputPaths, settings: VideoSettings,
                 detector: VideoDetector | None = None,
                 on_preview: Callable[[VideoPreview], None] | None = None):
        started_at = time.monotonic()
        self._stop = Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = self._executor.submit(
            _run, camera_id, outputs, settings, self._stop, started_at, 0.0, detector, on_preview
        )
        self._executor.shutdown(wait=False)

    @property
    def done(self) -> bool:
        return self._future.done()

    def stop_test(self, timeout: float | None = None) -> TestResult:
        self.request_stop()
        return self._future.result(timeout)

    def request_stop(self) -> None:
        """Request stop without blocking the monitoring interface."""
        self._stop.set()


def start_test(camera_id: int, *, outputs: OutputPaths,
               settings: VideoSettings = VideoSettings(),
               prepared_detector: VideoDetector | None = None,
               on_preview: Callable[[VideoPreview], None] | None = None) -> LiveTest:
    """Start a live test. Supply a preloaded detector to avoid loading weights after Start."""
    if type(camera_id) is not int or camera_id < 0:
        raise ValueError("camera_id must be a non-negative integer")
    return LiveTest(camera_id, outputs, settings, prepared_detector, on_preview)
