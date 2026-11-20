"""Water films must trigger without bead proposals, while rig changes must not."""
import json

import cv2
import numpy as np
import pytest

from droplet_detector import OutputPaths, VideoSettings, analyze_video
from droplet_detector import video
from droplet_detector.surface_water import SurfaceWaterDetector


def specimen():
    rng = np.random.default_rng(52)
    gray = np.clip(155+rng.normal(0, 2, (512, 512)), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(frame, (0, 290), (511, 230), (35, 35, 35), 3)
    cv2.line(frame, (210, 0), (290, 511), (35, 35, 35), 3)
    return frame


def with_water(opening):
    wet = opening.copy()
    yy, xx = np.ogrid[:512, :512]
    ring = ((xx-256)**2+(yy-256)**2 >= 175**2) & ((xx-256)**2+(yy-256)**2 < 237**2)
    wet[ring] = np.maximum(0, wet[ring].astype(int)-38).astype(np.uint8)
    cv2.circle(wet, (256, 256), 175, (200, 200, 200), 2)
    return wet


def test_spreading_water_needs_persistent_boundary_and_keeps_coordinates_inside():
    opening = specimen()
    detector = SurfaceWaterDetector(opening, (0, 0, 512, 512))
    wet = with_water(opening)
    assert detector.observe(opening, 0) is None
    assert detector.observe(wet, .25) is None
    assert detector.observe(wet, .5) is None
    result = detector.observe(wet, .75)
    assert result is not None
    assert result.detection_time_s == .25 and result.confirmation_time_s == .75
    assert all(((x-256)/256)**2+((y-256)/256)**2 < 1 for x, y in result.boundary)
    assert detector.observe(wet, 1) is None  # One presence event, not many droplets.


@pytest.mark.parametrize("change", ["lighting", "seams", "outside", "movement", "already_wet"])
def test_non_leak_changes_and_existing_water_do_not_trigger(change):
    opening = specimen()
    frame = opening.copy()
    if change == "lighting":
        frame = np.clip(frame.astype(int)+32, 0, 255).astype(np.uint8)
    elif change == "seams":
        frame = specimen()
        cv2.line(frame, (0, 300), (511, 240), (35, 35, 35), 3)
        cv2.line(frame, (220, 0), (300, 511), (35, 35, 35), 3)
    elif change == "outside":
        cv2.circle(frame, (256, 256), 254, (0, 0, 0), 10)
    elif change == "movement":
        frame = cv2.warpAffine(with_water(opening), np.float32([[1, 0, 16], [0, 1, 0]]),
                               (512, 512), borderMode=cv2.BORDER_REFLECT)
    elif change == "already_wet":
        opening = frame = with_water(opening)
    detector = SurfaceWaterDetector(opening, (0, 0, 512, 512))
    assert all(detector.observe(frame, t) is None for t in [0, .25, .5, .75, 1.])


def test_dry_interruption_resets_water_confirmation():
    opening = specimen()
    detector = SurfaceWaterDetector(opening, (0, 0, 512, 512))
    wet = with_water(opening)
    for frame, t in [(wet, 0), (wet, .25), (opening, .5), (wet, .75), (wet, 1.)]:
        assert detector.observe(frame, t) is None
    assert detector.observe(wet, 1.25).detection_time_s == .75


def test_video_reports_water_without_inventing_droplet_events(tmp_path, monkeypatch):
    opening = specimen()
    source = tmp_path / "film.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512))
    assert writer.isOpened()
    for index in range(35):
        writer.write(opening if index < 10 else with_water(opening))
    writer.release()
    monkeypatch.setattr(video, "FrameDetector", lambda *_: lambda frame: [])
    outputs = OutputPaths.in_directory(tmp_path / "result")
    result = analyze_video(source, outputs=outputs, settings=VideoSettings(roi=(0, 0, 512, 512)))
    assert result.status == "completed", result.errors
    assert result.leaks == []
    assert result.surface_water is not None
    assert result.leakage_detected
    assert 1 <= result.surface_water.detection_time_s < 2
    assert cv2.imread(result.surface_water.image_filename) is not None
    saved = json.loads(outputs.results.read_text())
    assert saved["leakage_detected"] and saved["surface_water"]["boundary"]
    assert any("not the exact leak onset" in warning for warning in result.warnings)


def test_invalid_frame_dimensions_fail_explicitly():
    detector = SurfaceWaterDetector(specimen(), (0, 0, 512, 512))
    with pytest.raises(ValueError, match="dimensions"):
        detector.observe(np.zeros((100, 100, 3), np.uint8), 0)


def test_water_boundary_must_fit_the_current_fabric_area():
    opening = specimen()
    detector = SurfaceWaterDetector(opening, (0, 0, 512, 512))
    wet = with_water(opening)
    assert detector.candidate(wet) is not None
    assert detector.candidate(wet, current_roi=(156, 156, 200, 200)) is None
