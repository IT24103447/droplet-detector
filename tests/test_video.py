"""Exercise the video contract using repeatable detections and real video files."""
import base64
import json

import cv2
import numpy as np
import pytest

from droplet_detector import DropletDetection, OutputPaths, VideoSettings, VisionLanguageDetector, analyze_video, start_test
from droplet_detector import video, vision_language


def test_smaller_preview_keeps_native_detection_and_area_coordinates():
    frame=np.zeros((1080,1920,3),np.uint8)
    leak=video.LeakEvent(droplet_number=1,x_px=1000,y_px=600,confidence=.9,
                         detection_time_s=1,confirmation_time_s=1.2)
    preview=video._preview_image(frame,[leak],(400,100,1000,900),'ellipse')
    assert preview.shape==(540,960,3)
    assert tuple(preview[300,500])==(0,255,0)
    assert leak.x_px==1000 and leak.y_px==600
    assert not frame.any()


def test_replay_records_three_distinct_leaks_and_keeps_recording(tmp_path, monkeypatch):
    source = tmp_path / "company_clip.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((120, 160, 3), index, dtype=np.uint8))
    writer.release()

    class Detector:
        def __init__(self, *_):
            self.index = 0

        def __call__(self, frame):
            assert frame.shape[:2] == (90, 130)
            i = self.index
            self.index += 1
            xs = []
            if i != 3:
                xs.append(16 if i < 3 else 20)  # One leak grows and briefly disappears.
            if i >= 4:
                xs.append(50)
            if i >= 7:
                xs.extend([85, 113])  # The fourth distinct leak must not be logged.
            return [DropletDetection(droplet_number=n, x_px=x, y_px=30,
                    radius_px=5 + i / 3, confidence=.9) for n, x in enumerate(xs, 1)]

    monkeypatch.setattr(video, "FrameDetector", Detector)
    outputs = OutputPaths.in_directory(tmp_path / "run")
    previews = []
    result = analyze_video(source, outputs=outputs, settings=VideoSettings(roi=(8, 10, 130, 90), require_new_appearance=False),
                           test_start_s=.2, on_preview=previews.append)
    assert result.status == "completed", result.errors
    assert [e.x_px for e in result.leaks] == [24, 58, 93]
    np.testing.assert_allclose([e.detection_time_s for e in result.leaks], [0, .4, .7], atol=.001)
    assert all(cv2.imread(e.image_filename) is not None for e in result.leaks)
    assert cv2.imread(result.final_image_filename) is not None
    assert json.loads(outputs.results.read_text())["status"] == "completed"
    assert previews and previews[0].elapsed_s == 0
    assert cv2.imdecode(np.frombuffer(previews[0].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
    recording = cv2.VideoCapture(result.video_filename)
    try:
        assert recording.get(cv2.CAP_PROP_FRAME_COUNT) == 18
    finally:
        recording.release()
    assert analyze_video(source, outputs=outputs).status == "error"  # Preserve completed outputs.


def test_live_session_can_stop_and_save_outputs(tmp_path, monkeypatch):
    import time

    class Camera:
        def __init__(self, camera_id):
            assert camera_id == 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 30

        def read(self):
            time.sleep(.005)
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def release(self):
            pass

    from threading import Event
    called = Event()

    def detect(frame):
        called.set()
        return [DropletDetection(droplet_number=1, x_px=40, y_px=50, radius_px=5, confidence=.9)]

    monkeypatch.setattr(video.cv2, "VideoCapture", Camera)
    monkeypatch.setattr(video, "FrameDetector", lambda *_: detect)
    session = start_test(0, outputs=OutputPaths.in_directory(tmp_path / "live"),
                         settings=VideoSettings(confirmation_hits=1, auto_fabric_area=False, require_new_appearance=False))
    try:
        assert called.wait(3)
    finally:
        result = session.stop_test(timeout=5)
    assert result.status == "stopped", result.errors
    assert len(result.leaks) == 1
    assert result.leaks[0].detection_time_s >= 0
    assert result.video_filename is not None


def test_sampled_replay_reports_source_time_and_reuses_saved_upload(tmp_path, monkeypatch):
    source = tmp_path / "uploaded.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((120, 160, 3), 15 * index, dtype=np.uint8))
    writer.release()

    sampled = []

    def detect(frame):
        sampled.append(round(float(frame.mean()) / 15))
        if len(sampled) == 1:
            return []
        return [DropletDetection(droplet_number=1, x_px=40, y_px=50,
                                 radius_px=5, confidence=.9)]

    monkeypatch.setattr(video, "FrameDetector", lambda *_: detect)
    outputs = OutputPaths.in_directory(tmp_path / "run")
    progress = []
    result = analyze_video(source, outputs=outputs, settings=VideoSettings(
        analysis_fps=4, confirmation_hits=2, auto_fabric_area=False, require_new_appearance=False,
    ), use_source_recording=True, on_progress=progress.append)

    assert result.status == "completed", result.errors
    assert sampled == [0, 3, 6, 9]
    assert result.leaks[0].detection_time_s == pytest.approx(.3)
    assert result.leaks[0].confirmation_time_s == pytest.approx(.6)
    assert result.video_filename == str(source.resolve())
    assert not outputs.video.exists()
    assert progress[0] == 0


def test_vision_replay_uses_opening_image_and_fabric_ellipse(tmp_path, monkeypatch):
    source = tmp_path / "vision.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for i in range(16):
        writer.write(np.full((120, 160, 3), 55 + 2 * i, dtype=np.uint8))
    writer.release()
    references = []

    def vision_service(path, payload):
        if path == "/api/show":
            return {"capabilities": ["vision"]}
        assert path == "/api/chat"
        opening, current = payload["messages"][0]["images"]
        assert opening != current
        references.append(opening)
        image = cv2.imdecode(np.frombuffer(base64.b64decode(current), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (80, 80)
        assert (image[0, 0] == 128).all()
        return {"message": {"content": json.dumps({"droplets": [
            {"bbox_2d": [500, 475, 600, 575], "confidence": .9},
            {"bbox_2d": [500, 475, 600, 575], "confidence": .85},
            {"bbox_2d": [0, 0, 40, 40], "confidence": .9},
        ]})}}

    monkeypatch.setattr(vision_language, "_ollama", vision_service)
    result = analyze_video(source, outputs=OutputPaths.in_directory(tmp_path / "run"),
                           settings=VideoSettings(detector_backend="vision", vision_interval_s=.3,
                                                  confirmation_hits=2, roi=(40, 20, 80, 80),
                                                  roi_shape="ellipse"))

    assert result.status == "completed", result.errors
    assert len(references) > 1 and len(set(references)) == 1
    assert len(result.leaks) == 1
    leak = result.leaks[0]
    assert (leak.x_px, leak.y_px) == (84, 62)
    assert leak.detection_time_s == pytest.approx(.3)
    assert leak.confirmation_time_s >= .6
    assert cv2.imread(leak.image_filename) is not None


def test_vision_still_image_rejects_invalid_boxes(monkeypatch):
    def vision_service(path, payload):
        assert path == "/api/chat"
        assert len(payload["messages"][0]["images"]) == 1
        return {"message": {"content": '{"droplets":[{"bbox_2d":[600,10,100,20],"confidence":0.9}]}'}}

    monkeypatch.setattr(vision_language, "_ollama", vision_service)
    with pytest.raises(RuntimeError, match="invalid droplet box"):
        VisionLanguageDetector()(np.zeros((80, 80, 3), dtype=np.uint8))
