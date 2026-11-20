"""Exercise the video contract using repeatable detections and real video files."""
import json

import cv2
import numpy as np

from droplet_detector import DropletDetection, OutputPaths, VideoSettings, analyze_video, start_test
from droplet_detector import video


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
    result = analyze_video(source, outputs=outputs, settings=VideoSettings(roi=(8, 10, 130, 90)),
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
                         settings=VideoSettings(confirmation_hits=1))
    try:
        assert called.wait(3)
    finally:
        result = session.stop_test(timeout=5)
    assert result.status == "stopped", result.errors
    assert len(result.leaks) == 1
    assert result.leaks[0].detection_time_s >= 0
    assert result.video_filename is not None
