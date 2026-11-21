"""Save manually labelled frames in the fabric training dataset."""
import math
import re
from pathlib import Path

import cv2
import numpy as np


def video_info(path: Path) -> tuple[int, float]:
    """Read the frame count and nominal frame rate for timeline navigation."""
    capture = cv2.VideoCapture(str(path))
    try:
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(count) or count < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Could not read the recording's timeline.")
        return int(count), fps
    finally:
        capture.release()


def read_frame(path: Path, index: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("This recording could not be opened.")
        if index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {index + 1}.")
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        fps = capture.get(cv2.CAP_PROP_FPS)
        if index and (not math.isfinite(timestamp) or timestamp <= 0):
            timestamp = index / fps if math.isfinite(fps) and fps > 0 else 0.0
        return frame, timestamp
    finally:
        capture.release()


def label_path(root: Path, split: str, clip_id: str, index: int) -> Path:
    if split not in ("train", "val") or not re.fullmatch(r"[0-9a-f]{64}", clip_id) or index < 0:
        raise ValueError("Invalid annotation location.")
    return root / "labels" / split / f"{clip_id}_{index:08d}.txt"


def load_boxes(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, w, h = map(float, line.split())
        boxes.append([cx - w / 2, cy - h / 2, w, h])
    return boxes


def save_frame(root: Path, split: str, clip_id: str, index: int, frame: np.ndarray,
               boxes: list[list[float]], *, dry: bool) -> None:
    path = label_path(root, split, clip_id, index)
    other = "val" if split == "train" else "train"
    if any((root / "labels" / other).glob(f"{clip_id}_*.txt")):
        raise ValueError("This recording is already in the other split. Use a different recording.")
    if dry == bool(boxes):
        raise ValueError("Draw a box around each droplet or explicitly mark the frame as dry.")
    if len(boxes) > 1000:
        raise ValueError("Too many boxes on one frame.")
    height, width = frame.shape[:2]
    lines = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid droplet box.")
        try:
            x, y, w, h = map(float, box)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid droplet box.") from exc
        if (not all(math.isfinite(v) for v in (x, y, w, h))
                or min(x, y) < 0 or x + w > 1.000001 or y + h > 1.000001
                or w * width < 2 or h * height < 2):
            raise ValueError("Keep each box inside the image and at least two pixels wide and tall.")
        lines.append(f"0 {x + w / 2:.8f} {y + h / 2:.8f} {w:.8f} {h:.8f}")
    ok, data = cv2.imencode(".png", frame)
    if not ok:
        raise OSError("Could not encode this video frame.")
    image_path = root / "images" / split / f"{path.stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(data.tobytes())
    path.write_text("\n".join(lines), encoding="utf-8")
