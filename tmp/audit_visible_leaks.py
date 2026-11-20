"""Audit current model on the visible-leak archive without writing video copies."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from droplet_detector.leak_tracking import LeakTracker
from droplet_detector.models import DropletDetection
from droplet_detector.single_image import FrameDetector


ROOT = Path(r"D:\TransitionX\test_records\test_records\test_records\Visible leaks")
REVIEW = Path("tmp/leak_archive_review")
OUTPUT = REVIEW / "current-model-visible"
THRESHOLD = 0.25


def inside(x, y, frame_width, frame_height):
    return ((x / frame_width - .51) / .221) ** 2 + ((y / frame_height - .524) / .403) ** 2 <= 1


def read_clip(entry):
    capture = cv2.VideoCapture(str(ROOT / entry["file"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    opening, samples = [], []
    try:
        if not math.isfinite(fps) or fps <= 0 or count < 1:
            raise OSError(f"Cannot read {entry['file']}")
        for _ in range(min(8, count)):
            ok, frame = capture.read()
            if not ok:
                break
            opening.append(frame)
        height, width = opening[0].shape[:2]
        left, right = round(width * .267), round(width * .753)
        top, bottom = round(height * .087), round(height * .961)
        sample_count = min(43, max(12, math.ceil(count / fps / 4)))
        frame_indices = set(np.linspace(0, count - 1, sample_count, dtype=int))
        frame_indices.update(int((count - 1) * f) for f in (.01, .05, .18, .32, .48, .64, .8, .9, .97, .995))
        if entry["index"] == 80:
            frame_indices.update(round(t * fps) for t in (3, 5, 7, 13, 20, 38, 60, 73, 82))
        for index in sorted(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                samples.append((index / fps, frame[top:bottom, left:right].copy()))
        return entry, fps, width, height, left, top, opening, samples
    finally:
        capture.release()


def sample_predictions(model, frames):
    for start in range(0, len(frames), 8):
        yield from model.predict(frames[start:start + 8], conf=THRESHOLD, verbose=False, device="cpu")


def detections(prediction, offset_x, offset_y):
    for coords, score in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist()):
        x1, y1, x2, y2 = coords
        yield dict(x=(x1 + x2) / 2 + offset_x, y=(y1 + y2) / 2 + offset_y,
                   confidence=round(score, 3), bounds=[round(x1), round(y1), round(x2), round(y2)])


def save_review(entry, samples, boxes):
    width, height = 194, 183
    columns = 7
    rows = math.ceil(len(samples) / columns)
    sheet = Image.new("RGB", (width * columns, 22 + height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), f"{entry['index']:03d} | {entry['file']}", fill="black")
    for number, ((seconds, crop), hits) in enumerate(zip(samples, boxes)):
        x = number % columns * width
        y = 22 + number // columns * height
        preview = cv2.resize(crop, (width - 2, height - 18), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = preview.shape[1] / crop.shape[1], preview.shape[0] / crop.shape[0]
        for hit in hits:
            x1, y1, x2, y2 = hit["bounds"]
            color = (40, 215, 255) if hit["in_fabric"] else (30, 80, 255)
            cv2.rectangle(preview, (round(x1 * scale_x), round(y1 * scale_y)),
                          (round(x2 * scale_x), round(y2 * scale_y)), color, 1)
        sheet.paste(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), (x, y))
        draw.text((x + 3, y + height - 15), f"{seconds:.1f}s  •  {sum(d['in_fabric'] for d in hits)}",
                  fill="black")
    sheet.save(OUTPUT / f"{entry['index']:03d}.jpg", quality=93)


def audit(model, entry, fps, width, height, left, top, opening, samples):
    tracker = LeakTracker(min_hits=3, match_px=10, max_gap_s=.5)
    opening_events = []
    for index, prediction in enumerate(sample_predictions(model, opening)):
        candidates = list(detections(prediction, 0, 0))
        tracked = tracker.update([
            DropletDetection(droplet_number=n, x_px=d["x"], y_px=d["y"],
                             confidence=d["confidence"], radius_px=max(d["bounds"][2] - d["bounds"][0],
                                                                       d["bounds"][3] - d["bounds"][1]) / 2)
            for n, d in enumerate(candidates, 1)
        ], index / fps)
        for track in tracked[:3 - len(opening_events)]:
            d = track.detection
            opening_events.append(dict(seconds=round(track.first_seen, 3),
                                       x=round(d.x_px, 1), y=round(d.y_px, 1),
                                       confidence=d.confidence, in_fabric=inside(d.x_px, d.y_px, width, height)))
    boxes = []
    for prediction in sample_predictions(model, [frame for _, frame in samples]):
        hits = list(detections(prediction, left, top))
        for hit in hits:
            hit["in_fabric"] = inside(hit["x"], hit["y"], width, height)
        boxes.append(hits)
    save_review(entry, samples, boxes)
    return {**entry, "opening_events": opening_events, "frame_size": [width, height],
            "sampled_frames": [{"seconds": round(t, 3), "detections": d}
                               for (t, _), d in zip(samples, boxes)]}


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    journal_path = OUTPUT / "predictions.jsonl"
    done = {json.loads(line)["file"] for line in journal_path.read_text().splitlines()} if journal_path.exists() else set()
    files = {path.name for path in ROOT.glob("*.mp4")}
    entries = [r for r in json.loads((REVIEW / "move-plan.json").read_text()) if r["file"] in files]
    if {r["file"] for r in entries} != files:
        raise ValueError("Visible-leak inventory does not match the folder")
    model = FrameDetector("models/droplet-seg.pt", THRESHOLD).model
    start = monotonic()
    work = [r for r in entries if r["file"] not in done]
    with ThreadPoolExecutor(max_workers=3) as pool, journal_path.open("a", encoding="utf-8") as journal:
        for offset in range(0, len(work), 3):
            for item in pool.map(read_clip, work[offset:offset + 3]):
                result = audit(model, *item)
                journal.write(json.dumps(result) + "\n")
                journal.flush()
                done.add(result["file"])
                print(f"{len(done)}/{len(files)} | clip {result['index']:03d} | "
                      f"{len(result['sampled_frames'])} samples | {monotonic() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
