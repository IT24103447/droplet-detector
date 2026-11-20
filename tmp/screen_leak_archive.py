"""Read-only screening of hydrotester MP4s. Outputs compact review evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

SOURCE = Path(r"D:\TransitionX\test_records\test_records\test_records")
OUTPUT = Path("tmp/leak_archive_review")
FRACTIONS = (.01, .05, .18, .32, .48, .64, .8, .9, .97, .995)


def read_samples(item):
    index, path = item
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if count < 1 or not math.isfinite(fps) or fps <= 0:
            return index, path, None, []
        for number in sorted({int(f * (count - 1)) for f in FRACTIONS}):
            if number:
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                # These cameras show the circular specimen in the centre.
                roi = frame[round(.08 * h):round(.95 * h), round(.24 * w):round(.76 * w)]
                frames.append((round(number / fps, 2), cv2.resize(roi, (544, 512))))
        return index, path, round(count / fps, 2), frames
    finally:
        cap.release()


def inspect(index, path, duration, samples, model):
    entry = {"index": index, "file": path.name, "duration_s": duration, "samples": len(samples),
             "detections": []}
    if not samples:
        return entry
    images = [image for _, image in samples]
    results = model.predict(images, imgsz=448, batch=10, conf=.13, max_det=12, verbose=False,
                            device="cpu")
    for (seconds, _), prediction in zip(samples, results):
        candidates = []
        for box, score in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist()):
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # Reject most detections on the illuminated clamp.
            if ((x - 272) / 252) ** 2 + ((y - 247) / 244) ** 2 < 1:
                candidates.append({"box": [round(v) for v in box], "confidence": round(score, 3)})
        entry["detections"].append({"seconds": seconds, "boxes": candidates})
    selected = list(dict.fromkeys([0, len(images) // 3, 2 * len(images) // 3, len(images) - 1]))
    best = sorted(range(len(images)),
                  key=lambda i: max((v["confidence"] for v in entry["detections"][i]["boxes"]), default=0),
                  reverse=True)
    tile = Image.new("RGB", (1056, 196), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((2, 0), f"{index:03d} | {path.name}", fill="black")
    for col, i in enumerate(selected + best[:2]):
        img = images[i]
        if col >= len(selected):
            img = img.copy()
            for candidate in entry["detections"][i]["boxes"]:
                x1, y1, x2, y2 = candidate["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (30, 210, 245), 2)
        offset = col * 176
        tile.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img, (176, 165)), cv2.COLOR_BGR2RGB)),
                   (offset, 15))
        draw.text((offset + 2, 181), f"{samples[i][0]:.1f}s", fill="black")
    tile.save(OUTPUT / f"{index:03d}.jpg", quality=88)
    return entry


def main():
    import torch
    from ultralytics import YOLO

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = YOLO("models/droplet-seg.pt")
    start = monotonic()
    videos = sorted(SOURCE.glob("*.mp4"))
    existing = OUTPUT / "screening.jsonl"
    done = {json.loads(line)["file"] for line in existing.read_text().splitlines()} if existing.exists() else set()
    with ThreadPoolExecutor(max_workers=5) as pool, existing.open("a", encoding="utf-8") as journal:
        work = [(i, path) for i, path in enumerate(videos) if path.name not in done]
        for offset in range(0, len(work), 10):
            for item in pool.map(read_samples, work[offset:offset + 10]):
                entry = inspect(*item, model)
                journal.write(json.dumps(entry) + "\n")
                journal.flush()
            print(f"Screened {len(done) + min(len(work), offset + 10)}/{len(videos)} in {monotonic()-start:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
