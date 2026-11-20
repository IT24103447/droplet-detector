"""Inspect complete recordings for spreading-water warnings, without running YOLO.

Folder names are weak video-level labels. This produces observations and review
snapshots, not an accuracy score or expert-verified ground truth.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2

from droplet_detector.fabric_area import find_fabric_area
from droplet_detector.surface_water import SurfaceWaterDetector
from droplet_detector.video import SurfaceWaterEvent, _annotate, _save_image


def evaluate(source: Path, output: Path) -> dict:
    started = time.perf_counter()
    result = {"source": str(source.resolve()), "event": None, "sampled_frames": 0,
              "rim_pauses": 0, "error": None}
    capture = cv2.VideoCapture(str(source))
    try:
        ok, opening = capture.read()
        if not ok:
            raise ValueError("Cannot read the opening frame")
        roi = find_fabric_area(opening)
        detector = SurfaceWaterDetector(opening, roi)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Video frame rate unavailable")
        frame_index = 1
        while capture.grab():
            elapsed = frame_index/fps
            frame_index += 1
            if elapsed < detector.next_sample:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError("Cannot decode a sampled frame")
            result["sampled_frames"] += 1
            try:
                current_roi = find_fabric_area(frame)
            except ValueError:
                result["rim_pauses"] += 1
                detector.next_sample = elapsed+detector.interval_s
                continue
            event = detector.observe(frame, elapsed, current_roi)
            if event is not None:
                result["event"] = asdict(event)
                snapshot = output / f"{source.stem}_water.jpg"
                _save_image(snapshot, _annotate(frame, [], roi, "ellipse",
                                               SurfaceWaterEvent(**asdict(event)), True))
                result["snapshot"] = str(snapshot.resolve())
                break
    except (ValueError, cv2.error) as exc:
        result["error"] = str(exc)
    finally:
        capture.release()
    result["wall_s"] = round(time.perf_counter()-started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in args.videos:
        row = evaluate(source, args.output_dir)
        rows.append(row)
        (args.output_dir / "observations.json").write_text(json.dumps({
            "note": "Visual heuristic, dry opening assumed. Folder labels are not verified frame labels. "
                    "No warning does not prove absence of water. Processing stops after the first warning.",
            "sample_interval_s": SurfaceWaterDetector.interval_s, "videos": rows,
        }, indent=2), encoding="utf-8")
        event = row["event"]
        print(f"{source.name}: " + (f"water {event['detection_time_s']:.2f}s" if event else "no water warning")
              + f"; {row['wall_s']:.1f}s; error={row['error']}", flush=True)
    return int(any(row["error"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
