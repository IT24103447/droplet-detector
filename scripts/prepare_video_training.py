"""Extract an unlabelled review set; never infer droplet boxes from folder names.

All recordings from a recording date stay together. Exact duplicate video
contents also stay together, including duplicates with different filenames.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
from PIL import Image, ImageDraw


def recording_group(path: Path) -> str:
    match = re.search(r"video-(\d{6})-", path.name)
    return match[1] if match else path.stem


def assign_splits(records: list[dict]) -> None:
    parents = {r["group"]: r["group"] for r in records}

    def find(group):
        while parents[group] != group:
            group = parents[group]
        return group

    seen = {}
    for record in records:
        group = record["group"]
        if record["sha256"] in seen:
            parents[find(group)] = find(seen[record["sha256"]])
        seen[record["sha256"]] = group
    groups = sorted({find(r["group"]) for r in records})
    if len(groups) < 3:
        raise ValueError("Need at least three independent recording groups.")
    holdout = max(1, round(len(groups) * .15))
    rng = random.Random(42)
    for _ in range(1000):
        rng.shuffle(groups)
        mapping = {g: "test" if i < holdout else "val" if i < holdout * 2 else "train"
                   for i, g in enumerate(groups)}
        for record in records:
            record["split"] = mapping[find(record["group"])]
        if all(len({r["source_category"] for r in records if r["split"] == split}) == 2
               for split in ("train", "val", "test")):
            return
    raise ValueError("Cannot place both video categories in each independent split.")


def inventory(source: Path) -> list[dict]:
    records = []
    for folder in ("Visible leaks", "No visible leaks in sampled frames"):
        for path in sorted((source / folder).glob("*.mp4")):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append(dict(source=str(path.resolve()), source_category=folder,
                                group=recording_group(path), sha256=digest))
    if not records:
        raise ValueError("No MP4 recordings found in the two expected folders.")
    assign_splits(records)
    return records


def extract(record: dict, output: Path) -> list[dict]:
    capture = cv2.VideoCapture(record["source"], cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 1])
    frames = []
    previews = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count < 1:
            raise ValueError(f"Unreadable video: {record['source']}")
        for fraction in (.05, .4, .75, .98):
            index = round((count - 1) * fraction)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {record['source']}")
            name = f"{record['sha256'][:20]}_{index:08d}.jpg"
            path = output / "images" / record["split"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not save {path}")
            frames.append(dict(**record, image=str(path.resolve()), frame_index=index,
                               time_s=round(index / fps, 3), width=frame.shape[1],
                               height=frame.shape[0], review_status="unreviewed"))
            preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            preview.thumbnail((400, 300))
            previews.append((preview, index / fps))
    finally:
        capture.release()
    sheet = Image.new("RGB", (1600, 340), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 3), f"{Path(record['source']).name} | {record['split']} | UNREVIEWED", fill="black")
    for i, (preview, seconds) in enumerate(previews):
        sheet.paste(preview, (i * 400, 22))
        draw.text((i * 400 + 5, 322), f"{seconds:.2f} s", fill="black")
    sheet.save(output / "review_sheets" / f"{record['sha256'][:20]}.jpg")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve earlier review work.")
    records = inventory(args.source)
    args.output.mkdir(parents=True)
    (args.output / "review_sheets").mkdir()
    (args.output / "videos.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    unique = {r["sha256"]: r for r in records}
    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, batch in enumerate(pool.map(lambda r: extract(r, args.output), unique.values()), 1):
            frames.extend(batch)
            if i % 20 == 0:
                print(f"Prepared {i}/{len(unique)} recordings", flush=True)
    (args.output / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
    summary = dict(videos=len(records), unique_videos=len(unique), frames=len(frames),
                   splits={s: dict(videos=sum(r["split"] == s for r in unique.values()),
                                   frames=sum(r["split"] == s for r in frames))
                           for s in ("train", "val", "test")},
                   status="awaiting_review_and_bounding_boxes", trained=False)
    for split in ("train", "val", "test"):
        with ZipFile(args.output / f"cvat_{split}_images.zip", "w", ZIP_DEFLATED) as archive:
            for path in sorted((args.output / "images" / split).glob("*.jpg")):
                archive.write(path, path.name)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "REVIEW_INSTRUCTIONS.txt").write_text(
        "These frames are UNLABELLED. Folder categories are video-level hints only.\n"
        "No empty label files or model-generated boxes have been supplied.\n"
        "Import each cvat_*_images.zip as a separate CVAT task. Label water_droplet\n"
        "with tight boxes around EVERY visible individual droplet. Review frames\n"
        "from both folders; no visible leaks in sampled frames is not proof of dry video.\n"
        "Explicitly review and accept genuinely droplet-free frames as negatives.\n"
        "Skip ambiguous frames, pooled water without individual droplets, and unreadable views.\n"
        "Export Ultralytics YOLO Detection labels, keeping the supplied splits.\n"
        "Train using train; tune using val; keep test sealed until final evaluation.\n"
        "Recording dates and exact video duplicates stay in the same split.\n"
        "Fabric identity is unknown: verify independent specimen runs before evaluation.\n"
        "Sparse sampling may miss leak onset; add reviewed frames around first visibility.\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
