"""Fine-tune a separate candidate using explicitly reviewed frame annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw


def build_dataset(manifest: Path, destination: Path, specimen_crop: bool = False) -> Path:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if destination.exists():
        raise ValueError("Use a new dataset directory to preserve previous annotations.")
    groups = {}
    hashes = {}
    positive_counts = {"train": 0, "val": 0}
    for record in records:
        split = record["split"]
        if split not in positive_counts or record["review_status"] != "ai_visually_reviewed_provisional":
            raise ValueError("Every training/validation frame must have an explicit review.")
        for mapping, key in ((groups, record["group"]), (hashes, record["sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Recording dates or video contents cross splits.")
            mapping[key] = split
        image_path = Path(record["image"])
        with Image.open(image_path) as image:
            width, height = image.size
        x0, y0, x1, y1 = record["crop"]
        labels = []
        for box in record["boxes_preview_xyxy"]:
            if len(box) != 4 or any(not math.isfinite(v) for v in box):
                raise ValueError("Invalid reviewed box.")
            left, top, right, bottom = box
            if not (0 <= left < right <= 500 and 0 <= top < bottom <= 500):
                raise ValueError("Box must fit the reviewed 500-pixel specimen preview.")
            left, right = (x0 + left / 500 * (x1-x0)), (x0 + right / 500 * (x1-x0))
            top, bottom = (y0 + top / 500 * (y1-y0)), (y0 + bottom / 500 * (y1-y0))
            if specimen_crop:
                left, right, top, bottom = left-x0, right-x0, top-y0, bottom-y0
                width, height = x1-x0, y1-y0
            labels.append(f"0 {(left+right)/2/width:.8f} {(top+bottom)/2/height:.8f} "
                          f"{(right-left)/width:.8f} {(bottom-top)/height:.8f}")
        positive_counts[split] += bool(labels)
        for kind in ("images", "labels", "review"):
            (destination / kind / split).mkdir(parents=True, exist_ok=True)
        if specimen_crop:
            with Image.open(image_path) as image:
                image.crop(tuple(record["crop"])).save(destination / "images" / split / image_path.name, quality=95)
        else:
            shutil.copyfile(image_path, destination / "images" / split / image_path.name)
        (destination / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(labels))
        with Image.open(image_path) as image:
            preview = image.crop(tuple(record["crop"])).resize((500,500))
        draw = ImageDraw.Draw(preview)
        for box in record["boxes_preview_xyxy"]:
            draw.rectangle(tuple(box), outline="lime", width=2)
        preview.save(destination / "review" / split / image_path.name)
    if not all(positive_counts.values()):
        raise ValueError("Training and validation both need reviewed positive frames.")
    config = destination / "dataset.yaml"
    config.write_text(json.dumps(dict(path=str(destination.resolve()), train="images/train",
                                     val="images/val", names=["water_droplet"])))
    shutil.copyfile(manifest, destination / "annotation_provenance.json")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("models/droplet-seg.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--specimen-crop", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(4)
    source_hash = hashlib.sha256(args.base_model.read_bytes()).hexdigest()
    config = build_dataset(args.annotations, args.output / "dataset", args.specimen_crop)
    records = json.loads(args.annotations.read_text())
    run_info = dict(base_model=str(args.base_model.resolve()), base_sha256=source_hash,
                    annotations="AI visually reviewed, provisional; not expert ground truth",
                    splits={s: dict(frames=sum(r["split"]==s for r in records),
                                    positive_frames=sum(r["split"]==s and bool(r["boxes_preview_xyxy"]) for r in records),
                                    videos=len({r["sha256"] for r in records if r["split"]==s}))
                            for s in ("train", "val")},
                    specimen_crop=args.specimen_crop,
                    normalized_crop=[.267,.087,.753,.961] if args.specimen_crop else None,
                    production_model_replaced=False)
    (args.output / "run_info.json").write_text(json.dumps(run_info, indent=2))
    model = YOLO(str(args.base_model))
    model.train(data=str(config.resolve()), epochs=args.epochs, imgsz=640,
                batch=4, device=0 if torch.cuda.is_available() else "cpu", workers=0,
                project=str(args.output.resolve()), name="training", exist_ok=False,
                freeze=10, optimizer="AdamW", lr0=.001, patience=args.patience, seed=42,
                mosaic=0, mixup=0, degrees=5, translate=.05, scale=.15,
                fliplr=.5, plots=True, amp=False)
    if hashlib.sha256(args.base_model.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Base weights unexpectedly changed.")
    print("Candidate training finished; original weights preserved.")


if __name__ == "__main__":
    main()
