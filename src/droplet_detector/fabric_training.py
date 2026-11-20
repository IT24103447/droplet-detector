"""Import CVAT YOLO boxes and train a selectable fabric-specific model."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2

from .dataset import IMAGE_SUFFIXES


def import_dataset(archive_path: Path, root: Path) -> dict[str, int]:
    """Copy only matched image/label files into a fresh, controlled layout."""
    counts = {}
    with ZipFile(archive_path) as archive:
        items = archive.infolist()
        if len(items) > 10_000 or sum(item.file_size for item in items) > 1_000_000_000:
            raise ValueError("The expanded dataset exceeds the import limit.")
        files = {}
        for item in items:
            path = PurePosixPath(item.filename)
            if item.is_dir() or len(path.parts) < 3:
                continue
            kind, split, filename = path.parts[-3:]
            if kind not in ("images", "labels") or split not in ("train", "val"):
                continue
            suffix = path.suffix.lower()
            if (kind == "images" and suffix not in IMAGE_SUFFIXES) or (kind == "labels" and suffix != ".txt"):
                continue
            if len(path.stem) == 0 or "\\" in filename or ":" in filename or ".." in path.parts:
                raise ValueError("The archive contains an invalid dataset filename.")
            key = (kind, split, path.stem.casefold())
            if key in files:
                raise ValueError(f"Duplicate dataset entry: {filename}")
            files[key] = item

        train_hashes = set()
        for split in ("train", "val"):
            images = [(stem, item) for (kind, group, stem), item in files.items()
                      if kind == "images" and group == split]
            positives = 0
            for index, (stem, item) in enumerate(images, 1):
                label_item = files.get(("labels", split, stem))
                if label_item and label_item.file_size > 100_000:
                    raise ValueError(f"Label file too large: {label_item.filename}")
                label = archive.read(label_item).decode("utf-8-sig").strip() if label_item else ""
                for line in label.splitlines():
                    parts = line.split()
                    if len(parts) != 5 or parts[0] != "0":
                        raise ValueError(f"{stem}: use only class 0 water_droplet bounding boxes.")
                    coords = [float(value) for value in parts[1:]]
                    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                        raise ValueError(f"{stem}: invalid normalized bounding box.")
                positives += bool(label)
                suffix = PurePosixPath(item.filename).suffix.lower()
                if item.file_size > 25_000_000:
                    raise ValueError(f"Image too large: {item.filename}")
                blob = archive.read(item)
                fingerprint = hashlib.sha256(blob).digest()
                if split == "train":
                    train_hashes.add(fingerprint)
                elif fingerprint in train_hashes:
                    raise ValueError("The same image was found in training and validation.")
                image_dir, label_dir = root / "images" / split, root / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                path = image_dir / f"{index:05d}{suffix}"
                path.write_bytes(blob)
                if cv2.imread(str(path)) is None:
                    raise ValueError(f"Could not decode image: {item.filename}")
                (label_dir / f"{index:05d}.txt").write_text(label, encoding="utf-8")
            if not positives:
                raise ValueError(f"The {split} split needs at least one image with a droplet annotation.")
            if any(kind == "labels" and group == split and ("images", split, stem) not in files
                   for kind, group, stem in files):
                raise ValueError(f"The {split} split has labels without matching images.")
            counts[split] = len(images)
    return counts


def train_profile(job: Path, destination: Path, base_model: Path, epochs: int) -> Path:
    """Train within a job folder, then publish only the completed best weights."""
    from ultralytics import YOLO

    dataset = job / "dataset"
    import_dataset(job / "dataset.zip", dataset)
    config = job / "dataset.yaml"
    config.write_text(json.dumps({
        "path": str(dataset.resolve()), "train": "images/train", "val": "images/val",
        "names": ["water_droplet"],
    }), encoding="utf-8")
    model = YOLO(str(base_model))
    result = model.train(data=str(config), epochs=epochs, imgsz=640,
                         project=str(job.resolve()), name="training", workers=0, plots=False)
    weights = Path(result.save_dir) / "weights" / "best.pt"
    if not weights.is_file():
        raise OSError("Training did not produce best.pt.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(".pending")
    shutil.copyfile(weights, staged)
    staged.replace(destination)
    return destination


def train_annotated_profile(job: Path, destination: Path, base_model: Path,
                            epochs: int, annotations: Path) -> Path:
    """Use the same dataset checks and training path as an imported CVAT ZIP."""
    job.mkdir(parents=True, exist_ok=True)
    with ZipFile(job / "dataset.zip", "w") as archive:
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                for path in sorted((annotations / kind / split).glob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(annotations).as_posix())
    return train_profile(job, destination, base_model, epochs)
