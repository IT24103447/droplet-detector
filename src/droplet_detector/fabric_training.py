"""Import labelled frames and fine-tune a local droplet detector."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np

from .dataset import IMAGE_SUFFIXES
from .fabric_area import crop_fabric, find_fabric_area
from .single_image import FrameDetector
from .video import OutputPaths, VideoSettings, analyze_video


def _read_image(path: Path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path.name}")
    return image


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


def _count_matches(dataset: Path, weights: Path, split: str) -> dict[str, int]:
    detector = FrameDetector(weights, .25)
    counts = {"matched": 0, "labels": 0, "predictions": 0}
    for image in sorted((dataset / "images" / split).iterdir()):
        frame = _read_image(image)
        h, w = frame.shape[:2]
        label_file = dataset / "labels" / split / f"{image.stem}.txt"
        labels = [tuple(map(float, line.split()[1:])) for line in label_file.read_text().splitlines()]
        predictions = sorted(detector(frame), key=lambda prediction: prediction.confidence, reverse=True)
        counts["labels"] += len(labels)
        counts["predictions"] += len(predictions)
        for prediction in predictions:
            matches = []
            for index, (x, y, bw, bh) in enumerate(labels):
                distance = math.hypot((prediction.x_px / w - x) / bw, (prediction.y_px / h - y) / bh)
                radius = max(bw * w, bh * h) / 2
                if distance <= .5 and .5 * radius <= prediction.radius_px <= 2 * radius:
                    matches.append((distance, index))
            if matches:
                labels.pop(min(matches)[1])
                counts["matched"] += 1
    return counts


def _score(counts: dict[str, int]) -> float:
    return 2 * counts["matched"] / max(1, counts["labels"] + counts["predictions"])


def _publish_checkpoint(weights: Path, destination: Path, job: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.{job.name}.pending")
    try:
        shutil.copyfile(weights, staged)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination


def train_profile(job: Path, destination: Path, base_model: Path, epochs: int) -> Path:
    """Fine-tune locally and publish only when the saved model passes validation."""
    from ultralytics import YOLO

    dataset = job / "dataset"
    import_dataset(job / "dataset.zip", dataset)
    config = job / "dataset.yaml"
    config.write_text(json.dumps({
        "path": str(dataset.resolve()), "train": "images/train", "val": "images/val",
        "names": ["water_droplet"],
    }), encoding="utf-8")
    model = YOLO(str(base_model))
    result = model.train(data=str(config), epochs=epochs, imgsz=640, batch=4,
                         project=str(job.resolve()), name="training", workers=0, plots=False,
                         optimizer="AdamW", lr0=.0002, freeze=10, mosaic=0, scale=.15,
                         translate=.05, patience=8)
    weights = Path(result.save_dir) / "weights" / "best.pt"
    if not weights.is_file():
        raise OSError("Training did not produce best.pt.")
    trained = _count_matches(dataset, weights, "train")
    validated = _count_matches(dataset, weights, "val")
    base = _count_matches(dataset, base_model, "val")
    report = {"train": trained, "validation": validated, "previous_validation": base}
    (job / "training_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if (not trained["matched"] or 2 * validated["matched"] < validated["labels"]
            or _score(validated) < _score(base)):
        raise ValueError(
            f"The candidate found {validated['matched']} of {validated['labels']} labelled validation droplets. "
            "The current model is still active. Check the boxes and add clear frames showing each droplet."
        )
    return _publish_checkpoint(weights, destination, job)


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


def _crop_labels(text: str, shape, roi) -> str:
    height, width = shape[:2]
    left, top, w, h = roi
    remapped = []
    for line in text.splitlines():
        category, *coords = line.split()
        x, y, bw, bh = map(float, coords)
        x, y = (x * width - left) / w, (y * height - top) / h
        bw, bh = bw * width / w, bh * height / h
        radius = max(bw * w, bh * h) / 2
        if radius > .1 * min(w, h) or any(
            (2 * (x + dx * radius / w) - 1) ** 2 + (2 * (y + dy * radius / h) - 1) ** 2 > 1
            for dx in (-1, 1) for dy in (-1, 1)
        ):
            raise ValueError("A box crosses the fabric boundary or is too large for an individual droplet.")
        remapped.append(f"{category} {x:.8f} {y:.8f} {bw:.8f} {bh:.8f}")
    return "\n".join(remapped)


def _validate_recording(source: Path, job: Path, candidate: Path,
                        manual_roi: tuple[int, int, int, int] | None,
                        areas: list[tuple[tuple[int, int, int, int], str]],
                        analysis_fps: float) -> None:
    outputs = OutputPaths.in_directory(job / "replay-validation")
    settings = VideoSettings(model_path=candidate, roi=manual_roi, analysis_fps=analysis_fps)
    result = analyze_video(source, outputs=outputs, settings=settings, use_source_recording=True)
    targets = []
    for (left, top, w, h), lines in areas:
        for line in lines.splitlines():
            x, y, bw, bh = map(float, line.split()[1:])
            targets.append((left + x * w, top + y * h, max(10., max(bw * w, bh * h) / 2)))
    if result.status != "completed":
        reason = result.errors[0] if result.errors else result.status
        raise ValueError(f"The recording check could not finish. The current model is still active. {reason}")
    if not any(math.hypot(leak.x_px - x, leak.y_px - y) <= radius
               for leak in result.leaks for x, y, radius in targets):
        raise ValueError(
            "The candidate matched labelled frames but could not confirm a marked leak in the recording. "
            "The current model is still active. Use a video that starts dry and label the droplets near when they appear."
        )


def train_annotated_model(job: Path, destination: Path, base_model: Path,
                          epochs: int, annotations: Path, *, focus_clip: str | None = None,
                          manual_roi: tuple[int, int, int, int] | None = None,
                          analysis_fps: float = 6.) -> Path:
    """Prepare in-app frames in the same circular view used during video detection."""
    job.mkdir(parents=True, exist_ok=True)
    examples = {}
    skipped = {}
    for label in sorted((annotations / "labels" / "train").glob("*.txt")):
        clip, _ = label.stem.split("_", 1)
        image = annotations / "images" / "train" / f"{label.stem}.png"
        frame = _read_image(image)
        try:
            roi = manual_roi if manual_roi and clip == focus_clip else find_fabric_area(frame)
            crop_fabric(frame, roi)
            lines = _crop_labels(label.read_text(encoding="utf-8"), frame.shape, roi)
        except ValueError as exc:
            skipped[label.stem] = str(exc)
            continue
        examples[label] = (image, roi, lines)
    if skipped:
        (job / "skipped_frames.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    positive = [path for path, (_, _, lines) in examples.items() if lines]
    if len(positive) < 2:
        raise ValueError("Save droplets on at least two frames with a visible rim and boxes inside the fabric.")
    positive_clips = sorted({path.stem.split("_", 1)[0] for path in positive})
    validation_clip = focus_clip or positive_clips[-1]
    selected = [path for path in positive if path.stem.startswith(validation_clip + "_")]
    if focus_clip and len(selected) < 2:
        raise ValueError("Save at least two usable droplet frames from the selected recording.")
    # Keep some labelled material from the selected recording out of training.
    by_time = focus_clip is not None or len(positive_clips) == 1
    cutoff = selected[-max(1, math.ceil(len(selected) / 5))].stem if by_time else validation_clip
    with ZipFile(job / "dataset.zip", "x") as archive:
        for label, (image, roi, lines) in examples.items():
            clip = label.stem.split("_", 1)[0]
            validation = clip == validation_clip and label.stem >= cutoff
            split = "val" if validation else "train"
            ok, encoded = cv2.imencode(".png", crop_fabric(_read_image(image), roi))
            if not ok:
                raise OSError(f"Could not encode training frame: {image.name}")
            archive.writestr(f"images/{split}/{image.name}", encoded.tobytes())
            archive.writestr(f"labels/{split}/{label.name}", lines)
    checked_clip = validation_clip if by_time else None
    source = next((p for p in (annotations / "videos").iterdir()
                   if p.is_file() and p.stem == checked_clip), None) if (annotations / "videos").is_dir() else None
    if source is None:
        return train_profile(job, destination, base_model, epochs)
    candidate = train_profile(job, job / "candidate.pt", base_model, epochs)
    areas = [(roi, lines) for label, (_, roi, lines) in examples.items()
             if label.stem.startswith(checked_clip + "_")]
    _validate_recording(source, job, candidate, manual_roi, areas, analysis_fps)
    return _publish_checkpoint(candidate, destination, job)
