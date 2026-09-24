"""Local storage helpers for one-image droplet model training data."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping


DATASET_ROOT = Path("datasets")
VALID_SPLITS = {"train", "val"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def safe_filename(filename: str) -> str:
    """Return a local image filename without path components or unsafe text."""
    source = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source).stem).strip("_") or "image"
    suffix = Path(source).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        suffix = ".jpg"
    return f"{stem}{suffix}"


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    index = 2
    while candidate.exists():
        candidate = directory / f"{Path(filename).stem}_{index}{Path(filename).suffix}"
        index += 1
    return candidate


def save_labelled_image(
    filename: str,
    image_bytes: bytes,
    split: str,
    rectangles: Iterable[Mapping[str, float]],
    canvas_width: int,
    canvas_height: int,
) -> tuple[Path, Path, int]:
    """Save one training image and its YOLO rectangle labels.

    Empty ``rectangles`` intentionally creates an empty label file: that is a
    valid no-water example and teaches the model to ignore cloth texture.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive")

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_path = _unique_path(image_dir, safe_filename(filename))
    image_path.write_bytes(image_bytes)
    label_path = label_dir / f"{image_path.stem}.txt"

    lines: list[str] = []
    for rectangle in rectangles:
        width = float(rectangle.get("width", 0)) * float(rectangle.get("scaleX", 1))
        height = float(rectangle.get("height", 0)) * float(rectangle.get("scaleY", 1))
        left = float(rectangle.get("left", 0))
        top = float(rectangle.get("top", 0))
        if width < 2 or height < 2:
            continue
        center_x = min(1.0, max(0.0, (left + width / 2) / canvas_width))
        center_y = min(1.0, max(0.0, (top + height / 2) / canvas_height))
        relative_width = min(1.0, max(0.0, width / canvas_width))
        relative_height = min(1.0, max(0.0, height / canvas_height))
        lines.append(
            f"0 {center_x:.6f} {center_y:.6f} {relative_width:.6f} {relative_height:.6f}"
        )

    label_path.write_text("\n".join(lines), encoding="utf-8")
    return image_path, label_path, len(lines)


def dataset_counts() -> dict[str, int]:
    """Count real image files available for each training split."""
    return {
        split: len(_image_paths(split))
        for split in sorted(VALID_SPLITS)
    }


def _image_paths(split: str) -> list[Path]:
    """Return image files only, excluding placeholder files such as .gitkeep."""
    directory = DATASET_ROOT / "images" / split
    if not directory.is_dir():
        return []
    return [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]


def training_dataset_summary() -> dict[str, dict[str, int]]:
    """Return image and positive-droplet-label counts for training checks."""
    summary: dict[str, dict[str, int]] = {}
    for split in sorted(VALID_SPLITS):
        images = _image_paths(split)
        label_directory = DATASET_ROOT / "labels" / split
        positive_labels = sum(
            1
            for image in images
            if (label_directory / f"{image.stem}.txt").is_file()
            and (label_directory / f"{image.stem}.txt").read_text(encoding="utf-8").strip()
        )
        summary[split] = {"images": len(images), "positive_images": positive_labels}
    return summary


def validate_training_dataset() -> dict[str, dict[str, int]]:
    """Check that both splits contain useful examples before model training."""
    summary = training_dataset_summary()
    missing_splits = [split for split, values in summary.items() if values["images"] == 0]
    if missing_splits:
        names = " and ".join("validation" if split == "val" else "training" for split in missing_splits)
        raise ValueError(
            f"No real image was saved in the {names} set. Open the Label training photos "
            "tab, select that set, and save at least one image."
        )

    no_droplets = [split for split, values in summary.items() if values["positive_images"] == 0]
    if no_droplets:
        names = " and ".join("validation" if split == "val" else "training" for split in no_droplets)
        raise ValueError(
            f"The {names} set has no labelled droplets. Upload a photo containing water droplets, "
            "draw a rectangle around each droplet, and save it again. Empty rectangles are only "
            "for genuine no-water photos."
        )
    return summary
