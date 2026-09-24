"""Train the local one-image fabric-droplet detection model.

Usage:
    python scripts/train_detection.py --data datasets/droplets.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from droplet_detector.dataset import DATASET_ROOT, validate_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/droplets.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install training dependencies first: pip install -e ".[ml]"') from exc

    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")

    try:
        summary = validate_training_dataset()
    except ValueError as exc:
        raise SystemExit(f"Dataset is not ready for training: {exc}") from exc

    # Ultralytics caches labels. Remove only its generated cache so newly saved
    # rectangles are always read by this training run.
    for split in ("train", "val"):
        (DATASET_ROOT / "labels" / f"{split}.cache").unlink(missing_ok=True)

    print(
        "Dataset ready: "
        f"{summary['train']['images']} training images "
        f"({summary['train']['positive_images']} with droplets), "
        f"{summary['val']['images']} validation images "
        f"({summary['val']['positive_images']} with droplets)."
    )

    model = YOLO("yolo11n.pt")
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.image_size)


if __name__ == "__main__":
    main()
