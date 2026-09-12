#!/usr/bin/env python3
"""Run the Step 1 droplet detector on a folder of fabric photos.

Usage:
    python scripts/run_prototype.py \
        --fabric-dir data/raw/cotton \
        --dry-reference data/dry_reference/cotton_dry.jpg \
        --sensitivity 0.5 \
        --results-path data/results/reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from droplet_detector.config import DropletDetectorConfig
from droplet_detector.pipeline import detect_droplets_in_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Step 1 droplet detector on a folder of fabric photos."
    )
    parser.add_argument(
        "--fabric-dir",
        required=True,
        help="Folder containing test photos for one fabric",
    )
    parser.add_argument(
        "--dry-reference",
        required=True,
        help="Path to the dry reference photo for this fabric",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0–1.0, default: 0.5)",
    )
    parser.add_argument(
        "--results-path",
        default="data/results/reports",
        help="Directory to write JSON results to",
    )
    args = parser.parse_args()

    fabric_dir = Path(args.fabric_dir)
    if not fabric_dir.is_dir():
        print(f"Error: fabric directory not found: {fabric_dir}", file=sys.stderr)
        sys.exit(1)

    dry_ref = Path(args.dry_reference)
    if not dry_ref.is_file():
        print(f"Error: dry reference image not found: {dry_ref}", file=sys.stderr)
        sys.exit(1)

    config = DropletDetectorConfig(
        camera_id=0,
        video_save_path=Path("data/results/video"),
        image_save_path=Path("data/results/images"),
        results_save_path=Path(args.results_path),
        sensitivity=args.sensitivity,
    )

    image_paths = sorted(
        p for p in fabric_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    )
    if not image_paths:
        print(f"No image files found in {fabric_dir}", file=sys.stderr)
        sys.exit(1)

    all_results = {}
    total_detections = 0
    for image_path in image_paths:
        try:
            detections = detect_droplets_in_image(
                str(image_path), str(dry_ref), config
            )
            all_results[image_path.name] = [
                d.model_dump(mode="json") for d in detections
            ]
            total_detections += len(detections)
            print(f"  {image_path.name}: {len(detections)} droplet(s) detected")
        except Exception as e:
            print(f"  {image_path.name}: ERROR — {e}", file=sys.stderr)
            all_results[image_path.name] = {"error": str(e)}

    out_file = config.results_save_path / f"{fabric_dir.name}_detections.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nTotal: {total_detections} detection(s) across {len(image_paths)} image(s)")
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
