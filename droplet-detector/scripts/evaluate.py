#!/usr/bin/env python3
"""Evaluate the Step 1 pipeline against CVAT-exported ground truth.

Compares detections to labelled ground truth per fabric and writes a
markdown report to docs/step1_report.md with precision, recall,
false-positive, and miss counts.

Usage:
    python scripts/evaluate.py --data-root data --sensitivity 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from droplet_detector.config import DropletDetectorConfig
from droplet_detector.pipeline import detect_droplets_in_image
from droplet_detector.models import FabricAccuracyReport


def load_ground_truth(path: Path) -> dict[str, list[dict]]:
    """Load ground-truth annotations from a simple JSON file.

    Expected schema::

        {
            "image_001.jpg": [{"x": 120.0, "y": 88.0, "radius": 9.0}, ...],
            "image_002.jpg": [...]
        }

    TODO: Write a converter from CVAT's actual export format (CVAT XML 1.1
    or COCO JSON) to this schema once the export format is confirmed.
    """
    return json.loads(path.read_text())


def match_detections_to_ground_truth(
    detections: list,
    ground_truth_points: list[dict],
    match_dist_px: float = 15.0,
) -> tuple[int, int, int]:
    """Match detections to ground-truth points by proximity.

    A detection counts as a true positive if its centre is within
    ``match_dist_px`` pixels of an unmatched ground-truth point.

    Returns
    -------
    tuple[int, int, int]
        (true_positives, false_positives, missed)
    """
    tp = 0
    matched_gt: set[int] = set()
    for det in detections:
        for i, gt in enumerate(ground_truth_points):
            if i in matched_gt:
                continue
            dist = ((det.x_px - gt["x"]) ** 2 + (det.y_px - gt["y"]) ** 2) ** 0.5
            if dist <= match_dist_px:
                tp += 1
                matched_gt.add(i)
                break
    fp = len(detections) - tp
    missed = len(ground_truth_points) - tp
    return tp, fp, missed


def evaluate_fabric(
    fabric_dir: Path,
    dry_reference: Path,
    ground_truth_path: Path,
    config: DropletDetectorConfig,
) -> FabricAccuracyReport:
    """Evaluate detection accuracy for a single fabric type."""
    ground_truth = load_ground_truth(ground_truth_path)
    total_tp = total_fp = total_missed = total_gt = 0
    for image_path in sorted(fabric_dir.glob("*.jpg")):
        gt_points = ground_truth.get(image_path.name, [])
        detections = detect_droplets_in_image(
            str(image_path), str(dry_reference), config
        )
        tp, fp, missed = match_detections_to_ground_truth(detections, gt_points)
        total_tp += tp
        total_fp += fp
        total_missed += missed
        total_gt += len(gt_points)
    return FabricAccuracyReport(
        fabric_name=fabric_dir.name,
        total_labelled_droplets=total_gt,
        true_positives=total_tp,
        false_positives=total_fp,
        missed=total_missed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Step 1 pipeline against ground-truth annotations."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    fabrics = ["cotton", "silk", "two_layer", "three_layer", "nylon"]
    lines = [
        "# Step 1 — Accuracy Report",
        "",
        f"Sensitivity threshold: {args.sensitivity}",
        "",
        "| Fabric | Labelled | TP | FP | Missed | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
    ]

    evaluated = 0
    for fabric in fabrics:
        fabric_dir = data_root / "raw" / fabric
        dry_ref = data_root / "dry_reference" / f"{fabric}_dry.jpg"
        gt_path = data_root / "annotations" / f"{fabric}.json"
        if not (fabric_dir.exists() and dry_ref.exists() and gt_path.exists()):
            print(f"Skipping {fabric} — missing data/reference/annotations")
            continue
        config = DropletDetectorConfig(
            video_save_path=data_root / "results" / "video",
            image_save_path=data_root / "results" / "images",
            results_save_path=data_root / "results" / "reports",
            sensitivity=args.sensitivity,
        )
        report = evaluate_fabric(fabric_dir, dry_ref, gt_path, config)
        lines.append(
            f"| {report.fabric_name} | {report.total_labelled_droplets} "
            f"| {report.true_positives} | {report.false_positives} "
            f"| {report.missed} | {report.precision:.2f} | {report.recall:.2f} |"
        )
        evaluated += 1
        print(
            f"{report.fabric_name}: P={report.precision:.2f} R={report.recall:.2f} "
            f"TP={report.true_positives} FP={report.false_positives} Missed={report.missed}"
        )

    if evaluated == 0:
        lines.append("| *(no data available yet)* | — | — | — | — | — | — |")
        print("\nNo fabric data found. Report generated with placeholder row.")

    out_path = Path("docs/step1_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
