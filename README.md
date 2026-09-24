# Droplet Detector

Local one-image water-droplet detection and live-camera training pipeline for fabric samples during hydrostatic-pressure testing.

## Overview

The production path trains one local model to recognise water droplets from a single fabric photo. It does not need a dry reference at runtime. The project also retains a classical dry-reference prototype for comparison and data-collection experiments.

The one-image model will learn from labelled examples of:

- real water droplets;
- dry fabric and normal weave patterns;
- reflections, folds, shadows, and existing wet areas.

## Supported Fabrics

The pipeline dynamically discovers available fabrics by scanning the `data/raw/` directory. Any subfolder added there will be automatically picked up for evaluation. Currently, our working set includes household fabrics:

| Fabric | Notes |
|---|---|
| Denim | Heavyweight |
| T-shirt Cotton | Cotton jersey/knit (stretchy) |
| Shirt Cotton | Woven poplin/oxford (stiffer) |

*Note: The eventual target is to process industrial samples (Cotton, Silk, Two-layer, Three-layer, Nylon) once they become available.*

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Launch Web UI Dashboard

Label training photos, train the local model, and detect droplets in one uploaded photo:

```bash
python scripts/run_ui.py
# or
streamlit run app.py
```

### Train the Single-Image Droplet Model

The final live-camera workflow does not need a dry reference. Label each
droplet in representative fabric photos with a tight rectangle, then place
image/label pairs in the matching folders below:

```text
datasets/
  images/train/   labels/train/
  images/val/     labels/val/
  images/test/    labels/test/
```

Use separate test runs for training and validation so near-identical video
frames do not appear in both sets. Include dry fabric, reflections, folds,
existing wet areas, and real droplets across every target fabric.

Install the optional local ML tools and train:

```bash
pip install -e ".[ml]"
python scripts/train_detection.py --data datasets/droplets.yaml
```

Copy the resulting `best.pt` model to `models/droplet-seg.pt`. The dashboard's
**Detect One Photo** tab then identifies droplets from a single image without
a dry reference. The same detector is the one that will run on each live
camera frame before multi-frame leak tracking is added.

### Run Detection on a Fabric (CLI)

```bash
python scripts/run_prototype.py \
    --fabric-dir data/raw/denim \
    --dry-reference data/dry_reference/denim_dry.jpg
```

### Evaluate Against Ground Truth

```bash
python scripts/evaluate.py --data-root data
```

### Run Tests

```bash
pytest
```

## Project Structure

```
droplet-detector/
├── src/droplet_detector/    # Core library
│   ├── config.py            # Pydantic configuration & validation
│   ├── models.py            # Data models (DropletDetection, FabricAccuracyReport)
│   ├── preprocessing.py     # Illumination flattening, noise suppression, differencing
│   ├── candidate_detection.py  # Hough circles + MSER blobs + merge
│   ├── scoring.py           # Confidence scoring (size, bright-centre, circularity)
│   ├── consistency.py       # Frame-to-frame consistency check
│   └── pipeline.py          # Top-level detection entry point
├── scripts/                 # CLI tools
│   ├── run_prototype.py     # Run detection on a folder of fabric photos
│   └── evaluate.py          # Compare detections to CVAT ground truth
├── tests/                   # Pytest suite with synthetic fixtures
├── datasets/                # Labelled images and YOLO training labels
├── data/                    # Uploaded images and legacy prototype outputs
└── docs/                    # Documentation (MkDocs, step reports)
```

## Step 1 Status

This is the initial classical-CV prototype (Step 1 of 8). Later steps will add:
- Live camera streaming (Step 2)
- Multi-frame tracking with droplet-count stopping (Step 3)
- YOLO-based detection for robustness (Step 4)
- Video/image saving pipeline (Step 5)
- Library packaging with clean API (Step 6)
- Validation metrics (Step 7)
- Full documentation (Step 8)

## License

Internal / proprietary — not yet published.
