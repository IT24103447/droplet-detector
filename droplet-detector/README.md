# Droplet Detector

Classical computer-vision pipeline for automatic water-droplet detection on fabric samples during hydrostatic-pressure testing.

## Overview

This library detects water droplets (1–5 mm) on fabric samples by comparing a "wet" photo against a pre-captured dry reference image. It uses a combination of:

- **Illumination flattening** — removes broad lighting gradients before differencing
- **Dry-reference differencing** — cancels the fabric's weave pattern and fixed reflections
- **Noise-floor suppression** — zeroes out sensor noise without contrast-stretching artefacts
- **Hough circle detection** — finds round shapes in the expected size range
- **MSER blob detection** — finds stable extremal regions with a circularity filter
- **Confidence scoring** — scores candidates on size match, bright-centre highlight, and roundness
- **Frame-to-frame consistency** — filters transient noise by requiring spatial persistence

## Supported Fabrics

| Fabric | Notes |
|---|---|
| Cotton | Standard baseline |
| Silk | Challenging — glossy surface, glare |
| Two-layer | Layered construction |
| Three-layer | Multi-layer, varied textures |
| Nylon | Synthetic, potential glare issues |

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Run Detection on a Fabric

```bash
python scripts/run_prototype.py \
    --fabric-dir data/raw/cotton \
    --dry-reference data/dry_reference/cotton_dry.jpg
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
├── data/                    # Raw images, references, annotations, results
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
