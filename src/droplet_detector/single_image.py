"""Single-image droplet inference backed by a custom detection model.

The model is trained on labelled images of this project's fabrics.  It does
not require a dry-reference image at runtime, so the same interface works for
an uploaded photo and for each frame of a live USB-camera stream.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from .models import DropletDetection


class ModelNotReadyError(RuntimeError):
    """Raised when single-image inference is requested without trained weights."""


def detect_droplets_in_single_image(
    image_path: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.5,
) -> list[DropletDetection]:
    """Detect independent water droplets in one fabric image.

    ``model_path`` must point to custom-trained Ultralytics detection
    weights (normally ``models/droplet-seg.pt``).  A generic pretrained model
    is deliberately not used because it has not learned this project's water
    droplets, fabrics, reflections, or camera setup.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    if not model_path.is_file():
        raise ModelNotReadyError(
            "No trained droplet model was found. Label fabric images and run "
            "scripts/train_detection.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelNotReadyError(
            "Single-image detection requires the ML dependencies. Install them "
            'with: pip install -e ".[ml]"'
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    result = YOLO(str(model_path)).predict(
        source=image, conf=confidence_threshold, verbose=False
    )[0]
    if result.boxes is None:
        return []

    detections: list[DropletDetection] = []
    for number, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            DropletDetection(
                droplet_number=number,
                x_px=(x1 + x2) / 2,
                y_px=(y1 + y2) / 2,
                radius_px=max(x2 - x1, y2 - y1) / 2,
                confidence=float(box.conf[0]),
            )
        )
    return detections
