"""Tests for the trained single-image detection entry point."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from droplet_detector.single_image import (
    ModelNotReadyError,
    detect_droplets_in_single_image,
)


def test_missing_model_explains_training_requirement(tmp_path: Path) -> None:
    image_path = tmp_path / "fabric.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

    with pytest.raises(ModelNotReadyError, match="Label fabric images"):
        detect_droplets_in_single_image(image_path, tmp_path / "missing.pt")
