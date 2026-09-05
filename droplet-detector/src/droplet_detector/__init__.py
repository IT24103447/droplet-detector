"""droplet-detector — classical CV pipeline for water-droplet detection on fabric samples."""

__version__ = "0.1.0"

from .config import DropletDetectorConfig, ROI
from .models import DropletDetection, FabricAccuracyReport
from .pipeline import detect_droplets_in_image

__all__ = [
    "DropletDetectorConfig",
    "ROI",
    "DropletDetection",
    "FabricAccuracyReport",
    "detect_droplets_in_image",
]
