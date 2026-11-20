"""droplet-detector — classical CV pipeline for water-droplet detection on fabric samples."""

__version__ = "0.1.0"

from .config import DropletDetectorConfig, ROI
from .models import DropletDetection, FabricAccuracyReport
from .pipeline import detect_droplets_in_image
from .single_image import FrameDetector, detect_droplets_in_single_image, ModelNotReadyError
from .video import LeakEvent, LiveTest, OutputPaths, TestResult, VideoSettings, analyze_video, start_test
from .vision_language import VisionLanguageDetector

__all__ = [
    "DropletDetectorConfig",
    "ROI",
    "DropletDetection",
    "FabricAccuracyReport",
    "detect_droplets_in_image",
    "detect_droplets_in_single_image",
    "ModelNotReadyError",
    "FrameDetector",
    "VisionLanguageDetector",
    "LeakEvent",
    "LiveTest",
    "OutputPaths",
    "TestResult",
    "VideoSettings",
    "analyze_video",
    "start_test",
]
