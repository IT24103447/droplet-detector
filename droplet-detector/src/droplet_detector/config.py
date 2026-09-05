"""Configuration and validation for the droplet-detector pipeline.

All user-supplied values (paths, sensitivity, ROI, resolution) are validated
here via Pydantic v2 so that bad input fails with a clear message rather than
a cryptic stack trace deep inside the image-processing code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class ROI(BaseModel):
    """Region of interest within the camera frame."""

    x: int = Field(..., ge=0, description="Top-left X coordinate in pixels")
    y: int = Field(..., ge=0, description="Top-left Y coordinate in pixels")
    width: int = Field(..., gt=0, description="ROI width in pixels")
    height: int = Field(..., gt=0, description="ROI height in pixels")


class DropletDetectorConfig(BaseModel):
    """Central configuration for a single detection run.

    Paths are auto-created if they don't exist yet.  ``mm_per_pixel`` is left
    as *None* until ArUco-based calibration lands in Step 2; the pipeline will
    still work — it just reports positions in pixels only.
    """

    camera_id: int = 0
    video_save_path: Path = Field(default=Path("data/results/video"))
    image_save_path: Path = Field(default=Path("data/results/images"))
    results_save_path: Path = Field(default=Path("data/results/reports"))
    roi: Optional[ROI] = None
    sensitivity: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold — lower = more detections but more FPs",
    )
    resolution: Tuple[int, int] = (1920, 1080)
    frame_rate: int = 30
    droplet_min_diameter_mm: float = Field(
        1.0, gt=0.0, description="Smallest real droplet we expect (mm)"
    )
    droplet_max_diameter_mm: float = Field(
        5.0, gt=0.0, description="Largest real droplet we expect (mm)"
    )
    mm_per_pixel: Optional[float] = Field(
        None,
        gt=0.0,
        description="Set once ArUco calibration lands in Step 2",
    )
    diff_noise_floor: int = Field(
        15,
        ge=0,
        le=255,
        description="Pixel-value differences below this are zeroed (sensor noise)",
    )

    @field_validator("video_save_path", "image_save_path", "results_save_path")
    @classmethod
    def _ensure_dir_exists(cls, v: Path) -> Path:
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
