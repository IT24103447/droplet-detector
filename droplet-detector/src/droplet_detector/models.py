"""Data models returned by the detection pipeline.

All results are Pydantic v2 models so they serialise cleanly to JSON and
integrate naturally with the calling application's data layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DropletDetection(BaseModel):
    """A single detected droplet candidate that passed the confidence threshold."""

    droplet_number: int = Field(..., description="1-based index within this frame")
    x_px: float = Field(..., description="Centre X in pixels")
    y_px: float = Field(..., description="Centre Y in pixels")
    x_mm: Optional[float] = Field(
        None, description="Centre X in mm (populated once mm_per_pixel is calibrated)"
    )
    y_mm: Optional[float] = Field(
        None, description="Centre Y in mm (populated once mm_per_pixel is calibrated)"
    )
    radius_px: float = Field(..., description="Estimated radius in pixels")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Composite confidence score"
    )
    detected_at: Optional[datetime] = Field(
        None, description="Populated from Step 2 onward (live timer)"
    )
    image_filename: Optional[str] = Field(
        None, description="Filename of the annotated snapshot saved for this detection"
    )


class FabricAccuracyReport(BaseModel):
    """Precision/recall summary for one fabric type."""

    fabric_name: str
    total_labelled_droplets: int
    true_positives: int
    false_positives: int
    missed: int

    @property
    def precision(self) -> float:
        """TP / (TP + FP), or 0.0 if no detections."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + missed), or 0.0 if no ground-truth labels."""
        denom = self.true_positives + self.missed
        return self.true_positives / denom if denom else 0.0
