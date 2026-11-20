"""Local vision-language inference for water droplets on textile specimens."""
from __future__ import annotations

import base64
import http.client
import json
from typing import Annotated

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .models import DropletDetection
from .single_image import ModelNotReadyError


MODEL_NAME = "qwen3-vl:2b-instruct"
REQUEST_TIMEOUT_S = 90

_INSTRUCTIONS = (
    "Find distinct raised beads of liquid water on the fabric in the last image. "
    "Look for a compact curved boundary with a glossy highlight, shading, or a local "
    "distortion of the cloth texture. A feature can be subtle. Reject stitching, tape, "
    "folds, rig lights, the clamp, writing, loose fibres, and flat wet patches. "
    "Ignore the blank border. Count a growing bead once. If no bead is supported by "
    "the image, return an empty droplets array. Report at most three high-evidence beads. "
    "Give each a tight box using "
    "[left, top, right, bottom] normalized to 0–1000 relative to the last image, and "
    "an evidence score (confidence) between 0 and 1. Return only the requested JSON."
)
_COMPARISON = (
    "Image 1 shows the same fabric at the start of the test. Image 2 is the current frame. "
    "Find only beads that newly appear in image 2. Ignore marks already present at the same "
    "location and changes caused by general lighting or motion. "
)

_Coordinate = Annotated[int, Field(ge=0, le=1000)]


class _Box(BaseModel):
    bbox_2d: tuple[_Coordinate, _Coordinate, _Coordinate, _Coordinate]
    confidence: float = Field(ge=0, le=1)


class _Answer(BaseModel):
    droplets: list[_Box] = Field(max_length=3)


_SCHEMA = _Answer.model_json_schema()


def _jpeg(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    if max(height, width) > 704:
        scale = 704 / max(height, width)
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise OSError("Could not encode the fabric image for local vision analysis")
    return base64.b64encode(encoded).decode("ascii")


def _ollama(path: str, payload: dict) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=REQUEST_TIMEOUT_S)
    try:
        connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status == 404:
            raise ModelNotReadyError(f"Local vision model unavailable. Run: ollama pull {MODEL_NAME}")
        if response.status != 200:
            raise RuntimeError(f"Local vision service returned HTTP {response.status}")
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise RuntimeError("Local vision service returned an oversized response")
        reply = json.loads(body)
        if not isinstance(reply, dict):
            raise RuntimeError("Local vision service returned an invalid response")
        return reply
    except ConnectionRefusedError as exc:
        raise ModelNotReadyError("Start Ollama on this computer to use the local vision model") from exc
    except TimeoutError as exc:
        raise RuntimeError("The local vision model timed out while analyzing a frame") from exc
    except (http.client.HTTPException, json.JSONDecodeError, UnicodeError) as exc:
        raise RuntimeError("Could not read the local vision service response") from exc
    finally:
        connection.close()


class VisionLanguageDetector:
    """Find droplets in one BGR image or compare a video against its opening frame."""

    def __init__(self, confidence_threshold: float = .25, *, compare_to_opening_frame: bool = False):
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Set a confidence threshold in (0, 1]")
        self.confidence_threshold = confidence_threshold
        self.compare_to_opening_frame = compare_to_opening_frame
        self._opening_image: str | None = None

    def check_available(self) -> None:
        details = _ollama("/api/show", {"model": MODEL_NAME})
        if "vision" not in details.get("capabilities", []):
            raise ModelNotReadyError(f"Install the local image-capable model with: ollama pull {MODEL_NAME}")

    def __call__(self, frame: np.ndarray) -> list[DropletDetection]:
        current = _jpeg(frame)
        if self.compare_to_opening_frame and self._opening_image is None:
            self._opening_image = current
            return []

        images = [current]
        prompt = _INSTRUCTIONS
        if self._opening_image is not None:
            images.insert(0, self._opening_image)
            prompt = _COMPARISON + prompt

        reply = _ollama("/api/chat", {
            "model": MODEL_NAME,
            "stream": False,
            "think": False,
            "format": _SCHEMA,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
            "messages": [{"role": "user", "content": prompt, "images": images}],
        })
        try:
            answer = _Answer.model_validate_json(reply["message"]["content"], strict=True)
        except (KeyError, TypeError, ValidationError) as exc:
            raise RuntimeError("The local vision model returned invalid droplet coordinates") from exc

        rectangles, scores = [], []
        for box in answer.droplets:
            x1, y1, x2, y2 = box.bbox_2d
            if x1 >= x2 or y1 >= y2:
                raise RuntimeError("The local vision model returned an invalid droplet box")
            if box.confidence >= self.confidence_threshold:
                rectangles.append((x1, y1, x2 - x1, y2 - y1))
                scores.append(box.confidence)

        # The model can emit the same bead more than once in a single answer.
        unique = cv2.dnn.NMSBoxes(rectangles, scores, 0.0, .5)
        height, width = frame.shape[:2]
        detections = []
        for number, index in enumerate(np.asarray(unique).ravel(), 1):
            x, y, w, h = rectangles[index]
            detections.append(DropletDetection(
                droplet_number=number,
                x_px=(x + w / 2) * width / 1000,
                y_px=(y + h / 2) * height / 1000,
                radius_px=max(w * width, h * height) / 2000,
                confidence=scores[index],
            ))
        return detections
