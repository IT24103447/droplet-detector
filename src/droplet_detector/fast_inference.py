"""Optional, verified float32 ONNX inference for the single-class YOLO detector."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import warnings

import cv2
import numpy as np

from .models import DropletDetection


CACHE_ROOT = Path(__file__).resolve().parents[2] / "data/results/inference_cache"


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class OnnxDetector:
    """Match YOLO's letterbox, class-0 confidence filter, NMS and scaling."""

    def __init__(self, path: Path, nms_iou: float = .5, *, cpu_threads: int = 8, allow_spinning: bool = False,
                 provider: str = "CPUExecutionProvider", device_id: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(cpu_threads, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if provider == "DmlExecutionProvider":
            if provider not in ort.get_available_providers():
                raise ValueError("Windows GPU inference is unavailable")
            options.enable_mem_pattern = False
            providers = [(provider,{"device_id":device_id}),"CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.provider = self.session.get_providers()[0]
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Optimized detector requires a float32 image input")
        self.input_name = inputs[0].name
        self.nms_iou = nms_iou

    def __call__(self, frame: np.ndarray, confidence: float) -> list[DropletDetection]:
        height, width = frame.shape[:2]
        ratio = 640 / max(height, width)
        new_w, new_h = max(1, round(width*ratio)), max(1, round(height*ratio))
        pad_w, pad_h = (640-new_w) % 32, (640-new_h) % 32
        left, top = round(pad_w/2-.1), round(pad_h/2-.1)
        image = cv2.resize(frame, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(image, top, round(pad_h/2+.1), left, round(pad_w/2+.1),
                                   cv2.BORDER_CONSTANT, value=(114,114,114))
        tensor = np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32)
        tensor /= 255.
        output = self.session.run(None, {self.input_name:tensor})[0]
        if output.ndim != 3 or output.shape[:2] != (1,5):
            raise ValueError("Optimized detector requires the single-class YOLO output layout")
        candidates = output[0].T
        candidates = candidates[candidates[:,4] > confidence]
        if not len(candidates):
            return []
        # NMSBoxes takes top-left/width/height, while YOLO emits centre/width/height.
        rectangles = candidates[:,:4].copy()
        rectangles[:,:2] -= rectangles[:,2:4]/2
        indices = cv2.dnn.NMSBoxes(rectangles.tolist(), candidates[:,4].tolist(), confidence, self.nms_iou)
        detections = []
        for number,index in enumerate(np.asarray(indices).ravel()[:300], 1):
            x,y,w,h = rectangles[index]
            x1,x2 = np.clip([(x-left)/(new_w/width),(x+w-left)/(new_w/width)],0,width)
            y1,y2 = np.clip([(y-top)/(new_h/height),(y+h-top)/(new_h/height)],0,height)
            detections.append(DropletDetection(droplet_number=number,x_px=float((x1+x2)/2),
                               y_px=float((y1+y2)/2),radius_px=float(max(x2-x1,y2-y1)/2),
                               confidence=float(candidates[index,4])))
        return detections


def load_cached_detector(source: Path) -> OnnxDetector | None:
    """Use only an intact export with matching weights and recorded parity checks."""
    if not source.is_file() or source.suffix.lower() != ".pt":
        return None
    digest = fingerprint(source)
    folder = CACHE_ROOT / digest
    manifest, exported = folder / "source.json", folder / "model.onnx"
    if not manifest.is_file() or not exported.is_file():
        return None
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if (info.get("source_sha256") != digest or not info.get("parity_checked")
                or info.get("precision") != "float32" or info.get("imgsz") != 640 or info.get("dynamic") is not True
                or info.get("onnx_sha256") != fingerprint(exported)):
            return None
        if (info.get("gpu_preferred", True) and info.get("gpu_parity_checked")
                and info.get("gpu_provider") == "DmlExecutionProvider"):
            try:
                return OnnxDetector(exported,nms_iou=info.get("nms_iou",.5),
                                    provider="DmlExecutionProvider",device_id=info["gpu_device_id"])
            except Exception:
                pass  # A moved project or unavailable adapter keeps its verified CPU path.
        return OnnxDetector(exported, nms_iou=info.get("nms_iou", .5))
    except ImportError:
        return None
    except Exception as exc:
        warnings.warn(f"Optimized detector could not load; using original weights: {exc}", RuntimeWarning)
        return None
