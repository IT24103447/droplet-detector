"""Export a float32 CPU runtime and enable it only after prediction parity checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from droplet_detector.fast_inference import CACHE_ROOT, OnnxDetector, fingerprint
from droplet_detector.fabric_area import find_fabric_area
from droplet_detector.single_image import FrameDetector
from droplet_detector.video import _ellipse_outside


def same_predictions(reference, optimized) -> bool:
    if len(reference) != len(optimized):
        return False
    remaining = list(optimized)
    for original in reference:
        closest = min(remaining, key=lambda d: np.hypot(d.x_px-original.x_px,d.y_px-original.y_px))
        if (max(abs(closest.x_px-original.x_px),abs(closest.y_px-original.y_px),
                abs(closest.radius_px-original.radius_px)) > .1
                or abs(closest.confidence-original.confidence) > .0001):
            return False
        remaining.remove(closest)
    return True


def optimize(source: Path, images: list[Path]) -> Path:
    from ultralytics import YOLO
    import torch

    if not images:
        raise ValueError("Provide real validation frames before enabling an optimized model.")
    digest = fingerprint(source)
    folder = CACHE_ROOT / digest
    folder.mkdir(parents=True, exist_ok=True)
    exported = folder / "model.onnx"
    old_info = {}
    if (folder / "source.json").is_file():
        old_info = json.loads((folder / "source.json").read_text())
    if not exported.is_file() or old_info.get("source_sha256") != digest:
        staged = folder / "model.pt"
        shutil.copyfile(source, staged)
        model = YOLO(str(staged))
        if model.task != "detect" or len(model.names) != 1:
            raise ValueError("Only single-class droplet detection weights are supported.")
        model.export(format="onnx", imgsz=640, dynamic=True, simplify=False, opset=17, device="cpu")
    reference = FrameDetector(source, .25, use_acceleration=False)
    optimized = OnnxDetector(exported)
    checked = 0
    for image in images:
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Unreadable validation image: {image}")
        inputs = [frame]
        try:
            x,y,w,h = find_fabric_area(frame)
            crop = frame[y:y+h,x:x+w].copy()
            crop[_ellipse_outside(h,w)] = 128
            inputs.append(crop)
        except ValueError:
            pass
        for sample in inputs:
            original = reference(sample)
            torch.set_num_threads(4)
            optimized.nms_iou = float(reference.model.predictor.args.iou)
            actual = optimized(sample,.25)
            if not same_predictions(original,actual):
                raise ValueError(f"Prediction parity failed on {image}; optimized runtime was not enabled.")
            checked += 1
    if fingerprint(source) != digest:
        raise RuntimeError("Source weights changed during optimization.")
    info = dict(source_sha256=digest,source_path=str(source.resolve()),onnx_sha256=fingerprint(exported),
                precision="float32",imgsz=640,dynamic=True,parity_checked=True,checked_inputs=checked,
                nms_iou=optimized.nms_iou,
                validation_images=[str(p.resolve()) for p in images],coordinate_tolerance_px=.1,
                confidence_tolerance=.0001)
    pending = folder / "source.pending.json"
    pending.write_text(json.dumps(info,indent=2),encoding="utf-8")
    pending.replace(folder / "source.json")
    print(f"Enabled optimized inference after {checked} prediction checks: {source}",flush=True)
    return exported


def enable_gpu(source: Path, images: list[Path], device: int) -> None:
    folder=CACHE_ROOT/fingerprint(source)
    manifest=folder/'source.json'
    info=json.loads(manifest.read_text())
    if not info.get('parity_checked') or info['onnx_sha256']!=fingerprint(folder/'model.onnx'):
        raise ValueError('CPU prediction parity must pass before GPU validation.')
    cpu=OnnxDetector(folder/'model.onnx',nms_iou=info['nms_iou'])
    gpu=OnnxDetector(folder/'model.onnx',nms_iou=info['nms_iou'],provider='DmlExecutionProvider',device_id=device)
    if gpu.provider!='DmlExecutionProvider':
        raise ValueError('The requested GPU is unavailable; keep the CPU runtime.')
    checked=0
    for path in images:
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError(f'Unreadable validation image: {path}')
        samples=[frame]
        try:
            x,y,w,h=find_fabric_area(frame);crop=frame[y:y+h,x:x+w].copy()
            crop[_ellipse_outside(h,w)]=128;samples.append(crop)
        except ValueError:pass
        for sample in samples:
            if not same_predictions(cpu(sample,.25),gpu(sample,.25)):
                raise ValueError(f'GPU prediction parity failed on {path}; GPU was not enabled.')
            checked+=1
    info.update(gpu_parity_checked=True,gpu_checked_inputs=checked,
                gpu_provider='DmlExecutionProvider',gpu_device_id=device)
    pending=folder/'source.pending.json';pending.write_text(json.dumps(info,indent=2),encoding='utf-8')
    pending.replace(manifest)
    print(f'Enabled GPU {device} after {checked} prediction checks: {source}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models",type=Path,nargs="+")
    parser.add_argument("--validation-manifest",type=Path,required=True)
    parser.add_argument("--max-images",type=int,default=12)
    parser.add_argument("--gpu-device",type=int)
    args=parser.parse_args()
    records=json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    images=[Path(r["image"]) for r in records[:args.max_images]]
    for source in args.models:
        optimize(source,images)
        if args.gpu_device is not None:
            enable_gpu(source,images,args.gpu_device)


if __name__=="__main__":
    main()
