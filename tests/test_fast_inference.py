"""Acceleration must preserve detector coordinates and never use stale weights."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from droplet_detector import fast_inference, single_image
from droplet_detector.fast_inference import OnnxDetector, fingerprint, load_cached_detector


def test_dynamic_letterbox_rgb_scaling_and_duplicate_suppression():
    detector=OnnxDetector.__new__(OnnxDetector)
    detector.input_name='images'
    detector.nms_iou=.5
    seen=[]
    output=np.array([[[320,322,50],[160,162,50],[64,64,10],[64,64,10],[.9,.8,.2]]],np.float32)
    def run(_,feed):
        seen.append(feed['images']);return [output]
    detector.session=SimpleNamespace(run=run)
    frame=np.full((100,200,3),(10,20,30),np.uint8)
    detections=detector(frame,.25)
    assert seen[0].shape==(1,3,320,640)
    np.testing.assert_allclose(seen[0][0,:,0,0],np.array([30,20,10])/255)
    assert len(detections)==1
    assert (detections[0].x_px,detections[0].y_px,detections[0].radius_px)==(100,50,10)
    assert (frame[0,0]==[10,20,30]).all()


def test_empty_output_respects_confidence_threshold():
    detector=OnnxDetector.__new__(OnnxDetector)
    detector.input_name='images';detector.nms_iou=.5
    detector.session=SimpleNamespace(run=lambda *_:[np.zeros((1,5,2),np.float32)])
    assert detector(np.zeros((50,50,3),np.uint8),.25)==[]


def test_cache_requires_matching_weights_export_integrity_and_parity(tmp_path,monkeypatch):
    source=tmp_path/'source.pt';source.write_bytes(b'original weights')
    monkeypatch.setattr(fast_inference,'CACHE_ROOT',tmp_path/'cache')
    digest=fingerprint(source);folder=fast_inference.CACHE_ROOT/digest;folder.mkdir(parents=True)
    export=folder/'model.onnx';export.write_bytes(b'verified export')
    info=dict(source_sha256=digest,onnx_sha256=fingerprint(export),precision='float32',imgsz=640,
              dynamic=True,parity_checked=True,nms_iou=.5)
    manifest=folder/'source.json';manifest.write_text(json.dumps(info))
    sentinel=object()
    monkeypatch.setattr(fast_inference,'OnnxDetector',lambda *_,**__:sentinel)
    assert load_cached_detector(source) is sentinel
    info['parity_checked']=False;manifest.write_text(json.dumps(info))
    assert load_cached_detector(source) is None
    info['parity_checked']=True;manifest.write_text(json.dumps(info));export.write_bytes(b'changed export')
    assert load_cached_detector(source) is None
    export.write_bytes(b'verified export');source.write_bytes(b'retrained weights')
    assert load_cached_detector(source) is None


def test_frame_detector_uses_verified_runtime_and_can_force_original(tmp_path,monkeypatch):
    source=tmp_path/'source.pt';source.touch()
    monkeypatch.setattr(fast_inference,'load_cached_detector',lambda _:lambda frame,confidence:['optimized'])
    monkeypatch.setattr(single_image,'_load_model',lambda _:SimpleNamespace(predict=lambda **_:[SimpleNamespace(boxes=[])]))
    detector=single_image.FrameDetector(source,.25)
    assert detector.backend=='onnx' and detector(np.zeros((2,2,3),np.uint8))==['optimized']
    original=single_image.FrameDetector(source,.25,use_acceleration=False)
    assert original.backend=='pytorch' and original(np.zeros((2,2,3),np.uint8))==[]


@pytest.mark.parametrize("prefer_gpu", [True, False])
def test_load_prefers_verified_cpu_when_gpu_disabled_or_unavailable(tmp_path,monkeypatch,prefer_gpu):
    source=tmp_path/'model.pt';source.write_bytes(b'weights')
    monkeypatch.setattr(fast_inference,'CACHE_ROOT',tmp_path/'cache')
    digest=fingerprint(source);folder=fast_inference.CACHE_ROOT/digest;folder.mkdir(parents=True)
    export=folder/'model.onnx';export.write_bytes(b'export')
    (folder/'source.json').write_text(json.dumps(dict(source_sha256=digest,onnx_sha256=fingerprint(export),
        precision='float32',imgsz=640,dynamic=True,parity_checked=True,gpu_parity_checked=True,
        gpu_provider='DmlExecutionProvider',gpu_device_id=1,gpu_preferred=prefer_gpu)))
    calls=[];cpu=object()
    def runtime(*_,provider='CPUExecutionProvider',**__):
        calls.append(provider)
        if provider=='DmlExecutionProvider':raise ValueError('GPU unavailable')
        return cpu
    monkeypatch.setattr(fast_inference,'OnnxDetector',runtime)
    assert load_cached_detector(source) is cpu
    assert calls == (['DmlExecutionProvider', 'CPUExecutionProvider'] if prefer_gpu else ['CPUExecutionProvider'])
