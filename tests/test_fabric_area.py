"""Checks for excluding fixtures, static marks, and uncertain calibration."""
import cv2
import numpy as np
import pytest

from droplet_detector.fabric_area import OpeningFrameEvidence, find_fabric_area
from droplet_detector.models import DropletDetection
from droplet_detector import video, VideoSettings, OutputPaths, analyze_video


def ring_frame():
    frame = np.full((480,640,3),30,np.uint8)
    cv2.circle(frame,(320,240),175,(245,245,245),-1)
    cv2.circle(frame,(320,240),155,(100,100,100),-1)
    for x,y in [(70,60),(570,420),(580,90)]:
        cv2.circle(frame,(x,y),10,(255,255,255),-1)
    return frame


def droplet(x,y,r=5):
    return DropletDetection(droplet_number=1,x_px=x,y_px=y,radius_px=r,confidence=.9)


def test_detects_inside_edge_not_white_rim_or_external_lights():
    x,y,w,h=find_fabric_area(ring_frame())
    assert abs(x+w/2-320)<4 and abs(y+h/2-240)<4
    assert 145 < w/2 < 157
    assert w==h


@pytest.mark.parametrize("value",[0,120,255])
def test_no_rim_fails_instead_of_using_full_frame(value):
    with pytest.raises(ValueError,match="white fabric rim"):
        find_fabric_area(np.full((480,640,3),value,np.uint8))


def test_masks_fixture_and_rejects_boxes_crossing_ellipse_or_pool_sized_boxes():
    frame=np.full((100,100,3),255,np.uint8)
    def detector(crop):
        assert (crop[0,0]==128).all()
        return [droplet(50,50),droplet(91,50,10),droplet(4,4),droplet(50,50,25)]
    result=video._detect_frame(detector,frame,(0,0,100,100),"ellipse")
    assert len(result)==1 and result[0].x_px==50


def test_static_texture_and_uniform_lighting_change_do_not_confirm_but_new_bead_can():
    rng=np.random.default_rng(42)
    before=rng.integers(60,110,(200,200,3),dtype=np.uint8)
    gate=OpeningFrameEvidence(before,(0,0,200,200))
    bead=droplet(100,100,8)
    assert gate.filter(before,[bead])==[]
    assert gate.filter(before+25,[bead])==[]
    after=before.copy()
    cv2.circle(after,(100,100),7,(230,230,230),-1)
    assert gate.filter(after,[bead])==[bead]
    reference=gate._reference.copy()
    for _ in range(3):
        assert gate.filter(after,[bead])==[bead]
        assert gate.filter(before,[bead])==[]
    np.testing.assert_array_equal(gate._reference,reference)


def test_large_camera_translation_is_rejected():
    rng=np.random.default_rng(8)
    before=rng.integers(40,200,(200,200,3),dtype=np.uint8)
    after=cv2.warpAffine(before,np.float32([[1,0,15],[0,1,0]]),(200,200))
    gate=OpeningFrameEvidence(before,(0,0,200,200))
    assert gate.filter(after,[droplet(100,100)])==[]
    assert gate.motion_rejections==1


def test_small_candidate_box_on_a_large_changed_pool_is_rejected():
    before=np.full((200,200,3),80,np.uint8)
    after=before.copy()
    cv2.circle(after,(100,100),60,(150,150,150),-1)
    cv2.circle(after,(100,100),7,(230,230,230),-1)
    gate=OpeningFrameEvidence(before,(0,0,200,200))
    assert gate.filter(after,[droplet(100,100,8)])==[]


def test_settled_reference_recovers_a_new_bead_after_a_broad_light_change():
    opening = np.full((200, 200, 3), 80, np.uint8)
    stable_light = opening.copy()
    stable_light[:, :85] += 45
    changed = stable_light.copy()
    cv2.circle(changed, (60, 100), 7, (215, 215, 215), -1)
    bead = droplet(60, 100, 8)
    gate = OpeningFrameEvidence(opening, (0, 0, 200, 200))
    assert gate.observe(opening, [], 0) == []
    assert gate.filter(changed, [bead]) == []
    assert gate.observe(stable_light, [], 6) == []
    assert gate.observe(stable_light, [bead], 7) == []
    assert gate.observe(changed, [bead.model_copy(update={"confidence": .35})], 8) == []
    assert gate.observe(changed, [bead], 8) == [bead]


def test_video_does_not_fall_back_to_unrestricted_detection(tmp_path,monkeypatch):
    source=tmp_path/'no_rim.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
    assert writer.isOpened()
    writer.write(np.zeros((120,160,3),np.uint8));writer.release()
    calls=[]
    monkeypatch.setattr(video,'FrameDetector',lambda *_:lambda frame:calls.append(frame) or [droplet(20,20)])
    result=analyze_video(source,outputs=OutputPaths.in_directory(tmp_path/'out'))
    assert result.status=='error' and not result.leaks and not calls
    assert 'white fabric rim' in result.errors[0]


def test_auto_calibrated_video_rejects_repeated_outside_hit_and_draws_area(tmp_path,monkeypatch):
    source=tmp_path/'rim.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
    assert writer.isOpened()
    for _ in range(5):writer.write(ring_frame())
    writer.release()
    monkeypatch.setattr(video,'FrameDetector',lambda *_:lambda frame:[droplet(5,5)])
    result=analyze_video(source,outputs=OutputPaths.in_directory(tmp_path/'out'),
                         settings=VideoSettings(require_new_appearance=False))
    assert result.status=='completed', result.errors
    assert result.fabric_roi and result.fabric_roi_shape=='ellipse'
    assert result.leaks==[]
    image=cv2.imread(result.final_image_filename)
    assert image is not None and (image[:,:,0]>200).any()


def test_lost_rim_pauses_detections_but_keeps_recording(tmp_path,monkeypatch):
    source=tmp_path/'lost_rim.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
    assert writer.isOpened()
    for _ in range(2):writer.write(ring_frame())
    for _ in range(3):writer.write(np.zeros((480,640,3),np.uint8))
    writer.release()
    calls=[]
    def detector(frame):
        calls.append(1)
        return [droplet(frame.shape[1]/2,frame.shape[0]/2)]
    monkeypatch.setattr(video,'FrameDetector',lambda *_:detector)
    result=analyze_video(source,outputs=OutputPaths.in_directory(tmp_path/'out'),
                         settings=VideoSettings(require_new_appearance=False))
    assert result.status=='completed',result.errors
    assert len(calls)==2 and not result.leaks
    assert any('paused on 3 frames' in w for w in result.warnings)
    capture=cv2.VideoCapture(result.video_filename)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT)==5
    capture.release()


def test_live_rim_rechecks_run_off_the_capture_thread(tmp_path,monkeypatch):
    import time
    from threading import Event,get_ident
    from droplet_detector import start_test
    reads=[]
    calibrations=[]
    called=Event()
    class Camera:
        def __init__(self,*_):pass
        def isOpened(self):return True
        def get(self,*_):return 30
        def release(self):pass
        def read(self):
            reads.append(get_ident());time.sleep(.01)
            return True,np.zeros((120,160,3),np.uint8)
    def locate(frame):
        calibrations.append(get_ident())
        return (40,20,80,80)
    def detector(frame):
        called.set();return []
    monkeypatch.setattr(video.cv2,'VideoCapture',Camera)
    monkeypatch.setattr(video,'find_fabric_area',locate)
    session=start_test(0,outputs=OutputPaths.in_directory(tmp_path/'live'),
                       settings=VideoSettings(require_new_appearance=False),prepared_detector=detector)
    try:
        assert called.wait(3)
    finally:
        result=session.stop_test(timeout=5)
    assert result.status=='stopped',result.errors
    assert calibrations.count(reads[0])==1  # Startup only; subsequent checks belong to inference.
    assert any(thread!=reads[0] for thread in calibrations)
