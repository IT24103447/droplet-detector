"""Check how locally labelled frames become training examples and active weights."""
import json
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from droplet_detector import fabric_training
from droplet_detector.video_annotation import save_frame


@pytest.mark.parametrize("mode", ["one video", "different videos", "selected video"])
def test_training_uses_circular_crops_and_holds_back_frames(tmp_path, monkeypatch, mode):
    annotations = tmp_path / "annotations"
    a, b = "a" * 64, "b" * 64
    other = b if mode == "different videos" else a
    frames = [(a, 3, 40), (other, 30, 120)]
    if mode == "selected video":
        frames.append((b, 20, 70))
    monkeypatch.setattr(fabric_training, "find_fabric_area", lambda _: (15, 5, 60, 60))
    for clip, index, color in frames:
        save_frame(annotations, "train", clip, index, np.full((80, 100, 3), color, np.uint8),
                   [[.47, .36, .06, .08]], dry=False)

    def check_split(job, destination, base_model, epochs):
        counts = fabric_training.import_dataset(job / "dataset.zip", job / "dataset")
        assert counts == {"train": len(frames) - 1, "val": 1}
        with ZipFile(job / "dataset.zip") as archive:
            assert f"images/train/{a}_00000003.png" in archive.namelist()
            assert f"images/val/{other}_00000030.png" in archive.namelist()
            image = cv2.imdecode(np.frombuffer(archive.read(f"images/train/{a}_00000003.png"), dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            assert image.shape == (60, 60, 3)
            assert np.all(image[0, 0] == 128)
            assert np.all(image[30, 30] == 40)
            label = archive.read(f"labels/train/{a}_00000003.txt").decode().split()
            assert [float(v) for v in label] == pytest.approx([0, 35 / 60, 27 / 60, 6 / 60, 6.4 / 60])
        return destination

    monkeypatch.setattr(fabric_training, "train_profile", check_split)
    destination = tmp_path / "active.pt"
    assert fabric_training.train_annotated_model(
        tmp_path / "job", destination, tmp_path / "base.pt", 1, annotations,
        focus_clip=a if mode == "selected video" else None,
    ) == destination


@pytest.mark.parametrize("matched", [0, 2])
def test_publishes_only_a_validated_candidate(tmp_path, monkeypatch, matched):
    job = tmp_path / "job"
    weights = job / "training" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"trained example")
    destination = tmp_path / "active.pt"
    destination.write_bytes(b"previous checkpoint")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda _: SimpleNamespace(train=lambda **_: SimpleNamespace(save_dir=job / "training")),
    ))
    monkeypatch.setattr(fabric_training, "import_dataset", lambda *_: None)
    monkeypatch.setattr(fabric_training, "_count_matches", lambda data, path, split: {
        "matched": matched if path == weights and split == "val" else 1,
        "predictions": 2,
        "labels": 2,
    })
    if matched:
        assert fabric_training.train_profile(job, destination, destination, 1) == destination
        assert destination.read_bytes() == b"trained example"
    else:
        with pytest.raises(ValueError, match="current model is still active"):
            fabric_training.train_profile(job, destination, destination, 1)
        assert destination.read_bytes() == b"previous checkpoint"
    assert json.loads((job / "training_check.json").read_text())["validation"]["matched"] == matched


@pytest.mark.parametrize("matches_video", [False, True])
def test_video_training_checks_a_marked_position_before_switching(tmp_path, monkeypatch, matches_video):
    annotations = tmp_path / "annotations"
    clip = "a" * 64
    source = annotations / "videos" / f"{clip}.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    for index, color in [(3, 40), (30, 120)]:
        save_frame(annotations, "train", clip, index, np.full((80, 100, 3), color, np.uint8),
                   [[.47, .36, .06, .08]], dry=False)
    destination = tmp_path / "active.pt"
    destination.write_bytes(b"working model")
    monkeypatch.setattr(fabric_training, "find_fabric_area", lambda _: (15, 5, 60, 60))

    def train(job, weights, base_model, epochs):
        assert weights != destination
        weights.write_bytes(b"candidate")
        return weights

    def replay(path, *, outputs, settings, use_source_recording):
        assert path == source and use_source_recording and settings.analysis_fps == 6
        assert destination.read_bytes() == b"working model"
        leaks = [SimpleNamespace(x_px=50 if matches_video else 18, y_px=32)]
        return SimpleNamespace(status="completed", leaks=leaks)

    monkeypatch.setattr(fabric_training, "train_profile", train)
    monkeypatch.setattr(fabric_training, "analyze_video", replay)
    args = (tmp_path / "job", destination, destination, 1, annotations)
    if matches_video:
        assert fabric_training.train_annotated_model(*args, focus_clip=clip) == destination
        assert destination.read_bytes() == b"candidate"
    else:
        with pytest.raises(ValueError, match="could not confirm a marked leak"):
            fabric_training.train_annotated_model(*args, focus_clip=clip)
        assert destination.read_bytes() == b"working model"
