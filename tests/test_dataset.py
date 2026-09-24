from pathlib import Path

from droplet_detector import dataset


def test_save_labelled_image_writes_yolo_box_and_empty_negative(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dataset, "DATASET_ROOT", tmp_path / "datasets")
    image_path, label_path, count = dataset.save_labelled_image(
        "cotton sample.jpg",
        b"image-data",
        "train",
        [{"left": 10, "top": 20, "width": 30, "height": 40, "scaleX": 1, "scaleY": 1}],
        100,
        100,
    )
    assert image_path.name == "cotton_sample.jpg"
    assert count == 1
    assert label_path.read_text() == "0 0.250000 0.400000 0.300000 0.400000"

    _, empty_label, empty_count = dataset.save_labelled_image(
        "dry.jpg", b"image-data", "val", [], 100, 100
    )
    assert empty_count == 0
    assert empty_label.read_text() == ""


def test_validate_training_dataset_requires_images_and_droplet_labels(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dataset, "DATASET_ROOT", tmp_path / "datasets")
    for split in ("train", "val"):
        dataset.save_labelled_image(
            f"{split}.jpg",
            b"image-data",
            split,
            [{"left": 10, "top": 10, "width": 20, "height": 20}],
            100,
            100,
        )

    summary = dataset.validate_training_dataset()
    assert summary["train"] == {"images": 1, "positive_images": 1}
    assert summary["val"] == {"images": 1, "positive_images": 1}
