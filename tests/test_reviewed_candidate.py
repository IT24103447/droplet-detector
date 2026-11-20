import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.train_reviewed_candidate import build_dataset


class ReviewedDatasetTests(unittest.TestCase):
    def records(self, root):
        records=[]
        for split in ("train", "val"):
            path=root / f"{split}.jpg"
            Image.new("RGB", (1000,800)).save(path)
            records.append(dict(image=str(path), split=split, group=split, sha256=split,
                                crop=[100,100,600,600], boxes_preview_xyxy=[[100,100,200,200]],
                                review_status="ai_visually_reviewed_provisional"))
        return records

    def test_preview_boxes_map_to_original_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root / "annotations.json"
            manifest.write_text(json.dumps(self.records(root)))
            build_dataset(manifest, root / "dataset")
            values=(root / "dataset/labels/train/train.txt").read_text().split()
            self.assertEqual(values[0], "0")
            self.assertEqual(list(map(float,values[1:])), [.25,.3125,.1,.125])
            build_dataset(manifest, root / "crop_dataset", specimen_crop=True)
            values=(root / "crop_dataset/labels/train/train.txt").read_text().split()
            self.assertEqual(list(map(float,values[1:])), [.3,.3,.2,.2])
            with Image.open(root / "crop_dataset/images/train/train.jpg") as image:
                self.assertEqual(image.size,(500,500))

    def test_rejects_video_leakage_and_unreviewed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            records=self.records(root)
            records[1]["sha256"]=records[0]["sha256"]
            manifest=root / "annotations.json"
            manifest.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "cross splits"):
                build_dataset(manifest,root / "leakage")
            records=self.records(root)
            records[0]["review_status"]="unreviewed"
            manifest.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError,"explicit review"):
                build_dataset(manifest,root / "unreviewed")


if __name__ == "__main__":
    unittest.main()
