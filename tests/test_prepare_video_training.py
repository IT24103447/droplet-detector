"""Guard against video leakage and accidental invention of detection labels."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_video_training import assign_splits, extract, recording_group


class PreparationTests(unittest.TestCase):
    def test_dates_and_duplicate_contents_never_cross_splits(self):
        records = [dict(group=f"day{i}", sha256=f"{i}-{kind}", source_category=kind)
                   for i in range(12) for kind in ("visible", "unconfirmed_negative")]
        records[2]["sha256"] = records[0]["sha256"]
        assign_splits(records)
        original = [r["split"] for r in records]
        for field in ("group", "sha256"):
            for value in {r[field] for r in records}:
                self.assertEqual(len({r["split"] for r in records if r[field] == value}), 1)
        self.assertEqual({r["split"] for r in records}, {"train", "val", "test"})
        assign_splits(records)
        self.assertEqual(original, [r["split"] for r in records])

    def test_extract_keeps_frames_unlabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for i in range(20):
                writer.write(np.full((64, 64, 3), i * 10, dtype=np.uint8))
            writer.release()
            (root / "review_sheets").mkdir()
            frames = extract(dict(source=str(video), sha256="a" * 64, split="train",
                                  source_category="visible", group="sample"), root)
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(Path(r["image"]).exists() for r in frames))
            self.assertTrue(all(r["review_status"] == "unreviewed" for r in frames))
            self.assertFalse((root / "labels").exists())
            self.assertEqual(recording_group(Path("test-video-101926-1234.mp4")), "101926")


if __name__ == "__main__":
    unittest.main()
