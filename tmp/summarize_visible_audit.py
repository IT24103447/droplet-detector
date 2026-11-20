"""Assemble compact sheets for manual comparison of model boxes with the clips."""
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("tmp/leak_archive_review/current-model-visible")
WIDTH, HEIGHT = 194, 183
ENTRIES = [json.loads(line) for line in (ROOT / "predictions.jsonl").read_text().splitlines()]


def representative_frames(entry):
    frames = entry["sampled_frames"]
    selected = {0, len(frames) - 1}
    with_hits = [
        (i, [d for d in frame["detections"] if d["in_fabric"]])
        for i, frame in enumerate(frames)
    ]
    with_hits = [(i, hits) for i, hits in with_hits if hits]
    for i, _ in (with_hits[:1] + with_hits[-1:]):
        selected.add(i)
    for q in range(4):
        options = [
            (i, hits)
            for i, hits in with_hits
            if q * len(frames) <= 4 * i < (q + 1) * len(frames)
        ]
        if options:
            selected.add(max(options, key=lambda pair: max(d["confidence"] for d in pair[1]))[0])
    return sorted(selected)


for start in range(0, len(ENTRIES), 5):
    entries = ENTRIES[start:start + 5]
    sheet = Image.new("RGB", (8 * WIDTH, (HEIGHT + 17) * len(entries)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, entry in enumerate(entries):
        y = row * (HEIGHT + 17)
        draw.text((3, y + 2), f"{entry['index']:03d}  {entry['file']}", fill="black")
        with Image.open(ROOT / f"{entry['index']:03d}.jpg") as source:
            for col, index in enumerate(representative_frames(entry)):
                x, top = (index % 7) * WIDTH, 22 + (index // 7) * HEIGHT
                tile = source.crop((x, top, x + WIDTH, top + HEIGHT))
                sheet.paste(tile, (col * WIDTH, y + 17))
    sheet.save(ROOT / f"model-review-{start // 5}.jpg", quality=93)
