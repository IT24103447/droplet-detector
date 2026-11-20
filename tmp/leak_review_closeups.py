from pathlib import Path
import json
import sys
import cv2
from PIL import Image, ImageDraw

ROOT = Path(r"D:\TransitionX\test_records\test_records\test_records")
OUT = Path("tmp/leak_archive_review/recheck")
records = {row["index"]: row["file"] for row in
           json.loads((OUT.parent / "move-plan.json").read_text())}
targets = [
    (61, .56, .35, [2, 25, 45, 66, 80]),
    (120, .56, .78, [5, 20, 53, 72, 89]),
    (129, .52, .64, [6, 50, 120, 180, 224]),
    (151, .54, .52, [5, 50, 126, 190, 228]),
    (166, .73, .4, [4, 65, 100, 122, 133]),
    (185, .6, .65, [2, 15, 38, 59, 69]),
    (198, .50, .51, [6, 50, 125, 189, 226]),
    (202, .2, .83, [3, 26, 60, 98, 117]),
    (266, .52, .50, [65, 69, 71, 73, 75]),
]
if "--remaining" in sys.argv:
    OUT = OUT / "remaining"
    targets = [
        (68, .58, .51, [0, 20, 40, 60, 80]),
        (72, .55, .51, [0, 25, 50, 75, 100]),
        (168, .50, .52, [0, 15, 30, 45, 60]),
        (185, .60, .65, [2, 15, 38, 59, 69]),
        (200, .50, .52, [0, 25, 50, 75, 100]),
        (262, .64, .60, [12, 25, 40, 55, 67]),
        (259, .47, .71, [0, 20, 40, 60, 81]),
    ]
for start in range(0, len(targets), 5):
    selection = targets[start:start + 5]
    sheet = Image.new("RGB", (1500, 226 * len(selection)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (number, x, y, times) in enumerate(selection):
        file = ROOT / "Needs review" / records[number]
        if not file.exists():
            file = ROOT / records[number]
        cap = cv2.VideoCapture(str(file), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        draw.text((3, i * 226), str(number) + " " + file.name, fill="black")
        for j, second in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                cx, cy = round((.23 + .54 * x) * w), round((.055 + .91 * y) * h)
                roi = frame[max(0, cy-200):cy+200, max(0, cx-300):cx+300]
                crop = cv2.resize(roi, (300, 200))
                sheet.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (j * 300, i * 226 + 14))
                draw.text((j * 300 + 4, i * 226 + 214), f"{second}s", fill="black")
        cap.release()
    sheet.save(OUT / f"closeups-{start}.jpg", quality=95)
