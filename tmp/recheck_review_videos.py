from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import cv2
from PIL import Image, ImageDraw


ROOT = Path(r"D:\TransitionX\test_records\test_records\test_records")
OUT = Path("tmp/leak_archive_review/recheck")
PLAN = Path("tmp/leak_archive_review/move-plan.json")
TIMES = (.025, .22, .55, .83, .995)
W, H = 275, 250
if "--remaining" in sys.argv:
    OUT = OUT / "remaining"
    TIMES = (0, .06, .16, .31, .49, .66, .8, .91, .975, .999)
    W, H = 190, 181


def render(entry):
    src = ROOT / "Needs review" / entry["file"]
    if not src.is_file():
        src = ROOT / entry["file"]
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sheet = Image.new("RGB", (W * len(TIMES), H + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 1), f'{entry["index"]:03d} | {entry["file"]}', fill="black")
    for col, fraction in enumerate(TIMES):
        frame_number = int((n - 1) * fraction)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            draw.text((col * W + 5, 60), "Could not decode frame", fill="black")
            continue
        h, w = frame.shape[:2]
        crop = frame[round(.055 * h):round(.965 * h), round(.23 * w):round(.77 * w)]
        thumb = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
        sheet.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (col * W, 17))
        draw.text((col * W + 5, H + 18), f"{frame_number / fps:.1f} s", fill="black")
    cap.release()
    return sheet


cv2.setNumThreads(1)
OUT.mkdir(parents=True, exist_ok=True)
entries = [e for e in json.loads(PLAN.read_text()) if e["bucket"] == "Needs review"
           and ("--remaining" not in sys.argv or (ROOT / "Needs review" / e["file"]).is_file())]
with ThreadPoolExecutor(max_workers=6) as pool:
    for start in range(0, len(entries), 6):
        rows = list(pool.map(render, entries[start:start + 6]))
        sheet = Image.new("RGB", (W * len(TIMES), (H + 34) * len(rows)))
        for row, tile in enumerate(rows):
            sheet.paste(tile, (0, (H + 34) * row))
        sheet.save(OUT / f"batch-{start // 6:02d}.jpg", quality=92)
        print(f"Decoded {start + len(rows)} / {len(entries)}", flush=True)
