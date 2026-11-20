from pathlib import Path
from PIL import Image

root = Path("tmp/leak_archive_review")
for first in range(0, 268, 10):
    files = [root / f"{i:03d}.jpg" for i in range(first, min(first + 10, 268))]
    target = root / f"sheet-{first:03d}.jpg"
    if target.exists() or not all(f.exists() for f in files):
        continue
    sheet = Image.new("RGB", (1056, 196 * len(files)), "white")
    for row, file in enumerate(files):
        sheet.paste(Image.open(file), (0, row * 196))
    sheet.save(target, quality=90)
