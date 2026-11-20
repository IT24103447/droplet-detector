from pathlib import Path
import cv2
from PIL import Image, ImageDraw

root = Path(r"D:\TransitionX\test_records\test_records\test_records")
videos = sorted(root.glob("*.mp4"))
sheet = Image.new("RGB", (960, 156 * 8), "white")
draw = ImageDraw.Draw(sheet)
for row, index in enumerate([5, 30, 66, 91, 129, 170, 218, 260]):
    path = videos[index]
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    draw.text((2, row * 156), f"{index:03} {path.name}", fill="black")
    for col, fraction in enumerate([0.01, .4, .7, .97]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((count - 1) * fraction))
        ok, frame = cap.read()
        if ok:
            thumb = cv2.resize(frame, (240, 135))
            sheet.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (col * 240, row * 156 + 17))
    cap.release()
sheet.save("tmp/leak_inventory_probe.jpg", quality=92)
