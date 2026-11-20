import argparse
import json
from pathlib import Path

SOURCE = Path(r"D:\TransitionX\test_records\test_records\test_records")
REVIEW = Path("tmp/leak_archive_review")

parser = argparse.ArgumentParser()
parser.add_argument("--wet", nargs="*", type=int, default=[])
parser.add_argument("--review", nargs="*", type=int, default=[])
args = parser.parse_args()
wet, review = set(args.wet), set(args.review)
entries = [json.loads(s) for s in (REVIEW / "screening.jsonl").read_text().splitlines()]
assert len(entries) == 268
assert {entry["index"] for entry in entries} == set(range(268))
assert not wet & review
assert wet | review <= set(range(268))
assert {entry["file"] for entry in entries} == {p.name for p in SOURCE.glob("*.mp4")}

for entry in entries:
    index = entry["index"]
    entry["bytes"] = (SOURCE / entry["file"]).stat().st_size
    if index in wet:
        entry["bucket"] = "Visible leaks"
    elif index in review or entry["samples"] < 6 or (entry["duration_s"] or 0) < 5:
        entry["bucket"] = "Needs review"
    else:
        entry["bucket"] = "No visible leaks in sampled frames"
    entry.pop("detections")

assert sum(entry["bytes"] for entry in entries) == 17_932_838_148
(REVIEW / "move-plan.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
