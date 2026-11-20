import argparse
import json
from pathlib import Path

root = Path(r"D:\TransitionX\test_records\test_records\test_records")
work = Path("tmp/leak_archive_review")
parser = argparse.ArgumentParser()
parser.add_argument("--wet", nargs="+", type=int, required=True)
parser.add_argument("--dry", nargs="+", type=int, required=True)
args = parser.parse_args()
wet, dry = set(args.wet), set(args.dry)
entries = [e for e in json.loads((work / "move-plan.json").read_text()) if e["bucket"] == "Needs review"]
assert not wet & dry and wet | dry <= {e["index"] for e in entries}
changes = []
for entry in entries:
    index = entry["index"]
    previous = "Needs review" if (root / "Needs review" / entry["file"]).is_file() else None
    if index not in wet | dry and previous:
        continue
    bucket = "Visible leaks" if index in wet else (
        "No visible leaks in sampled frames" if index in dry else "Needs review"
    )
    changes.append({**entry, "previous_bucket": previous, "bucket": bucket})
(work / "recheck" / "move-plan.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
