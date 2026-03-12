"""
One-time script to add original_data/ prefix to file_name in annotation JSONs.
Run from project root.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PREFIX = "original_data/"


def update_file(path: Path) -> int:
    """Update file_name in images array. Returns count of updated paths."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    count = 0
    for img in images:
        fn = img.get("file_name", "")
        if fn and not fn.startswith(PREFIX):
            img["file_name"] = PREFIX + fn
            count += 1
    if count > 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return count


def main():
    files = [
        PROJECT_ROOT / "data" / "original_data" / "annotations_cleaned.json",
        PROJECT_ROOT / "data" / "original_data" / "annotations_raw.json",
        PROJECT_ROOT / "data" / "splits" / "train.json",
        PROJECT_ROOT / "data" / "splits" / "val.json",
        PROJECT_ROOT / "data" / "splits" / "test.json",
        PROJECT_ROOT / "data" / "augmented" / "annotations_augmented.json",
    ]
    total = 0
    for p in files:
        n = update_file(p)
        if n > 0:
            print(f"Updated {n} paths in {p.relative_to(PROJECT_ROOT)}")
            total += n
    print(f"Total: {total} paths updated")
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
