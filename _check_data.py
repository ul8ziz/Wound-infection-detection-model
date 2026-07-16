import json
from pathlib import Path

root = Path("e:/GitHub/Wound-infection-detection-model")

# augmented/ dir
aug = root / "data/wound_focus_clean/augmented"
print("augmented/ contents:")
for item in sorted(aug.iterdir()):
    sub = list(item.iterdir()) if item.is_dir() else []
    first = sub[0].name if sub else None
    kind = "DIR" if item.is_dir() else "FILE"
    print(f"  {kind} {item.name}/ ({len(sub)} items) first={first}" if item.is_dir() else f"  {kind} {item.name}")

# train_augmented.json
j = json.loads((root / "data/wound_focus_clean/augmented_marker/train_augmented.json").read_text(encoding="utf-8"))
imgs = j["images"]
print(f"\ntrain_augmented.json: {len(imgs)} images")
print("First 3 file_name:")
for im in imgs[:3]:
    fn = im["file_name"]
    full = root / "data/wound_focus_clean/augmented_marker" / fn
    print(f"  {fn}  exists={full.exists()}")

# Check what the labels reference
print("\nLabel files sample:")
ldir = root / "data/wound_focus_clean/augmented_marker/labels"
for lf in sorted(ldir.iterdir())[:3]:
    print(f"  {lf.name}")
