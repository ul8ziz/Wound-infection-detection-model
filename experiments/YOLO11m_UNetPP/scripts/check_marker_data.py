#!/usr/bin/env python3
import json

for fn in ["data/wound_focus_clean/train_wound_marker.json",
           "data/wound_focus_clean/val_wound_marker.json",
           "data/wound_focus_clean/test_wound_marker.json"]:
    with open(fn, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_counts = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        name = cats.get(cid, f"unknown_{cid}")
        cat_counts[name] = cat_counts.get(name, 0) + 1
    n_img = len(coco["images"])
    print(f"{fn}: {n_img} images, cats={cats}, counts={cat_counts}")
