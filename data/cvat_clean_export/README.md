# CVAT exports (new / cleaned data only)

Use this tree for **new** CVAT exports and cleaned annotations.  
Do **not** modify `data/original_data/` from this workflow; keep the legacy dataset intact.

Subfolders:

- `tasks/` — task-style exports (e.g. per-task folders or archives).
- `coco/` — COCO-format dumps.
- `splits/` — optional train/val/test file lists or JSON for the new pipeline.

See [cvat/CVAT_SETUP.md](../../cvat/CVAT_SETUP.md) for Docker and [cvat/setup_cvat.py](../../cvat/setup_cvat.py).
