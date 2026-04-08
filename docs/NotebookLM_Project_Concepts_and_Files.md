# Wound Infection Detection Project — Concepts and File Guide for NotebookLM

This document is meant to be imported into [NotebookLM](https://notebooklm.google.com) as a knowledge source (course notes / summaries) for the project.

**Part 1:** High-level overview and a concept glossary. **Part 2:** Per-file (or per-group) descriptions for code and documentation in the repository. Image and per-task annotation assets are described at folder level, not as thousands of individual image/JSON paths.

---

## 1. Quick introduction

- **Domain:** Medical computer vision — clinical photographs of postoperative wounds.
- **Tasks in current scope:** Wound isolation (detection + segmentation), **infected vs. non-infected** labeling derived from filenames where applicable, **wound area in cm²** using a **3×3 cm reference marker** when available in training or inference.
- **Important limitations:** Research/educational project, not a certified clinical tool. Subclass annotation quality (fibrin, granulation, etc.) is limited; recommended pipelines focus on **wound-only** segmentation and infection presence from naming conventions.
- **Core stack:** PyTorch, torchvision (Mask R-CNN), Ultralytics YOLO (YOLO11-seg), U-Net++ via `segmentation-models-pytorch`, **COCO** annotations, **Albumentations**, **pycocotools**, OpenCV.

---

## 2. Concept glossary (study vocabulary)

### 2.1 Computer vision and deep learning

| Concept | Short meaning in this project |
|--------|--------------------------------|
| **Instance segmentation** | Per object (wound, optionally marker): pixel mask + bounding box. |
| **Semantic segmentation** | Per-pixel classification; here U-Net++ on a wound **ROI** crop. |
| **Object detection** | Bounding boxes + confidence scores; YOLO adds coarse segmentation masks. |
| **Mask R-CNN** | Faster R-CNN + mask head; here ResNet-50-FPN, single wound class (wound-only). |
| **YOLO (Ultralytics)** | Fast detector; **yolo11m-seg** for detection with instance masks. |
| **U-Net / U-Net++** | Encoder–decoder for dense segmentation; encoder e.g. EfficientNet-B1 with ImageNet weights. |
| **Transfer learning** | Pretrained encoder, then fine-tuning on wound data. |
| **ROI crop** | Crop around a YOLO box (with padding) as U-Net++ input. |
| **Two-stage pipeline** | Stage 1: YOLO localizes; stage 2: U-Net++ refines the mask inside the crop. |

### 2.2 Data and annotations

| Concept | Short meaning |
|--------|----------------|
| **CVAT** | Annotation tool; many exported tasks (`task_*`) in COCO-like JSON. |
| **COCO format** | JSON: `images`, `annotations`, `categories`; polygons in `segmentation`; `bbox` in xywh. |
| **Polygon → mask** | Rasterize polygon to a binary mask for training and evaluation. |
| **Train / val / test split** | Often 70% / 15% / 15% with a fixed seed for reproducibility. |
| **Infection label from filename** | Filenames containing `-not-` usually mean **non-infected**; absence is treated as infected in binary analysis paths. |
| **wound_focus_clean** | Standardized image names and paths after the cleanup pipeline scripts. |
| **Offline augmentation** | Extra images written to disk (with a new JSON) once before training. |

### 2.3 Evaluation

| Concept | Short meaning |
|--------|----------------|
| **Average Precision (AP)** | Precision–recall summary; **AP50** at IoU ≥ 0.5; **AP75** stricter. |
| **COCOeval** | From `pycocotools`: bbox and segmentation AP. |
| **IoU** | Intersection over union of two masks or boxes. |
| **Dice coefficient** | Common medical overlap metric: \(2|A \cap B| / (|A| + |B|)\). |
| **Balanced score** | In the YOLO + U-Net experiment: weighted mix of bbox/segm AP and mean Dice for hyperparameter tuning. |
| **TTA (test-time augmentation)** | e.g. horizontal flip on U-Net++ input and average probabilities. |

### 2.4 Imaging and geometry

| Concept | Short meaning |
|--------|----------------|
| **Normalization (ImageNet)** | Fixed mean/std for compatibility with pretrained encoders. |
| **Bilinear upscale of probabilities** | Resize probability map before thresholding vs. nearest-neighbor on hard masks only. |
| **Morphological post-processing** | Binary open/close, small-component removal, hole filling — see `postprocess.py`. |
| **pixels_per_cm** | Scale calibration from the 3×3 cm marker or a default from config. |

### 2.5 Ethics and safety

- Do not claim clinical readiness; state limitations and assumptions in reports and demos.

---

## 3. Repository layout (mental map)

- **`data/`**: Raw exports (`original_data/` CVAT tasks), splits, **`wound_focus_clean/`** (standardized images + wound-only JSON + splits), optional `augmented/` trees.
- **`scripts/`**: Dataset builders, augmentation, COCO viewer, path helpers.
- **`experiments/maskrcnn/`**: **Mask R-CNN** wound-only pipeline.
- **`experiments/YOLO11m_UNetPP/`**: **YOLO11m-seg + U-Net++** hybrid, combined inference, evaluation.
- **`cvat/`**: Optional Docker/CVAT setup and clean export folders.
- **`docs/`**: Extended documentation and reports.

---

## 4. Part 2 — File by file (and grouped assets)

Order: repo root → `docs` → `scripts` → `cvat` → `experiments/maskrcnn` → `experiments/YOLO11m_UNetPP` → notebooks → `data`.

### 4.1 Repository root

| File | Concepts and role |
|------|-------------------|
| **`README.md`** | Project overview, revised scope, dataset interpretation, folder layout, installation, Mask R-CNN wound-only status. |
| **`requirements.txt`** | Dependencies: torch, torchvision, opencv, albumentations, numpy/pandas, scikit-learn, pycocotools, **ultralytics**, **segmentation-models-pytorch**, pyyaml, jupyter, python-docx. |
| **`INSTALL_PYTHON313.md`** | Python 3.13+ environment notes if present. |

### 4.2 `docs/`

| File | Concepts and role |
|------|-------------------|
| **`PROJECT_OVERVIEW.md`** | Single reference: context, goals, constraints, technical requirements, data layout, CVAT category table (Russian / English roles). |
| **`DATASET_BUILD_PIPELINE.md`** | Stages of building the dataset (stages 1 and 2). |
| **`WOUND_FOCUS_DATASET_DOCUMENTATION.md`** | Standardization stage: naming, mappings, reports. |
| **`DATA_AUGMENTATION_GUIDE.md`** | Medically plausible augmentation strategies. |
| **`MODEL_SELECTION_AND_CHECKPOINTS.md`** | Model choice and checkpoint handling. |
| **`MODEL_SELECTION_AND_EARLY_STOPPING.md`** | Early stopping and criteria. |
| **`CVAT_SETUP.md`** | Pointer to CVAT setup (often under `cvat/`). |
| **`wound_only_training_report.md`** | Wound-only training report. |
| **`review_summary_for_chatgpt.md`** | Review summary. |
| **`Project_Progress_Report_Wound_Detection_EN.md`** | English progress report if present. |
| **`NotebookLM_Project_Concepts_and_Files.md`** | This file — NotebookLM concept index. |

### 4.3 `scripts/`

| File | Concepts and role |
|------|-------------------|
| **`build_wound_focus_dataset.py`** | Scan `task_*`, `manifest.jsonl`, infection naming rules, safe renaming, `image_mapping.json/csv`, `skipped/ambiguous` reports; optional image copy; clinical regex; raw data unchanged without `--copy`. |
| **`build_wound_only_dataset.py`** | From `annotations_cleaned` + mapping: **wound-only** COCO, `labels_infection.json/csv`, train/val/test split, validation reports. |
| **`apply_augmentation_only.py`** | One-off offline augmentation; calls `get_medical_augmentation_pipeline`; images + `annotations_augmented.json`; comments may reference legacy paths — align with current repo layout. |
| **`augmentation_strategy.py`** | Medical Albumentations design: light/moderate/aggressive intensity, preserve marker geometry, class balance notes, LongestMaxSize, PadIfNeeded, etc. |
| **`coco_dataset_viewer.py`** | Tkinter/PIL viewer for images + boxes + COCO masks; `--check` mode for dataset QA. |
| **`update_annotation_paths.py`** | One-time prefix `original_data/` on `file_name` in selected JSON files. |
| **`export_markdown_to_docx.py`** | Markdown → Word (`python-docx`), tables and paragraphs. |

### 4.4 `cvat/`

| File | Concepts and role |
|------|-------------------|
| **`setup_cvat.py`** | Docker, clone CVAT, run containers, export folders under `data/cvat_clean_export/` without touching `original_data/`. |
| **`CVAT_SETUP.md`**, **`README.md`** | Install and usage instructions. |
| **`setup_cvat_ubuntu.sh`** | Optional Ubuntu/Debian Docker helper. |

### 4.5 `experiments/maskrcnn/`

| File | Concepts and role |
|------|-------------------|
| **`train_model.py`** | Train **Mask R-CNN ResNet50-FPN** with one foreground class (wound), COCO loading, train/val loops, COCOeval, early stopping, `--config improved`, checkpoints and history. |
| **`pipeline_utils.py`** | `WoundDataset` with Albumentations, original Russian CVAT class names, `WOUND_ONLY_CLASSES`, UTF-8 workaround for pycocotools via temp file, DataLoader, transforms. |
| **`training_pipeline.ipynb`** | Main interactive Mask R-CNN (wound-only) notebook with CONFIG. |
| **`training_pipeline_ru.ipynb`** | Russian-language variant if used. |
| **`README.md`** | Commands, hyperparameter table, outputs. |

### 4.6 `experiments/YOLO11m_UNetPP/`

| File | Concepts and role |
|------|-------------------|
| **`config.yaml`** | Source of truth: data paths, YOLO classes (`wound` or `wound`+`marker`), YOLO hyperparameters (image size, epochs, SGD, Ultralytics aug), U-Net++ (encoder, focal+dice, ROI padding), **`combined`** section (thresholds, box strategy, TTA, postprocess, balanced_score_weights, area calibration). |
| **`train_model.py`** | Stages: `convert` (COCO→YOLO), `yolo`, `unet`, `combined`, optional `infection`, `all`; model builders, training loops, Markdown reports, combined eval via `combined.coco_eval`. |
| **`pipeline_utils.py`** | `coco_to_yolo_seg`, `prepare_yolo_dataset`, `WoundROIDataset`, `get_unet_transforms`, `imread_bgr_ultralytics_safe` (avoid Ultralytics `cv2.imread` patch issues), `load_config` with optional `validate_combined`. |
| **`augment_offline.py`** | Several augmented copies per training image + new COCO; medically safe Albumentations (no harsh perspective). |
| **`build_wound_marker_dataset.py`** | Two-class COCO JSONs: wound + 3×3 marker from `annotations_cleaned` with image mapping. |
| **`yolo_data/dataset.yaml`** | YOLO dataset paths for training (generated/updated with conversion). |

#### 4.6.1 `experiments/YOLO11m_UNetPP/combined/`

| File | Concepts and role |
|------|-------------------|
| **`__init__.py`** | Package marker. |
| **`config.py`** | `CombinedInferenceConfig`, `BalancedScoreWeights`, `combined_config_from_dict` — all combined-inference keys. |
| **`geometry.py`** | `xyxy_to_padded_roi`, `tight_bbox_from_binary_mask`, COCO xywh conversion. |
| **`marker.py`** | `calculate_pixels_per_cm_from_marker` from YOLO marker-class box. |
| **`postprocess.py`** | Ordered binary-mask ops: connected components, open/close, hole fill, contour smoothing; `PRESETS` for tuning. |
| **`inference.py`** | `combined_inference`: YOLO → ROI → U-Net++ → prob upscale → threshold → mask merge/NMS; internal TTA. |
| **`coco_eval.py`** | Build COCO-format predictions, run COCOeval, `balanced_score`, optional pixel metrics. |
| **`debug_viz.py`** | Save intermediate steps (ROI, prob map, GT overlays) for debugging. |
| **`error_analysis.py`** | Rule-based error taxonomy: missed_detection, poor_bbox_localization, over/under-segmentation, fragmented_mask, shifted_roi_or_mask, etc.; CSV, reports, qualitative samples. |

#### 4.6.2 `experiments/YOLO11m_UNetPP/scripts/`

| File | Concepts and role |
|------|-------------------|
| **`tune_combined.py`** | Staged grid search on combined hyperparameters; cache one low-conf YOLO forward per image; write best configs. |
| **`run_error_analysis.py`** | CLI wrapper for `run_error_analysis` on val/test. |
| **`run_combined_debug.py`** | Generate debug figure grids for a subset of images. |

#### 4.6.3 `experiments/YOLO11m_UNetPP/` — output directories (concepts)

| Path | Concepts |
|------|----------|
| **`checkpoints/yolo/`**, **`checkpoints/unet/`** | `best` / `last` weights. |
| **`results/yolo/`**, **`results/unet/`**, **`results/combined/`** | Curves, predictions, `coco_metrics.json`. |
| **`reports/`** | `training_report.md`, hybrid optimization reports if present. |

### 4.7 Jupyter notebooks (`.ipynb`)

| File | Concepts and role |
|------|-------------------|
| **`experiments/maskrcnn/training_pipeline.ipynb`** | Interactive Mask R-CNN wound-only training workflow. |
| **`experiments/maskrcnn/training_pipeline_ru.ipynb`** | Russian-language / alternate notebook. |
| **`experiments/YOLO11m_UNetPP/training_pipeline.ipynb`** | **Primary entry** for YOLO + U-Net++: load `config.yaml`, COCO→YOLO, datasets, train YOLO then U-Net++, curves, combined inference, full eval; kernel **cwd** should be `experiments/YOLO11m_UNetPP`. |

### 4.8 `data/` (as concepts, not every per-task JSON)

| Component | Concepts and role |
|-----------|-------------------|
| **`original_data/task_*`** | CVAT tasks; typically `annotations.json`, `task.json`, image folder — merge/cleanup source. |
| **`original_data/annotations_cleaned.json`** (and raw) | Project-level merged/cleaned annotations. |
| **`splits/`** | `train.json`, `val.json`, `test.json` — general COCO splits. |
| **`wound_focus_clean/images/`** | Standardized filenames. |
| **`wound_focus_clean/*_wound_only.json`** | Wound-only COCO splits. |
| **`wound_focus_clean/labels_infection.*`** | Binary infection labels from paths/names. |
| **`wound_focus_clean/mappings/`** | Old path ↔ new path mapping. |
| **`wound_focus_clean/augmented/`** | If present: expanded training for YOLO11m path per `config.yaml`. |
| **`augmented/`** (under `data/`) | Legacy path for the general augmentation script. |

### 4.9 Cursor rules (optional, for developers)

| File | Role |
|------|------|
| **`.cursor/rules/projact-roles.mdc`** | Medical project context and AI assistant constraints. |
| **`.cursor/rules/yolo11m-training-notebook.mdc`** | Keep `config.yaml` / `combined` changes in sync with `training_pipeline.ipynb`. |

---

## 5. Two suggested learning paths in NotebookLM

1. **Data path:** `README.md` → `PROJECT_OVERVIEW.md` → `DATASET_BUILD_PIPELINE.md` → `build_wound_focus_dataset.py` + `build_wound_only_dataset.py` → COCO viewer.
2. **Models path:** Mask R-CNN (`maskrcnn/README.md` + `train_model.py`) then YOLO + U-Net++ (`config.yaml` → `train_model.py` → `combined/*` → `training_pipeline.ipynb`).

---

*Index generated from the current repository layout. When new files are added, update this document or re-import it into NotebookLM.*
