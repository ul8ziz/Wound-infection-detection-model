# Project Overview: Postoperative Wound Infection Detection

**Purpose of this document:** This document is the single reference for understanding the project, its requirements, and the dataset. It can be given to an assignee or an AI model to understand the task, choose the appropriate technique (e.g. Mask R-CNN, YOLO, or others), and run experiments.

---

## 1. Project Context and Objectives

- **Context:** Master's thesis on detecting signs of infection in postoperative wounds using computer vision and deep learning. The project is **research/educational** and is not a clinical diagnostic tool.
- **Main task:**
  1. **Detect and segment (instance segmentation)** the full wound and a **3×3 cm reference marker** in clinical photographs.
  2. **Compute wound area in cm²** using the marker as a scale reference (marker area = 9 cm²).
- **Secondary tasks (depending on data availability):**
  - Segment clinical regions: granulation, fibrin, necrosis, edema, hyperemia, pus, etc.
  - Derive **infection indicators** from these segmentations (edema, hyperemia, necrosis = clinical infection indicators).
  - Final outputs: JSON with infection status, confidence scores, wound area in cm², and presence/absence of each indicator.
- **Important constraints:**
  - The 3×3 cm marker shape must be preserved under any augmentation or transform; distorting it invalidates area calculation.
  - Single-GPU setup (e.g. RTX 4060); reasonable image sizes (e.g. 512–1024 px) and batch size.

---

## 2. Technical Requirements (What the Chosen Technique Must Provide)

- **Input:** Clinical images (file paths or arrays), with or without augmentation.
- **Required outputs:**
  - Detection and segmentation (bounding box + mask) for the full wound and the 3×3 cm marker.
  - Optional detection/segmentation of clinical regions (edema, hyperemia, necrosis, granulation, fibrin, pus) where annotations exist.
  - **Wound area in cm²** from the ratio (wound area in pixels / marker area in pixels) × 9.
  - **Infection status** (yes/no or confidence score) based on presence of infection indicators.
  - Results exported as structured JSON (area, infection status, confidence, findings per indicator).
- **Annotation format for training:** COCO (polygons in `segmentation`, plus `bbox`, `category_id`, `image_id`). Current data is exported from CVAT and converted to COCO.
- **Environment:** Python 3.12+, PyTorch 2.x, torchvision; libraries: albumentations, opencv-python, pycocotools, etc. (see `requirements.txt`). CUDA preferred for training and inference.

---

## 3. Dataset: Structure and Components

### 3.1 Root and Core Files

- **Data root:** `data/` at the project root. All components below live inside or under it.
- **`data/project.json`:** Defines the annotation labels used in CVAT (Russian names and colors). Used when converting CVAT → COCO to map names to `category_id`.

### 3.2 Category List (from project.json)

| Name (Russian) | English / Role |
|----------------|----------------|
| ВсяРана | Full wound (primary class for detection and segmentation) |
| Метка для размерности | 3×3 cm marker (scale reference for area) |
| Зона отека вокруг раны | Edema around wound — **infection indicator** |
| Зона гиперемии вокруг | Hyperemia — **infection indicator** |
| Зона некроза | Necrosis — **infection indicator** |
| Зона грануляций | Granulation (tissue) |
| Фибрин | Fibrin (tissue) |
| Гнойное отделяемое | Pus (infection indicator) |
| Зона шва | Suture zone |
| Other | Металлоконструкция, Вторичная пигментация, Подкожная жир.кл., Фасция, Губка ВАК, Глубины раны, Сухожилие — may not all be used in current training |

In code, a subset of these is used (see `TARGET_CLASSES_NAMES` in `pipeline_utils.py`): wound, marker, edema, hyperemia, necrosis, granulation, fibrin, pus.

### 3.3 Task Folders (CVAT Export)

- **Structure:** `data/task_0/`, `data/task_1/`, … `data/task_239/` (~240 tasks).
- **Per folder:**
  - `data/`: task images (jpg/png).
  - `annotations.json`: annotations for this task only (CVAT format; shapes with points, labels).
  - `task.json`: task description in CVAT.
- Annotations in CVAT are polygons (segmentation) per class; they are later converted to COCO.

### 3.4 Unified COCO File (After Conversion)

- **File:** e.g. `data/annotations.json` (or whatever path the converter outputs).
- **Structure (COCO format):**
  - `images[]`: each entry has `id`, `file_name` (path relative to data root), `width`, `height`; a field like `infection_status` may be added depending on code.
  - `annotations[]`: each entry has `id`, `image_id`, `category_id`, `segmentation` (list of polygons), `bbox` [x, y, width, height], `area`, `iscrowd`.
  - `categories[]`: each entry has `id`, `name` (category name).
- **How it is produced:** The `convert_cvat_to_coco(data_root, output_file)` function in the notebook merges all tasks and outputs a single COCO file.

### 3.5 Train / Val / Test Splits

- **Files:** `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`.
- **Content:** Same COCO structure with images (and thus annotations) split into three sets. Default ratios: 70% train, 15% val, 15% test.
- **Usage:** Training reads from `train` (and optionally augmented data), validation from `val` (validation loss and metrics), final evaluation on `test`.
- **How produced:** `split_dataset(coco_file, output_dir, train_r=0.7, val_r=0.15, test_r=0.15)` reads the unified COCO file, shuffles images, and writes the three files.

### 3.6 Augmented Data (Optional)

- **Path:** `data/augmented/` — folder `images/` and file `annotations_augmented.json` in COCO format.
- **Content:** Augmented copies of original images with annotations copied/transformed per copy. Produced by a script such as `scripts/apply_augmentation_only.py`.
- **Usage:** Can be used for training instead of (or with) original data to increase diversity while respecting medical constraints (no strong distortion of the marker).

### 3.7 Infection Regions and Class Balance

- **Infection regions in annotations:** The dataset already contains annotated regions (polygon/segmentation) for infection indicators and tissues: edema, hyperemia, necrosis, granulation, fibrin, pus. Each region has `category_id` and `segmentation` in COCO files.
- **Class balance:** Annotation counts differ by class (e.g. edema is very sparse; hyperemia and necrosis are more frequent). This can affect performance on rare classes; augmentation or class-balancing strategies may help.

### 3.8 Infection Status Convention in Filenames

- Images whose filename contains **"-not-"** are considered **no infection**; otherwise they may contain infection indicators. This can be used as an extra signal for analysis or metrics.

---

## 4. Success Criteria for the Chosen Technique

- **Reliable detection of wound and marker** with segmentation (mask) accurate enough for area calculation.
- **Wound area in cm²** using the 3×3 cm marker (9 cm²) as reference.
- **Detection of infection indicators** (edema, hyperemia, necrosis, pus, etc.) where annotations allow, with structured JSON output (infection status, confidence, findings).
- **Runnable** on a single GPU with reasonable training/inference time.
- **Comparability:** Save metrics (e.g. bbox_AP50, segm_AP50, or equivalents) and training outputs (training_results.json, training_report.md) to compare different techniques (e.g. Mask R-CNN vs YOLO).

---

## 5. Current Technique and Alternatives

- **Current:** Mask R-CNN (ResNet50-FPN) from torchvision — detection and segmentation (bbox + masks), compatible with COCO and pycocotools. Training and inference live in `experiments/maskrcnn/` with checkpoints, results, and reports saved.
- **Suggested alternatives:** YOLO (e.g. YOLOv8-seg) for speed and reasonable accuracy; other instance-segmentation models compatible with COCO. Choice depends on the trade-off between accuracy, speed, and ease of deployment — multiple techniques can be tried in separate folders under `experiments/` using the same dataset.

---

## 6. Execution Steps (Summary)

1. **Environment setup:** Python 3.12+, PyTorch + CUDA, `pip install -r requirements.txt`.
2. **Data setup:** Ensure `data/`, `project.json`, and task folders exist → convert CVAT → COCO → call `split_dataset` → (optional) create `data/augmented/`.
3. **Training:** From the experiment folder (e.g. `experiments/maskrcnn`) run the notebook or `python train_model.py`; outputs in `checkpoints/` (last.pt, best_model.pth, training_results.json, training_report.md).
4. **Review:** `python train_model.py --review checkpoints` or open the reports and JSON.
5. **Inference:** Load best model, call inference functions, save results in `results/` as JSON.
6. **Further experiments:** Copy an experiment folder, change technique/config, run training and inference, and compare results.

---

## 7. Quick Summary for Assignee / Model

| Item | Description |
|------|-------------|
| **Goal** | Detect and segment wound and 3×3 cm marker, compute area in cm², derive infection indicators (edema, hyperemia, necrosis, etc.) from segmentations; output structured JSON. |
| **Dataset** | Clinical images from CVAT (task_0…task_239); COCO annotations after conversion with multiple classes and annotated infection regions; train/val/test splits; optional augmented data. |
| **Required from technique** | Instance segmentation of wound, marker, and clinical regions; area in cm²; infection classification; JSON output; single-GPU run; preserve marker shape under augmentation. |
| **Current pipeline** | Mask R-CNN in `experiments/maskrcnn/`; data setup → training → review → inference; option to add experiments (e.g. YOLO) in separate folders with the same dataset. |

---

This document summarizes the project, requirements, and dataset; it can be used as a complete brief for an assignee or model to understand the task, choose the right technique, and run experiments.
