# Project Overview: Postoperative Wound Infection Detection

**Purpose of this document:** This document is the single reference for understanding the project, its requirements, and the dataset. It can be given to an assignee or an AI model to understand the task, choose the appropriate technique (e.g. Mask R-CNN, YOLO, or others), and run experiments.

---

## 1. Project Context and Objectives

### Context

Master's thesis on detecting signs of infection in postoperative wounds using computer vision and deep learning. The project is **research/educational** and is not a clinical diagnostic tool.

### Revised Scope (Post-Dataset Review)

The project scope was refined after reviewing training results and manual inspection of dataset annotations. The dataset is more suitable for:

1. **Wound isolation / wound segmentation** — segmenting the wound region from surrounding tissue
2. **Infected vs. non-infected wound analysis** — distinguishing infected from non-infected cases
3. **Studying visual signs** — texture and wound-related patterns that differentiate infection status

It is **not currently reliable enough** to present as a robust fine-grained multi-class segmentation system for detailed infection subclasses (fibrin, granulation, edema, hyperemia, necrosis). Manual inspection suggests that many annotations for secondary subclasses are inconsistent or not precise enough for dependable multi-class segmentation training.

### Primary Objectives

1. **Detect and segment** the full wound and a **3×3 cm reference marker** in clinical photographs.
2. **Compute wound area in cm²** using the marker as a scale reference (marker area = 9 cm²).
3. **Assess infection presence** — distinguish infected vs. non-infected wounds (file names with `-not-` indicate no infection).
4. **Build a reproducible training and evaluation pipeline** — structured experiments with Mask R-CNN.
5. **Identify dataset limitations** — assess annotation quality and suitability for different tasks.

### Secondary Objectives (Data-Dependent)

- Segment clinical regions (granulation, fibrin, necrosis, edema, hyperemia, pus) **where annotations allow** — note that current annotation quality limits reliability for these subclasses.
- Derive infection indicators from segmentations where feasible.
- Output structured JSON with infection status, confidence scores, wound area in cm², and findings.

### Important Constraints

- The 3×3 cm marker shape must be preserved under any augmentation or transform; distorting it invalidates area calculation.
- Single-GPU setup (e.g. RTX 4060); reasonable image sizes (e.g. 512–1024 px) and batch size.

---

## 2. Technical Requirements (What the Chosen Technique Must Provide)

- **Input:** Clinical images (file paths or arrays), with or without augmentation.
- **Required outputs:**
  - Detection and segmentation (bounding box + mask) for the full wound and the 3×3 cm marker.
  - **Wound area in cm²** from the ratio (wound area in pixels / marker area in pixels) × 9.
  - **Infection status** (yes/no or confidence score) — derived from file naming (`-not-` convention) and/or model predictions.
  - Optional detection/segmentation of clinical regions where annotations exist and are reliable.
  - Results exported as structured JSON (area, infection status, confidence, findings).
- **Annotation format for training:** COCO (polygons in `segmentation`, plus `bbox`, `category_id`, `image_id`). Current data is exported from CVAT and converted to COCO.
- **Environment:** Python 3.12+, PyTorch 2.x, torchvision; libraries: albumentations, opencv-python, pycocotools, etc. (see `requirements.txt`). CUDA preferred for training and inference.

---

## 3. Dataset: Structure and Components

### 3.1 Dataset Interpretation

- **Polygon annotations** — available for wound regions and related structures.
- **Infection status** — file names containing `-not-` indicate **no infection**; absence of `-not-` indicates **infected**.
- **Label semantics** — many labels appear to represent broader wound-related zones rather than precise pathology masks. Manual inspection suggests that secondary subclasses (fibrin, granulation, edema, hyperemia, necrosis) are inconsistent or incomplete.
- **Conclusion:** The dataset is more suitable for **wound-region analysis** and **infection presence assessment** than for reliable fine-grained subclass segmentation. Training for detailed subclass masks may require additional annotation cleaning, relabeling, or a better source of annotations.

### 3.2 Root and Core Files

- **Data root:** `data/` at the project root. Raw data lives under `data/original_data/`.
- **`data/original_data/project.json`:** Defines the annotation labels used in CVAT (Russian names and colors). Used when converting CVAT → COCO to map names to `category_id`.

### 3.3 Category List (from project.json)

| Name (Russian) | English / Role |
|----------------|----------------|
| ВсяРана | Full wound (primary class for detection and segmentation) |
| Метка для размерности | 3×3 cm marker (scale reference for area) |
| Зона отека вокруг раны | Edema around wound — infection indicator |
| Зона гиперемии вокруг | Hyperemia — infection indicator |
| Зона некроза | Necrosis — infection indicator |
| Зона грануляций | Granulation (tissue) |
| Фибрин | Fibrin (tissue) |
| Гнойное отделяемое | Pus (infection indicator) |
| Зона шва | Suture zone |

**Note:** Annotation quality for secondary subclasses (edema, hyperemia, necrosis, granulation, fibrin, pus) varies; manual inspection suggests these are not consistently precise enough for reliable multi-class segmentation.

### 3.4 Task Folders (CVAT Export)

- **Structure:** `data/original_data/task_0/`, `data/original_data/task_1/`, … `data/original_data/task_240/` (~241 tasks).
- **Per folder:**
  - `data/`: task images (jpg/png).
  - `annotations.json`: annotations for this task only (CVAT format; shapes with points, labels).
  - `task.json`: task description in CVAT.
- Annotations in CVAT are polygons (segmentation) per class; they are later converted to COCO.

### 3.5 Unified COCO File (After Conversion)

- **File:** e.g. `data/original_data/annotations_cleaned.json` (or `annotations_raw.json`).
- **Structure (COCO format):**
  - `images[]`: each entry has `id`, `file_name` (path relative to data root, e.g. `original_data/task_0/data/2.jpg`), `width`, `height`; a field like `infection_status` may be added depending on code.
  - `annotations[]`: each entry has `id`, `image_id`, `category_id`, `segmentation` (list of polygons), `bbox` [x, y, width, height], `area`, `iscrowd`.
  - `categories[]`: each entry has `id`, `name` (category name).

### 3.6 Train / Val / Test Splits

- **Files:** `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`.
- **Content:** Same COCO structure with images (and thus annotations) split into three sets. Default ratios: 70% train, 15% val, 15% test.
- **Usage:** Training reads from `train` (and optionally augmented data), validation from `val` (validation loss and metrics), final evaluation on `test`.

### 3.7 Infection Status Convention in Filenames

- Images whose filename contains **"-not-"** are considered **no infection**; otherwise they may contain infection indicators. This convention is used for infection presence assessment and dataset analysis.

### 3.8 Wound Focus Clean Dataset

- **Path:** `data/wound_focus_clean/` — folder `images/` and `mappings/` with standardized filenames and metadata.
- **Source:** Produced by `scripts/build_wound_focus_dataset.py` from `data/original_data/`. See `docs/WOUND_FOCUS_DATASET_DOCUMENTATION.md`.

---

## 4. Success Criteria for the Chosen Technique

- **Reliable detection of wound and marker** with segmentation (mask) accurate enough for area calculation.
- **Wound area in cm²** using the 3×3 cm marker (9 cm²) as reference.
- **Infection presence assessment** — distinguish infected vs. non-infected wounds where feasible.
- **Detection of infection indicators** (edema, hyperemia, necrosis, pus, etc.) where annotations allow — **note:** current annotation quality limits reliability for these subclasses.
- **Runnable** on a single GPU with reasonable training/inference time.
- **Comparability:** Save metrics (e.g. bbox_AP50, segm_AP50, or equivalents) and training outputs (training_results.json, training_report.md) to compare different techniques.

---

## 5. Current Technique and Alternatives

- **Current:** Mask R-CNN (ResNet50-FPN) from torchvision — detection and segmentation (bbox + masks), compatible with COCO and pycocotools. Training and inference live in `experiments/maskrcnn/` with checkpoints, results, and reports saved.
- **Experimental findings:** Training runs successfully with stable loss; detection metrics are weak; segmentation metrics for detailed subclasses are near-zero. Dataset annotation quality is a major limiting factor.
- **Suggested alternatives:** YOLO (e.g. YOLOv8-seg) for speed and reasonable accuracy; other instance-segmentation models compatible with COCO. Choice depends on the trade-off between accuracy, speed, and ease of deployment — multiple techniques can be tried in separate folders under `experiments/` using the same dataset.

---

## 6. Execution Steps (Summary)

1. **Environment setup:** Python 3.12+, PyTorch + CUDA, `pip install -r requirements.txt`.
2. **Data setup:** Ensure `data/original_data/` with task folders and `project.json` exist. Run `build_wound_focus_dataset.py` for standardized image set. Optional: `apply_augmentation_only.py` for augmented data.
3. **Training:** From the experiment folder (e.g. `experiments/maskrcnn`) run the notebook or `python train_model.py`; outputs in `checkpoints/` (last.pt, best_model.pth, training_results.json, training_report.md).
4. **Review:** `python train_model.py --review checkpoints` or open the reports and JSON.
5. **Inference:** Load best model, call inference functions, save results in `results/` as JSON.
6. **Further experiments:** Copy an experiment folder, change technique/config, run training and inference, and compare results.

---

## 7. Quick Summary for Assignee / Model

| Item | Description |
|------|-------------|
| **Goal** | Detect and segment wound and 3×3 cm marker, compute area in cm², assess infection presence (infected vs. non-infected). Secondary: segment clinical regions where annotation quality allows. |
| **Dataset** | Clinical images from CVAT (task_0…task_240) under `data/original_data/`; COCO annotations; `-not-` in filename = no infection. Annotation quality limits reliable multi-class subclass segmentation. |
| **Required from technique** | Instance segmentation of wound and marker; area in cm²; infection presence assessment; JSON output; single-GPU run; preserve marker shape under augmentation. |
| **Current pipeline** | Mask R-CNN in `experiments/maskrcnn/`; data setup → training → review → inference. Training stable; detection weak; subclass segmentation near-zero due to dataset limitations. |

---

This document summarizes the project, requirements, and dataset; it can be used as a complete brief for an assignee or model to understand the task, choose the right technique, and run experiments.
