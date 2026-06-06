# Wound Infection Detection

**Detecting and analyzing postoperative wound infections from clinical photographs using deep learning**

---

## Overview

This repository contains an experimental pipeline for **wound detection**, **wound segmentation**, and **infected vs. non-infected wound analysis** from clinical photographs. The project uses Mask R-CNN for instance segmentation and focuses on:

1. **Wound isolation** — segmenting the wound region from surrounding tissue  
2. **Infection presence assessment** — distinguishing infected from non-infected wounds  
3. **Dataset review and annotation-quality assessment** — evaluating the suitability of available annotations for training  
4. **Reproducible training and evaluation** — a structured pipeline for experiments

The work is part of a Master's thesis on postoperative wound infection detection. It is **research-oriented** and not intended for clinical use without proper validation.

---

## Revised Project Scope

This project's scope was refined after reviewing training results and dataset annotations. Manual inspection of the dataset revealed that:

- **Wound isolation** and **infection presence** (infected vs. non-infected) are realistic and well-supported goals.
- **Fine-grained multi-class segmentation** of infection subclasses (e.g., fibrin, granulation, edema, hyperemia, necrosis) is **not reliably achievable** with the current dataset due to inconsistent and imprecise annotations.

The project therefore focuses on wound detection, wound segmentation, and infection presence analysis rather than detailed subclass segmentation.

---

## Dataset Interpretation

The dataset consists of clinical photographs with polygon annotations in COCO format:

- **Polygon annotations** — available for wound regions and related structures.
- **Infection status** — file names containing `-not-` indicate **no infection**; absence of `-not-` indicates **infected**.
- **Label semantics** — many labels appear to represent broader wound-related zones rather than precise pathology masks. Manual inspection of annotations suggests that secondary subclasses (fibrin, granulation, edema, hyperemia, necrosis) are inconsistent or incomplete.

**Conclusion:** The dataset is more suitable for **wound-region analysis** and **infection presence assessment** than for reliable fine-grained subclass segmentation. Training for detailed subclass masks may require additional annotation cleaning, relabeling, or a better source of annotations.

---

## Objectives

1. **Isolate the wound region** — segment the wound boundary from surrounding tissue.
2. **Analyze infected vs. non-infected wounds** — study visual differences between infected and non-infected cases.
3. **Build a reproducible training and evaluation pipeline** — structured experiments with Mask R-CNN.
4. **Identify dataset limitations** — assess annotation quality and suitability for different tasks.

---

## Current Status / Experimental Findings

- **Training pipeline** — fully refactored to wound-only segmentation; `experiments/maskrcnn/training_pipeline.ipynb` is the single main training notebook.
- **Model** — Mask R-CNN ResNet-50-FPN with `num_classes=2` (background + wound).
- **Dataset** — `data/wound_focus_clean/` with pre-built train/val/test splits; validated before training.
- **Training loss** — stable convergence during training.
- **Manual dataset inspection** — performed using a COCO dataset viewer; findings suggest that annotation quality is a major limiting factor for multi-class tasks, hence the wound-only focus.
- **Multi-class subclass segmentation** — removed from the main pipeline; annotation quality was insufficient for reliable fine-grained subclass segmentation.

---

## Repository Structure

```
Wound-infection-detection-model/
├── data/
│   ├── original_data/             # Raw data (241 CVAT tasks)
│   │   ├── task_0/ ... task_240/  # Task folders with images and annotations
│   │   ├── project.json
│   │   ├── annotations_cleaned.json
│   │   └── annotations_raw.json
│   ├── splits/                    # Train/val/test splits
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── wound_focus_clean/         # Standardized dataset (wound-only, infection labels)
│   │   ├── images/
│   │   ├── annotations_wound_only.json
│   │   ├── labels_infection.json
│   │   ├── train/val/test_wound_only.json, *_images.txt
│   │   ├── mappings/
│   │   └── reports/
│   └── augmented/                 # Augmented data (optional)
│       ├── annotations_augmented.json
│       └── images/
│   └── cvat_clean_export/         # New CVAT exports only (does not replace original_data/)
│       ├── tasks/
│       ├── coco/
│       └── splits/
│
├── cvat/                          # Self-hosted CVAT tooling (Docker helper + docs)
│   ├── setup_cvat.py              # Clone upstream CVAT + docker compose; export folders
│   ├── setup_cvat_ubuntu.sh       # Optional: apt install Docker on Ubuntu/Debian
│   ├── CVAT_SETUP.md              # Installation and troubleshooting
│   └── README.md
│
├── experiments/
│   └── maskrcnn/                  # Mask R-CNN experiments
│       ├── training_pipeline.ipynb # Main wound-only training notebook (recommended)
│       ├── train_model.py         # Wound-only training (CLI + validation + helpers)
│       ├── pipeline_utils.py
│       ├── checkpoints/
│       ├── results/
│       ├── reports/
│       └── reports_wound_only/   # Improved pipeline reports
│
├── notebooks/
│   ├── pipeline_utils.py          # Data utilities and dataset classes
│   └── INFERENCE_GUIDE.md         # Inference usage guide
│
├── scripts/
│   ├── build_wound_focus_dataset.py  # Safe image renaming and mapping pipeline
│   ├── build_wound_only_dataset.py   # Wound-only COCO, infection labels, splits
│   ├── apply_augmentation_only.py    # Offline augmentation
│   └── augmentation_strategy.py     # Augmentation strategy
│
├── docs/
│   ├── DATASET_BUILD_PIPELINE.md              # Full dataset build pipeline (stages 1 & 2)
│   ├── WOUND_FOCUS_DATASET_DOCUMENTATION.md   # Stage 1: standardization details
│   ├── DATA_AUGMENTATION_GUIDE.md             # Augmentation guide
│   ├── CVAT_SETUP.md                          # Pointer to cvat/CVAT_SETUP.md
│   └── thesis/                                # Thesis manuscript, figures, and defense presentations
│
├── checkpoints/                   # Saved models (created during training)
├── requirements.txt
└── README.md
```

---

## Installation

**Recommended:** Python 3.12+ with PyTorch and CUDA support.

**Windows:**
```powershell
py -3.12 -m venv .venv_cuda
.venv_cuda\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3.12 -m venv .venv_cuda
source .venv_cuda/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Verify CUDA:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
```

### CVAT (optional, self-hosted)

To run [CVAT](https://github.com/cvat-ai/cvat) locally for new or cleaned annotations, use Docker and see **[cvat/CVAT_SETUP.md](cvat/CVAT_SETUP.md)**. Quick helper (creates `data/cvat_clean_export/` for exports **without** touching `data/original_data/`):

```bash
python cvat/setup_cvat.py --only-folders   # folders only
# or: python cvat/setup_cvat.py            # tries winget/choco to install Docker if missing (Windows), then clone + compose
#     python cvat/setup_cvat.py --no-install-docker   # skip automatic Docker install attempt
```

---

## Training Pipeline

### Wound-only training notebook (recommended)

The main training workflow is in `experiments/maskrcnn/training_pipeline.ipynb`. Run from `experiments/maskrcnn`:

```bash
cd experiments/maskrcnn
jupyter notebook training_pipeline.ipynb
```

**Kernel cwd:** Start Jupyter with `experiments/maskrcnn` as the working directory so paths resolve correctly.

### Wound-only CLI (alternative)

```bash
cd experiments/maskrcnn
python train_model.py
python train_model.py --config improved   # Improved pipeline (768px, cosine LR, lighter aug)
```

**Config presets:** Use `--config baseline` (default) or `--config improved`:

| Preset | image_size | LR schedule | Augmentation | Reports |
|--------|------------|-------------|--------------|---------|
| baseline | 512×512 | StepLR | moderate | `reports/` |
| improved | 768×768 | CosineAnnealingLR | light | `reports_wound_only/` |

**Configuration:** Edit `CONFIG` in `training_pipeline.ipynb` or `train_model.py`:

```python
CONFIG = {
    "data_root": "data/wound_focus_clean",
    "ann_file_train": "data/wound_focus_clean/train_wound_only.json",
    "ann_file_val": "data/wound_focus_clean/val_wound_only.json",
    "ann_file_test": "data/wound_focus_clean/test_wound_only.json",
    "output_dir": "checkpoints",
    "results_dir": "results",
    "reports_dir": "reports",
    "batch_size": 2,
    "epochs": 50,
    "lr": 0.001,
    "image_size": (512, 512),
    "pixels_per_cm": 26.0,  # For wound area in cm² (calibration; ~20cm FOV)
    "use_medical_augmentation": True,
    "preserve_marker": True,
}
```

**Prerequisites:** Run `build_wound_focus_dataset.py` and `build_wound_only_dataset.py` first.

**Outputs:**
- `checkpoints/` — best model, last checkpoint, training history
- `results/` — metrics, plots, qualitative predictions, `baseline_vs_improved_comparison.json` (when using improved)
- `reports/` — baseline: `wound_only_training_report.md`, `review_summary_for_chatgpt.md` (include interpretation and baseline comparison)
- `reports_wound_only/` — improved: `wound_only_improved_training_report.md`, `review_summary_for_chatgpt_improved.md`

**Validation:** Run `python train_model.py --validate-only` before training, or let training run it automatically.

**Quick test:** `python train_model.py --epochs 1` for a single-epoch sanity check.

---

## Data Preparation

The dataset build has two stages. See `docs/DATASET_BUILD_PIPELINE.md` for full documentation.

### Stage 1: Build wound focus dataset (standardization)

```bash
cd scripts
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean --copy
```

### Stage 2: Build wound-only annotations and splits

```bash
cd scripts
python build_wound_only_dataset.py --data-root ../data
```

Creates `annotations_wound_only.json`, `labels_infection.json`, `train/val/test_wound_only.json`, and split lists. See `data/wound_focus_clean/reports/dataset_build_report.md`.

### Apply augmentation

```bash
cd scripts
python apply_augmentation_only.py
```

---

## Results Summary

| Aspect | Status |
|--------|--------|
| Training | Runs successfully; loss converges |
| Wound-only segmentation | Active focus; clean baseline established |
| Multi-class subclasses | Removed from main pipeline (annotation quality) |
| Outputs | Checkpoints, COCO metrics, training curves, qualitative predictions, markdown reports |

---

## Limitations

1. **Dataset annotation quality** — subclasses are inconsistent or imprecise; not suitable for reliable fine-grained segmentation.
2. **Model performance** — detection and segmentation metrics are weak; near-zero segmentation AP for subclasses.
3. **Research use only** — not validated for clinical deployment.
4. **Infection status** — derived from file naming (`-not-` convention); no independent clinical labels.

---

## Future Work

- **Dataset cleaning** — improve annotation consistency and precision.
- **Annotation verification** — manual review and correction of subclasses.
- **Simplification** — focus on wound-only segmentation if subclass data remains insufficient.
- **Infection classification** — binary infected vs. non-infected classification as a primary task.
- **Subclass segmentation** — consider detailed subclass segmentation only if better annotations become available.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `experiments/maskrcnn/training_pipeline.ipynb` | **Main wound-only training notebook** |
| `experiments/maskrcnn/train_model.py` | Wound-only training (CLI); build_model, train/val, predict_image, visualize_prediction, reports |
| `experiments/maskrcnn/train_model.py --validate-only` | Pre-training dataset validation |
| `experiments/maskrcnn/pipeline_utils.py` | Dataset classes, augmentation, dataloader utilities |
| `scripts/build_wound_focus_dataset.py` | Safe image renaming and mapping pipeline |
| `scripts/build_wound_only_dataset.py` | Wound-only COCO, infection labels, splits |
| `scripts/apply_augmentation_only.py` | Offline augmentation |
| `scripts/augmentation_strategy.py` | Medical augmentation strategy |
| `cvat/setup_cvat.py` | Optional CVAT Docker helper; creates `data/cvat_clean_export/` |
| `cvat/setup_cvat_ubuntu.sh` | Optional: install Docker via apt on Ubuntu/Debian (`sudo`), then use `cvat/setup_cvat.py` |
| `cvat/CVAT_SETUP.md` | Self-hosted CVAT installation and export paths |

---

## Troubleshooting

**CUDA not available:**
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Data loading errors:** Ensure `data/original_data/` contains `task_0/` ... `task_240/` and `project.json`.

**Out of memory:** Reduce `batch_size` or `image_size` in CONFIG.

---

## References

- **Mask R-CNN** — Instance segmentation
- **PyTorch** — Training framework
- **COCO format** — Annotation format

---

## Disclaimer

This is a **research project.** Do not use it for clinical decisions without proper validation and medical supervision.
