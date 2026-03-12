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

- **Training pipeline** — improved and functional; training runs complete successfully.
- **Training loss** — stable convergence during training.
- **Detection metrics** — weak; detection performance (e.g., bbox AP) is below expectations.
- **Segmentation metrics** — near-zero segmentation performance (segm AP) for detailed subclasses, indicating that the current annotations are not sufficient for reliable multi-class segmentation.
- **Manual dataset inspection** — performed using a COCO dataset viewer; findings suggest that annotation quality is a major limiting factor.
- **Dataset status** — the dataset may require cleaning, relabeling, or a better source for detailed subclass segmentation.

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
│
├── notebooks/
│   ├── train_model.py             # Unified training script
│   ├── training_pipeline.ipynb    # Training and analysis notebook
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
│   └── DATA_AUGMENTATION_GUIDE.md             # Augmentation guide
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

---

## Training Pipeline

### Quick start

```bash
cd notebooks
python train_model.py
```

Or use the Jupyter notebook:

```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

**Configuration:** Edit `CONFIG` in `train_model.py` or `training_pipeline.ipynb`:

```python
CONFIG = {
    "data_root": "../data",
    "ann_file_train": "../data/splits/train.json",
    "ann_file_val": "../data/splits/val.json",
    "output_dir": "../checkpoints_medical_aug",
    "batch_size": 4,
    "epochs": 50,
    "lr": 0.005,
    "image_size": (512, 512),
    "use_medical_augmentation": True,
    "preserve_marker": True,
}
```

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
| Detection (bbox) | Weak; below expectations |
| Segmentation (subclasses) | Near-zero; annotation quality limits performance |
| Inference | Pipeline outputs wound area, infection presence, and confidence |

**Note:** The inference output structure includes `findings` for subclasses (edema, hyperemia, necrosis, etc.), but these should not be interpreted as reliable multi-class segmentations given the current dataset limitations.

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
| `notebooks/train_model.py` | Unified training, evaluation, and inference |
| `notebooks/training_pipeline.ipynb` | Interactive training and analysis |
| `scripts/build_wound_focus_dataset.py` | Safe image renaming and mapping pipeline |
| `scripts/build_wound_only_dataset.py` | Wound-only COCO, infection labels, splits |
| `scripts/apply_augmentation_only.py` | Offline augmentation |
| `scripts/augmentation_strategy.py` | Medical augmentation strategy |

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
