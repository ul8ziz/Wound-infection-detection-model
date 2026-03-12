# 🏥 Wound Infection Detection

**Detecting infection signs in surgical wounds using Deep Learning**

## ⚡ Recommended Environment

**⚠️ Important:** This project uses `.venv_cuda` with:
- **Python 3.12.10**
- **PyTorch 2.5.1+cu121** (with CUDA support)
- **CUDA 12.1**

Environment ready for GPU (NVIDIA GeForce RTX 4060 or better).

## ⭐ Organized project with Python scripts and Jupyter Notebooks

**`experiments/maskrcnn/train_model.py`** - Unified training script  
**`experiments/maskrcnn/training_pipeline.ipynb`** - Notebook for training and analysis  
**`docs/PROJECT_OVERVIEW.md`** - Detailed project overview, requirements, and dataset — reference for understanding the task and choosing the right approach (Mask R-CNN, YOLO, etc.)  
**`docs/IMPROVEMENT_ROADMAP.md`** - 4-phase improvement roadmap for low segm_AP50; coordinate/size fix, dataset, training, and model improvements

---

## 📁 Project Structure

```
Wound-infection-detection-model/
├── data/                          # Data (241 tasks)
│   ├── task_0/ ... task_240/      # Original data
│   ├── project.json
│   ├── annotations.json           # All data (COCO format)
│   ├── annotations_cleaned.json   # Cleaned annotations (after clean_dataset.py)
│   ├── splits/                    # Data splits
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── augmented_clean/             # Augmented data from cleaned (optional; do NOT use 
│
├── scripts/                        # Helper scripts
│   ├── clean_dataset.py            # Clean annotations (filter, simplify, validate)
│   ├── validate_cleaned_dataset.py # Validate annotations_cleaned.json
│   ├── visualize_cleaned_dataset.py # Visualize cleaned annotations
│   ├── coco_dataset_viewer.py      # GUI viewer for COCO bboxes/masks; --check to validate
│   ├── apply_augmentation_only.py  # Apply augmentation to data
│   └── augmentation_strategy.py   # Augmentation strategy
│
├── docs/                           # Documentation
│   ├── PROJECT_OVERVIEW.md         # Detailed project and dataset overview
│   ├── DATA_AUGMENTATION_GUIDE.md  # Augmentation guide
│   ├── IMPROVEMENT_ROADMAP.md      # 4-phase roadmap for low segm_AP50
│   └── SEGMENTATION_DEBUG_PLAN.md  # Segmentation-specific debug checks
│
├── experiments/                    # Each experiment has its folder: code + outputs (shared dataset)
│   ├── maskrcnn/                   # Mask R-CNN experiment
│   │   ├── checkpoints/            # last.pt, best_model.pth, training_results.json, training_report.md
│   │   ├── results/                 # Inference results (*_result.json)
│   │   ├── training_pipeline.ipynb  # Notebook for this experiment
│   │   ├── train_model.py           # Training script for this experiment
│   │   └── pipeline_utils.py        # Data utilities (uses scripts/augmentation_strategy.py)
│   └── yolo/                        # YOLO experiment (when added): same structure
│
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Installation

#### 🐍 Recommended: Python environment with CUDA support

**⚠️ Important:** This project uses `.venv_cuda` with Python 3.12 and PyTorch with CUDA support.

**Windows:**
```powershell
# Create environment with Python 3.12 (if not exists)
py -3.12 -m venv .venv_cuda

# Activate environment
.venv_cuda\Scripts\Activate.ps1

# Install PyTorch with CUDA
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
python -m pip install -r requirements.txt

# Setup Jupyter Kernel
python -m ipykernel install --user --name=venv_cuda --display-name="Python 3.12 (CUDA)"
```

**Linux/Mac:**
```bash
# Create environment with Python 3.12
python3.12 -m venv .venv_cuda

# Activate environment
source .venv_cuda/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install -r requirements.txt

# Setup Jupyter Kernel
python -m ipykernel install --user --name=venv_cuda --display-name="Python 3.12 (CUDA)"
```

> **💡 Best practice:** Use a separate Python environment per project with CUDA support

### Dataset cleaning (recommended before training)

To fix noisy polygons, invalid masks, and class mismatches that cause near-zero AP:

```bash
cd scripts
# Clean from CVAT tasks (default)
python clean_dataset.py --input-mode cvat --data-root ../data
# Or from existing COCO JSON
python clean_dataset.py --input-mode coco --input-file ../data/annotations.json
# Regenerate train/val/test splits from cleaned data
python clean_dataset.py --input-mode cvat --split
```

Output: `data/annotations_cleaned.json`, `data/cleaning_report.txt`. Validate and visualize:

```bash
python validate_cleaned_dataset.py
python visualize_cleaned_dataset.py --num-samples 8
python coco_dataset_viewer.py --check -a ../data/splits/val.json   # Validate dataset
python coco_dataset_viewer.py -i ../data -a ../data/splits/val.json # GUI viewer (bboxes/masks)
```

**Train on cleaned data:** If splits are missing, `train_model.py` automatically uses `annotations_cleaned.json`. After `clean_dataset.py --split`, use `data/splits/train.json` (derived from cleaned data).

See [docs/DATASET_CLEANING_REPORT.md](docs/DATASET_CLEANING_REPORT.md) for a full report of what was broken, cleaned, and how to use the pipeline.

#### Augmentation (online recommended)

Training uses **online augmentation** by default (Mode 2: clean + online). Set `data_mode` in CONFIG:

| Mode | data_mode | Description |
|------|-----------|-------------|
| 1 | `clean_only` | No augmentation |
| 2 | `clean_online_aug` | Clean + online augmentation (recommended) |
| 3 | `clean_offline_aug` | Clean + offline augmented_clean + online |

**Regenerate offline augmentation (optional):**
```bash
cd scripts
python apply_augmentation_only.py
```
Requires `data/annotations_cleaned.json`. Output: `data/augmented_clean/`. Do NOT use old `data/augmented/` — it is contaminated.

See [docs/augmentation_pipeline.md](docs/augmentation_pipeline.md) for full documentation. For `augmented_clean` inputs, outputs, and workflow, see [docs/AUGMENTED_CLEAN.md](docs/AUGMENTED_CLEAN.md).

**Exact commands (from project root):**
```bash
# 1. Clean dataset
cd scripts
python clean_dataset.py --input-mode cvat --data-root ../data

# 2. Regenerate splits (optional)
python clean_dataset.py --input-mode coco --input-file ../data/annotations_cleaned.json --split

# 3. Regenerate offline augmentation (optional, Mode 3 only)
python apply_augmentation_only.py

# 4. Training
cd ../experiments/maskrcnn
python train_model.py
# Or override data mode: python train_model.py --data-mode clean_online_aug
```

#### 📝 Verify CUDA

After installation, verify CUDA works:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

#### 📝 Manual method (Anaconda)

If using Anaconda:
```bash
# 1. PyTorch (with CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 2. Other packages
conda install opencv numpy pandas matplotlib seaborn -y
pip install -r requirements.txt
```

> **💡 Tip:** Prefer using `.venv_cuda` with Python 3.12

### 2. Usage

#### Method 1: Jupyter Notebook (recommended for training) 📓

**Use `training_pipeline.ipynb` for training — interactive interface with editable CONFIG:**
```powershell
# Activate environment
.venv_cuda\Scripts\Activate.ps1

# Run Notebook from experiment folder
cd experiments/maskrcnn
jupyter notebook training_pipeline.ipynb
```

**⚠️ Important:** In Jupyter Notebook:
1. Open `training_pipeline.ipynb`
2. Select **Kernel → Change Kernel → Python 3.12 (CUDA)**
3. Run cells — GPU will be used automatically

#### Method 2: Python script (alternative)

```bash
cd experiments/maskrcnn
python train_model.py
# Or: python train_model.py --data-mode clean_offline_aug
```

### 3. Run cells in order (in Notebook)

1. ✅ **Setup**: Import + Config
2. ⭐ **Data Loading**: Load data
3. ✅ **Model Building**: Build model
4. ⭐⭐ **Training**: Training (4-6 hours)
5. ✅ **Evaluation**: Evaluation
6. ⭐ **Inference**: Prediction and analysis

### 4. Training outputs and review

After training (from Notebook or `train_model.py` script), outputs are saved in the experiment folder under `experiments/` (e.g. `experiments/maskrcnn/checkpoints/`):

| File | Description |
|-------|--------|
| `last.pt` | Last checkpoint (for resuming or comparison) |
| `best_model.pth` | Best model by combined_AP50 (AP-based, not loss) |
| `training_results.json` | Training config, train/val loss per epoch, COCO metrics (bbox_AP50, segm_AP50), best epoch |
| `training_report.md` | Markdown report: config, loss/metric tables, loss improvement analysis |

**Review previous training outputs from command line:**
```bash
cd experiments/maskrcnn
python train_model.py --review checkpoints
# Or without plotting: --no-plot
python train_model.py --review checkpoints --no-plot
```

**From Notebook:** After running the training cell, run the next cell (plot loss and metrics curves). You can also open `training_report.md` or load `training_results.json` to review results.

---

## 📝 Project Contents

### `train_model.py` - Unified training script

This file consolidates all training functions in one place:

**Model building:**
- `build_model()` - Build Mask R-CNN model

**Training:**
- `train_one_epoch()` - Train one epoch
- `validate_one_epoch()` - Validate one epoch
- `main()` - Full training pipeline

**Evaluation:**
- `evaluate_metrics()` - Evaluate COCO metrics

**Checkpoints:**
- `save_checkpoint()` - Save checkpoint
- `load_checkpoint()` - Load checkpoint

**Inference:**
- `run_inference()` - Run inference on single image
- `run_wound_inference()` - Wound-specific inference (area + infection)

**Reports:**
- `generate_report()` - Generate comprehensive Markdown report
- `review_training_results()` - Review previous training results (load training_results.json, print summary, plot curves)

### `training_pipeline.ipynb` - Training notebook

**Setup & Configuration:**
- Import libraries
- CONFIG dictionary - Edit settings here

**Data Loading:**
- Load data from `data/splits/` or `data/annotations_cleaned.json` (do NOT use old `data/augmented/`)
- Support for augmented data

**Model Building:**
- Build model using `train_model.build_model()`
- Setup Optimizer & Scheduler

**Training:**
- Full training loop with COCO metrics evaluation each epoch
- Auto-save checkpoints (`last.pt`, `best_model.pth`)
- Save `training_results.json` and `training_report.md` in checkpoints folder
- Cell to plot train/val loss and metrics after training

**Evaluation & Inference:**
- Evaluate model
- **Section 6 (Prediction)** loads `best_model.pth` by default (best model by combined_AP50)
- You can change `INFERENCE_CHECKPOINT` to an epoch number (e.g. 15) if it gives better predictions than best_model.pth (overfitting case)
- Training saves `checkpoint_epoch_N.pth` per epoch for manual epoch selection
- Run inference on new images
- Compute wound area and detect infection

---

## ⚙️ Customization

### In `train_model.py`:

Edit `CONFIG` in the file:

```python
CONFIG = {
    # Data mode: clean_only | clean_online_aug (recommended) | clean_offline_aug
    "data_mode": "clean_online_aug",
    
    # Data paths
    "data_root": "../data",
    "ann_file_train": "../data/splits/train.json",
    "ann_file_val": "../data/splits/val.json",
    
    # Training settings (uses GPU/CUDA when available)
    "device_prefer_cuda": True,
    "output_dir": "../experiments/maskrcnn/checkpoints",
    "seed": 42,
    "batch_size": 4,
    "epochs": 50,
    "lr": 0.005,
    "image_size": (512, 512),
    
    # Medical Augmentation (ignored when data_mode="clean_only")
    "use_medical_augmentation": True,
    "preserve_marker": True,
    "intensity": "moderate"  # "light", "moderate", "aggressive"
}
```

### In `training_pipeline.ipynb`:

Current training CONFIG (after fixes applied 2026-02-28):

```python
CONFIG = {
    'epochs': 80,            # Raised from 50 for full convergence
    'learning_rate': 0.001,  # SGD linear-scaled for batch_size=2
    'batch_size': 2,
    'image_size': [1024, 1024],
    'early_stop_patience': 15,   # Raised from 7
    'early_stop_min_delta': 0.005,
    # Scheduler: LinearLR warmup (5 epochs) -> CosineAnnealingLR (75 epochs)
    # Val set: 82/18 split from data/annotations_cleaned.json (~106 val images)
}
```

---

## 📊 Outputs

### After data preparation:
- `data/annotations.json` - All data (COCO format)
- `data/splits/train.json` - Training data
- `data/splits/val.json` - Validation data
- `data/splits/test.json` - Test data
- `data/augmented_clean/` - Augmented data from cleaned (optional; do NOT use old data/augmented/)

### After training:
- `experiments/maskrcnn/checkpoints/best_model.pth` - Best model (by combined_AP50)
- `experiments/maskrcnn/checkpoints/last_checkpoint.pth` - Last checkpoint (for resume)
- `experiments/maskrcnn/checkpoints/training_results.json` - Training results
- `experiments/maskrcnn/checkpoints/training_report.md` - Full report

(Change `EXPERIMENT_NAME` in CONFIG or `train_model.py` for another experiment; dataset `data/` is shared across all experiments.)

### After inference:
```json
{
  "wound_area_cm2": 25.3,
  "has_infection": true,
  "infection_confidence": 0.87,
  "findings": {
    "edema": true,
    "hyperemia": true,
    "necrosis": false,
    "granulation": true,
    "fibrin": true
  }
}
```

---

## 🎯 What the system detects

### The 16 labels:

1. **AllWound** - Entire wound
2. **Fibrin** - Fibrin
3. **SutureZone** - Suture zone
4. **EdemaZone** - Edema (infection sign) ⚠️
5. **HyperemiaZone** - Hyperemia (infection sign) ⚠️
6. **NecrosisZone** - Necrosis (infection sign) ⚠️
7. **GranulationZone** - Granulation
8. **SizeMarker** - Size marker (3×3 cm)
9. And more...

---

## 💡 Tips

### If you get CUDA Out of Memory:
```python
# In Part 2, edit CONFIG:
CONFIG['batch_size'] = 1
CONFIG['image_size'] = [800, 800]
```

### For quick training:
```python
CONFIG['epochs'] = 10  # Instead of 50
```

### To monitor training:
Watch the output in the Notebook — you'll see the loss decrease!

---

## 📈 Expected results

With GPU (RTX 4060 or better):
- ⏱️ **Training**: 4-6 hours (50 epochs) on GPU
- ⏱️ **Training on CPU**: 20-30 hours (50 epochs) - **not recommended**
- 🎯 **mAP**: ~70-80%
- 🔍 **Infection Detection**: ~85%

**⚠️ Important:** Use `.venv_cuda` to leverage GPU and significantly reduce training time!

---

## 🆘 Troubleshooting

### ❌ CUDA not available / PyTorch CPU-only

**Problem:** PyTorch installed without CUDA support

**Fix:**
1. Ensure you use `.venv_cuda` (Python 3.12)
2. Reinstall PyTorch with CUDA:
   ```powershell
   .venv_cuda\Scripts\Activate.ps1
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

### ❌ ERROR: Unknown compiler / Preparing metadata failed

**Problem:** numpy tries to build from source (requires Visual Studio)

**Fix:**
1. Use `.venv_cuda` (Python 3.12) - has pre-built wheels
2. Or run: `pip install --only-binary :all: numpy scipy`

### ❌ ERROR: Could not install packages - WinError 32

**Problem:** pip cannot access files (in use by another process)

**Fix:**
1. **Close Jupyter Notebook** if open
2. **Close all Terminal windows**
3. Retry after closing all processes
4. Or use: `taskkill /F /IM python.exe` then retry

### ❌ ValueError: numpy.dtype size changed

**Problem:** Conflict between numpy and scipy

**Fix:**
1. Run **Part 0.5** in Notebook (fixes automatically)
2. Restart Kernel: `Kernel → Restart`

### Data loading error?
Ensure `data/` folder contains:
- `task_0/`, `task_1/`, ... `task_240/`
- `project.json`

### Loss not decreasing?
- Reduce `learning_rate` to 0.0005
- Increase `epochs` to 100
- Ensure data is correct

### segm_AP50 near zero but bbox_AP50 improves?
- **Coordinate/size mismatch**: Predictions at 1024x1024 vs GT at original image size. Fixed in `train_model.py` (resize masks, scale bboxes before COCO eval).
- See [docs/IMPROVEMENT_ROADMAP.md](docs/IMPROVEMENT_ROADMAP.md) and [docs/SEGMENTATION_DEBUG_PLAN.md](docs/SEGMENTATION_DEBUG_PLAN.md) for full diagnosis and fixes.

### Model too slow?
- Reduce `image_size`
- Reduce `batch_size`
- Use a stronger GPU

---

## 📚 References

- **Mask R-CNN**: Instance Segmentation
- **PyTorch**: Training framework
- **COCO Format**: Data format

---

## 👨‍💻 Developer

Master's thesis project - Infection detection in surgical wounds

---

**Note:** This is a research project. Do not use for real medical decisions without medical consultation!

---

## 🎉 Summary

```
1 Jupyter Notebook = Complete project
Everything organized and clear
Ready to use immediately
```

**Start now!** 🚀

**Quick method (Python script):**
```powershell
# Activate environment
.venv_cuda\Scripts\Activate.ps1

# Run training
cd experiments/maskrcnn
python train_model.py
```

**Or using Jupyter Notebook:**
```powershell
# Activate environment
.venv_cuda\Scripts\Activate.ps1

# Run Jupyter
cd experiments/maskrcnn
jupyter notebook training_pipeline.ipynb

# ⚠️ Important: Select Kernel → Change Kernel → Python 3.12 (CUDA)
```

---

## 📚 Main files

### `experiments/maskrcnn/train_model.py`
Unified Python script with all training, evaluation, and inference functions. Can be run directly or imported from other notebooks.

**Usage:**
```python
# Direct run
python experiments/maskrcnn/train_model.py

# Or import functions
from train_model import build_model, train_one_epoch, evaluate_metrics
```

### `experiments/maskrcnn/pipeline_utils.py`
Data processing and dataset creation:
- `create_dataset()` - Create PyTorch Dataset
- `make_dataloaders()` - Create DataLoaders
- `get_transforms()` - Image transforms
- `WoundDataset` - Dataset class

### `scripts/apply_augmentation_only.py`
Script to apply augmentation to data and save:
```bash
cd scripts
python apply_augmentation_only.py
```

### `docs/DATA_AUGMENTATION_GUIDE.md`
Comprehensive guide for medical augmentation strategy.

### `experiments/maskrcnn/INFERENCE_GUIDE.md` (if exists)
Guide for inference usage and analysis.
