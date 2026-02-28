# Mask R-CNN experiment

This folder contains the **code and outputs** for the Mask R-CNN wound detection experiment.

- **Code:** `training_pipeline.ipynb`, `train_model.py`, `pipeline_utils.py`, `augmentation_strategy.py`
- **Outputs:** `checkpoints/` (models, training_results.json, training_report.md), `results/` (inference JSONs)
- **Data:** Shared at `../../data` (not copied here)

Run from this directory:
- Jupyter: open `training_pipeline.ipynb` (kernel cwd = this folder)
- CLI: `python train_model.py`
- Review: `python train_model.py --review checkpoints`
