# Wound-Only Segmentation Baseline — Review Summary for ChatGPT

**Status:** Pipeline prepared. Final training results must be populated after running `python train_wound_only.py`.

## What Was Implemented

A clean wound-only segmentation baseline using Mask R-CNN on the standardized `wound_focus_clean` dataset.  
The revised scope focuses on a single segmentation target (`wound`) instead of fine-grained multi-class infection-region segmentation.

Implemented components include:
- wound-only COCO dataset files
- train/validation/test split support
- Mask R-CNN wound-only training pipeline
- evaluation pipeline for validation and test sets
- training plots, checkpoints, and qualitative prediction outputs
- wound-only reporting files for technical review

## Which Files Were Used

- Train: `train_wound_only.json` (257 images)
- Val: `val_wound_only.json` (57 images)
- Test: `test_wound_only.json` (55 images)
- Images root: `data/wound_focus_clean/images/`

## Best Key Metrics

**Not populated yet.**  
Run the wound-only baseline training first, then update this section with:
- best validation bbox AP50
- best validation segm AP50
- best combined metric if used
- final test metrics

## Whether Wound-Only Direction Appears More Viable

The wound-only direction is expected to be more viable than the previous multi-class segmentation attempt because it removes noisy secondary subclass annotations and focuses only on the most consistent target class: the wound region itself.

This direction is better aligned with:
- the revised project scope,
- the dataset review findings,
- and the original dataset interpretation.

Final confirmation still depends on actual wound-only training results and qualitative prediction review.

## Any Unresolved Issues

- Final wound-only training metrics have not yet been populated in this summary.
- 11 standardized images do not contain wound annotations and are excluded from the wound-only segmentation set.
- Prior review showed that secondary subclass annotations were unreliable; this should remain documented as a project limitation.
- The quality of wound-only segmentation still needs to be confirmed empirically through training and visual evaluation.

## Recommended Next Action

1. Run `python train_wound_only.py`
2. Populate the final validation and test metrics
3. Review qualitative predictions in `results_wound_only/predictions/`
4. Compare wound-only segmentation results against the earlier multi-class baseline
5. If wound-only results are acceptable, proceed to infected vs non-infected classification