# Segmentation improvement experiments

Generated: 2026-04-09 02:58:22

## Candidate table

| experiment_name | architecture | input_size | loss_type | roi_crop_mode | multi_scale_refinement | refinement_postprocess | val_best_dice | val_combined_dice | val_segm_AP75 | test_combined_dice | test_segm_AP75 | test_combined_AP75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| group_A_baseline | unetplusplus | 384x384 | focal_dice | gt_only | False | none | None | None | None | None | None | None |

CSV: `E:\GitHub\Wound-infection-detection-model\experiments\YOLO11m_UNetPP\reports\segmentation_candidate_table.csv`

## Experiment notes

### group_A_baseline

- Results dir: `E:\GitHub\Wound-infection-detection-model\experiments\YOLO11m_UNetPP\results\combined\group_A_baseline`
- Architecture: `unetplusplus`
- Input size: `384x384`
- ROI mode: `gt_only`
- Loss: `focal_dice`
- Multi-scale: `False`
- Refinement postprocess: `none`

