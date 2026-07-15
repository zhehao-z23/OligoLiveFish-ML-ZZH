# Locked Cell-Crop Pipeline

This document records the locked `cell_crop` stage after the fov7 migration
check and the fov17 QC review. It is project-level documentation; run-specific
closeout folders are audit artifacts, not the canonical project interface.

## Project Location and Runtime

Use the repository-local venv documented in
[`REQUIREMENTS_ND2_TO_TRAJ.md`](REQUIREMENTS_ND2_TO_TRAJ.md). No developer-specific
project or Conda path is part of the locked workflow:

```powershell
$Repo = "D:\path\to\OligoLiveFish-ML-ZZH"                 # REPLACE: local repository clone
$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe"    # AUTO after creating the documented venv
$Work = "D:\path\to\nd2_to_trajectory_results"           # REPLACE: run-specific output parent
```

## Locked Crop Parameters

```text
device=auto
nucleus_channel=0
margin=30
min_area=1000
max_area=200000
border_margin=5
mask_border_margin=0
segmentation_mode=apg
model_type=vit_b_lm
```

Cell IDs are assigned after final filtering and deduplication by spatial order:
centroid Y from top to bottom, then centroid X from left to right:

```text
cell_001, cell_002, cell_003, ...
```

## Fixed Processing Order

The final structural mask order is:

1. Load ND2 and normalize axes to `T, Z, C, Y, X`.
2. Build the time-averaged channel-0 nucleus image.
3. Run micro-SAM instance segmentation.
4. Apply pre-merge filters:
   - `min_area`
   - `max_area * 2`
   - centroid `border_margin`
   - solidity / watershed split
5. Defer `mask_border_margin` at this stage.
6. Run `merge_adjacent_masks()`.
7. Apply `mask_border_margin` after merge.
8. Deduplicate overlapping masks.
9. Sort final masks spatially and assign `cell_###` IDs.
10. Crop each accepted nucleus with neighbor suppression.
11. Save crop JSON, mask TIFFs, crop TIFFs, metadata, and visual QC.

### Reason For The Order Fix

In fov17, current `cell_001` survived because its border-touching neighbor was
removed by `mask_border` before merge was attempted.

Observed raw micro-SAM instances:

```text
raw label 4: area=9925,  centroid=(47.19, 1455.64),  mask_edge_dist=0,  removed by mask_border
raw label 5: area=10150, centroid=(148.08, 1473.24), mask_edge_dist=88, kept as current cell_001
```

When `mask_border` is deferred until after merge, raw labels `4` and `5` merge
into one mask:

```text
area=20075, centroid=(98.20, 1464.54), solidity=0.9466, mask_edge_dist=0
```

The merged mask is then correctly removed by post-merge `mask_border`.

## Locked Posthoc Nuclear-Quality Rule

The final-crop nuclear-quality review rule is:

```python
bad_qc = (ch0_contrast < 0.085) & (ch0_boundary_grad < 1.70)
```

This is a posthoc crop-quality rule. It is not part of micro-SAM instance
generation and does not change structural mask generation.

Metric meaning:

- `ch0_contrast`: median channel-0 intensity inside the nucleus mask relative
  to the local background ring.
- `ch0_boundary_grad`: median channel-0 gradient magnitude on the nucleus-mask
  boundary.

The rule targets cells with both low nuclear contrast and weak boundary signal.

Validation:

- fov17 hits: `cell_008`, `cell_011`, `cell_013`, `cell_018`, `cell_019`,
  `cell_031`
- fov17 current non-problem false positives: `0`
- fov7 accepted-crop validation hits: `0`
- fov17 `cell_001` is a structural split/border-contact case, not a bad_qc
  calibration case.

## Visualization Convention

Future run folders should keep cell-crop visual QC under the FOV output
`visualizations/` directory, for example:

```text
<run_folder>\input\<FOV_stem>\visualizations\qc_metric_review
```

Run-specific `cell_crop_closeout/` folders may exist as historical audit
artifacts, but the final project code should not require a separate closeout
directory because it duplicates material that belongs under `visualizations/`.

For fov17, the locked bad_qc visualizations are:

```text
2026-07-07_test_crop_gpu_fov17\input\LiveFISH 3h_DSB016\visualizations\qc_metric_review\bad_qc_scatter_fov17_only.png
2026-07-07_test_crop_gpu_fov17\input\LiveFISH 3h_DSB016\visualizations\qc_metric_review\bad_qc_scatter_fov17_with_fov7_validation.png
```

The fov17 `cell_001` filter-order diagnostic is stored under:

```text
2026-07-07_test_crop_gpu_fov17\input\LiveFISH 3h_DSB016\visualizations\qc_order_review
```

## Core Files

Canonical crop code:

```text
OligoLiveFish-ML\nucleus_segmentation\crop_nuclei_sam.py
```

Canonical bad_qc code:

```text
OligoLiveFish-ML\nucleus_segmentation\cell_crop_qc.py
```

The former copies under `trajectory_extraction/pipeline/` were removed in v4;
one canonical implementation prevents segmentation and trajectory code from
silently diverging.

Project-level lock document:

```text
OligoLiveFish-ML\CELL_CROP_LOCKED.md
```

## Implementation Notes From Final FOV17 Rerun

The `2026-07-08_final_crop_gpu_fov17` rerun exposed two memory-heavy
visualization/export paths that are now fixed in the core code:

- full-FOV mask overlays use a single integer label image instead of allocating
  one full-size RGBA array per mask
- fallback ImageJ display-range estimation in `save_crops.py` uses sparse
  integer samples instead of full-FOV float arrays

These implementation fixes affect memory use only. They do not change
segmentation, mask filtering, crop pixel values, or the locked QC thresholds.
