# Trajectory CLI quick reference — v4.0.0-anchor-roi

The root [README.md](README.md) is authoritative. These commands assume the
venv, Fiji, and MATLAB variables defined there.

## One cell from a Step-3 crop

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --anchor-channel purple `
  --matlab-workers 1
```

## Reuse existing Fiji outputs

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" `
  --no-fiji $Analysis `
  --crop-tif $CropTif `
  --matlab-bin $Matlab
```

## Explicit max-step override

The default is the audited metadata/physical model. Override only with a
documented scientific reason:

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --max-step-px 2.75
```

`audit\max_step_model.json` records whether the operational value came from the
model or the override.

## FOV batch preflight and run

```powershell
& $Python "$Repo\trajectory_extraction\run_batch_pipeline_v4.py" $FovCropDir `
  --crop-glob "$Stem`_*.tif" `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --anchor-channel purple `
  --cell-workers 1 `
  --matlab-workers 1 `
  --dry-run

# After checking the accepted files, repeat without --dry-run.
```

## Useful inspection commands

```powershell
Get-Content "$Analysis\anchor_roi_v4\anchor_roi_v4_summary.json"
Get-Content "$Analysis\anchor_roi_v4\audit\baseline_selection.csv"
Get-Content "$Analysis\anchor_roi_v4\audit\max_step_model.json"
Get-ChildItem "$Analysis\anchor_roi_v4\baseline_longest\*_cleaned.csv"
Get-ChildItem "$Analysis\anchor_roi_v4\figures" -Recurse
```

Do not call `match_m2DGaussian_to_reference.py` after v4: reference-distance
matching is not part of the fixed anchor-ROI baseline.
