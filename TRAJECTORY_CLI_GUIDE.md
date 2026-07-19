# Trajectory CLI quick reference — v4.1.1 experiment profiles

The root [README.md](README.md) is authoritative. Every production trajectory
run must select exactly one locked biological profile. The anchor is derived
from that profile; there is no `--anchor-channel` production override.

## One cell from a Step-3 crop

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --experiment-profile chr3_sites_2_3_4 `
  --matlab-workers 1
```

For a verified three-channel DSB/53BP1 crop, replace the profile with
`dsb_53bp1_site1_site2`. Do not select a profile merely to satisfy validation.

## Reuse existing Fiji outputs

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" `
  --no-fiji $Analysis `
  --crop-tif $CropTif `
  --matlab-bin $Matlab `
  --experiment-profile chr3_sites_2_3_4
```

## Explicit max-step override

The default is the audited metadata/physical model. Override only with a
documented scientific reason:

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --experiment-profile chr3_sites_2_3_4 `
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
  --experiment-profile chr3_sites_2_3_4 `
  --cell-workers 1 `
  --matlab-workers 1 `
  --dry-run

# After checking the accepted files, repeat without --dry-run.
```

## Useful inspection commands

```powershell
$Result = "$Analysis\anchor_roi_v4_chr3_sites_2_3_4"
Get-Content "$Result\run_manifest.json"
Get-Content "$Result\audit\baseline_selection.csv"
Get-Content "$Result\audit\max_step_model.json"
Get-ChildItem "$Result\baseline_longest\*_cleaned.csv"
Get-ChildItem "$Result\figures" -Recurse
```

Do not call `match_m2DGaussian_to_reference.py` after v4. The legacy
reference-distance matcher is retained only for a separately namespaced
published-style comparison and is not part of the v4 baseline.
