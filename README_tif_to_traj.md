# TIFF to trajectories (v4.0.0-anchor-roi)

Use the root [README.md](README.md) for the canonical ND2-to-trajectory
tutorial. This page covers the special case where Fiji channel TIFFs already
exist.

v4 still requires the original multi-channel single-cell crop and its exact
micro-SAM instance mask. They are used to align the biological nucleus support
to the existing Fiji-corrected frames. An existing `*_Nucleus.tif` is not a
substitute for the micro-SAM mask.

## Required files

```text
<cell>.tif                         # original TCYX/TZCYX crop
<cell>_metadata.json               # recommended; records microsam_mask
<ND2_stem>_mask_<N>.tif            # exact associated instance mask
<cell>\                             # existing Fiji analysis directory
  <cell>_Nucleus.tif
  <cell>_green.tif
  <cell>_red.tif
  <cell>_purple.tif
```

G/R/P TIFF metadata must agree in frame interval, X/Y pixel size, unit, frame
count, and shape.

## Run

```powershell
$Repo = "D:\path\to\OligoLiveFish-ML-ZZH"             # REPLACE
$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe"
$Analysis = "D:\path\to\<cell>"                       # REPLACE
$CropTif = "D:\path\to\<cell>.tif"                    # REPLACE
$Matlab = "matlab"                                      # KEEP or REPLACE

& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" `
  --no-fiji $Analysis `
  --crop-tif $CropTif `
  --matlab-bin $Matlab `
  --anchor-channel purple `
  --matlab-workers 1
```

If the crop sidecar is missing or predates v4, pass the verified exact mask:

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" `
  --no-fiji $Analysis `
  --crop-tif $CropTif `
  --microsam-mask "D:\path\to\<ND2_stem>_mask_<N>.tif" `
  --matlab-bin $Matlab
```

The output is `$Analysis\anchor_roi_v4\`. The parameter definitions, physical
max-step formula, output tree, QC checklist, and rerun rules are maintained only
in the root README to avoid divergent instructions.
