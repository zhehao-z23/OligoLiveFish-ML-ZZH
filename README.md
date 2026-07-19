# OligoLiveFish: ND2 to cleaned SPT trajectories

Current production version: **v4.1.5-experiment-profiles**.

This repository converts a time-lapse Oligo-LiveFISH ND2 acquisition into
single-cell, sub-pixel trajectories. The documented scope ends at trajectory
QC; machine learning and deep learning are intentionally excluded.

```text
ND2
  -> micro-SAM nucleus instances and crop QC
  -> multi-channel single-cell TIFF + exact micro-SAM mask association
  -> Fiji drift correction and channel separation
  -> drift-aligned micro-SAM nucleus support
  -> experiment-profile-locked anchor trajectories
  -> one static irregular ROI per accepted anchor
  -> ROI-restricted G/R/P 2-D Gaussian SPT
  -> deterministic longest-track cleaned baseline
  -> Python spatial QC + MATLAB time-coloured PNG/SVG
```

v4 replaces the v3 production path. In particular, it does **not** rebuild the
nucleus boundary from Fiji intensity with Otsu, and it does **not** use
reference/candidate distance matching or the old 2,000 nm filter for the final
baseline. See [CHANGELOG.md](CHANGELOG.md) for the migration record.

## Requirements

- Windows PowerShell examples below; macOS/Linux may use equivalent shell paths.
- Python 3.13 (verified with 3.13.14).
- MATLAB with `matlab -batch` support.
- Fiji/ImageJ with **Correct 3D drift** installed.
- CUDA is recommended for micro-SAM, but CPU is supported.

Python dependencies are pinned in
[requirements_nd2_to_traj.txt](requirements_nd2_to_traj.txt). The complete venv
guide is [REQUIREMENTS_ND2_TO_TRAJ.md](REQUIREMENTS_ND2_TO_TRAJ.md).

## Locked experiment profiles

The ND2 must normalize to `T,Z,C,Y,X` and contain calibrated pixel size and
frame interval metadata. Every trajectory command requires exactly one of the
two profiles below; the profile validates the crop before Fiji/MATLAB starts,
sets the biological labels, and locks the anchor. There is no free-form anchor
override.

| Profile | C0 | C1 | C2 | C3 | Locked anchor |
| --- | --- | --- | --- | --- | --- |
| `chr3_sites_2_3_4` | Hoechst nucleus | Site 4, chr3:198M, A647 | Site 2, chr3:195M, A488 | Site 3, chr3:195.7M, A565 | C2 / green / Site 2 |
| `dsb_53bp1_site1_site2` | 53BP1 / green | Site 1 / yellow | Site 2 / purple | — | C2 / purple / Site 2 |

Although the biological meanings differ, both profiles deliberately lock raw
zero-based channel `C2` as the anchor. The four-channel Chr3 profile requires
the filename evidence `chr3`, `195M`, `195.7M`, `198M`, `488`, `565`, and `647`.
The three-channel DSB profile requires `DSB` or `53BP1` in the filename. A
channel-count or filename mismatch is a hard error.

The Step-3 `*_metadata.json` sidecar is also mandatory. Profile validation
checks the original ND2 channel-name order: `405/640/488/561` for the Chr3
profile and `GFP/RFP/Cy5` for the DSB profile. The TIFF channel count, sidecar
count, raw indices, and source-name evidence must all agree.

All G/R/P channels must have the same frame count, frame interval, pixel size,
and corrected image shape. A different acquisition order requires editing the
channel mapping in
`trajectory_extraction/pipeline/headless_Macro_first_steps_for_published.ijm`;
do not relabel unknown channels just to make a run complete.

## One-time setup

Run from the cloned repository root. Replace every value marked `REPLACE`.

```powershell
$Repo = "D:\path\to\OligoLiveFish-ML-ZZH"            # REPLACE
$Raw = "D:\path\to\raw_nd2"                         # REPLACE
$Work = "D:\path\to\analysis"                       # REPLACE
$Fiji = "D:\path\to\Fiji.app\ImageJ-win64.exe"      # REPLACE
$Matlab = "matlab"                                     # KEEP if MATLAB is on PATH; otherwise REPLACE

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$BasePython = (Get-Command python -ErrorAction Stop).Source
& $BasePython -m venv "$Repo\.venv_nd2_to_traj"
$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "venv creation failed" }

& $Python -m pip install --upgrade pip
& $Python -m pip install -r "$Repo\requirements_nd2_to_traj.txt"
& $Python -c "import sys, nd2, micro_sam, numpy, scipy, skimage, tifffile, torch; assert sys.prefix != sys.base_prefix; print(sys.executable)"
& $Python -m pip check
& $Matlab -batch "disp(version)"
if (-not (Test-Path -LiteralPath $Fiji)) { throw "Fiji not found: $Fiji" }
```

Activation is optional because every command calls the venv executable through
`$Python`. In each new PowerShell session, set `$Repo`, `$Raw`, `$Work`, `$Fiji`,
`$Matlab`, `$Python`, and the two UTF-8 environment variables again.

## Step 1 — segment nuclei from one ND2

Use one representative FOV first. `<ND2_stem>` is a placeholder, not a literal
filename.

```powershell
$Nd2 = "$Raw\<ND2_stem>.nd2"                     # REPLACE
$CropRoot = "$Work\01_cell_crops"                # KEEP or REPLACE
if (-not (Test-Path -LiteralPath $Nd2)) { throw "ND2 not found: $Nd2" }
New-Item -ItemType Directory -Force -Path $CropRoot | Out-Null

& $Python "$Repo\nucleus_segmentation\crop_nuclei_sam.py" $Nd2 `
  --device auto `
  --nucleus-channel 0 `
  --margin 30 `
  --min-area 1000 `
  --max-area 200000 `
  --border-margin 5 `
  --mask-border-margin 0 `
  --segmentation-mode apg `
  --model-type vit_b_lm `
  --output-root $CropRoot
```

The script creates `$CropRoot\<ND2_stem>\`; do not create that stem-specific
folder yourself. The first run may download the selected micro-SAM model.

Important outputs:

```text
<ND2_stem>_crops.json
<ND2_stem>_mask_<N>.tif
cell_id_mapping.csv
cell_crop_quality_metrics.csv
filtered_bad_qc_candidates.csv
visualizations\
```

Every crop JSON row now records its exact `microsam_mask`. The mask TIFF is a
repeated `TYX` instance mask, not a new intensity-based segmentation.

### Step 1 parameters

| Parameter | Default/tutorial | Meaning |
| --- | ---: | --- |
| `input` | required | One ND2 file, or a directory searched recursively for ND2 files. |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `--nucleus-channel` | `0` | Zero-based nucleus channel. |
| `--margin` | `30` px | Context around an accepted nucleus crop. |
| `--min-area` | `1000` px | Minimum nucleus/split-piece area. |
| `--max-area` | `200000` px | Maximum final nucleus area. |
| `--border-margin` | `5` px | Reject centroids close to the FOV edge. |
| `--mask-border-margin` | `0` px | Reject a mask touching the image edge; `-1` disables. |
| `--segmentation-mode` | `apg` | `apg` or `amg`. |
| `--model-type` | `vit_b_lm` | micro-SAM checkpoint. |
| `--output-root` | ND2 parent | Parent of the automatically created FOV folder. |

## Step 2 — inspect nucleus and crop QC

Before exporting or batching, inspect:

```text
visualizations\seg_overview.png
visualizations\crop_grid.png
visualizations\suppression_demo.png
visualizations\all_channels_demo.png
cell_crop_quality_metrics.csv
filtered_bad_qc_candidates.csv
```

Confirm that nuclei are separated, non-truncated, and not debris; touching
nuclei should be split and fragmented masks should be merged appropriately.
The locked low-quality rule rejects only when both are true:

```text
ch0_contrast < 0.085 AND ch0_boundary_grad < 1.70
```

If QC fails, change only the relevant Step 1 parameter and rerun into a fresh
`$Work` directory. Do not mix parameter sets in one folder.

## Step 3 — export single-cell TIFFs

```powershell
$FovCropDir = "$CropRoot\<ND2_stem>"              # REPLACE
if (-not (Test-Path -LiteralPath $FovCropDir)) { throw "FOV folder not found" }
& $Python "$Repo\nucleus_segmentation\save_crops.py" $FovCropDir
```

This creates each `<ND2_stem>_<N>.tif` and
`<ND2_stem>_<N>_metadata.json`. The sidecar includes the exact source ND2,
crop geometry, calibration, and associated micro-SAM mask. Do not move the ND2
between Steps 1 and 3. Treat a missing or implausible pixel size/frame interval
as a failed export.

To export all already-QC-approved FOVs, pass `$CropRoot` instead.

## Step 4 — run v4 on one cell

Use a real TIFF written by Step 3 and explicitly select its biological profile.
For the manuscript Chr3 batch, Site 2 / A488 / raw C2 is locked automatically.

```powershell
$CropTif = "$FovCropDir\<ND2_stem>_<N>.tif"       # REPLACE
if (-not (Test-Path -LiteralPath $CropTif)) { throw "Crop TIFF not found: $CropTif" }
$Analysis = [System.IO.Path]::ChangeExtension($CropTif, $null)

& $Python "$Repo\trajectory_extraction\run_full_pipeline_v4.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --matlab-workers 1 `
  --experiment-profile chr3_sites_2_3_4
```

Do not create `$Analysis`; Fiji and the runner create it beside the crop TIFF.
On Windows, a Unicode/Chinese parent directory is supported through a temporary
ASCII directory link. The TIFF filename itself must be ASCII and retain `.tif`.

### Fixed v4 analysis rules

1. Fiji drift-corrects and separates G/R/P channels.
2. The crop-associated micro-SAM instance mask is aligned independently to
   every Fiji-corrected frame using phase correlation and dilated by 5 px.
3. Stage 1 receives that aligned mask through `--nucleus-mask`; it does not
   replace it with Otsu. The profile-locked anchor is rejected when more than 10% of its
   checkable positions are outside the mask.
4. For every retained anchor, its complete path is rasterized into one connected
   centerline, dilated by 5 px, intersected with frame 1 of the aligned/dilated
   micro-SAM support, and reused unchanged for all frames. This is a **static
   irregular Union ROI**, not a moving per-frame box.
5. MATLAB peak centers are admitted only inside that ROI. Pixels outside the ROI
   are not zeroed before Gaussian fitting, so a near-edge detection retains its
   full fitting window. In the bundled MATLAB code, `boxr=9` is the odd fit-box
   width (half-width 4 px); therefore ROI dilation may not be below 5 px.
6. All ROI-restricted SPT candidates are exported. They are not filtered by a
   reference-distance threshold and no greedy reference assignment is used.
7. The cleaned baseline for each allele/channel is the candidate with the most
   points. Ties use: largest frame span, earliest first frame, then smallest
   candidate number. Missing channels are recorded explicitly.

### Metadata-to-physics max-step model

Unless `--max-step-px` is supplied, v4 derives the linking radius from TIFF
metadata and declared physical priors:

```text
tau = frame_gap * frame_interval_s
r_p(tau) = sqrt[-4 ln(1-p) * (D_star * tau^alpha + sigma_loc^2)]
max_step_px = ceil[(r_p / pixel_size_um) / rounding_increment] * rounding_increment
```

Locked defaults are `D*=0.0041 um^2/s^alpha`, `alpha=0.38`, `p=0.995`,
`sigma_loc=0 nm`, adjacent `frame_gap=1`, and upward rounding to `0.05 px`.
For the validated FOV15 metadata (`1.0114487 s/frame`, `0.1083333 um/px`), the
theoretical value is `2.726893 px` and the operational value is `2.75 px`.

`trackMem=3` is unchanged. The current MATLAB linker uses the same operational
radius after a gap; gap-specific radii are written to the audit JSON as
sensitivity values but are **not** silently applied.

### Step 4 parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `input_path` | required | Step 3 crop TIFF; with `--no-fiji`, an existing Fiji analysis directory. |
| `--fiji-bin` | `fiji` | Fiji executable. |
| `--matlab-bin` | `matlab` | MATLAB executable/command. |
| `--matlab-workers` | `1` | Concurrent G/R/P MATLAB groups (`1`–`3`). |
| `--matlab-save-filter-images` | off | Retain large filtered movies in MAT files. |
| `--experiment-profile` | required | `chr3_sites_2_3_4` or `dsb_53bp1_site1_site2`; locks channel identity and anchor. |
| `--microsam-mask` | associated mask | Explicit mask override; normally do not use. |
| `--mask-dilation-px` | `5` | Expansion after drift alignment. |
| `--roi-dilation-px` | `5` | Anchor centerline expansion; minimum `5`. |
| `--d-star` | `0.0041` | Anomalous diffusion prior in `um^2/s^alpha`. |
| `--alpha` | `0.38` | Anomalous exponent. |
| `--coverage-probability` | `0.995` | Radial quantile `p`. |
| `--localization-error-nm` | `0` | Per-axis localization error term. |
| `--max-step-frame-gap` | `1` | Lag used to calculate the one operational radius. |
| `--max-step-rounding-px` | `0.05` | Upward rounding increment. |
| `--max-step-px` | model | Explicit positive override; recorded in the audit. |
| `--no-fiji` | off | Reuse existing Fiji outputs. The original crop is still required for mask alignment. |
| `--crop-tif` | inferred | Original crop for `--no-fiji` when `<analysis>.tif` is unavailable. |

Changing a scientific parameter requires a fresh run and a recorded reason.
The runner removes only its own generated profile-namespaced v4 subdirectories before
a rerun; archive a completed result first if it must be retained.

## Step 5 — inspect automatic QC and cleaned baselines

Step 4 generates QC automatically. No review script is required.

```text
$Analysis\anchor_roi_v4_<experiment_profile>\figures\python_spatial_qc\
  01_anchor_mask_roi_overview.png
  02_all_candidates_fixed_coordinates.png
  03_candidate_length_and_baseline.png

$Analysis\anchor_roi_v4_<experiment_profile>\figures\matlab_longest\
  allele_<N>_<marker>_longest_spt.png
  allele_<N>_<marker>_longest_spt.svg
```

The Python figures show the corrected-channel max projection, aligned micro-SAM
support, static ROI, all candidates in one fixed corrected coordinate system,
and the chosen longest tracks. The reference path is shown as a dashed gray
trajectory; it is a spatial prior, not an SPT result.

MATLAB produces one plot only when a cleaned baseline exists. Coordinates are
centered on the trajectory median, Y is inverted to Cartesian orientation, and
line colour encodes acquisition time. SVGs use explicit RGB segments to prevent
the trajectory from becoming black during vector export.

Before batching, verify:

- the accepted profile-locked anchors lie inside the biological nucleus/support;
- each static ROI follows one anchor and is a single connected component;
- candidate trajectories lie inside their ROI and on visible signal;
- the longest baseline is biologically plausible, not merely long;
- `audit/max_step_model.json` contains the expected calibration and priors.

Trajectory length is a baseline selection rule, not proof of trajectory quality.

## Step 6 — batch the remaining cells

First list and validate crops without running analysis:

```powershell
$Stem = "<ND2_stem>"                              # REPLACE
& $Python "$Repo\trajectory_extraction\run_batch_pipeline_v4.py" $FovCropDir `
  --crop-glob "$Stem`_*.tif" `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --experiment-profile chr3_sites_2_3_4 `
  --cell-workers 1 `
  --matlab-workers 1 `
  --dry-run
```

Every accepted crop must be a `TCYX` or `TZCYX` TIFF with the exact channel
count required by the selected profile (four for Chr3 sites 2/3/4; three for
DSB/53BP1) and an associated micro-SAM mask. Remove `--dry-run` after the list
is correct.

`--resume` is enabled by default and skips only a completed v4 manifest with
the same scientific and MATLAB-worker settings. The batch writes
`trajectory_batch_v4_<experiment_profile>_summary.csv`. Peak MATLAB processes equal
`cell-workers × matlab-workers`; start with only one parallel dimension above 1.

The batch runner exposes the same profile, mask/ROI dilation, max-step model,
explicit max-step override, MATLAB filter-image, Fiji, and MATLAB options as the
single-cell runner, plus `--crop-glob`, `--cell-workers`, `--resume/--no-resume`,
and `--dry-run`.

## Output tree

```text
<cell>.tif
<cell>_metadata.json
<associated micro-SAM mask>.tif
<cell>\
  <cell>_Nucleus.tif
  <cell>_green.tif
  <cell>_red.tif
  <cell>_purple.tif
  anchor_roi_v4_<experiment_profile>\
    run_manifest.json
    log_anchor_roi_v4.txt
    mask_alignment\
      microsam_mask_aligned_raw.tif
      microsam_mask_aligned_dilated_5px.tif
      drift_alignment.csv
      mask_alignment_audit.json
      microsam_mask_alignment_qc.png
    anchor_stage1\
      Nucleus_masks.tif
      RoiSet_<profile-locked-anchor>.zip
      <anchor-prefix>_loci<N>_traj_rela2wholeimg.csv
    static_union_rois\
      allele_<N>_loci<M>_static_anchor_roi.tif
    roi_spt\
      allele_<N>_loci<M>\matlab_result\
        <cell>_<channel>.mat
        matlab_trajectory\[GPR]_m2DGaussian_traj<N>.csv
    baseline_longest\
      allele_<N>_<marker>_longest_spt_cleaned.csv
      baseline_manifest.csv
    audit\
      max_step_model.json
      static_anchor_roi_geometry.csv
      all_candidate_trajectories.csv
      baseline_selection.csv
    figures\
      python_spatial_qc\
      matlab_longest\
    anchor_roi_v4_summary.json
```

Each trajectory CSV has 1-indexed `frame`, `x_nm`, and `y_nm`. Gaps are valid.
The `*_longest_spt_cleaned.csv` files are the v4 automated deliverable. Here,
“cleaned” means ROI-restricted SPT followed by the documented deterministic
longest-track rule; it does not mean old reference-distance matching.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `example_fov.nd2` or `<ND2_stem>` not found | Replace placeholders with an exact real filename and keep the preflight `Test-Path`. |
| Crop path has no `.tif` | Use the Step 3 TIFF, not the analysis directory, JSON, or mask. |
| No associated micro-SAM mask | Rerun Steps 1 and 3 with v4, or pass the verified exact mask with `--microsam-mask`. |
| Anchor appears outside the nucleus | Inspect `mask_alignment_qc.png` and `drift_alignment.csv`; never substitute Otsu output. |
| No profile-locked anchor remains | Inspect segmentation/alignment and signal. Do not bypass the mask merely to create output. |
| Profile validation fails | Stop. Confirm the acquisition, channel count, and filename evidence; never choose the other profile merely to make the job run. |
| ROI has more than one component | The run stops; inspect the anchor coordinates and mask intersection. |
| Fiji fails on a Chinese path | Use `run_full_pipeline_v4.py`; it creates the temporary ASCII parent-path bridge. Keep the filename ASCII. |
| No `*_Nucleus.tif` | Fiji failed before channel export; inspect the log and Correct 3D drift installation. |
| Metadata mismatch | Stop. Confirm G/R/P originate from the same corrected crop and preserve TIFF calibration. |
| MATLAB seems idle | 2-D Gaussian fitting can be silent; the runner prints a heartbeat after 60 s. |
| MATLAB process count is too high | Reduce `cell-workers` or `matlab-workers`; their product is the peak MATLAB process count. |
| SVG trajectory is black | Regenerate with `plot_longest_trajectories.m` from v4; it exports explicit RGB segments. |
| `UnicodeEncodeError` | Set `PYTHONUTF8` and `PYTHONIOENCODING` as shown; v4 entry points also configure UTF-8 streams. |

## Reproducibility

For every scientific run, retain the crop TIFF/sidecar/mask association,
`run_manifest.json`, `mask_alignment_audit.json`, `max_step_model.json`, all
audit CSVs, and the QC figures. Record any non-default parameter and why it was
changed. Do not use manually cleaned trajectories to select or filter automated
candidates; manual files may be introduced only after the run for validation.
