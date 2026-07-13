# DNA trajectory extraction CLI guide

This guide is a compact trajectory-only reference. It does not start
deep-learning training. The canonical runnable ND2 tutorial and complete
parameter tables are in [README.md](README.md); environment details are in
[REQUIREMENTS_ND2_TO_TRAJ.md](REQUIREMENTS_ND2_TO_TRAJ.md).

## 1. Environment

Use the maintained venv and force UTF-8 output before every run:

```powershell
$repo = "D:\path\to\OligoLiveFish-ML-ZZH"             # REPLACE: this cloned repository
$py = "$repo\.venv_nd2_to_traj\Scripts\python.exe"    # AUTO after following REQUIREMENTS_ND2_TO_TRAJ.md
$matlab = "matlab"                                      # KEEP if on PATH; otherwise REPLACE with full matlab.exe path
$fiji = "C:\path\to\Fiji.app\ImageJ-win64.exe"         # REPLACE: Fiji executable
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Install Python dependencies in a fresh environment with:

```powershell
& $py -m pip install -r "$repo\requirements_nd2_to_traj.txt"
```

MATLAB must pass `& $matlab -batch "disp(version)"`. Fiji/ImageJ is needed only
when starting from a single multi-channel crop and using Fiji preprocessing.

## 2. Essential physical-scale rule

Stage 1 uses `auto_roi_for_published_v2.13.py`, which reads the TIFF physical
scale from metadata. Before trusting a run, confirm the Stage 1 and Stage 2
logs report the same px/um scale. If they disagree, stop the run: nm-based
matching and filtering are not scientifically valid.

## 3. Reference-channel automatic extraction

Stage 1 detects loci in one reference channel, tracks those loci through time,
then automatically treats the other two colour channels as targets. The default
remains `green`, preserving the historical workflow.

```powershell
# Default: green is reference; red and purple are targets.
& $py "$repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $analysis --matlab-bin $matlab

# Use red as reference; green and purple become targets automatically.
& $py "$repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $analysis `
  --matlab-bin $matlab `
  --reference-channel red

# Use purple as reference; green and red become targets automatically.
& $py "$repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $analysis `
  --matlab-bin $matlab `
  --reference-channel purple
```

`--reference-channel` accepts only `green`, `red`, or `purple`. It changes
Stage 1 anchoring only. Stage 2 still performs global SPT in all three colour
channels, and Stage 3 still matches each reference trajectory solely against
SPT candidates from that same colour channel.

The three stages are:

1. Stage 1 detects and tracks reference-channel loci, then tracks the other two
   channels near each reference locus.
2. Stage 2 runs MATLAB 2D-Gaussian SPT globally and writes every candidate.
3. Stage 3 makes one-to-one, same-channel reference/candidate assignments and
   writes only accepted `*_traj_m2DGaussian_cleaned.csv` files.

Important Stage 1/3 settings:

| Parameter | Meaning | Guidance |
| --- | --- | --- |
| `INTER_FRAME_MAX_NM` | Normal target-channel frame-to-frame continuity limit | Default 500 nm. |
| `INTER_FRAME_MAX_NM_RED` | Red target-channel continuity limit | Default 750 nm. |
| `REFERENCE_PROX_MAX_UM` | Maximum target distance from its reference locus | Default 3.0 um. |
| `MIN_OVERLAP_FRAMES` | Shared frames required for Stage 3 comparison | Default 5. |
| `MAX_AVG_DIST_NM` | Stage 3 maximum average reference/candidate distance | Default 2000 nm. |

If a target has no biological relation to the selected reference channel, this
reference-anchored workflow is not an appropriate selection method for it.

## 4. Starting from an ND2 field of view

Create nuclear crops first:

```powershell
$nd2 = "D:\path\to\<ND2_stem>.nd2"                    # REPLACE: one real ND2 file
$cropRoot = "D:\path\to\cell_crops"                   # REPLACE: output parent; created below
$device = "auto"                                        # KEEP for single test; use cuda for validated GPU batch
New-Item -ItemType Directory -Force -Path $cropRoot | Out-Null

& $py "$repo\nucleus_segmentation\crop_nuclei_sam.py" $nd2 `
  --device $device --nucleus-channel 0 --margin 30 `
  --min-area 1000 --max-area 200000 --border-margin 5 `
  --mask-border-margin 0 --segmentation-mode apg --model-type vit_b_lm `
  --output-root $cropRoot

$fovCropDir = "$cropRoot\<ND2_stem>"                    # REPLACE <ND2_stem>: folder created by segmentation
& $py "$repo\nucleus_segmentation\save_crops.py" $fovCropDir
```

Review `visualizations/seg_overview.png`, `crop_grid.png`, and
`cell_crop_quality_metrics.csv` before trajectory extraction.

## 5. Starting from registered three-channel TIFFs

Prepare an analysis directory without changing the original source files:

```powershell
$green = "G:\path\to\green.tif"                        # REPLACE: registered green TYX TIFF
$red = "G:\path\to\red.tif"                            # REPLACE: registered red TYX TIFF
$purple = "G:\path\to\purple.tif"                      # REPLACE: registered purple TYX TIFF
$analysis = "G:\path\to\analysis"                      # REPLACE: new output directory
& $py "$repo\trajectory_extraction\pipeline\prepare_single_channel_analysis.py" `
  --green $green `
  --red $red `
  --purple $purple `
  --output-dir $analysis `
  --stem "experiment_cell1"                              # REPLACE: ASCII output stem
```

The tool requires registered TYX inputs with equal shape, XY calibration, and
frame interval. It creates a synthetic `*_Nucleus.tif` if a true nucleus input
is not supplied. A synthetic nucleus is only a Stage 1 processing scaffold;
never use it for nucleus morphology or chromatin-density features.

## 6. Resume and batch operation

Reuse complete stages after an interruption:

```powershell
& $py "$repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $analysis `
  --matlab-bin $matlab `
  --skip-stage1 `
  --skip-stage2
```

For production crop batches, validate the list first and then run the same
top-level Fiji path used for a single cell:

```powershell
$stem = "<ND2_stem>"                                     # REPLACE: common crop filename stem
$cellWorkers = 2                                         # REPLACE after benchmarking; start with 1 or 2
$matlabWorkers = 1                                       # KEEP for cell batching; one reused MATLAB session per cell
& $py "$repo\trajectory_extraction\run_batch_pipeline_v3.py" $fovCropDir `
  --crop-glob "$stem`_*.tif" `
  --fiji-bin $fiji `
  --matlab-bin $matlab `
  --reference-channel red `
  --cell-workers $cellWorkers `
  --matlab-workers $matlabWorkers `
  --dry-run
```

Remove `--dry-run` after the accepted file list is correct. The default resume
mode skips compatible completed manifests. Total possible MATLAB processes are
`cellWorkers * matlabWorkers`; increase only one dimension at a time. MATLAB SPT
is CPU-based, whereas CUDA accelerates only the earlier micro-SAM segmentation.

## 7. Review cleaned trajectories before deleting temporary products

After all three stages have succeeded, run the read-only review utility before
removing the analysis directory. It reconstructs the Stage-3 decisions while
the reference and MATLAB candidate tracks still exist, and writes a compact
set of review artifacts elsewhere.

```powershell
& $py "$repo\trajectory_extraction\pipeline\review_trajectory_results.py" `
  --analysis-dir $analysis --output-dir $review
```

It writes `match_filter_summary.csv`, an average-distance threshold plot,
whole-image and start-aligned cleaned trajectory plots, per-channel maximum
projection overlays, a trajectory summary CSV, and a metadata manifest. The
utility does not modify the analysis inputs and does not invoke deep learning.

## 8. Hand-off checklist

1. Never edit source TIFFs on `G:`.
2. Verify Stage 1 and Stage 2 report the same physical scale.
3. Select a biologically meaningful `--reference-channel` before Stage 1.
4. Run one cell first and inspect logs and cleaned CSVs.
5. Validate representative results manually before changing thresholds or
   launching a batch.
6. Do not start DL training as part of this procedure.
