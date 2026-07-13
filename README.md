# OligoLiveFish: ND2 to Cleaned DNA Trajectories

This repository converts raw, time-lapse Oligo-LiveFISH `.nd2` files into
cleaned, sub-pixel DNA-locus trajectories. This README is the canonical
tutorial for the completed preprocessing workflow only:

```text
raw ND2
  -> nucleus segmentation and single-cell crops
  -> Fiji drift correction and channel separation
  -> reference-locus tracking
  -> MATLAB 2D-Gaussian single-particle tracking
  -> same-channel trajectory matching
  -> cleaned trajectory CSV files
```

Machine learning and deep learning are intentionally outside the scope of this
guide.

## Contents

- [Requirements](#requirements)
- [Input-data contract](#input-data-contract)
- [One-time setup](#one-time-setup)
- [Step 1: segment nuclei from ND2](#step-1-segment-nuclei-from-nd2)
- [Step 2: inspect the crop QC](#step-2-inspect-the-crop-qc)
- [Step 3: export single-cell TIFFs](#step-3-export-single-cell-tiffs)
- [Step 4: extract trajectories from one cell](#step-4-extract-trajectories-from-one-cell)
- [Step 5: review the cleaned trajectories](#step-5-review-the-cleaned-trajectories)
- [Step 6: process the remaining cells](#step-6-process-the-remaining-cells)
- [All adjustable parameters](#all-adjustable-parameters)
- [Outputs](#outputs)
- [Resume and rerun safely](#resume-and-rerun-safely)
- [Troubleshooting](#troubleshooting)

## Requirements

The current production path requires all three applications below. Python and
MATLAB alone are not sufficient because the trajectory runner calls Fiji for
drift correction and channel separation.

- Python 3.13. The pinned preprocessing stack in
  `requirements_nd2_to_traj.txt` was verified with Python 3.13.14. Other Python
  versions may work, but they are not the documented reproducible environment.
- MATLAB with the `matlab` command available on `PATH`. The required `.m`
  helper files are already bundled in `trajectory_extraction/pipeline/`.
- Fiji/ImageJ with the **Correct 3D drift** command installed. On Windows, pass
  the full path to `ImageJ-win64.exe` with `--fiji-bin`; Fiji does not have to
  be on `PATH` in that case.
- A CUDA GPU is recommended for micro-SAM nucleus segmentation, but CPU and
  Apple MPS execution are supported and slower.

Only the preprocessing dependencies are listed in
[`requirements_nd2_to_traj.txt`](requirements_nd2_to_traj.txt); it
intentionally excludes ML/DL analysis packages and notebook tools. See
[`REQUIREMENTS_ND2_TO_TRAJ.md`](REQUIREMENTS_ND2_TO_TRAJ.md) for the standalone
venv setup, verification, update, and removal guide.

## Input-data contract

Before starting, confirm which supported channel contract each ND2 file uses:

1. The acquisition is a time-lapse stack that the `nd2` package can normalize
   to `T, Z, C, Y, X`.
2. Four-channel acquisitions use this order:

   | Zero-based ND2 index | Fiji channel | Role |
   | ---: | ---: | --- |
   | `0` | `C1` | nucleus stain |
   | `1` | `C2` | red locus |
   | `2` | `C3` | green locus |
   | `3` | `C4` | purple/magenta locus |

   Three-channel acquisitions use this order:

   | Zero-based ND2 index | Fiji channel | Role |
   | ---: | ---: | --- |
   | `0` | `C1` | green locus |
   | `1` | `C2` | red locus |
   | `2` | `C3` | purple/magenta locus |

   For a three-channel acquisition, the trajectory runner automatically creates
   a synthetic full-frame nucleus image required only as a Stage 1 scaffold.
   Never use that synthetic image for morphology or chromatin-density features.

3. XY pixel size and frame interval are present in the ND2 metadata. They are
   copied into the TIFF files and are required for physical-unit tracking.
4. The red, green, and purple channels image the same registered field and have
   the same shape, pixel calibration, and frame interval.

If the channel order differs, do not run with the defaults. Change both the
nucleus channel passed to Step 1 and the channel-to-colour assignments near the
end of
`trajectory_extraction/pipeline/headless_Macro_first_steps_for_published.ijm`.
Record that mapping with the run. A wrong mapping can produce plausible-looking
but scientifically incorrect trajectories.

## One-time setup

All Windows examples use PowerShell. Run them from the repository root. Replace
the paths marked `REPLACE` below with paths on the current computer; these are the only
machine-specific paths used by the tutorial.

```powershell
$Repo = "D:\path\to\OligoLiveFish-ML-ZZH"                 # REPLACE: this cloned repository
$Raw = "D:\path\to\raw_nd2"                              # REPLACE: folder containing the source ND2 files
$Work = "D:\path\to\nd2_to_trajectory_results"           # REPLACE: new/existing output parent with enough free space
$Fiji = "C:\path\to\Fiji.app\ImageJ-win64.exe"            # REPLACE: Fiji executable, not the Fiji.app directory
$Matlab = "matlab"                                         # KEEP if MATLAB is on PATH; otherwise REPLACE with matlab.exe

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
```

- `$Repo` contains this README and the code. Do not substitute the current
  repository author's path.
- `$Raw` contains the original ND2 files. The scripts read these files; they do
  not modify them.
- `$Work` is a new output parent. Create it before using `--output-root`.
- `$Fiji` is the Fiji executable, not the Fiji directory.
- `$Matlab` may remain `matlab` when that command is on `PATH`; otherwise use
  the full local `matlab.exe` path.
- The two UTF-8 variables prevent scientific-unit characters in script output
  from failing on Windows consoles configured for GBK or another legacy code
  page. Set them again in every new PowerShell session.

### Create and use a virtual environment

Create the environment once inside the repository. These commands deliberately
use the venv's full Python path for every installation; they work when the
optional Windows `py` launcher is not installed and do not require activation:

```powershell
$BasePython = (Get-Command python -ErrorAction Stop).Source # AUTO: base python.exe used only to create the venv
& $BasePython --version
& $BasePython -m venv "$Repo\.venv_nd2_to_traj"
if ($LASTEXITCODE -ne 0) { throw "venv creation failed; do not install packages" }

$Python = "$Repo\.venv_nd2_to_traj\Scripts\python.exe" # AUTO: isolated interpreter used by every later Python command
if (-not (Test-Path -LiteralPath $Python)) { throw "venv Python was not created: $Python" }

& $Python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix, 'Not running inside a venv'"
if ($LASTEXITCODE -ne 0) { throw "venv isolation check failed" }

& $Python -m pip install --upgrade pip
& $Python -m pip install -r "$Repo\requirements_nd2_to_traj.txt"
```

If either venv check fails, stop. Do **not** fall back to an unqualified
`python -m pip install`: on Microsoft Store Python it may print `Defaulting to
user installation` and modify the user's global package set.

For later sessions, set the path variables and UTF-8 settings, then reset
`$Python` to the existing venv executable. Activation is optional because every
command in this README invokes `$Python` explicitly. Do not recreate the
environment for every dataset.

On macOS or Linux, create it with `python3.13 -m venv .venv_nd2_to_traj`, activate
with `source .venv_nd2_to_traj/bin/activate`, and replace PowerShell path syntax
with the equivalent shell paths.

### Verify all executables

```powershell
& $Python --version
& $Python -c "import sys, nd2, micro_sam, numpy, scipy, skimage, tifffile, torch; assert sys.prefix != sys.base_prefix; print('Python dependencies: OK'); print('nd2:', nd2.__version__); print('CUDA available:', torch.cuda.is_available()); print(sys.executable)"
& $Python -m pip check
& $Matlab -batch "disp(['MATLAB ', version])"
Test-Path $Fiji
```

Open Fiji once in graphical mode and confirm that **Correct 3D drift** is an
available command. The headless macro will fail later if the plugin is absent.

## Step 1: segment nuclei from ND2

Start with one representative ND2 file. Do not start a folder-wide batch until
the visual QC in Step 2 passes.

```powershell
$Nd2 = "$Raw\<ND2_stem>.nd2"                   # REPLACE <ND2_stem>: one real ND2 filename without .nd2
if (-not (Test-Path -LiteralPath $Nd2)) { throw "ND2 file not found: $Nd2" }

$CropRoot = "$Work\01_cell_crops"              # KEEP or REPLACE: this parent is created by the next line
$Device = "auto"                                 # KEEP: CUDA/MPS if available, otherwise CPU; use "cpu" to force CPU
New-Item -ItemType Directory -Force -Path $CropRoot | Out-Null

& $Python "$Repo\nucleus_segmentation\crop_nuclei_sam.py" $Nd2 `
  --device $Device `
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

The command creates `$CropRoot\<ND2_stem>\`; do not create the stem-specific
subfolder yourself. The model may be downloaded into the user's cache on its
first run, so the first execution can take longer.

Step 1 writes metadata, masks, mapping/QC tables, and previews, but not the final
multi-channel crop TIFFs:

```text
$CropRoot\<ND2_stem>\
  <ND2_stem>_crops.json
  <ND2_stem>_mask_<N>.tif
  cell_id_mapping.csv
  cell_crop_quality_metrics.csv
  filtered_bad_qc_candidates.csv
  visualizations\
```

For a directory batch after QC passes, use the block below. The model is loaded
once and reused across files. CUDA usually provides the largest speedup in this
workflow, so explicitly select it for a large batch after the CUDA preflight
succeeds; CPU remains fully supported.

```powershell
$Device = "cuda"                                 # RECOMMENDED for a large batch; REPLACE with "cpu" if no compatible GPU
& $Python "$Repo\nucleus_segmentation\crop_nuclei_sam.py" $Raw `
  --device $Device `
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

`--device cuda` now fails immediately with an actionable message if the current
PyTorch build cannot access CUDA instead of failing later during model loading.

## Step 2: inspect the crop QC

Open these files before exporting TIFFs or running trajectory extraction:

```text
$CropRoot\<ND2_stem>\visualizations\seg_overview.png
$CropRoot\<ND2_stem>\visualizations\crop_grid.png
$CropRoot\<ND2_stem>\visualizations\suppression_demo.png
$CropRoot\<ND2_stem>\visualizations\all_channels_demo.png
$CropRoot\<ND2_stem>\cell_crop_quality_metrics.csv
$CropRoot\<ND2_stem>\filtered_bad_qc_candidates.csv
```

Check that nuclei are individually segmented, edge-truncated nuclei and debris
are absent, touching nuclei are split correctly, fragmented masks are merged,
and neighbouring nuclei are suppressed without erasing the target nucleus.

The locked final low-quality rule removes a candidate only when both conditions
are true:

```text
ch0_contrast < 0.085 AND ch0_boundary_grad < 1.70
```

`cell_id_mapping.csv` assigns stable IDs in top-to-bottom, then left-to-right
centroid order. The crop suffix and cell number agree for new runs:
`<stem>_1.tif` corresponds to `cell_001` after Step 3.

If QC fails, change the relevant Step 1 parameter, rerun Step 1, and inspect the
new QC before continuing. Use a new `$Work` run directory when comparing
parameter sets so stale products are never mixed.

## Step 3: export single-cell TIFFs

`save_crops.py` reads every `*_crops.json` recursively below its one input path,
reloads the immutable source ND2 recorded in the JSON, and writes the actual
multi-channel crop TIFFs beside the JSON.

The JSON stores an absolute path to its source ND2. Do not move or rename the
ND2 between Steps 1 and 3. If the raw-data location changes, rerun Step 1 from
the new location rather than silently exporting from a different file.

Export only the tested FOV first:

```powershell
$FovCropDir = "$CropRoot\<ND2_stem>"             # REPLACE <ND2_stem>: tested FOV folder created by Step 1
if (-not (Test-Path -LiteralPath $FovCropDir)) { throw "FOV crop folder not found: $FovCropDir" }
& $Python "$Repo\nucleus_segmentation\save_crops.py" $FovCropDir
```

After all FOVs pass QC, export all of them in one call:

```powershell
& $Python "$Repo\nucleus_segmentation\save_crops.py" $CropRoot
```

This command has no tunable options other than its required input directory. It
creates, for every accepted nucleus:

```text
<stem>_<N>.tif
<stem>_<N>_metadata.json
```

It also verifies the first exported TIFF's ImageJ axes, time interval, display
ranges, and LUT count in the console. Treat a missing or implausible frame
interval/pixel size as a failed export; do not continue to physical-unit
tracking.

## Step 4: extract trajectories from one cell

Run one cell first. The input is a crop TIFF from Step 3, not the ND2, mask TIFF,
JSON sidecar, or FOV directory.

```powershell
$CropTif = "$FovCropDir\<ND2_stem>_<N>.tif"      # REPLACE both placeholders: one real Step 3 crop TIFF
$ReferenceChannel = "green"                       # REPLACE if needed: green, red, or purple
$MatlabWorkers = 1                                 # KEEP for the first test: one MATLAB session handles all 3 channels
$Analysis = [System.IO.Path]::ChangeExtension($CropTif, $null) # AUTO: runner-created sibling folder
if (-not (Test-Path -LiteralPath $CropTif)) { throw "Crop TIFF not found: $CropTif" }

& $Python "$Repo\trajectory_extraction\run_full_pipeline_v3.py" $CropTif `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --matlab-workers $MatlabWorkers `
  --reference-channel $ReferenceChannel
```

No analysis directory needs to be created. The runner creates a sibling folder
named after the TIFF without its extension:

```text
<ND2_stem>_<N>.tif      # Step 3 input
<ND2_stem>_<N>\         # automatically created analysis directory
```

On Windows, `$CropTif` may remain in a G-drive/cloud directory whose parent
folders contain Chinese or other non-ASCII characters. The runner automatically
creates a temporary ASCII-only directory link for Fiji, while both the input
TIFF and generated analysis directory remain physically in their original G
drive location. Keep the TIFF filename itself ASCII (spaces are allowed) and
retain the `.tif` extension. No manual copy or path alias is required.

The runner performs four operations:

1. Fiji corrects drift, makes a Z maximum projection, and writes registered
   `*_Nucleus.tif`, `*_red.tif`, `*_green.tif`, and `*_purple.tif` stacks.
2. Python detects and tracks loci using the selected reference channel, then
   tracks the two target channels near those reference loci.
3. MATLAB detects and links global 2D-Gaussian SPT candidates independently in
   all three locus channels.
4. Python makes one-to-one, same-channel reference/candidate assignments and
   writes the accepted cleaned trajectories.

`--reference-channel` accepts `green`, `red`, or `purple`; the other two channels
automatically become targets. It changes Stage 1 anchoring only. Choose a
reference channel with a biologically meaningful spatial relationship to both
targets and record the choice. MATLAB still processes all three channels, and
Stage 3 never matches across colours.

### Required calibration check

Read `$Analysis\log_trajectory_v3.txt` after the run. Stage 1 prints the
TIFF scale in `px/um`. Stage 2 prints `XResolution` in `px/um` and its reciprocal
`pixl` in `um/px`. The Stage 1 value and Stage 2 `XResolution` must agree. Also
confirm that the frame interval is the expected acquisition interval. Stop if
these values are missing or inconsistent; distance-based matching is then not
scientifically valid.

## Step 5: review the cleaned trajectories

Review the first cell before starting a batch. Run the read-only QC tool while
the Stage 1 reference tracks and Stage 2 MATLAB candidates still exist:

```powershell
$Review = "$Analysis\review"                                  # AUTO: review output folder; the tool creates it

& $Python "$Repo\trajectory_extraction\pipeline\review_trajectory_results.py" `
  --analysis-dir $Analysis `
  --output-dir $Review
```

Inspect at minimum:

- `match_filter_summary.csv` and `match_filter_avg_distance.png`;
- `cleaned_trajectories_absolute_and_start_aligned.png`;
- `cleaned_tracks_on_channel_mip.png`;
- `cleaned_trajectory_summary.csv`;
- `review_manifest.json`.

The channel-overlay plot should place each trajectory on a real locus signal.
The review tool does not alter or re-fit any trajectory.

## Step 6: process the remaining cells

Use the production batch wrapper below. It inspects TIFF axes/shape and therefore
rejects mask/preview TIFFs even when the glob also finds them. Valid crop axes
are `TCYX` (single Z) or `TZCYX` (multiple Z). Start with
`--dry-run`; no analysis is performed during that check.

```powershell
$Stem = "<ND2_stem>"                              # REPLACE: exact common filename stem before _<N>.tif
$CellWorkers = 2                                   # REPLACE for hardware: concurrent cell pipelines; start with 1 or 2
$MatlabWorkers = 1                                 # KEEP for cell batching; total MATLAB processes = both worker values multiplied

& $Python "$Repo\trajectory_extraction\run_batch_pipeline_v3.py" $FovCropDir `
  --crop-glob "$Stem`_*.tif" `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --reference-channel $ReferenceChannel `
  --cell-workers $CellWorkers `
  --matlab-workers $MatlabWorkers `
  --dry-run
```

After the accepted filenames are correct, rerun the same command without the
last `--dry-run` line (remove the preceding continuation backtick as well):

```powershell
& $Python "$Repo\trajectory_extraction\run_batch_pipeline_v3.py" $FovCropDir `
  --crop-glob "$Stem`_*.tif" `
  --fiji-bin $Fiji `
  --matlab-bin $Matlab `
  --reference-channel $ReferenceChannel `
  --cell-workers $CellWorkers `
  --matlab-workers $MatlabWorkers
```

The default `--resume` skips compatible completed runs and writes
`$FovCropDir\trajectory_batch_summary.csv`. Use `--no-resume` only when a full
rerun is intentional. Run the review command for each completed analysis
directory, or at least for a documented representative sample plus every
anomalous cell.

There are two independent CPU-parallelism controls:

- For one cell, `--cell-workers 1 --matlab-workers 2` or `3` processes channel
  groups concurrently.
- For many cells, prefer `--cell-workers 2 --matlab-workers 1` initially. This
  runs two complete cells at once while each cell reuses one MATLAB session.
- Do not maximize both controls blindly. Peak MATLAB process count is
  `cell-workers x matlab-workers`; each process consumes RAM and MATLAB license
  capacity, and concurrent cloud-drive I/O can erase the speedup.
- This process-level method does not require MATLAB Parallel Computing Toolbox.
  Benchmark two representative cells before selecting the production worker
  counts.

Do **not** use `trajectory_extraction/pipeline/run_crop_trajectory_batch.py` as
the standard entry point. That legacy smoke-test/compact-batch utility skips the
Fiji drift-correction path. The new top-level batch wrapper calls the same
production runner used in Step 4.

## All adjustable parameters

Parameters exposed on the command line are the supported user interface.
Advanced constants require editing the named source file. If a code-level
scientific parameter is changed, record the file, old value, new value, and
reason with the run.

### Nucleus segmentation CLI

Script: `nucleus_segmentation/crop_nuclei_sam.py`

| Parameter | CLI default | Tutorial value | Meaning |
| --- | ---: | ---: | --- |
| `input` | required | one ND2, then `$Raw` | ND2 file or directory searched recursively for `*.nd2`. |
| `--nucleus-channel` | `0` | `0` | Zero-based channel used to form the time-averaged, Z-max nucleus image. |
| `--margin` | `30` px | `30` px | Padding around each accepted nucleus bounding box. |
| `--min-area` | `1000` px | `1000` px | Masks smaller than this are rejected; watershed pieces must also meet it. |
| `--max-area` | `200000` px | `200000` px | Maximum final/merged area. Pre-split filtering permits up to `2 x max-area`. |
| `--border-margin` | `5` px | `5` px | Reject a mask whose centroid is this close to the FOV edge. |
| `--mask-border-margin` | `-1` | `0` px | Reject masks whose pixels are at or within this distance of the edge, after merging. `-1` disables this filter; the locked workflow uses `0`. |
| `--segmentation-mode` | `apg` | `apg` | micro-SAM mode: `apg` or `amg`. |
| `--model-type` | `vit_b_lm` | `vit_b_lm` | micro-SAM checkpoint name. A different valid installed model may be supplied. |
| `--device` | `auto` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. `auto` selects the best available backend. |
| `--output-root` | ND2 parent | `$CropRoot` | Existing parent under which one `<ND2_stem>` folder is created per FOV. |

Useful adjustments are conservative: raise `--min-area` for debris; lower it
for genuinely small nuclei; increase `--margin` if a valid locus can leave the
crop; and inspect the QC after every change.

### Advanced nucleus-segmentation constants

File: `nucleus_segmentation/crop_nuclei_sam.py`

| Constant | Current value | Meaning |
| --- | ---: | --- |
| `norm_u8(..., lo_pct, hi_pct)` | `1`, `99` | Percentile stretch used only for the micro-SAM input image. |
| `MIN_SOLIDITY` | `0.70` | A lower-solidity mask triggers watershed splitting. |
| `SPLIT_SIGMA` | `5` px | Gaussian smoothing of the distance transform before watershed seeds. |
| `SPLIT_MIN_DIST` | `20` px | Minimum distance between the two watershed peaks. |
| `MIN_CIRC` | `0.3` | Minimum circularity accepted for each split piece. |
| `MERGE_PROXIMITY` | `2` px | Maximum gap used to propose merging adjacent fragments. |
| `MERGE_MIN_SOLID` | `0.60` | Minimum solidity required for a proposed merged union. |
| `IOU_THRESH` | `0.3` | IoU above which the smaller overlapping mask is treated as a duplicate. |
| `CONTAIN_THRESH` | `0.5` | Containment above which the smaller mask is treated as a duplicate. |
| `BAD_QC_CH0_CONTRAST_MAX` | `0.085` | First half of the final low-quality rule. |
| `BAD_QC_CH0_BOUNDARY_GRAD_MAX` | `1.70` | Second half of the final low-quality rule; both comparisons must pass to reject. |

After changing any value in these two segmentation tables, rerun Steps 1-3 in a
fresh run directory.

### Single-cell trajectory runner CLI

Script: `trajectory_extraction/run_full_pipeline_v3.py`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `input_path` | required | Crop TIFF, or an existing analysis directory when `--no-fiji` is used. |
| `--fiji-bin` | `fiji` | Fiji executable used for preprocessing. Use the full local path on Windows. |
| `--matlab-bin` | `matlab` | MATLAB executable/command. Use `$Matlab` to support either `PATH` lookup or a full local path. |
| `--matlab-workers` | `1` | Concurrent MATLAB processes: `1`, `2`, or `3`. One process handles all channels in one reused session; higher values partition channel groups across processes. |
| `--matlab-save-filter-images` | off | Retain the full band-pass-filtered movies in MAT files. They are not read by final matching and greatly increase MAT size/cloud writes. |
| `--reference-channel` | `green` | Stage 1 anchor: `green`, `red`, or `purple`. |
| `--no-fiji` | off | Treat `input_path` as an already-preprocessed analysis directory and do not run Fiji. |
| `--skip-stage1` | off | Reuse existing `*_traj_rela2wholeimg.csv` reference tracks. The runner verifies that at least one exists. |
| `--skip-stage2` | off | Reuse existing MATLAB candidate CSVs. The runner verifies that at least one exists. |
| `--skip-stage3` | off | Stop without rerunning final matching. |

### Multi-cell batch runner CLI

Script: `trajectory_extraction/run_batch_pipeline_v3.py`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `crop_dir` | required | One Step 3 FOV crop directory. |
| `--crop-glob` | `*.tif` | Non-recursive filename glob. Every match is still validated as a `TCYX`/`TZCYX` TIFF with 3 or 4 channels. |
| `--cell-workers` | `1` | Concurrent full cell pipelines. Must be at least 1. |
| `--matlab-workers` | `1` | MATLAB processes per active cell: `1`, `2`, or `3`. |
| `--fiji-bin` | `fiji` | Fiji executable forwarded to every cell runner. |
| `--matlab-bin` | `matlab` | MATLAB executable/command forwarded to every cell runner. |
| `--reference-channel` | `green` | Common Stage 1 reference choice for the batch. |
| `--resume` / `--no-resume` | resume | Skip completed runs whose manifest has compatible reference/MATLAB settings, or force reruns. |
| `--matlab-save-filter-images` | off | Forward the large-filter-stack retention option. |
| `--dry-run` | off | Validate and list crop TIFFs without starting Fiji, Python tracking, or MATLAB. |

### Fiji preprocessing constants

File: `trajectory_extraction/pipeline/headless_Macro_first_steps_for_published.ijm`

| Setting | Current value | Meaning |
| --- | --- | --- |
| `nChannels` | auto-detected `3` or `4` | Three channels map to green/red/purple; four map to nucleus/red/green/purple. |
| First contrast `saturated` | `0.35`% | Display contrast before drift correction. |
| Drift reference `channel` | `1` | C1 drives drift estimation: nucleus for four-channel input, green for three-channel input. |
| Drift options | `correct`, `multi_time_scale`, `sub_pixel`, `edge_enhance` | Enabled plugin modes. |
| Drift channel range | `lowest=1`, `highest=nChannels` | Apply correction through every detected channel. |
| Maximum drift | `10` px in X, Y, and Z | Search limit passed to Correct 3D drift. |
| Second contrast `saturated` | `0.15`% | Composite display contrast after correction. |
| Output mapping | 4ch: C1 nucleus/C2 red/C3 green/C4 purple; 3ch: C1 green/C2 red/C3 purple | Split-channel naming used downstream. |

Changing a Fiji setting requires rerunning Step 4 from the crop TIFF in a fresh
analysis directory.

### Stage 1 reference-tracking constants

File: `trajectory_extraction/pipeline/auto_roi_for_published_v2.13.py`

| Constant | Current value | Meaning |
| --- | ---: | --- |
| `K_SIGNAL['green']` | `2.0` | Green threshold multiplier in `mean + k x std`. |
| `K_SIGNAL['red']` | `0.5` | Red threshold multiplier. |
| `K_SIGNAL['purple']` | `0.5` | Purple threshold multiplier. |
| `NUCLEUS_SIGMA` | `2.0` px | Gaussian blur before nucleus-mask thresholding. |
| `NUCLEUS_OUTSIDE_FRAC` | `0.10` | Maximum fraction of reference-track frames allowed outside the nucleus mask. |
| `FILL_RATIO` | `0.85` | Border-connected drift-fill detection relative to the image 75th percentile. |
| `MIN_SPOT_PX` | `10` px | Minimum connected-component area for a spot candidate. |
| `N_MAX` | `5` | Maximum number of reference loci retained. |
| `PADDING` | `20` px | Padding around each reference trajectory ROI. |
| `ADJACENCY_PX` | `30` px | ROI-overlap grouping distance. |
| `INTER_FRAME_MAX_NM` | `500` nm | Normal target-channel frame-to-frame displacement limit. |
| `INTER_FRAME_MAX_NM_RED` | `750` nm | Red target-channel displacement limit. It applies only when red is a target. |
| `REFERENCE_PROX_MAX_UM` | `3.0` um | Maximum target distance from the selected reference locus. |
| `SEED_MAX_FRAME` | `5` | Number of early frames searched for a target seed. |
| `MAX_BLOB_PX` | `120` px | Size above which a seed component is re-examined as a merged blob. |
| `_ADAPTIVE_K_STEPS` | `[1.0, 1.5, 2.0]` | Successive local thresholds tried to split a merged blob. |
| `PIXEL_SIZE_UM` | metadata-derived | Despite its historical name, this stores `px/um`; do not hard-code it. |

Changing one of these constants requires rerunning Stages 1 and 3. A fresh
analysis directory is recommended to prevent old locus files from surviving.

### Stage 2 MATLAB SPT constants

File: `trajectory_extraction/pipeline/spt_batch.m`

| MATLAB setting | Current value/source | Meaning |
| --- | --- | --- |
| `sptpara.f_rate` | `1 / finterval` metadata | Frame rate in Hz. |
| `sptpara.pixl` | `1 / XResolution` metadata | Pixel width in um/px. |
| `K` | `0.5` | Threshold is `mean(nonzero filtered pixels) + K x std(...)` on frame 1. |
| `sptpara.mtl` | `3` frames | Minimum retained trajectory length. |
| `sptpara.dia` | `5` px | Particle diameter; keep odd. Band-pass scale is `dia + 2`. |
| `sptpara.estD` | `0.001` um^2/s | Expected diffusion used to derive `max_disp`. |
| `sptpara.max_disp` | derived | `round(3 x sqrt(4 x estD / f_rate) / pixl)` pixels. |
| `sptpara.fitmethod` | `0` | `0` = 2D Gaussian, `1` = centroid. |
| `sptpara.boxr` | `9` px | Sub-pixel fitting window radius; keep odd. |
| `sptpara.trackMem` | `3` frames | Allowed blinking gap during linking. |
| `sptpara.IntTh` | `0` | Integrated-intensity threshold; zero disables it. |
| `sptpara.saveFilterImgMode` | `0` by default | Filtered images are used during detection but removed before the MAT file is saved. `--matlab-save-filter-images` changes this to `1` for diagnostic retention. Trajectories are unchanged. |
| `sptpara.cell_num` | `1` | Legacy metadata field; one crop is processed per run. |

Changing a Stage 2 constant requires rerunning Stages 2 and 3.

### Stage 3 matching constants

File: `trajectory_extraction/pipeline/match_m2DGaussian_to_reference.py`

| Constant | Current value | Meaning |
| --- | ---: | --- |
| `MIN_OVERLAP_FRAMES` | `5` | Minimum shared frames needed to score a reference/candidate pair. |
| `MAX_AVG_DIST_NM` | `2000` nm | Reject an assigned pair whose mean shared-frame distance is larger. |
| Red early-frame rule | frames `1-3` | A red MATLAB candidate is rejected unless it contains at least one of these frames. |

Stage 3 scores same-colour pairs, sorts by average distance, performs greedy
one-to-one assignment, then applies the distance and red early-frame filters.
Changing these rules requires rerunning Stage 3.

## Outputs

For a crop named `<cell>.tif`, the main analysis tree is:

```text
<cell>/
  log_trajectory_v3.txt
  trajectory_run_manifest.json
  <cell>_Nucleus.tif
  <cell>_green.tif
  <cell>_red.tif
  <cell>_purple.tif
  Nucleus_masks.tif
  RoiSet_<reference-channel>.zip
  G_loci<N>_traj_rela2wholeimg.csv
  P_loci<N>_traj_rela2wholeimg.csv
  R_loci<N>_traj_rela2wholeimg.csv
  matlab_result/
    <cell>_green.mat
    <cell>_red.mat
    <cell>_purple.mat
    matlab_trajectory/
      G_m2DGaussian_traj<N>.csv
      P_m2DGaussian_traj<N>.csv
      R_m2DGaussian_traj<N>.csv
  G_loci<N>_traj_m2DGaussian_cleaned.csv
  P_loci<N>_traj_m2DGaussian_cleaned.csv
  R_loci<N>_traj_m2DGaussian_cleaned.csv
  review/
```

For a multi-cell run, `trajectory_batch_summary.csv` is additionally written in
the FOV crop directory. `trajectory_run_manifest.json` records the input,
reference channel, MATLAB concurrency, executable choices, timestamps, and
completion status used by safe resume checks.

The `*_traj_m2DGaussian_cleaned.csv` files are the final deliverable. Each has:

| Column | Meaning |
| --- | --- |
| `frame` | 1-indexed acquisition frame. Gaps are possible. |
| `x_nm` | X coordinate in nanometres in the whole single-cell TIFF coordinate system. |
| `y_nm` | Y coordinate in nanometres in the whole single-cell TIFF coordinate system. |

Missing cleaned files are not automatically an error: a reference may have no
candidate with enough overlap, may exceed 2000 nm average distance, or, for red,
may fail the early-frame rule. Use the review summary and log to distinguish a
scientific rejection from a failed stage.

## Resume and rerun safely

For an analysis directory already produced by Fiji:

```powershell
& $Python "$Repo\trajectory_extraction\run_full_pipeline_v3.py" --no-fiji $Analysis `
  --matlab-bin $Matlab `
  --skip-stage1 `
  --skip-stage2
```

This example reruns Stage 3 only. Other valid combinations follow the runner
table above. Never use a skip option unless the reused files came from the same
input, channel mapping, reference-channel choice, calibration, and parameter
set.

The runner overwrites `log_trajectory_v3.txt` on every invocation and stage
scripts may not remove obsolete files from an older parameterization. Before a
scientific rerun, archive the old analysis folder and generate a fresh one. Do
not compare parameter sets inside the same directory.

## Troubleshooting

### Resolved edge cases in the current runner

The following failures encountered during manual validation are now handled in
code and are retained here as regression expectations:

| Observed problem | Current behavior |
| --- | --- |
| Literal `example_fov.nd2` or `<ND2_stem>` was treated as a real filename | Every user-edited name/path is marked `REPLACE` at its assignment and guarded with `Test-Path`. Placeholders must never be copied literally. |
| Removing `.tif` after a failed crop command | The runner explicitly requires the real Step 3 `.tif/.tiff` file; an extensionless path is rejected with guidance. |
| Fiji corrupted a Chinese G-drive path and truncated `.tif` to `.t` | The top-level runner creates a temporary ASCII directory link while all data and outputs remain in the original directory. |
| The Fiji macro assumed four channels but the crop had green/red/purple only | The macro detects exactly three or four channels; the three-channel path exports the three colours and Python creates a Stage 1-only synthetic nucleus scaffold. |
| Z projection on a `Z=1, T>1` acquisition collapsed time | The macro skips Z projection when there is only one Z slice, preserving the full `TYX` movie. |
| `nd2` time-loop metadata changed shape in newer releases | `save_crops.py` reads the current `periods[0]` representation and falls back safely; `nd2==0.11.3` is pinned because 0.11.1 was yanked for loop-detection regression. |
| MATLAB appeared frozen during a long silent SPT calculation | Stage 2 prints a heartbeat every 60 seconds and reports elapsed time for each MATLAB process group. |
| Rerunning Stage 2 left higher-numbered candidate CSVs from an older run | Stage 2 removes only old `G/P/R_m2DGaussian_traj*.csv` candidates immediately before exporting the new candidates. |
| A GBK PowerShell raised `UnicodeEncodeError` on `µ`, arrows, or box characters | User-facing Python entry points configure UTF-8 streams, and child processes receive UTF-8 environment settings. The setup variables remain recommended for consistent rendering. |

### `py` is not recognized, or `Activate.ps1` does not exist

The Windows `py` launcher is optional and is not installed with every Python
distribution. Use the documented `$BasePython -m venv ...` command instead.
`Activate.ps1` is missing because venv creation did not complete; activation is
not the fix. Stop, create the venv successfully, confirm that `$Python` exists,
and run the isolation assertion before installing anything.

### pip says `Defaulting to user installation`

Stop the command. It is using the base Microsoft Store Python rather than the
project venv. A correct command starts with `& $Python -m pip`, and this must
print a pip path below `.venv_nd2_to_traj`:

```powershell
& $Python -m pip --version
& $Python -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix); assert sys.prefix != sys.base_prefix"
```

If packages were already installed into the user site, do not mass-uninstall
them: some may predate this project and be required elsewhere. The normal venv
is isolated from the user site, so create and use it as above. Only clean the
global user site after reviewing `python -m pip list --user` and determining
package ownership outside this workflow.

### Installed command-line scripts are reported as “not on PATH”

When this warning points to a Microsoft Store `LocalCache\local-packages`
directory, it is another sign that pip is installing outside the venv. Stop and
repeat the isolation checks above. The pipeline itself calls modules and script
files through `$Python`; adding that user-level Scripts directory to `PATH` does
not repair the missing venv.

### `matlab` is not recognized

Add the MATLAB `bin` directory to the current shell's `PATH`, restart the shell,
and verify `& $Matlab -batch "disp(version)"`. Alternatively, set `$Matlab` to
the full `matlab.exe` path and pass `--matlab-bin $Matlab`. No separate checkout
of the SPT `.m` files is needed.

### MATLAB prints no new results for several minutes

This is normal while 2D-Gaussian fitting is running. The Stage 1 `Done.` message
does not mean the full pipeline has ended if a subsequent `Running:
...run_pipeline_v3.py` block is visible. The new runner prints `MATLAB still
running` every 60 seconds. Completion requires `Pipeline complete` and the
PowerShell `PS ...>` prompt to return. If no heartbeat appears for more than two
minutes, inspect MATLAB process CPU use and the end of `log_trajectory_v3.txt`.

### A parallel MATLAB run is slower or fails to start

Return to `--cell-workers 1 --matlab-workers 1`. Parallel processes need
additional CPU, RAM, MATLAB license capacity, and concurrent disk bandwidth.
The default still avoids the previous three-startup overhead by processing all
channels in one MATLAB session. Increase only one worker dimension at a time.

### Python raises `UnicodeEncodeError` before analysis starts

Set `$env:PYTHONUTF8 = "1"` and `$env:PYTHONIOENCODING = "utf-8"` in the current
PowerShell session, then rerun the command. These variables are part of the
one-time setup block above because several logs contain non-ASCII scientific
unit symbols such as `µm`.

### Fiji reports an unknown `Correct 3D drift` command

Install/enable that Fiji plugin, restart Fiji, and verify it in graphical mode.
Passing a valid Fiji executable cannot compensate for a missing plugin.

### Fiji cannot find the split-channel windows

Confirm that the crop contains exactly three or four channels and follows one
of the documented input channel contracts. Do not relabel an unknown order just
to make the macro complete.

### Fiji prints a corrupted path or truncates `.tif` to `.t`

This is a legacy Windows Fiji-launcher limitation for non-ASCII macro
arguments. Use the repository's top-level
`trajectory_extraction/run_full_pipeline_v3.py` runner rather than invoking the
`.ijm` macro directly. The runner automatically bridges a non-ASCII parent path
to a temporary ASCII path without copying the TIFF or moving outputs. The TIFF
filename itself must remain ASCII and end in `.tif` or `.tiff`.

### No `*_Nucleus.tif` is found

This normally means Fiji failed before channel export. Read the beginning of
`log_trajectory_v3.txt`, verify `$Fiji`, the plugin, and the crop axes, then run
the crop again in a fresh analysis directory.

### Stage 1 and Stage 2 scales disagree

Stop. Inspect the exported TIFF's `XResolution`, ImageJ unit, and `finterval`
metadata. Do not tune matching thresholds to hide a calibration problem.

### CUDA runs out of memory during segmentation

Close other GPU jobs and retry. The script already uses a CUDA memory guard for
micro-SAM embeddings. If needed, use `--device cpu` for a slower, lower-risk
run; changing the device should not be used as a substitute for QC.

### `--device cuda` says CUDA is unavailable

The GPU hardware alone is insufficient: the active venv must contain a
CUDA-enabled PyTorch build compatible with the installed NVIDIA driver. Follow
the optional GPU section in `REQUIREMENTS_ND2_TO_TRAJ.md`, then require this
preflight to print `True` before batching:

```powershell
& $Python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Too many or too few nucleus crops

Inspect `seg_overview.png` first. Adjust `--min-area`, `--max-area`, or the
border filters only when the rejected/accepted masks support that change. For
touching or fragmented nuclei, use the advanced split/merge constants and
create a separate comparison run.

### No cleaned trajectory is produced

Check, in order: Stage 1 reference CSVs, Stage 2 candidate CSVs, Stage 3 log,
`match_filter_summary.csv`, overlap count, average distance, and the red
early-frame rule. Absence of a cleaned file can be the intended filter outcome.

## Reproducibility checklist

- Preserve raw ND2 files unchanged.
- Save the exact repository revision and `& $Python -m pip freeze` output.
- Record all CLI arguments, code-level parameter edits, channel mapping, and
  reference channel.
- Keep each parameter set in a separate run directory.
- Verify pixel size and frame interval in both Python and MATLAB logs.
- Inspect segmentation QC before TIFF export and trajectory QC before batching.
- Retain the run log, review artifacts, cleaned CSVs, and the crop metadata
  sidecar used for each analyzed cell.

## Associated manuscript

Chen*, X., Zhu*, Y., Washington, K., Chan, T.-K., Shamsher, N., and Qi, L. S.
*Genomic DNA Dynamics Predict Chromatin Density Features in Living Cells.*
(* equal contribution)

## Contributors

- Xinyi Chen (Stanford University)
- Yanyu Zhu (Stanford University)
- Kenaj Washington (Stanford University)
- Tse-Kai (Kevin) Chan (Stanford University)
- Nikhiya Shamsher (Stanford University)
