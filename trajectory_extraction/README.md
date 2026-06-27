# OligoLIVEFISH Analysis Pipeline — v3

Automated reference trajectory extraction and single-particle tracking (SPT) pipeline for live-cell oligoLiveFISH imaging. Tracks DNA loci labelled with green (reference), red, and purple fluorophores across multi-timeframe time-lapse acquisitions.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/xxx/OligoLiveFish-ML.git
cd OligoLiveFish-ML/trajectory_extraction/pipeline

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Add MATLAB binary to PATH (add to ~/.zshrc to make permanent)
export PATH="/Applications/MATLAB_R20XXx.app/bin:$PATH"

# 4. Run (can use any of the try_analysis/ folder in example_data/)
python3 run_full_pipeline_v3.py /path/to/try_analysis
```

No other path configuration is needed. All MATLAB `.m` dependencies are bundled in `matlab_deps/` and added to MATLAB's path automatically by the script.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Input files](#input-files)
4. [Pipeline scripts](#pipeline-scripts)
5. [Running the pipeline](#running-the-pipeline)
6. [Parameters](#parameters)
7. [Output files](#output-files)
8. [Algorithm details](#algorithm-details)
9. [Validation scripts](#validation-scripts)

---

## Overview

The pipeline has three stages:

```
Stage 1  auto_roi_for_published_v2.12.py
         ↳ Detects green loci, tracks green/purple/red reference trajectories,
           saves per-locus reference CSVs.

Stage 2  run_pipeline_v3.py
         ↳ Calls MATLAB spt_batch.m on each channel TIFF → .mat files,
           then exports every MATLAB trajectory to CSV.

Stage 3  match_m2DGaussian_to_reference.py
         ↳ Matches each MATLAB trajectory to the reference track of the same
           locus by spatial overlap, saves the cleaned final trajectory.
```

The full pipeline (all three stages) is orchestrated by `run_full_pipeline_v3.py`. 

---

## Requirements

### Python packages

```
Pillow==12.1.0
numpy==2.4.2
scipy==1.17.1
```

Install with:

```bash
pip install -r requirements.txt
```

All other imports (`csv`, `math`, `pathlib`, `subprocess`, `struct`, `zipfile`, `re`, `collections`, `datetime`) are Python standard library.

### External dependency

**MATLAB** must be installed and on your `$PATH` (test with `matlab -batch "disp('ok')"`).

All required MATLAB `.m` files are bundled in `matlab_deps/` and added to MATLAB's search path automatically — no manual path configuration needed.

---

## Input files

Each analysis directory must contain the following files with a common stem `<stem>`:

| File | Description |
|------|-------------|
| `<stem>_Nucleus.tif` | Multi-frame nucleus channel (Hoechst/DAPI), used for nucleus boundary masking |
| `<stem>_green.tif` | Multi-frame green channel (DNA locus with the strongest signal).|
| `<stem>_red.tif` | Multi-frame red channel (a different DNA locus adjacent to the green signal). |
| `<stem>_purple.tif` | Multi-frame purple channel (a different DNA locus adjacent to the green signal). |

All TIFFs must carry ImageJ-format metadata with `finterval` (frame interval in seconds) and `XResolution` (pixels per µm) tags.

---

## Pipeline scripts

| Script | Role |
|--------|------|
| `auto_roi_for_published_v2.12.py` | Stage 1 — reference trajectory extraction |
| `run_pipeline_v3.py` | Stage 2 — MATLAB SPT + CSV export (uses bundled `matlab_deps/`) |
| `match_m2DGaussian_to_reference.py` | Stage 3 — trajectory matching and final output |
| `run_full_pipeline_v3.py` | Runs all three stages sequentially for one dataset |

---

## Running the pipeline

### Single dataset

```bash
python3 run_full_pipeline_v3.py "<path_to_analysis_dir>"
```

Example:

```bash
python3 run_full_pipeline_v3.py "example_data/try_analysis1"
```

A log file `log_trajectory_v3.txt` is written to the analysis directory.

---

## Parameters

All tunable parameters are at the top of `auto_roi_for_published_v2.12.py`.

### Detection thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_SIGNAL['green']` | `2.0` | Threshold multiplier for green channel: `mean + k × std` . The values have been optimized to maximize signal detection.|
| `K_SIGNAL['red']` | `0.5` | Threshold multiplier for red channel |
| `K_SIGNAL['purple']` | `0.5` | Threshold multiplier for purple channel |

### Nucleus masking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUCLEUS_SIGMA` | `2.0` | Gaussian blur σ (px) before Otsu thresholding of nucleus channel |
| `NUCLEUS_OUTSIDE_FRAC` | `0.10` | Maximum fraction of frames a green locus may be outside the nucleus before it is rejected. The parameter has been introduced to increase tolerance for inaccurate nuclear mask generation for a small subset of timeframes. |
| `FILL_RATIO` | `0.85` | Drift-correction fill detection: pixels below `75th-percentile × FILL_RATIO` on the image border are masked as fill |

### Spot detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SPOT_PX` | `10` | Minimum connected-component area (px) for a candidate to be accepted |
| `N_MAX` | `5` | Maximum number of green loci to detect. Assuming no more than 5 real green DNA loci per cell. |
| `PADDING` | `20` | Pixels added around the green trajectory bounding box on each side to define the per-locus ROI |

### Tracking constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_SIZE_UM` | `5.45` | Pixel size in px/µm (read from TIFF metadata; used to convert nm ↔ px) |
| `INTER_FRAME_MAX_NM` | `500` | Maximum frame-to-frame displacement (nm) for purple tracking, assuming any DNA loci move a limited step per frame |
| `INTER_FRAME_MAX_NM_RED` | `750` | Maximum frame-to-frame displacement (nm) for red tracking (relaxed; red loci are more mobile) |
| `GREEN_PROX_MAX_UM` | `3.0` | Maximum distance (µm) from the green locus for a purple/red candidate to be accepted; assuming any purple or red loci will not be too far from green loci |
| `SEED_MAX_FRAME` | `5` | Number of early frames searched to find the initial seed position. If signal not found in any of the first 5 frames, the signal is unlikely to be real |

### Adaptive k for merged blobs (overlap groups only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_BLOB_PX` | `120` | Connected-component area threshold (px) above which a blob is considered potentially merged. When 2 real signals are close to each other, they might be detected as a connected large "blob". This parameter detects overly large blobs, and elevate k_signal locally to try spliting the blob into real signal clusters. |
| `_ADAPTIVE_K_STEPS` | `[1.0, 1.5, 2.0]` | Progressive k values tried to split a large blob into sub-components. The adaptive K_signal only applies locally, and to all time frames, without compromising the sensitivity to detect weaker signals in other areas of the same image. |

### MATLAB SPT parameters (Stage 2)

These parameters are set inside `spt_batch.m`. Two are read from TIFF metadata; one is auto-computed from the image data; the rest are fixed constants.

**From TIFF metadata** (read by `run_pipeline_v3.py` and passed to MATLAB):

| Parameter | Source | Description |
|-----------|--------|-------------|
| `f_rate` | `finterval` tag in ImageDescription | Frame rate (Hz) = 1 / finterval |
| `pixl_um` | `XResolution` TIFF tag | Pixel size (µm/px) = 1 / XResolution |

**Auto-thresholded from image data:**

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `thresh` | `mean(nz) + 0.5 × std(nz)` | Detection threshold, where `nz` = non-zero pixels of the band-pass filtered first frame (`bpass` filter, lp=1, bp=`dia+2`=7) |

**Derived parameter:**

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `max_disp` | `round(3 × sqrt(4 × estD / f_rate) / pixl_um)` | Maximum allowed inter-frame displacement (px); set to 3 standard deviations of expected diffusive motion |

**Fixed constants:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dia` | `5` | Particle diameter (px, must be odd) |
| `boxr` | `9` | Sub-pixel fitting window radius (px, must be odd) |
| `mtl` | `3` | Minimum trajectory length (frames) |
| `trackMem` | `3` | Frames a particle may blink off and still be linked |
| `estD` | `0.001` | Estimated diffusion constant (µm²/s), used only to compute `max_disp` |
| `fitmethod` | `0` | Fitting method: 0 = 2D Gaussian, 1 = centroid |
| `IntTh` | `0` | Integrated intensity threshold (disabled) |

### Matching (Stage 3)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_OVERLAP_FRAMES` | `5` | Minimum shared frames between reference and MATLAB trajectory for a match to be considered |
| `MAX_AVG_DIST_NM` | `2000` | Maximum average distance (nm) between matched trajectories; pairs exceeding this are rejected |

---

## Output files

All output files are written to the analysis directory.

### From Stage 1

| File | Description |
|------|-------------|
| `Nucleus_masks.tif` | Per-frame binary nucleus masks |
| `RoiSet_green.zip` | ImageJ ROI zip containing the per-locus green bounding boxes |
| `G_loci{N}_traj_rela2wholeimg.csv` | Green reference trajectory for locus N (whole-image coordinates) |
| `P_loci{N}_traj_rela2wholeimg.csv` | Purple reference trajectory for locus N |
| `R_loci{N}_traj_rela2wholeimg.csv` | Red reference trajectory for locus N |

### From Stage 2

| File | Description |
|------|-------------|
| `matlab_result/<stem>_green.mat` | MATLAB SPT output for green channel |
| `matlab_result/<stem>_red.mat` | MATLAB SPT output for red channel |
| `matlab_result/<stem>_purple.mat` | MATLAB SPT output for purple channel |
| `matlab_result/matlab_trajectory/G_m2DGaussian_traj{N}.csv` | Individual MATLAB-tracked green trajectories |
| `matlab_result/matlab_trajectory/R_m2DGaussian_traj{N}.csv` | Individual MATLAB-tracked red trajectories |
| `matlab_result/matlab_trajectory/P_m2DGaussian_traj{N}.csv` | Individual MATLAB-tracked purple trajectories |

### From Stage 3 (final outputs)

| File | Description |
|------|-------------|
| `G_loci{N}_traj_m2DGaussian_cleaned.csv` | Final green trajectory for locus N |
| `P_loci{N}_traj_m2DGaussian_cleaned.csv` | Final purple trajectory for locus N |
| `R_loci{N}_traj_m2DGaussian_cleaned.csv` | Final red trajectory for locus N |

All trajectory CSVs have three columns: `frame` (1-indexed), `x_nm`, `y_nm`. The coordinates is relative to the whole .tif image, not to individual ROIs. 

### Log file

| File | Description |
|------|-------------|
| `log_trajectory_v3.txt` | Full console output from all three stages |

---

## Algorithm details

### Stage 1 — Reference trajectory extraction

**Pass 1 — Green tracking**

1. The time-averaged green image is thresholded at `mean + K_SIGNAL['green'] × std` to detect up to `N_MAX` green loci clusters.
2. Each cluster is tracked frame-by-frame across the stack using a nearest-neighbour approach anchored to the cluster centroid.
3. Loci whose tracked positions are outside the nucleus mask in more than `NUCLEUS_OUTSIDE_FRAC` of frames are rejected.
4. Each accepted locus gets a bounding-box ROI: the convex hull of its trajectory ± `PADDING` pixels.

**Overlap detection**

ROIs that overlap by more than `ADJACENCY_PX` pixels are grouped. Loci in the same group share a union ROI for seed finding (joint seeding). Loci with no overlap are tracked independently (singletons).

**Pass 2 — Purple and red tracking**

*Overlap groups — joint seeding:*

1. A union mask is computed from all ROIs in the group.
2. Candidates are detected in each seed frame within the union mask. Large connected components (area > `MAX_BLOB_PX`) are re-examined locally at higher k values (`_ADAPTIVE_K_STEPS = [1.0, 1.5, 2.0]`) to split potentially merged blobs. Other components keep the base-k detection.
3. Candidates are linked greedily across seed frames into seed trajectories (inter-frame gap ≤ 1 frame, displacement ≤ inter-frame limit).
4. Seed trajectories are assigned to loci one-to-one using the Hungarian algorithm (minimising average distance to each locus's green trajectory).
5. Each assigned seed trajectory is then propagated forward through the full stack from `propagate_from_seed`:
   - **Primary**: accept a candidate that is within the inter-frame displacement limit AND within `GREEN_PROX_MAX_UM` of the green locus.
   - **Fallback**: if no candidate passes the inter-frame constraint, accept the candidate closest to the green locus (within `GREEN_PROX_MAX_UM`). This handles highly mobile loci whose frame-to-frame displacement exceeds the limit but whose signal stays near the green anchor.

*Singletons — standard tracking:*

`track_channel_in_roi` applies the same primary + fallback logic as above, starting from a seed found in the first `SEED_MAX_FRAME` frames.

### Stage 2 — MATLAB SPT

MATLAB `spt_batch.m` runs 2D-Gaussian sub-pixel fitting on the full-field channel TIFFs to detect and link sub-diffraction spots independently of any ROI.

1. **Band-pass filtering**: each frame is pre-filtered with `bpass` (low-pass 1 px, band-pass 7 px) to suppress camera noise and background.
2. **Auto-thresholding**: the detection threshold is computed from the filtered first frame as `mean(nz) + 0.5 × std(nz)`, where `nz` is all non-zero filtered pixels. This adapts to per-dataset signal levels without manual tuning.
3. **Spot detection**: candidate spots above threshold are localised with `pkfnd` and refined to sub-pixel precision using 2D Gaussian fitting (`pkRefnd`, window radius 9 px, particle diameter 5 px).
4. **Linking**: detected positions are linked frame-to-frame into trajectories using a nearest-neighbour tracker (`track`) with a maximum displacement computed as 3 standard deviations of expected diffusive motion (`max_disp = round(3 × sqrt(4 × estD / f_rate) / pixl_um)`). A particle may blink off for up to 3 frames (`trackMem = 3`) and still be re-linked. Only trajectories spanning ≥ 3 frames (`mtl = 3`) are kept.

Each channel produces a `.mat` file whose trajectories are exported to individual CSVs by `run_pipeline_v3.py`.

### Stage 3 — Trajectory matching

For each reference locus track (from Stage 1), the best-matching MATLAB trajectory (from Stage 2) of the same channel is found by:

1. Computing the average Euclidean distance between the two tracks over all shared frames.
2. Requiring at least `MIN_OVERLAP_FRAMES` shared frames.
3. Performing greedy one-to-one assignment (lowest average distance first).
4. Rejecting pairs whose average distance exceeds `MAX_AVG_DIST_NM`.
5. **For red channel only**: rejecting matched MATLAB trajectories that have no points in frames 1–3 (1-indexed), as these indicate the track started too late to be reliable.

The matched MATLAB trajectory is written as the final `*_traj_m2DGaussian_cleaned.csv`.

---

## Validation scripts

The output trajectory should be the same as manual and interactive analysis result produced by `spt.m`. Note that the numbering of loci will not be the same following the automated pipeline and following the interactive script.

| Script | Description |
|--------|-------------|
| `spt.m` | Interactive 2D Gaussian fitting code written by Yanyu Zhu. Taking any .tif image as input, generating .mat as output. |
| `export_trajectories.py` | Run `python3 export_trajectories.py XX.mat` with the output from `spt.m` to generate .csv files with trajectories saved in matlab_trajectory/ |
