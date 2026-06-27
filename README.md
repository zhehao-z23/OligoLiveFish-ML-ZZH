# OligoLiveFish-ML 

OligoLiveFish-ML is an analysis and modeling repository for live-cell
Oligo-LiveFISH chromatin dynamics data. It provides a workflow for turning
multi-channel microscopy movies into cleaned single-particle DNA trajectories,
then using those trajectories to model nuclear and local chromatin features.

The repository is organized around four stages:

1. `nucleus_segmentation/` segments nuclei in `.nd2` field-of-view files and
   exports one single-nucleus TIFF per crop.
2. `trajectory_extraction/` runs the single-nucleus DNA locus trajectory extraction
   workflow: Fiji preprocessing, reference trajectory detection, MATLAB 2D
   Gaussian fitting, and reference-based filtering of real DNA signals in each
   fluorescence channel.
3. `Cellular_feature_extraction/` extracts the cellular and nuclear features from single-nucleus TIFF files.
4. `trajectory_to_nuclear_features/` contains traditional machine-learning and
   deep-learning experiments for predicting nuclear and local chromatin
   features from trajectory dynamics.

Raw microscopy files are not committed to this repository. The modeling folder
documents how to download the small derived data archive needed to reproduce the
prediction tasks.

## Repository Layout

```text
nucleus_segmentation/
  crop_nuclei_sam.py       # segment nuclei from ND2 field-of-view files
  save_crops.py            # export single-nucleus TIFF crops
  crop_nucleus.md          # segmentation and crop-export usage notes

trajectory_extraction/
  run_full_pipeline_v3.py  # production trajectory-extraction entry point
  internal/                # runtime helper scripts and MATLAB dependencies

Cellular_feature_extraction/
  extract_features.py      # Core feature extractor from images of nucleus
  extract_nuclear_features.py  # Standalone nuclear morphology extraction from binary masks

trajectory_to_nuclear_features/
  traditional_ml/          # grouped Elastic Net and Random Forest baselines
  deep_learning/           # CNN/LSTM and engineered-feature neural models
  README.md                # modeling data download and reproduction notes
```

## Stage 1: Nucleus Segmentation

The segmentation workflow starts from multi-channel `.nd2` field-of-view files.
`crop_nuclei_sam.py` uses micro-SAM to segment nuclei from the nuclear stain
channel, applies custom post-processing to split, merge, and filter masks, and
writes crop metadata. `save_crops.py` then exports one single-nucleus TIFF per
accepted nucleus.

See `nucleus_segmentation/crop_nucleus.md` for setup, parameters, and
recommended visual QC.

## Stage 2: Trajectory Extraction

After single-nucleus TIFFs are generated, run:

```bash
python trajectory_extraction/run_full_pipeline_v3.py /path/to/single_nucleus.tif
```

This runner performs:

- Fiji preprocessing and drift correction.
- Reference trajectory detection from the green channel.
- MATLAB 2D Gaussian single-particle tracking on each fluorescence channel.
- Matching and filtering of MATLAB tracks against the reference trajectories.

The final per-locus trajectory files are written next to the single-nucleus
analysis directory as:

```text
G_loci{N}_traj_m2DGaussian_cleaned.csv
P_loci{N}_traj_m2DGaussian_cleaned.csv
R_loci{N}_traj_m2DGaussian_cleaned.csv
```

Each trajectory CSV contains `frame`, `x_nm`, and `y_nm` columns in whole-image
coordinates.

## Stage 3: Cellular Feature Extraction

Extracts spatial and morphological features from nuclear segmentation masks and the location of DNA locus.

```bash
python3 extract_features.py /path/to/nucleus_folder [--pixel-size 108.33]
```

## Stage 4: Trajectory-To-Feature Modeling

The modeling workflows ask whether chromatin motion encodes information about
nuclear morphology, local chromatin density, and locus spatial context.

`trajectory_to_nuclear_features/traditional_ml/` contains grouped traditional
ML baselines on engineered trajectory features, including tuned Random Forest
models. `trajectory_to_nuclear_features/deep_learning/` contains CNN/LSTM
experiments that operate directly on raw step-vector sequences.

See `trajectory_to_nuclear_features/README.md` for the modeling data download,
expected data layout, and commands for rerunning the experiments.

## External Requirements

The full workflow uses a mix of Python, Fiji/ImageJ, and MATLAB:

- Python for segmentation, trajectory filtering, and modeling.
- Fiji/ImageJ for trajectory-extraction preprocessing and drift correction.
- MATLAB for 2D Gaussian single-particle tracking.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For GPU-specific PyTorch builds, install the appropriate `torch` package from
the official PyTorch instructions before or after installing this file.

The trajectory-extraction runner bundles its MATLAB helper code under
`trajectory_extraction/internal/`; no separate SPT code checkout is required.
