# Trajectory Analysis

Downstream analysis of Oligo-LiveFISH trajectory data: feature extraction from nuclear segmentation masks and tracked loci, synthetic trajectory generation for motion classification, and MSD-based diffusion analysis.

## Folder Structure

```
trajectory_analysis/
  feature_extraction/     Spatial and morphological feature extraction from imaging data
  synthetic_classification/   Synthetic trajectory generation + RF/LSTM/CNN classification
  msd_analysis/           Mean squared displacement and anomalous diffusion exponent fitting
```

## Data

These scripts operate on the outputs of the upstream imaging pipeline (`pipeline/`) and nuclear segmentation (`nucleus_segmentation/`). Specifically:

- **Nuclear masks**: binary per-frame TIF masks from segmentation
- **Trajectory CSVs**: locus coordinates (frame, x_nm, y_nm) from ThunderSTORM tracking
- **Nucleus channel TIFs**: DAPI intensity images for chromatin density measurements

The three fluorescence channels correspond to probes on chromosome 3: G (chr3:195M, 488nm), R (chr3:195.7M, 565nm), P (chr3:198M, 647nm).

## Imaging Parameters

| Parameter | Value |
|-----------|-------|
| Frame interval | 24.5 s |
| Pixel size | 183.3 nm/px |
| Localization precision | ~15-40 nm |

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
fbm          # for synthetic subdiffusion (fractional Brownian motion)
torch        # for LSTM/CNN models (optional, RF works without it)
Pillow       # for reading TIF masks
```
