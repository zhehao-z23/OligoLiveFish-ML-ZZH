# Trajectory To Nuclear Features

This folder contains modeling workflows that test whether OligoLiveFISH
trajectory dynamics predict nucleus-level and local chromatin features.

The folder is organized by method family:

- `traditional_ml/`: grouped Elastic Net and Random Forest baselines on
  engineered trajectory features.
- `deep_learning/`: CNN/LSTM experiments on raw trajectory step sequences,
  plus engineered-feature neural baselines.
- `data/`: local derived CSV data needed to rerun the modeling workflows.

Raw microscopy files are not stored here. The derived modeling data are stored
as an external ZIP archive and should be extracted locally into `data/`.

## Data Download

Download the modeling data ZIP from Google Drive:

<https://drive.google.com/file/d/157jxYWHadL8MKUz0s2Z_TATcJxMVxGKP/view?usp=sharing>

After downloading, extract it so this folder contains a `data/` directory:

```bash
cd OligoLiveFish/trajectory_to_nuclear_features
unzip /path/to/downloaded_modeling_data.zip
```

If the ZIP extracts into a wrapper folder, move or copy its `data/` directory
to `OligoLiveFish/trajectory_to_nuclear_features/data/`.

The `data/` directory and local ZIP files are ignored by Git.

## Data Layout

The default data root is this folder's local `data/` directory. You can
override it by setting `DATA_ROOT` to another directory with the same structure.

```text
data/
  chr3/
    engineered_feature_table.csv
    locus_feature_table.csv
    nucleus_feature_table.csv
    traditional_ml_results/
      grouped_model_comparison.csv
      best_model_by_target.csv
      heldout_test_nuclei.csv
  trajectories/
    batch1/
      Nuc_number_mapping.csv
      *_traj_m2DGaussian_cleaned.csv
    chr3_batch/
      Nuc_number_mapping.csv
      *_traj_m2DGaussian_cleaned.csv
```

`engineered_feature_table.csv` is the compact table used by the traditional ML
workflow. It contains one row per trajectory/locus, engineered movement summary
features, nucleus-level targets, local-intensity targets, and locus geometry
targets.

`locus_feature_table.csv`, `nucleus_feature_table.csv`, and the cleaned
trajectory CSV folders are used by the deep-learning workflow to rebuild raw
step-sequence tensors and align them with target features.

## Running

Traditional ML grouped baselines:

```bash
cd trajectory_to_nuclear_features
jupyter notebook traditional_ml/traditional_ml_grouped_baselines.ipynb
```

Deep-learning experiment suite:

```bash
cd trajectory_to_nuclear_features
python deep_learning/run_deep_learning_experiments.py
```

To use external data:

```bash
DATA_ROOT="/path/to/modeling/data" python deep_learning/run_deep_learning_experiments.py
```

Deep-learning outputs are written to `outputs/deep_learning/` by default. Set
`OUTPUT_DIR` to write them elsewhere.

## Modeling Notes

The traditional ML workflow deliberately excludes area-normalized trajectory
features that would leak `area_um2` into the inputs. Both method families use
nucleus-level grouping for train/test splits so loci from the same nucleus do
not appear on both sides of an evaluation split.
