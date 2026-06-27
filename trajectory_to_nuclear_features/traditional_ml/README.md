# Traditional ML Baselines

`traditional_ml_grouped_baselines.ipynb` reruns the corrected grouped baseline
analysis on `../data/chr3/engineered_feature_table.csv`.

The notebook trains:

- `mean_baseline`
- `elastic_net`
- `random_forest`

For each target, it drops rows missing that target, tunes Elastic Net and
Random Forest with grouped cross-validation over training nuclei, then reports
performance on held-out nuclei.

Committed reference outputs are in:

```text
../data/chr3/traditional_ml_results/
```

The most useful summary file is `grouped_model_comparison.csv`, which is also
referenced by the deep-learning results document as the tuned Random Forest
baseline.

## Feature Extraction

`engineered_traj_feature_extraction.py` extracts biophysical motion features from per-locus trajectory CSVs and prepares a merged feature table for downstream modeling.

For each DNA locus trajectory, it computes 21 features capturing spatial spread (convex hull area, radius of gyration), step statistics (mean/max step size, straightness index), turning angle statistics, autocorrelations (speed, direction, persistence length), and frequency/wavelet features from the step size and turning angle signals.

The script runs in three steps: feature extraction from trajectory CSVs, normalization by nucleus area, and merging with per-locus spatial measurements (distance to membrane, local chromatin intensity, etc.). See the docstring at the top of the file for the expected folder layout and the two manual cleaning steps required between runs.
