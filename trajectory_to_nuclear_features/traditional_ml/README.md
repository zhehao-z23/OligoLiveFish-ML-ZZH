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
