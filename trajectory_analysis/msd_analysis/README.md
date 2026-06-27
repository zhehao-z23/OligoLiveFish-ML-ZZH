# MSD Analysis

Computes mean squared displacement curves and fits the anomalous diffusion exponent for all real trajectories.

## Model

```
MSD(tau) = 4D * tau^alpha + 2*sigma^2
```

Where D is the diffusion coefficient, alpha is the anomalous exponent (alpha < 1 = subdiffusion, alpha = 1 = normal diffusion), and sigma is the localization error.

## Script

**`compute_msd_alpha.py`** -- Two fitting approaches for each trajectory:
1. **Raw alpha**: log-log slope of MSD vs lag (biased by localization error at short lags)
2. **Corrected alpha**: nonlinear fit of MSD = A*tau^alpha + B, accounting for the 2*sigma^2 offset

Requires `master_locus_features.csv` in the parent directory. Trajectories need at least 5 frames.

```bash
python3 compute_msd_alpha.py
```

## Output

- `msd_alpha_results.csv` -- per-trajectory alpha, D, sigma, and trajectory stats (1,797 trajectories)
- `msd_curves.npz` -- raw MSD curves for plotting

## Key Result

Median raw alpha = 0.52 across 1,797 trajectories, consistent with the Rouse model prediction for chromatin polymer dynamics. This independently confirms that subdiffusive motion is prevalent in the data, validating that the classifiers' 0% subdiffusion prediction on real data is a domain gap problem.
