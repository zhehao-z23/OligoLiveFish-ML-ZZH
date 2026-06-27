# Synthetic Trajectory Classification

Generates physics-based synthetic trajectories matched to our imaging conditions, then trains classifiers to distinguish four motion types.

## Motion Classes

| Class | Model | Key Parameter |
|-------|-------|---------------|
| Normal diffusion | Independent Gaussian steps | alpha = 1.0 |
| Confined diffusion | Ornstein-Uhlenbeck process | R = 200-800 nm |
| Directed motion | Constant drift + Brownian noise | v = 5-40 nm/s |
| Subdiffusion | Fractional Brownian motion | alpha = 0.2-0.8 |

All trajectories are corrupted with position-level localization noise (15-40 nm), which introduces realistic correlated errors between consecutive displacements.

## Scripts

**`synthetic_classification_data.py`** -- Generates balanced synthetic datasets (20K train, 4K val, 4K test) and computes 18 hand-crafted motion features for the Random Forest baseline. Features include step-size statistics, MSD ratios, displacement autocorrelation, turning angles, and spatial extent measures.

```bash
python3 synthetic_classification_data.py --full
```

**`train_trajectory_classifier.py`** -- Trains and evaluates three classifiers, then applies them to real LiveFISH trajectories.

```bash
# Train
python3 train_trajectory_classifier.py --model rf     # Random Forest (no PyTorch needed)
python3 train_trajectory_classifier.py --model lstm   # Bidirectional LSTM
python3 train_trajectory_classifier.py --model cnn    # 1D-CNN

# Predict on real data
python3 train_trajectory_classifier.py --predict --model lstm
```

## Results

All three models achieve ~68% accuracy on synthetic data. Directed motion is trivially separable (>96% recall); normal and confined diffusion are the main source of confusion. When applied to 1,640 real trajectories, all models predict 0% subdiffusion, despite MSD analysis confirming subdiffusive motion (median alpha = 0.52) -- indicating a synthetic-to-real domain gap rather than a model limitation.

## Requirements

```
numpy, scikit-learn, matplotlib
fbm           # pip install fbm (fractional Brownian motion generator)
torch         # optional, needed for LSTM and CNN only
```
