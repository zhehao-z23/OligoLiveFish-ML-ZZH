# Extended DL Experiment Documentation

## OligoLiveFISH — Trajectory → Nuclear Feature Prediction (v3, extended)

**Script:** `run_deep_learning_experiments.py`  
**Date:** May 2026  
**Results CSV:** `outputs/deep_learning/dl_extended_raw_results.csv` (315 rows)  
**Summary CSV:** `outputs/deep_learning/dl_final_comprehensive_results.csv`

### Running

The script uses the repo-local `trajectory_to_nuclear_features/data/` directory by
default. Set `DATA_ROOT` only if you want to use a different directory with the
same layout. Run from the `trajectory_to_nuclear_features/` directory:

```bash
python deep_learning/run_deep_learning_experiments.py
```

The two notebooks (`deep_learning_architecture_sweep.ipynb`,
`engineered_feature_mlp_baseline.ipynb`) read the same `DATA_ROOT` variable
locally, or mount Google Drive when run on Colab.

---

## 1. Dataset


|                     |                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **Source**          | OligoLiveFISH — live-cell FISH, U2OS cells, chr3 loci                                             |
| **Trajectories**    | 751 (skipped 208 because <3 consecutive steps)                                                    |
| **Nuclei**          | 356 unique                                                                                        |
| **Locus**           | Green channel (G loci) only — see rationale below                                                |
| **Split**           | `GroupShuffleSplit(test_size=0.20)` by `nucleus_id` — zero nucleus overlap between train and test |
| **Train / Test**    | ~608 / ~143 trajectories                                                                          |
| **Seeds evaluated** | 0, 1, 2 — all results reported as mean ± std                                                      |


### Why G loci only

The chr3 dataset has three imaging channels targeting loci at chr3:195M (G, 488nm), chr3:195.7M (P, 565nm), and chr3:198M (R, 647nm). Only G loci are used for the following reasons:

**1. Data abundance.** G has 2,601 unique (nucleus, locus) trajectory pairs in the feature dataset vs 1,296 for P and 506 for R. Using G maximizes the training set while keeping a consistent input domain.

**2. Channel-specific intensity baselines.** The three fluorescent dyes have different brightnesses, different probe binding efficiencies, and probe different genomic positions with different local chromatin environments. Empirically, local_intensity_mean distributions differ substantially across channels: G median ≈ 152 a.u., P median ≈ 122 a.u., R median ≈ 113 a.u., and G has thousands of near-saturated measurements absent in P and R. If channels were pooled, a model predicting local_intensity_mean would primarily learn "which channel is this" rather than chromatin compaction state — a confound, not a signal.

**3. Genomic position consistency.** Each locus sits at a distinct position on chr3, embedded in a different chromatin domain with different average mobility characteristics. The 195M, 195.7M, and 198M loci are ~3–6 Mb apart. Pooling would require the model to generalize across loci with systematically different biophysical properties, making the learning problem harder and less interpretable.

**4. Conceptual scope.** This work asks whether a *single locus trajectory* encodes nucleus-level state. That question is cleanest when the locus is held fixed. Multi-locus joint modeling — using all three channels as simultaneous inputs to a shared representation — is a natural extension that would require a different architecture (concatenated inputs, graph-based, or cross-attention) and is deferred to future work.

---

## 2. Input Representation

Each trajectory is a sequence of **consecutive step vectors**:

- Blinking gaps excluded: only steps where `frame_diff == 1`
- Step vector at time t: `(dx, dy)` in nm
- Maximum sequence length: **T_MAX = 29 steps**
- Shorter trajectories: zero-padded to T_MAX with a boolean mask tracking valid positions
- Longer trajectories: last 29 steps kept

Three representations tested (Task 2):


| Name          | Features per step | Contents                             |
| ------------- | ----------------- | ------------------------------------ |
| **Cartesian** | 2                 | `(dx, dy)`                           |
| **Polar**     | 2                 | `(step_size, turning_angle)`         |
| **Combined**  | 4                 | `(dx, dy, step_size, turning_angle)` |


Cartesian is the default for all tasks except Task 2.

---

## 3. Targets (7, predicted simultaneously)


| Target                 | Unit | Description                                              |
| ---------------------- | ---- | -------------------------------------------------------- |
| `area_um2`             | µm²  | Nucleus area (median over trajectory frames)             |
| `local_intensity_mean` | a.u. | Mean fluorescence intensity in local region around locus |
| `local_to_nuc_ratio`   | —    | Local intensity / nuclear mean intensity                 |
| `nuc_intensity_mean`   | a.u. | Mean nuclear fluorescence intensity                      |
| `dist_to_membrane_nm`  | nm   | Distance from locus to nuclear membrane                  |
| `dist_to_centroid_nm`  | nm   | Distance from locus to nuclear centroid                  |
| `norm_radial_pos`      | 0–1  | Normalized radial position within nucleus                |


Each target is a **median** over all frames of the trajectory (not per-frame prediction).

---

## 4. Loss Function

**Masked MSE** — handles missing/NaN targets per sample:

```python
for each target t:
    valid = ~isnan(Y[:, t])
    loss += MSE(pred[valid, t], Y[valid, t])
```

All 7 targets predicted in a single forward pass; only valid targets contribute gradient.

---

## 5. Training Configuration (shared across all DL models)


| Hyperparameter          | Value                                                                      |
| ----------------------- | -------------------------------------------------------------------------- |
| Optimizer               | AdamW                                                                      |
| Learning rate           | 1e-3                                                                       |
| Weight decay            | 1e-4                                                                       |
| Batch size              | 32                                                                         |
| Max epochs              | 150                                                                        |
| LR schedule             | CosineAnnealingLR (T_max = 150)                                            |
| Early stopping patience | 20 epochs on validation masked MSE                                         |
| Gradient clip           | global norm = 1.0                                                          |
| Internal val split      | 10% of training set (random, not grouped) for early stopping               |
| Dropout                 | 0.3 throughout                                                             |
| X normalization         | per-feature mean/std over training set                                     |
| Y normalization         | per-target mean/std over training set; denormalized for metric computation |
| Device                  | MPS (Apple Silicon)                                                        |


---

## 6. Architectures

### Design rationale

Two architecture families were chosen to test complementary inductive biases about what structure in a trajectory is predictive:

**TrajCNN** encodes a local-pattern hypothesis: the signal lives in short-range temporal motifs — bursts of fast steps, sudden direction reversals, brief periods of confinement. A kernel of size 3 captures velocity changes between adjacent steps (2nd-order temporal patterns); two stacked conv layers give an effective receptive field of 5 steps, sufficient to detect confinement bouts at the median trajectory length of 9 steps. 1D CNNs are also sample-efficient, which matters at N ≈ 600.

**TrajLSTM** encodes a sequential-memory hypothesis: the signal is in the overall temporal dynamics of the full trajectory rather than local motifs. A bidirectional LSTM reads the sequence forward and backward simultaneously before compressing it to a fixed vector, capturing dependencies across the full length without assumptions about where in the sequence the signal sits. `pack_padded_sequence` handles variable-length inputs exactly, with no contribution from padding to the hidden state.

Running both families lets us distinguish whether the signal is local or global. In practice, CNN and LSTM achieve similar test R² on intensity targets (0.60 vs 0.60–0.63), suggesting the signal is not strongly order-dependent at this sequence length.

**Why not more complex architectures:**

- **Transformer / self-attention:** Self-attention requires N >> T to learn meaningful attention patterns; with ~600 training trajectories and T_MAX=29, there is not enough data to populate the attention weight distribution reliably. Self-attention also offers minimal benefit at T=29, where every position is already within a small number of CNN hops of every other. We would revisit this with ≥5,000 trajectories.
- **Deeper / residual CNNs:** With T=29 and kernel=3, padding=1, two conv layers already give a receptive field covering the full sequence. A third layer adds no new context window and increases parameter count. The small/medium/large size sweep confirms this: CNN-medium (R²=0.602 ± 0.173) outperforms CNN-large (R²=0.484 ± 0.320) on multi-seed average for local_intensity_mean. At N=608 more capacity hurts via overfitting.
- **Temporal convolutional networks (dilated conv):** Dilation would extend the receptive field, but the full sequence is already covered at our sequence length. No coverage argument for dilation applies at T=29.
- **Graph neural networks:** The natural extension for this data — three loci per nucleus share a chromatin fiber and likely co-vary. GNNs would model joint locus dynamics. This work treats each locus independently as a deliberate simplification to isolate the single-locus signal; multi-locus modeling is an explicit future direction.

**Size sweep rationale (small / medium / large):**

Rather than tuning a single architecture, we swept three capacity levels to get empirical overfitting curves at this dataset size. The pattern that emerges — medium beats large on intensity, large doesn't help geometric targets either — gives direct evidence that the dataset is capacity-limited and validates not going deeper.


| Variant | Capacity     | Observation                                  |
| ------- | ------------ | -------------------------------------------- |
| Small   | ~10K params  | Baseline; competitive on intensity           |
| Medium  | ~40K params  | Best on intensity (R²=0.602); sweet spot     |
| Large   | ~160K params | Overfits; worse intensity, similar geometric |


---

### TrajCNN (1D Convolutional)

```
Input: (B, T_MAX=29, n_feat)
→ permute to (B, n_feat, T_MAX)
→ Conv1d(n_feat → n_ch, kernel=3, padding=1) → BatchNorm1d → ReLU → Dropout(0.3)
→ Conv1d(n_ch → n_ch*2, kernel=3, padding=1) → BatchNorm1d → ReLU → Dropout(0.3)
→ permute back to (B, T_MAX, n_ch*2)
→ masked global average pool: sum(x * mask) / sum(mask)   [ignores padding]
→ Linear(n_ch*2 → 64) → ReLU → Dropout(0.3)
→ Linear(64 → 7)
```


| Variant    | n_ch | Output channels after 2nd conv |
| ---------- | ---- | ------------------------------ |
| CNN-small  | 32   | 64                             |
| CNN-medium | 64   | 128                            |
| CNN-large  | 128  | 256                            |


### TrajLSTM (Bidirectional LSTM)

```
Input: (B, T_MAX=29, n_feat)
→ pack_padded_sequence (lengths from mask, enforce_sorted=False)
→ BiLSTM(n_feat, hidden, num_layers=2, dropout=0.3)
→ unpack → take final hidden state: concat [h_fwd[-1], h_bwd[-1]] → (B, hidden*2)
→ Linear(hidden*2 → 64) → ReLU → Dropout(0.3)
→ Linear(64 → 7)
```


| Variant     | hidden | Final concat dim |
| ----------- | ------ | ---------------- |
| LSTM-small  | 32     | 64               |
| LSTM-medium | 64     | 128              |
| LSTM-large  | 128    | 256              |


### TrajMLP (on engineered features — Task 5)

```
Input: (B, 18)   ← 18 engineered features, NaN-imputed with training median, standard-scaled
→ Linear(18 → 64) → ReLU → Dropout(0.3)
→ Linear(64 → 32) → ReLU → Dropout(0.3)
→ Linear(32 → 7)
```

### Random Forest (Task 4 baseline)

```
sklearn RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
Fit separately per target (univariate — one RF per target)
NaN imputed with training-set median
No normalization (RF is scale-invariant)
```

---

## 7. Engineered Features (RF and MLP inputs, 18 total)

```
x_variance_nm2            y_variance_nm2
mean_step_size_nm         max_step_size_nm
displacement_variance_nm2 total_path_length_nm
net_displacement_nm       straightness_index
turning_angle_mean        turning_angle_var
turning_angle_median      turning_angle_acf_lag1
turning_angle_acf_lag2    speed_autocorr_lag1
vector_autocorr_lag1      step_size_psd_total_power
step_size_psd_peak_frequency    step_size_psd_spectral_centroid
```

---

## 8. Tasks Run


| Task       | What was varied                                     | Models                                          | Seeds   | Training runs |
| ---------- | --------------------------------------------------- | ----------------------------------------------- | ------- | ------------- |
| **Task 1** | Architecture sweep                                  | CNN-small/medium/large, LSTM-small/medium/large | 0, 1, 2 | 18            |
| **Task 2** | Input representation (cartesian / polar / combined) | CNN-large, LSTM-large                           | 0, 1, 2 | 18            |
| **Task 3** | Augmentation on/off                                 | CNN-large, cartesian                            | 0, 1, 2 | 3             |
| **Task 4** | RF baseline, nucleus-level split                    | RF (per-target)                                 | 0, 1, 2 | 3             |
| **Task 5** | MLP baseline, nucleus-level split                   | TrajMLP                                         | 0, 1, 2 | 3             |


**Total:** 45 DL training runs + 3 RF fits = **315 rows** in raw CSV (45 × 7 targets).

All previously completed runs are checkpointed to `dl_extended_checkpoint.csv`; re-running the script skips cached runs.

---

## 9. Augmentation (Task 3)

Applied on-the-fly during training only (never at test time), Cartesian input only:

- **Time reversal** (50% probability per sample):
  ```python
  x[:n] = -x[:n].flip(0)
  ```
  Physically correct: if the forward trajectory is P₀→P₁→…→Pₙ, the reversed steps are −(forward steps in reverse order).
- **Random rotation** (applied every sample, uniform θ ∈ [0, 2π]):
  ```python
  dx' =  cos(θ)·dx − sin(θ)·dy
  dy' =  sin(θ)·dx + cos(θ)·dy
  ```
  For combined representation: polar channels (step_size, turning_angle) re-derived from rotated Cartesian after rotation.

---

## 10. Results Summary

### Master Table — Test R² (mean ± std, 3 seeds)


| Target               | RF (random split) | RF-eng tuned* | MLP-eng        | Best DL                        | Signal    |
| -------------------- | ----------------- | ------------- | -------------- | ------------------------------ | --------- |
| local_intensity_mean | 0.259             | 0.620         | −0.013 ± 0.055 | **0.602 ± 0.173** (CNN-medium) | ✓ real    |
| nuc_intensity_mean   | 0.255             | 0.657         | −0.000 ± 0.084 | **0.614 ± 0.126** (CNN-medium) | ✓ stable  |
| area_um2             | 0.393             | 0.252         | −0.064 ± 0.011 | 0.086 ± 0.285 (LSTM-large)     | uncertain |
| local_to_nuc_ratio   | —                 | 0.247         | −0.013 ± 0.060 | 0.096 ± 0.290 (LSTM-small)     | uncertain |
| dist_to_membrane_nm  | —                 | 0.165         | 0.003 ± 0.112  | 0.006 ± 0.090 (LSTM-medium)    | ≈ zero    |
| dist_to_centroid_nm  | 0.050             | 0.132         | −0.138 ± 0.098 | 0.049 ± 0.038 (LSTM-medium)    | ≈ zero    |
| norm_radial_pos      | —                 | 0.109         | −0.087 ± 0.044 | −0.022 ± 0.083 (CNN-medium)    | ≈ zero    |


RF (random split): batch 1 only (515 traj), random row split — partially inflated by same-nucleus leakage.  
*RF-eng tuned: Kevin's RandomizedSearchCV + GroupKFold on 18 engineered features, nucleus-level split, single seed (random_state=42) from the `grouped_model_comparison.csv`. MLP-eng and Best DL use nucleus-level split, mean ± std over 3 seeds.

---

## 11. Ablation Logic

The four model columns form a deliberate chain — each step changes exactly one thing:

```
RF baseline (random split)
    ↓ fix split to nucleus-level
RF-eng (nucleus-level)          ← isolates leakage contribution
    ↓ swap model class only (same features, same split)
MLP-eng (nucleus-level)         ← isolates architecture contribution 
    ↓ swap input representation only (same split, same model class)
Best DL (nucleus-level)         ← isolates representation contribution
```

**Finding:** MLP-eng ≈ 0 for intensity targets while both tuned RF-eng (0.62) and Best DL (0.60) succeed. The signal is not recoverable by a neural net on engineered summaries — it requires either the raw trajectory representation (DL) or a tree-based model with the right inductive bias (RF). The DL advantage over tuned RF is methodological convenience (no feature engineering required), not a performance gain.

---

## 12. Key Findings

1. **Raw trajectory dynamics encode chromatin compaction** (intensity targets, R² ≈ 0.60), but **not spatial position** within the nucleus (positional targets, R² ≈ 0 across all models).
2. **Single-seed evaluation is unreliable at this sample size.** CNN-large with one lucky seed gave R²=0.684 for local_intensity_mean; multi-seed gives 0.484 ± 0.320. CNN-medium is both more accurate (0.602) and more stable (±0.173).
3. **The leakage finding:** The original random-split RF R²=0.393 for area_um2 collapses to 0.252 under a properly tuned nucleus-level split (and to −0.108 ± 0.037 for our untuned 3-seed RF). A substantial fraction of the original reported advantage was same-nucleus information leakage; the remainder survives with proper tuning.
4. **Augmentation counterintuitively hurts intensity targets** (ΔR² = −0.179 for local_intensity_mean). Time reversal disrupts the temporal autocorrelation structure that carries the compaction signal. It slightly helps geometric targets (+0.014 to +0.058) where rotational symmetry is more relevant.
5. **Combined representation wins for CNN; Cartesian wins for LSTM.** CNNs can selectively use features per channel--apparantly LSTMs can't filter noisy additional features effectively at small N.

---

## 13. Interpretation

Intensity (local_intensity_mean, nuc_intensity_mean) is a proxy for local chromatin compaction. Dense chromatin confines the locus, producing sub-diffusive motion with small, autocorrelated step sizes. The CNN learns this temporal autocorrelation pattern directly from the raw step vector sequence — a pattern that is washed out when the trajectory is compressed into 18 scalar summary statistics.

Spatial position (membrane distance, centroid distance, radial position) leaves no trace in step vectors, which are relative displacements. The model cannot reconstruct an absolute nuclear coordinate from local dynamics alone.

---
