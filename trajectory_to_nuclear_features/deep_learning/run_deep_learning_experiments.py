"""
run_deep_learning_experiments.py
Tasks 1–5: multi-seed stability · input representations · augmentation · RMSE/MAE · final table

Task 1 — Multiple seeds (0, 1, 2): all 6 arch variants, final model only per seed
          (CV already done in arch sweep; seeds give variance estimates)
Task 2 — Input representations: Cartesian (dx,dy) | Polar (step_size, turning_angle)
          | Combined (all 4) — CNN-large and LSTM-large, 3 seeds each
Task 3 — Augmentation: time reversal + random rotation on/off — CNN-large, 3 seeds
Task 4 — RMSE and MAE: re-run RF on same nucleus-level split + collect from DL runs
Task 5 — Final comparison table
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
# By default, use the small derived modeling dataset shipped with this repo.
# Set DATA_ROOT to point at another directory with the same layout.
MODULE_ROOT = Path(__file__).resolve().parents[1]
BASE      = Path(os.environ.get("DATA_ROOT", MODULE_ROOT / "data")).resolve()
LOCUS_CSV = BASE / "chr3/locus_feature_table.csv"
NUC_CSV   = BASE / "chr3/nucleus_feature_table.csv"
B1_MAP    = BASE / "trajectories/batch1/Nuc_number_mapping.csv"
C3_MAP    = BASE / "trajectories/chr3_batch/Nuc_number_mapping.csv"
B1_DIR    = BASE / "trajectories/batch1"
C3_DIR    = BASE / "trajectories/chr3_batch"
OUT_DIR   = Path(os.environ.get("OUTPUT_DIR", MODULE_ROOT / "outputs" / "deep_learning")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENG_CSV   = BASE / "chr3/engineered_feature_table.csv"

# ── Config ─────────────────────────────────────────────────────────────────
TARGETS = [
    "area_um2", "local_intensity_mean", "local_to_nuc_ratio",
    "nuc_intensity_mean", "dist_to_membrane_nm", "dist_to_centroid_nm",
    "norm_radial_pos",
]
TRAJ_FEATURES = [
    "x_variance_nm2", "y_variance_nm2", "mean_step_size_nm", "max_step_size_nm",
    "displacement_variance_nm2", "total_path_length_nm", "net_displacement_nm",
    "straightness_index", "turning_angle_mean", "turning_angle_var",
    "turning_angle_median", "turning_angle_acf_lag1", "turning_angle_acf_lag2",
    "speed_autocorr_lag1", "vector_autocorr_lag1",
    "step_size_psd_total_power", "step_size_psd_peak_frequency",
    "step_size_psd_spectral_centroid",
]

T_MAX    = 29
MIN_STEPS = 3
BATCH_SZ  = 32
N_EPOCHS  = 150
LR        = 1e-3
WD        = 1e-4
PATIENCE  = 20
SEEDS     = [0, 1, 2]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

"""
build_chr3_canonical_map() picks one canonical nucleus number per (subdir, file).
Inputs:  c3_map_path — path to the chr3 Nuc_number_mapping.csv
Outputs: dict mapping (bare_subdir, original_filename) → canonical (min) Nuc_num

The same nucleus can appear under several Nuc_num values across mapping rows; we
keep the minimum so chr3 trajectories are deduplicated to one record per nucleus.
"""
def build_chr3_canonical_map(c3_map_path):
    df = pd.read_csv(c3_map_path)
    df["bare_subdir"] = df["source_subsub_directory"].str.split("/").str[-1]
    return (df.groupby(["bare_subdir", "original_filename"])["Nuc_num"]
              .min().to_dict())


"""
build_lookup() maps each (nucleus_id, locus_id) to its trajectory CSV on disk.
Inputs:  map_path — Nuc_number_mapping.csv; traj_dir — folder of trajectory CSVs;
         chr3_canon — optional canonical map from build_chr3_canonical_map() used
         to drop non-canonical chr3 duplicates
Outputs: dict {(nucleus_id, locus_id): Path to trajectory CSV} for files that exist

1. read the mapping rows
2. (chr3 only) skip rows whose Nuc_num isn't the canonical one for that nucleus
3. build the expected per-locus CSV path and keep it only if the file exists
"""
def build_lookup(map_path, traj_dir, chr3_canon=None):
    df = pd.read_csv(map_path)
    lookup = {}
    for _, row in df.iterrows():
        nid  = row["source_subsub_directory"]
        orig = row["original_filename"]
        nnum = row["Nuc_num"]
        bare = nid.split("/")[-1]
        if chr3_canon is not None:
            canon = chr3_canon.get((bare, orig))
            if canon is not None and nnum != canon:
                continue
        locus_id = orig.replace("_traj_m2DGaussian_cleaned.csv", "")
        fpath = traj_dir / f"Nuc{nnum}_{orig}"
        if fpath.exists():
            lookup[(nid, locus_id)] = fpath
    return lookup


"""
traj_to_steps() converts one trajectory CSV into a padded step-vector sequence.
Inputs:  df_traj — DataFrame with frame, x_nm, y_nm columns for one locus
Outputs: (padded, mask) where padded is (T_MAX, 2) float32 of (dx, dy) steps and
         mask is (T_MAX,) bool marking valid steps; (None, None) if too few steps

1. sort by frame and take frame-to-frame displacements
2. keep only steps between consecutive frames (drops fluorophore-blinking gaps)
3. require >= MIN_STEPS steps, else reject
4. truncate to the most recent T_MAX steps, then zero-pad to T_MAX with a mask
"""
def traj_to_steps(df_traj):
    df     = df_traj.sort_values("frame").reset_index(drop=True)
    frames = df["frame"].values.astype(int)
    x      = df["x_nm"].values.astype(float)
    y      = df["y_nm"].values.astype(float)
    consec = np.diff(frames) == 1
    dx     = np.diff(x)[consec]
    dy     = np.diff(y)[consec]
    n      = len(dx)
    if n < MIN_STEPS:
        return None, None
    steps = np.stack([dx, dy], axis=1).astype(np.float32)
    if n > T_MAX:
        steps, n = steps[-T_MAX:], T_MAX
    padded       = np.zeros((T_MAX, 2), dtype=np.float32)
    mask         = np.zeros(T_MAX, dtype=bool)
    padded[:n]   = steps
    mask[:n]     = True
    return padded, mask


"""
build_dataset() assembles the model-ready arrays from the green-channel loci.
Inputs:  lookup — (nucleus_id, locus_id) → trajectory CSV path; locus_df — per-frame
         locus feature table; nuc_df — per-nucleus feature table
Outputs: (X, masks, Y, nucleus_ids) where X is (N, T_MAX, 2) step vectors, masks is
         (N, T_MAX) bool, Y is (N, 7) targets, nucleus_ids is (N,) for grouped splits

1. restrict to G (green) loci and aggregate per-locus targets (median over frames)
2. require >= 5 frames per locus; merge in per-nucleus targets (area, intensity)
3. for each locus with a trajectory file, convert to step vectors via traj_to_steps
4. stack into arrays, carrying nucleus_id for nucleus-level GroupShuffleSplit
"""
def build_dataset(lookup, locus_df, nuc_df):
    g = locus_df[locus_df["locus_id"].str.startswith("G")].copy()
    traj_agg = (g.groupby(["nucleus_id", "locus_id"])
                  .agg(n_frames             = ("frame",                "count"),
                       local_intensity_mean = ("local_intensity_mean", "median"),
                       local_to_nuc_ratio   = ("local_to_nuc_ratio",   "median"),
                       dist_to_membrane_nm  = ("dist_to_membrane_nm",  "median"),
                       dist_to_centroid_nm  = ("dist_to_centroid_nm",  "median"),
                       norm_radial_pos      = ("norm_radial_pos",      "median"),
                       x_nm_mean            = ("x_nm",                 "mean"),
                       y_nm_mean            = ("y_nm",                 "mean"))
                  .reset_index())
    traj_agg = traj_agg[traj_agg["n_frames"] >= 5]
    nuc_agg  = (nuc_df.groupby("nucleus_id")[["area_um2", "nuc_intensity_mean"]]
                      .median().reset_index())
    traj_agg = traj_agg.merge(nuc_agg, on="nucleus_id", how="left")
    X_list, mask_list, Y_list, nid_list = [], [], [], []
    skip = 0
    for _, row in traj_agg.iterrows():
        key = (row["nucleus_id"], row["locus_id"])
        if key not in lookup:
            skip += 1; continue
        try:
            df_traj = pd.read_csv(lookup[key])
        except Exception:
            skip += 1; continue
        steps, mask = traj_to_steps(df_traj)
        if steps is None:
            skip += 1; continue
        X_list.append(steps)
        mask_list.append(mask)
        Y_list.append([
            row["area_um2"], row["local_intensity_mean"],
            row["local_to_nuc_ratio"], row["nuc_intensity_mean"],
            row["dist_to_membrane_nm"], row["dist_to_centroid_nm"],
            row["norm_radial_pos"],
        ])
        nid_list.append(row["nucleus_id"])
    print(f"  Built {len(X_list)} trajectories  (skipped {skip})")
    return (np.stack(X_list).astype(np.float32),
            np.stack(mask_list),
            np.array(Y_list, dtype=np.float32),
            np.array(nid_list))


# ═══════════════════════════════════════════════════════════════════════════
# INPUT REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
cart_to_polar() converts Cartesian step vectors to polar (step_size, turning_angle).
Inputs:  X_cart — (N, T, 2) (dx, dy) steps; masks — (N, T) bool valid-step masks
Outputs: (N, T, 2) array of (step_size, turning_angle)

turning_angle[0] = 0 by convention; subsequent entries are the signed angle between
consecutive steps. Padded positions stay 0.
"""
def cart_to_polar(X_cart, masks):
    N, T, _ = X_cart.shape
    X_polar  = np.zeros((N, T, 2), dtype=np.float32)
    for i in range(N):
        n  = int(masks[i].sum())
        if n == 0:
            continue
        dx = X_cart[i, :n, 0]
        dy = X_cart[i, :n, 1]
        X_polar[i, :n, 0] = np.sqrt(dx**2 + dy**2)         # step_size
        for j in range(1, n):
            cross = dx[j-1]*dy[j] - dy[j-1]*dx[j]
            dot   = dx[j-1]*dx[j] + dy[j-1]*dy[j]
            X_polar[i, j, 1] = np.arctan2(cross, dot)       # turning_angle
        # X_polar[i, 0, 1] stays 0 by convention
    return X_polar


"""
cart_to_combined() stacks Cartesian and polar features into one input.
Inputs:  X_cart — (N, T, 2) (dx, dy) steps; masks — (N, T) bool valid-step masks
Outputs: (N, T, 4) array of [dx, dy, step_size, turning_angle]
"""
def cart_to_combined(X_cart, masks):
    return np.concatenate([X_cart, cart_to_polar(X_cart, masks)], axis=2)


REPR_FNS = {
    "cartesian": lambda X, m: X,
    "polar":     cart_to_polar,
    "combined":  cart_to_combined,
}
REPR_NFEAT = {"cartesian": 2, "polar": 2, "combined": 4}


# ═══════════════════════════════════════════════════════════════════════════
# MODELS  (n_feat parameter added)
# ═══════════════════════════════════════════════════════════════════════════

"""
TrajCNN — 1D CNN over step-vector sequences for joint nuclear-feature regression.
Two Conv1d layers (kernel 3) + BN/ReLU/Dropout, masked global average pooling over
valid steps, then a two-layer head.
forward inputs:  x — (B, T, n_feat) steps; mask — (B, T) bool valid-step mask
forward outputs: (B, n_targets) predictions
"""
class TrajCNN(nn.Module):
    def __init__(self, n_targets, n_ch=64, drop=0.3, n_feat=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, n_ch,    3, padding=1),
            nn.BatchNorm1d(n_ch),    nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(n_ch,  n_ch*2, 3, padding=1),
            nn.BatchNorm1d(n_ch*2),  nn.ReLU(), nn.Dropout(drop),
        )
        self.head = nn.Sequential(
            nn.Linear(n_ch*2, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, n_targets),
        )
    def forward(self, x, mask):
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        m = mask.unsqueeze(-1).float()
        return self.head((x*m).sum(1) / m.sum(1).clamp(min=1))


"""
TrajLSTM — 2-layer bidirectional LSTM over step-vector sequences for joint regression.
Packs the padded sequence, concatenates the final forward/backward hidden states, then
applies the same two-layer head as TrajCNN.
forward inputs:  x — (B, T, n_feat) steps; mask — (B, T) bool valid-step mask
forward outputs: (B, n_targets) predictions
"""
class TrajLSTM(nn.Module):
    def __init__(self, n_targets, hidden=64, drop=0.3, n_feat=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, n_targets),
        )
    def forward(self, x, mask):
        lengths = mask.sum(1).cpu().clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-2], h[-1]], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION-AWARE DATASET
# ═══════════════════════════════════════════════════════════════════════════

"""
TrajDataset — step-vector dataset with optional on-the-fly augmentation.
__getitem__ inputs:  idx
__getitem__ outputs: (x (T, n_feat), mask (T,), y (n_targets,))

Augmentations (applied only when augment=True):
  - Time reversal: reversed steps = -(original steps in reverse order). This is the
    physically correct reversal: if P0→P1→...→Pn, then reversed is Pn→...→P0 with
    steps -(P[i]-P[i-1]) in reverse.
  - Random rotation: rotate (dx, dy) uniformly in [0, 2π]. Rotation-invariant
    quantities (step_size, turning_angle) are re-derived for combined input.
"""
class TrajDataset(Dataset):
    def __init__(self, X, masks, Y, augment=False, n_feat=2):
        self.X      = torch.tensor(X,     dtype=torch.float32)
        self.masks  = torch.tensor(masks, dtype=torch.bool)
        self.Y      = torch.tensor(Y,     dtype=torch.float32)
        self.augment = augment
        self.n_feat  = n_feat

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x    = self.X[idx].clone()
        mask = self.masks[idx]
        y    = self.Y[idx]

        if self.augment:
            n = int(mask.sum().item())
            if n > 1:
                # ── Time reversal (50 % probability) ─────────────────────
                # Physically correct: reversed trajectory steps are
                # -(forward steps in reverse order).
                if torch.rand(1).item() < 0.5:
                    x[:n] = -x[:n].flip(0)

                # ── Random rotation of (dx, dy) channels ─────────────────
                theta = torch.rand(1).item() * 2 * np.pi
                c, s  = float(np.cos(theta)), float(np.sin(theta))
                dx = x[:n, 0].clone()
                dy = x[:n, 1].clone()
                x[:n, 0] =  c * dx - s * dy
                x[:n, 1] =  s * dx + c * dy

                # For combined input: re-derive polar channels from rotated Cartesian
                if self.n_feat == 4:
                    dx_n = x[:n, 0].numpy()
                    dy_n = x[:n, 1].numpy()
                    ss   = np.sqrt(dx_n**2 + dy_n**2)
                    ta   = np.zeros(n, dtype=np.float32)
                    for j in range(1, n):
                        cross = dx_n[j-1]*dy_n[j] - dy_n[j-1]*dx_n[j]
                        dot   = dx_n[j-1]*dx_n[j] + dy_n[j-1]*dy_n[j]
                        ta[j] = np.arctan2(cross, dot)
                    x[:n, 2] = torch.tensor(ss)
                    x[:n, 3] = torch.tensor(ta)
        return x, mask, y


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

"""
masked_mse() computes MSE over only the non-NaN targets, summed across targets.
Inputs:  pred — (B, n_targets) predictions; target — (B, n_targets) with NaN for
         per-sample missing labels
Outputs: scalar loss tensor (skips targets with no valid samples in the batch)
"""
def masked_mse(pred, target):
    valid = ~torch.isnan(target)
    loss  = torch.tensor(0., device=pred.device)
    for t in range(target.shape[1]):
        m = valid[:, t]
        if m.sum() > 0:
            loss = loss + F.mse_loss(pred[m, t], target[m, t])
    return loss


"""
fit_model() trains a TrajCNN/TrajLSTM with early stopping on a validation split.
Inputs:  factory — zero-arg callable returning a fresh model; X_tr/m_tr/Y_tr and
         X_va/m_va/Y_va — train/validation steps, masks, targets; augment — enable
         augmentation on the training set; n_feat — input feature count
Outputs: the trained model with the best-validation weights restored

Uses masked MSE loss, AdamW + cosine annealing, grad clipping, and PATIENCE-based
early stopping on validation loss.
"""
def fit_model(factory, X_tr, m_tr, Y_tr, X_va, m_va, Y_va,
              augment=False, n_feat=2):
    model = factory().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    tr_ld = DataLoader(
        TrajDataset(X_tr, m_tr, Y_tr, augment=augment, n_feat=n_feat),
        BATCH_SZ, shuffle=True)
    va_ld = DataLoader(
        TrajDataset(X_va, m_va, Y_va, augment=False, n_feat=n_feat),
        BATCH_SZ)
    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(N_EPOCHS):
        model.train()
        for Xb, mb, Yb in tr_ld:
            opt.zero_grad()
            masked_mse(model(Xb.to(DEVICE), mb.to(DEVICE)),
                       Yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        val = 0.
        with torch.no_grad():
            for Xb, mb, Yb in va_ld:
                val += masked_mse(model(Xb.to(DEVICE), mb.to(DEVICE)),
                                  Yb.to(DEVICE)).item()
        val /= max(len(va_ld), 1)
        if val < best_val - 1e-6:
            best_val   = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


"""
predict() runs a trained TrajCNN/TrajLSTM over a dataset in eval mode.
Inputs:  model — trained model; X — (N, T, n_feat) steps; masks — (N, T) bool;
         Y — targets (only used to build the loader); n_feat — input feature count
Outputs: (N, n_targets) numpy array of normalized predictions
"""
@torch.no_grad()
def predict(model, X, masks, Y, n_feat=2):
    model.eval()
    out = []
    for Xb, mb, _ in DataLoader(TrajDataset(X, masks, Y, n_feat=n_feat), BATCH_SZ):
        out.append(model(Xb.to(DEVICE), mb.to(DEVICE)).cpu())
    return torch.cat(out).numpy()


"""
norm_X() standardizes step-vector inputs using training-set statistics.
Inputs:  X_tr, X_te — (N, T, n_feat) train/test step arrays
Outputs: (X_tr_norm, X_te_norm), both standardized by the train mean/std
"""
def norm_X(X_tr, X_te):
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, (X_te - mu) / std


"""
norm_Y() standardizes targets using NaN-aware training-set statistics.
Inputs:  Y_tr, Y_te — (N, n_targets) train/test target arrays (may contain NaN)
Outputs: (Y_tr_norm, Y_te_norm, mu, std) — normalized targets plus the train
         mean/std needed to denormalize predictions later
"""
def norm_Y(Y_tr, Y_te):
    mu  = np.nanmean(Y_tr, axis=0)
    std = np.nanstd(Y_tr,  axis=0) + 1e-8
    return (Y_tr - mu) / std, (Y_te - mu) / std, mu, std


"""
metrics() denormalizes predictions and scores each target.
Inputs:  Y_true — (N, n_targets) raw targets (may contain NaN); Y_pred_norm —
         (N, n_targets) normalized predictions; y_mu, y_std — target norm stats
Outputs: dict {target: {r2, corr, rmse, mae}}; NaN entries where <2 valid samples
"""
def metrics(Y_true, Y_pred_norm, y_mu, y_std):
    Y_pred = Y_pred_norm * y_std + y_mu
    out = {}
    for i, t in enumerate(TARGETS):
        yt = Y_true[:, i]
        yp = Y_pred[:, i]
        m  = ~np.isnan(yt)
        if m.sum() < 2:
            out[t] = dict(r2=np.nan, corr=np.nan, rmse=np.nan, mae=np.nan)
            continue
        yt, yp = yt[m], yp[m]
        out[t] = dict(
            r2   = float(r2_score(yt, yp)),
            corr = float(np.corrcoef(yt, yp)[0, 1]),
            rmse = float(np.sqrt(np.mean((yt - yp)**2))),
            mae  = float(np.mean(np.abs(yt - yp))),
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MLP ON ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

"""
EngDataset — minimal (X, Y) tensor dataset for the engineered-feature MLP.
__getitem__ outputs: (x (n_feat,), y (n_targets,))
"""
class EngDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


"""
TrajMLP — 3-layer MLP baseline on the 18 engineered features (64→32→n_targets).
forward inputs:  x — (B, n_feat) engineered feature vectors
forward outputs: (B, n_targets) predictions
"""
class TrajMLP(nn.Module):
    def __init__(self, n_feat, n_targets, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 32),    nn.ReLU(), nn.Dropout(drop),
            nn.Linear(32, n_targets),
        )
    def forward(self, x):
        return self.net(x)


"""
fit_mlp() trains the engineered-feature MLP with early stopping.
Inputs:  X_tr, Y_tr, X_va, Y_va — train/validation feature and target arrays
Outputs: the trained model with best-validation weights restored

Mirrors fit_model (masked MSE, AdamW + cosine annealing, grad clip, early stopping)
so the only difference vs the trajectory models is the input representation.
"""
def fit_mlp(X_tr, Y_tr, X_va, Y_va):
    n_feat = X_tr.shape[1]
    model  = TrajMLP(n_feat=n_feat, n_targets=len(TARGETS)).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    tr_ld  = DataLoader(EngDataset(X_tr, Y_tr), BATCH_SZ, shuffle=True)
    va_ld  = DataLoader(EngDataset(X_va, Y_va), BATCH_SZ)
    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(N_EPOCHS):
        model.train()
        for Xb, Yb in tr_ld:
            opt.zero_grad()
            masked_mse(model(Xb.to(DEVICE)), Yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        val = 0.
        with torch.no_grad():
            for Xb, Yb in va_ld:
                val += masked_mse(model(Xb.to(DEVICE)), Yb.to(DEVICE)).item()
        val /= max(len(va_ld), 1)
        if val < best_val - 1e-6:
            best_val   = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


"""
predict_mlp() runs the trained engineered-feature MLP in eval mode.
Inputs:  model — trained TrajMLP; X — (N, n_feat) features; Y — targets (loader only)
Outputs: (N, n_targets) numpy array of normalized predictions
"""
def predict_mlp(model, X, Y):
    model.eval()
    out = []
    with torch.no_grad():
        for Xb, _ in DataLoader(EngDataset(X, Y), BATCH_SZ):
            out.append(model(Xb.to(DEVICE)).cpu())
    return torch.cat(out).numpy()


"""
run_mlp_single() runs one full MLP experiment (one seed) on engineered features.
Inputs:  X_eng_raw — (N, n_feat) engineered features (may contain NaN); Y_eng —
         (N, n_targets) targets; nids_eng — (N,) nucleus ids; seed — random seed
Outputs: dict of per-target metrics from metrics()

1. nucleus-level GroupShuffleSplit into train/test
2. impute NaN features with the training median (matches the RF baseline)
3. standard-scale inputs and normalize targets
4. train with an internal early-stopping hold-out, then evaluate on the test set
"""
def run_mlp_single(X_eng_raw, Y_eng, nids_eng, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_eng_raw, Y_eng, groups=nids_eng))

    # Impute NaN with training median (same as RF baseline)
    tr_med    = np.nanmedian(X_eng_raw[tr_idx], axis=0)
    X_tr_imp  = np.where(np.isnan(X_eng_raw[tr_idx]), tr_med, X_eng_raw[tr_idx])
    X_te_imp  = np.where(np.isnan(X_eng_raw[te_idx]), tr_med, X_eng_raw[te_idx])
    Y_tr      = Y_eng[tr_idx]
    Y_te      = Y_eng[te_idx]

    # Standard-scale inputs; normalise targets same as DL pipeline
    mu_x      = X_tr_imp.mean(axis=0);  std_x = X_tr_imp.std(axis=0) + 1e-8
    X_tr_n    = ((X_tr_imp - mu_x) / std_x).astype(np.float32)
    X_te_n    = ((X_te_imp - mu_x) / std_x).astype(np.float32)
    Y_tr_n, Y_te_n, y_mu, y_std = norm_Y(Y_tr, Y_te)

    # Internal hold-out for early stopping
    fi, vi = train_test_split(np.arange(len(X_tr_n)), test_size=0.1,
                               random_state=seed)

    model    = fit_mlp(X_tr_n[fi], Y_tr_n[fi], X_tr_n[vi], Y_tr_n[vi])
    te_preds = predict_mlp(model, X_te_n, Y_te_n)
    return metrics(Y_te, te_preds, y_mu, y_std)


# ═══════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

"""
run_single() runs one full trajectory-model experiment (one seed/config).
Inputs:  X_cart — (N, T, 2) Cartesian steps; masks — (N, T) bool; Y — (N, n_targets);
         nucleus_ids — (N,) for grouped split; ModelCls — TrajCNN/TrajLSTM;
         model_kwargs — capacity kwargs; seed — random seed; repr_name —
         cartesian/polar/combined; augment — enable augmentation
Outputs: dict of per-target metrics from metrics()

1. nucleus-level GroupShuffleSplit into train/test
2. convert Cartesian steps to the chosen input representation
3. standardize inputs and normalize targets
4. train (with internal early-stopping hold-out) and evaluate on the test set
"""
def run_single(X_cart, masks, Y, nucleus_ids,
               ModelCls, model_kwargs,
               seed, repr_name="cartesian", augment=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_cart, Y, groups=nucleus_ids))

    # Input representation
    n_feat    = REPR_NFEAT[repr_name]
    repr_fn   = REPR_FNS[repr_name]
    X_repr    = repr_fn(X_cart, masks)   # (N, T, n_feat)

    X_tr_r, m_tr = X_repr[tr_idx], masks[tr_idx]
    X_te_r, m_te = X_repr[te_idx], masks[te_idx]
    Y_tr_r        = Y[tr_idx]; Y_te_r = Y[te_idx]

    X_tr_n, X_te_n              = norm_X(X_tr_r, X_te_r)
    Y_tr_n, Y_te_n, y_mu, y_std = norm_Y(Y_tr_r, Y_te_r)

    # Internal hold-out for early stopping
    fi, vi = train_test_split(np.arange(len(X_tr_n)), test_size=0.1,
                               random_state=seed)

    factory = lambda: ModelCls(len(TARGETS), **model_kwargs, n_feat=n_feat)
    model   = fit_model(factory,
                        X_tr_n[fi], m_tr[fi], Y_tr_n[fi],
                        X_tr_n[vi], m_tr[vi], Y_tr_n[vi],
                        augment=augment, n_feat=n_feat)

    te_preds = predict(model, X_te_n, m_te, Y_te_n, n_feat=n_feat)
    return metrics(Y_te_r, te_preds, y_mu, y_std)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

"""
main() runs the full experiment suite and writes the result tables.
Inputs:  none (reads data paths from the module-level path config)
Outputs: none (runs the seed/representation/augmentation experiments + the RF and
         MLP baselines, prints the master comparison table, and saves result CSVs)
"""
def main():
    # ── Load trajectory dataset ────────────────────────────────────────────
    print("Building lookup tables ...")
    c3_canon  = build_chr3_canonical_map(C3_MAP)
    lookup_b1 = build_lookup(B1_MAP, B1_DIR)
    lookup_c3 = build_lookup(C3_MAP, C3_DIR, chr3_canon=c3_canon)
    lookup    = {**lookup_b1, **lookup_c3}

    print("Loading feature tables ...")
    locus_df  = pd.read_csv(LOCUS_CSV)
    nuc_df    = pd.read_csv(NUC_CSV)

    print("Building dataset ...")
    X_cart, masks, Y, nucleus_ids = build_dataset(lookup, locus_df, nuc_df)
    print(f"  {len(X_cart)} trajectories | {len(set(nucleus_ids))} nuclei")

    # ── Checkpoint: load previously completed rows ──────────────────────────
    CKPT = OUT_DIR / "dl_extended_checkpoint.csv"
    if CKPT.exists():
        ckpt_df  = pd.read_csv(CKPT)
        all_rows = ckpt_df.to_dict("records")
        done_keys = set(
            zip(ckpt_df["task"], ckpt_df["variant"],
                ckpt_df["repr"], ckpt_df["augment"].astype(str),
                ckpt_df["seed"], ckpt_df["target"])
        )
        print(f"  Resuming: {len(ckpt_df)} rows already in checkpoint")
    else:
        all_rows  = []
        done_keys = set()

    def already_done(task, variant, repr_name, augment, seed):
        """True if ALL 7 targets for this run are already in the checkpoint."""
        return all(
            (task, variant, repr_name, str(augment), seed, t) in done_keys
            for t in TARGETS
        )

    def save_ckpt():
        pd.DataFrame(all_rows).to_csv(CKPT, index=False)

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1 — Multiple seeds across all 6 architecture variants
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("TASK 1 — Multi-seed (seeds 0,1,2), all 6 arch variants")
    print("═"*60)

    ARCH_VARIANTS = [
        ("CNN-small",   TrajCNN,  {"n_ch": 32}),
        ("CNN-medium",  TrajCNN,  {"n_ch": 64}),
        ("CNN-large",   TrajCNN,  {"n_ch": 128}),
        ("LSTM-small",  TrajLSTM, {"hidden": 32}),
        ("LSTM-medium", TrajLSTM, {"hidden": 64}),
        ("LSTM-large",  TrajLSTM, {"hidden": 128}),
    ]

    for variant, ModelCls, kwargs in ARCH_VARIANTS:
        for seed in SEEDS:
            if already_done("multiseed", variant, "cartesian", False, seed):
                print(f"  {variant}  seed={seed} ... (cached)")
                continue
            print(f"  {variant}  seed={seed} ...", end=" ", flush=True)
            m = run_single(X_cart, masks, Y, nucleus_ids,
                           ModelCls, kwargs, seed,
                           repr_name="cartesian", augment=False)
            for t in TARGETS:
                all_rows.append({"task": "multiseed", "variant": variant,
                                  "repr": "cartesian", "augment": False,
                                  "seed": seed, "target": t, **m[t]})
            save_ckpt()
            print("done")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2 — Input representation comparison
    # Using CNN-large and LSTM-large (best overall architectures)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("TASK 2 — Input representations (cartesian / polar / combined)")
    print("  Architectures: CNN-large, LSTM-large | seeds 0,1,2")
    print("═"*60)

    REPR_ARCHS = [
        ("CNN-large",  TrajCNN,  {"n_ch": 128}),
        ("LSTM-large", TrajLSTM, {"hidden": 128}),
    ]
    for repr_name in ["cartesian", "polar", "combined"]:
        for variant, ModelCls, kwargs in REPR_ARCHS:
            for seed in SEEDS:
                if already_done("representation", variant, repr_name, False, seed):
                    print(f"  {variant}  repr={repr_name}  seed={seed} ... (cached)")
                    continue
                print(f"  {variant}  repr={repr_name}  seed={seed} ...",
                      end=" ", flush=True)
                m = run_single(X_cart, masks, Y, nucleus_ids,
                               ModelCls, kwargs, seed,
                               repr_name=repr_name, augment=False)
                for t in TARGETS:
                    all_rows.append({"task": "representation", "variant": variant,
                                      "repr": repr_name, "augment": False,
                                      "seed": seed, "target": t, **m[t]})
                save_ckpt()
                print("done")

    # Note: cartesian results for CNN-large and LSTM-large are already in
    # the multiseed task above — but we re-run here per seed for repr table
    # consistency. The representation table uses only Task 2 rows.

    # ══════════════════════════════════════════════════════════════════════
    # TASK 3 — Data augmentation
    # CNN-large, cartesian input, augment=True vs False (already have False)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("TASK 3 — Augmentation: CNN-large, time-reversal + rotation, seeds 0,1,2")
    print("  (no-aug baseline comes from Task 1 CNN-large rows)")
    print("═"*60)

    for seed in SEEDS:
        if already_done("augmentation", "CNN-large", "cartesian", True, seed):
            print(f"  CNN-large  augment=True  seed={seed} ... (cached)")
            continue
        print(f"  CNN-large  augment=True  seed={seed} ...", end=" ", flush=True)
        m = run_single(X_cart, masks, Y, nucleus_ids,
                       TrajCNN, {"n_ch": 128}, seed,
                       repr_name="cartesian", augment=True)
        for t in TARGETS:
            all_rows.append({"task": "augmentation", "variant": "CNN-large",
                              "repr": "cartesian", "augment": True,
                              "seed": seed, "target": t, **m[t]})
        save_ckpt()
        print("done")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 4 — RF baseline RMSE/MAE on same nucleus-level split (seed=42)
    # Also collect RF on seeds 0,1,2 for completeness
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("TASK 4 — RF on engineered features, nucleus-level split, seeds 0,1,2")
    print("═"*60)

    df_eng = pd.read_csv(ENG_CSV)
    df_eng = df_eng[df_eng["locus_id"].str.startswith("G")].copy()
    df_eng = df_eng.dropna(subset=TRAJ_FEATURES, how="all")
    # rename nuc_mean_intensity → nuc_intensity_mean for consistency
    if "nuc_mean_intensity" in df_eng.columns and "nuc_intensity_mean" not in df_eng.columns:
        df_eng = df_eng.rename(columns={"nuc_mean_intensity": "nuc_intensity_mean"})
    # align target name
    eng_targets = [t if t in df_eng.columns else t for t in TARGETS]
    X_eng   = df_eng[TRAJ_FEATURES].values.astype(np.float32)
    Y_eng   = df_eng[eng_targets].values.astype(np.float32)
    nids_eng = df_eng["nucleus_id"].values

    for seed in SEEDS:
        if already_done("rf_baseline", "RF-engineered", "engineered", False, seed):
            print(f"  RF  seed={seed} ... (cached)")
            continue
        np.random.seed(seed)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, te_idx = next(gss.split(X_eng, Y_eng, groups=nids_eng))

        tr_med   = np.nanmedian(X_eng[tr_idx], axis=0)
        X_tr_imp = np.where(np.isnan(X_eng[tr_idx]), tr_med, X_eng[tr_idx])
        X_te_imp = np.where(np.isnan(X_eng[te_idx]), tr_med, X_eng[te_idx])
        Y_tr     = Y_eng[tr_idx]
        Y_te     = Y_eng[te_idx]

        print(f"  RF  seed={seed} ...", end=" ", flush=True)
        for i, t in enumerate(TARGETS):
            tr_valid = ~np.isnan(Y_tr[:, i])
            if tr_valid.sum() < 5:
                all_rows.append({"task": "rf_baseline", "variant": "RF-engineered",
                                  "repr": "engineered", "augment": False,
                                  "seed": seed, "target": t,
                                  "r2": np.nan, "corr": np.nan,
                                  "rmse": np.nan, "mae": np.nan})
                continue
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
            rf.fit(X_tr_imp[tr_valid], Y_tr[tr_valid, i])
            yp = rf.predict(X_te_imp)
            yt = Y_te[:, i]
            m  = ~np.isnan(yt)
            if m.sum() < 2:
                r2_, co_, rm_, ma_ = np.nan, np.nan, np.nan, np.nan
            else:
                r2_ = float(r2_score(yt[m], yp[m]))
                co_ = float(np.corrcoef(yt[m], yp[m])[0, 1])
                rm_ = float(np.sqrt(np.mean((yt[m] - yp[m])**2)))
                ma_ = float(np.mean(np.abs(yt[m] - yp[m])))
            all_rows.append({"task": "rf_baseline", "variant": "RF-engineered",
                              "repr": "engineered", "augment": False,
                              "seed": seed, "target": t,
                              "r2": r2_, "corr": co_,
                              "rmse": rm_, "mae": ma_})
        save_ckpt()
        print("done")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 5 — MLP on engineered features, nucleus-level split, seeds 0,1,2
    # Same inputs/split as RF baseline (Task 4) — ablates architecture vs repr
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("TASK 5 — MLP on engineered features, nucleus-level split, seeds 0,1,2")
    print("═"*60)

    for seed in SEEDS:
        if already_done("mlp_baseline", "MLP-engineered", "engineered", False, seed):
            print(f"  MLP  seed={seed} ... (cached)")
            continue
        print(f"  MLP  seed={seed} ...", end=" ", flush=True)
        m = run_mlp_single(X_eng, Y_eng, nids_eng, seed)
        for t in TARGETS:
            all_rows.append({"task": "mlp_baseline", "variant": "MLP-engineered",
                              "repr": "engineered", "augment": False,
                              "seed": seed, "target": t, **m[t]})
        save_ckpt()
        print("done")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RAW RESULTS
    # ══════════════════════════════════════════════════════════════════════
    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(OUT_DIR / "dl_extended_raw_results.csv", index=False)
    print(f"\nRaw results saved: {len(raw_df)} rows")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 5 — FINAL TABLES
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*70)
    print("TASK 6 — FINAL TABLES")
    print("═"*70)

    def agg(df):
        """Mean ± std over seeds for each (variant, repr, augment, target)."""
        return (df.groupby(["variant", "repr", "augment", "target"])
                  [["r2", "corr", "rmse", "mae"]]
                  .agg(["mean", "std"])
                  .reset_index())

    def fmt(mean, std):
        if np.isnan(mean): return "    —"
        if np.isnan(std):  return f"{mean:.3f}"
        return f"{mean:.3f}±{std:.3f}"

    # ── Table 1: Multi-seed — best per target ──────────────────────────────
    t1 = raw_df[raw_df.task == "multiseed"]
    t1_agg = agg(t1)

    print("\n── TABLE 1: Multi-seed Test R² (mean±std, 3 seeds) ──────────────")
    print("  Architecture×Repr=cartesian, no augmentation")
    print()

    variants_order = [v[0] for v in ARCH_VARIANTS]
    header = f"{'Target':<24}" + "".join(f"  {v:>18}" for v in variants_order)
    print(header)
    print("─" * (24 + 20 * len(variants_order)))

    for t in TARGETS:
        row_str = f"{t:<24}"
        for v in variants_order:
            sub = t1_agg[(t1_agg.variant == v) &
                         (t1_agg.repr == "cartesian") &
                         (t1_agg.augment == False) &
                         (t1_agg.target == t)]
            if len(sub) == 0:
                row_str += f"  {'—':>18}"
            else:
                mean_ = sub[("r2", "mean")].iloc[0]
                std_  = sub[("r2", "std")].iloc[0]
                row_str += f"  {fmt(mean_, std_):>18}"
        print(row_str)

    # Best per target across all variants
    best_per_target = {}
    for t in TARGETS:
        sub = t1_agg[(t1_agg.repr == "cartesian") &
                     (t1_agg.augment == False) &
                     (t1_agg.target == t)]
        if len(sub) == 0: continue
        idx = sub[("r2", "mean")].idxmax()
        best_per_target[t] = (sub.loc[idx, "variant"],
                               sub.loc[idx, ("r2",   "mean")],
                               sub.loc[idx, ("r2",   "std")],
                               sub.loc[idx, ("corr", "mean")],
                               sub.loc[idx, ("corr", "std")],
                               sub.loc[idx, ("rmse", "mean")],
                               sub.loc[idx, ("rmse", "std")],
                               sub.loc[idx, ("mae",  "mean")],
                               sub.loc[idx, ("mae",  "std")])

    print("\n── TABLE 2: Best DL architecture per target (mean±std, 3 seeds) ──")
    print(f"{'Target':<24} {'Best model':<14} {'R² mean±std':>14} "
          f"{'Corr mean±std':>15} {'RMSE mean±std':>15} {'MAE mean±std':>14}")
    print("─" * 100)
    for t in TARGETS:
        if t not in best_per_target: continue
        v, r2m, r2s, cm, cs, rmm, rms, mam, mas = best_per_target[t]
        print(f"{t:<24} {v:<14} {fmt(r2m,r2s):>14} "
              f"{fmt(cm,cs):>15} {fmt(rmm,rms):>15} {fmt(mam,mas):>14}")

    # ── Table 3: Input representation comparison ──────────────────────────
    t2 = raw_df[raw_df.task == "representation"]
    t2_agg = agg(t2)

    print("\n── TABLE 3: Input representation Test R² (mean±std, 3 seeds) ────")
    for arch in ["CNN-large", "LSTM-large"]:
        print(f"\n  {arch}")
        print(f"  {'Target':<24} {'Cartesian':>14} {'Polar':>14} {'Combined':>14}")
        print("  " + "─" * 68)
        for t in TARGETS:
            row_str = f"  {t:<24}"
            for repr_name in ["cartesian", "polar", "combined"]:
                sub = t2_agg[(t2_agg.variant == arch) &
                             (t2_agg.repr == repr_name) &
                             (t2_agg.target == t)]
                if len(sub) == 0:
                    row_str += f"  {'—':>12}"
                else:
                    m_ = sub[("r2", "mean")].iloc[0]
                    s_ = sub[("r2", "std")].iloc[0]
                    row_str += f"  {fmt(m_, s_):>12}"
            print(row_str)

    # ── Table 4: Augmentation comparison ─────────────────────────────────
    # no-aug = Task 1 CNN-large; aug = Task 3
    t3_aug   = raw_df[(raw_df.task == "augmentation") & (raw_df.augment == True)]
    t3_noaug = raw_df[(raw_df.task == "multiseed") &
                      (raw_df.variant == "CNN-large") &
                      (raw_df.repr == "cartesian")]
    t3_agg_aug   = agg(t3_aug)
    t3_agg_noaug = agg(t3_noaug)

    print("\n── TABLE 4: Augmentation effect — CNN-large, cartesian ──────────")
    print(f"  {'Target':<24} {'No aug R²':>12} {'Augmented R²':>14} {'Δ R²':>8}")
    print("  " + "─" * 62)
    for t in TARGETS:
        sub_n = t3_agg_noaug[(t3_agg_noaug.target == t)]
        sub_a = t3_agg_aug[(t3_agg_aug.target == t)]
        mn = sub_n[("r2","mean")].iloc[0] if len(sub_n) else np.nan
        sn = sub_n[("r2","std")].iloc[0]  if len(sub_n) else np.nan
        ma = sub_a[("r2","mean")].iloc[0] if len(sub_a) else np.nan
        sa = sub_a[("r2","std")].iloc[0]  if len(sub_a) else np.nan
        delta = "" if np.isnan(mn) or np.isnan(ma) else f"{ma-mn:+.3f}"
        print(f"  {t:<24} {fmt(mn,sn):>12} {fmt(ma,sa):>14} {delta:>8}")

    # ── Table 5: RF baseline ──────────────────────────────────────────────
    t4 = raw_df[raw_df.task == "rf_baseline"]
    t4_agg = agg(t4)

    print("\n── TABLE 5: RF on engineered features, nucleus-level split ──────")
    print(f"  {'Target':<24} {'R² mean±std':>14} {'Corr mean±std':>15} "
          f"{'RMSE mean±std':>15} {'MAE mean±std':>14}")
    print("  " + "─" * 82)
    for t in TARGETS:
        sub = t4_agg[(t4_agg.target == t)]
        if len(sub) == 0:
            print(f"  {t:<24} {'—':>14}")
            continue
        r2m, r2s = sub[("r2","mean")].iloc[0], sub[("r2","std")].iloc[0]
        cm,  cs  = sub[("corr","mean")].iloc[0], sub[("corr","std")].iloc[0]
        rmm, rms = sub[("rmse","mean")].iloc[0], sub[("rmse","std")].iloc[0]
        mam, mas = sub[("mae","mean")].iloc[0],  sub[("mae","std")].iloc[0]
        print(f"  {t:<24} {fmt(r2m,r2s):>14} {fmt(cm,cs):>15} "
              f"{fmt(rmm,rms):>15} {fmt(mam,mas):>14}")

    # ── Table 6: MLP baseline ─────────────────────────────────────────────
    t5 = raw_df[raw_df.task == "mlp_baseline"]
    t5_agg = agg(t5)

    print("\n── TABLE 6: MLP on engineered features, nucleus-level split ─────")
    print(f"  {'Target':<24} {'R² mean±std':>14} {'Corr mean±std':>15} "
          f"{'RMSE mean±std':>15} {'MAE mean±std':>14}")
    print("  " + "─" * 82)
    for t in TARGETS:
        sub = t5_agg[(t5_agg.target == t)]
        if len(sub) == 0:
            print(f"  {t:<24} {'—':>14}")
            continue
        r2m, r2s = sub[("r2","mean")].iloc[0], sub[("r2","std")].iloc[0]
        cm,  cs  = sub[("corr","mean")].iloc[0], sub[("corr","std")].iloc[0]
        rmm, rms = sub[("rmse","mean")].iloc[0], sub[("rmse","std")].iloc[0]
        mam, mas = sub[("mae","mean")].iloc[0],  sub[("mae","std")].iloc[0]
        print(f"  {t:<24} {fmt(r2m,r2s):>14} {fmt(cm,cs):>15} "
              f"{fmt(rmm,rms):>15} {fmt(mam,mas):>14}")

    # ── Master table: RF vs MLP vs Best-DL vs Augmented-DL ───────────────
    print("\n" + "═"*110)
    print("MASTER TABLE — Test R² (mean±std over 3 seeds)")
    print("  RF-baseline† = batch-1 random split; RF-eng & MLP = nucleus-level split (same as DL)")
    print("═"*110)

    # RF baseline numbers (batch 1 only, random row-level split — inflated by
    # same-nucleus leakage; kept here for reference against the nucleus-level runs).
    rf_baseline = {
        "area_um2":            0.393, "local_intensity_mean": 0.259,
        "local_to_nuc_ratio":  None,  "nuc_intensity_mean":   0.255,
        "dist_to_membrane_nm": None,  "dist_to_centroid_nm":  0.050,
        "norm_radial_pos":     None,
    }

    print(f"\n{'Target':<24} {'RF-base†':>10} {'RF-eng':>14} {'MLP-eng':>14}  "
          f"{'Best-DL (no aug)':>22}  {'CNN-L+aug':>14}")
    print("─" * 110)
    for t in TARGETS:
        xrf   = rf_baseline.get(t)
        xrf_s = f"{xrf:.3f}" if xrf else "(new)"

        rf_sub = t4_agg[t4_agg.target == t]
        rf_s   = fmt(rf_sub[("r2","mean")].iloc[0],
                     rf_sub[("r2","std")].iloc[0]) if len(rf_sub) else "—"

        mlp_sub = t5_agg[t5_agg.target == t]
        mlp_s   = fmt(mlp_sub[("r2","mean")].iloc[0],
                      mlp_sub[("r2","std")].iloc[0]) if len(mlp_sub) else "—"

        if t in best_per_target:
            v, r2m, r2s = best_per_target[t][:3]
            dl_s = f"{fmt(r2m,r2s)} ({v})"
        else:
            dl_s = "—"

        aug_sub = t3_agg_aug[t3_agg_aug.target == t]
        aug_s   = fmt(aug_sub[("r2","mean")].iloc[0],
                      aug_sub[("r2","std")].iloc[0]) if len(aug_sub) else "—"

        print(f"{t:<24} {xrf_s:>10} {rf_s:>14} {mlp_s:>14}  {dl_s:<22}  {aug_s:>14}")
    print("═"*110)
    print("† RF baseline: batch 1 only (515 traj), random split — slightly inflated vs DL\n")

    # Save final summary
    summary_rows = []
    for t in TARGETS:
        rf_sub = t4_agg[t4_agg.target == t]
        aug_sub = t3_agg_aug[t3_agg_aug.target == t]
        row = {"target": t, "rf_baseline_r2": rf_baseline.get(t)}
        if len(rf_sub):
            row.update({
                "rf_eng_r2_mean":   rf_sub[("r2","mean")].iloc[0],
                "rf_eng_r2_std":    rf_sub[("r2","std")].iloc[0],
                "rf_eng_corr_mean": rf_sub[("corr","mean")].iloc[0],
                "rf_eng_rmse_mean": rf_sub[("rmse","mean")].iloc[0],
                "rf_eng_mae_mean":  rf_sub[("mae","mean")].iloc[0],
            })
        if t in best_per_target:
            v, r2m, r2s, cm, cs, rmm, rms, mam, mas = best_per_target[t]
            row.update({
                "best_dl_model": v,
                "best_dl_r2_mean": r2m, "best_dl_r2_std": r2s,
                "best_dl_corr_mean": cm, "best_dl_corr_std": cs,
                "best_dl_rmse_mean": rmm, "best_dl_rmse_std": rms,
                "best_dl_mae_mean": mam, "best_dl_mae_std": mas,
            })
        if len(aug_sub):
            row.update({
                "cnn_large_aug_r2_mean": aug_sub[("r2","mean")].iloc[0],
                "cnn_large_aug_r2_std":  aug_sub[("r2","std")].iloc[0],
            })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "dl_final_comprehensive_results.csv", index=False)
    print(f"Summary saved to: {OUT_DIR / 'dl_final_comprehensive_results.csv'}")
    raw_df.to_csv(OUT_DIR / "dl_extended_raw_results.csv", index=False)
    print(f"Raw results saved to: {OUT_DIR / 'dl_extended_raw_results.csv'}")


if __name__ == "__main__":
    main()
