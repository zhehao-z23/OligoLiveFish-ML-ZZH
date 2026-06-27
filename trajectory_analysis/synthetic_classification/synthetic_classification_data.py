"""
synthetic_classification_data.py — Generate synthetic trajectories for motion classification.

Generates 4 classes of 2D particle motion:
    0 = Normal diffusion (Brownian motion, alpha ~ 1.0)
    1 = Confined diffusion (particle in a harmonic potential well)
    2 = Directed motion (drift + diffusion)
    3 = Anomalous subdiffusion (fBm with alpha < 1)

Parameters are matched to Oligo-LiveFISH data:
    - dt = 24.5 s (frame interval)
    - pixel_size = 183.3 nm
    - trajectory lengths: 5-30 steps (matching real data distribution)
    - localization noise: 15-40 nm (ThunderSTORM-level)

Usage:
    python3 synthetic_classification_data.py              # generate small test batch + verify
    python3 synthetic_classification_data.py --full       # generate full training dataset
"""

import numpy as np
from fbm import fbm as generate_fbm
import os
import sys
from typing import Tuple, Dict


# ── Physical constants matched to our data ──────────────────────────────────
DT = 24.5           # seconds per frame
PIXEL_SIZE_NM = 183.3
MIN_STEPS = 5       # minimum trajectory length (steps, not positions)
MAX_STEPS = 30      # maximum trajectory length
MEDIAN_STEPS = 10   # peak of length distribution

CLASS_NAMES = {0: "normal", 1: "confined", 2: "directed", 3: "subdiffusion"}
N_CLASSES = 4


def sample_trajectory_length(rng: np.random.Generator) -> int:
    """Sample trajectory length matching real data distribution.

    Real data: median ~10 frames, range 5-30, right-skewed.
    We use a shifted geometric-like distribution.
    """
    # Log-normal centered around median
    length = int(rng.lognormal(mean=np.log(MEDIAN_STEPS), sigma=0.4))
    return np.clip(length, MIN_STEPS, MAX_STEPS)


def generate_normal_diffusion(n_steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Class 0: Pure Brownian motion (alpha ≈ 1.0).

    MSD(tau) = 4 * D * tau  (2D)
    """
    # D in nm^2/s — range relevant to chromatin loci (~100-5000 nm^2/s)
    D = 10 ** rng.uniform(2.0, 3.7)  # 100 to ~5000 nm^2/s

    sigma_step = np.sqrt(2 * D * DT)  # per-dimension step std
    displacements = rng.normal(0, sigma_step, size=(n_steps, 2))

    params = {"D": D, "alpha": 1.0}
    return displacements, params


def generate_confined_diffusion(n_steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Class 1: Confined diffusion (Ornstein-Uhlenbeck process).

    Particle in a harmonic potential well with spring constant k.
    dx = -k * x * dt + sqrt(2*D*dt) * noise
    Confinement radius ~ sqrt(D/k) in nm.
    """
    D = 10 ** rng.uniform(2.0, 3.5)       # nm^2/s
    conf_radius = rng.uniform(200, 800)    # nm — confinement radius
    k = D / (conf_radius ** 2)            # spring constant (1/s)

    positions = np.zeros((n_steps + 1, 2))
    sigma_step = np.sqrt(2 * D * DT)

    for t in range(n_steps):
        drift = -k * positions[t] * DT
        noise = rng.normal(0, sigma_step, size=2)
        positions[t + 1] = positions[t] + drift + noise

    displacements = np.diff(positions, axis=0)
    params = {"D": D, "conf_radius_nm": conf_radius, "k": k}
    return displacements, params


def generate_directed_motion(n_steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Class 2: Directed motion (drift + diffusion).

    dx = v * dt + sqrt(2*D*dt) * noise
    MSD(tau) = 4*D*tau + v^2 * tau^2  (ballistic at long tau)
    """
    D = 10 ** rng.uniform(1.5, 3.0)       # nm^2/s (can be lower since drift dominates)
    speed = rng.uniform(5, 40)             # nm/s — slow chromatin drift
    angle = rng.uniform(0, 2 * np.pi)

    v = speed * np.array([np.cos(angle), np.sin(angle)])
    sigma_step = np.sqrt(2 * D * DT)

    displacements = np.zeros((n_steps, 2))
    for t in range(n_steps):
        displacements[t] = v * DT + rng.normal(0, sigma_step, size=2)

    params = {"D": D, "speed_nm_s": speed, "angle_rad": angle}
    return displacements, params


def generate_anomalous_subdiffusion(n_steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Class 3: Anomalous subdiffusion (fractional Brownian motion, alpha < 1).

    MSD(tau) = 4 * D_alpha * tau^alpha   with alpha in [0.2, 0.8]
    Uses fbm library for correlated increments.
    """
    alpha = rng.uniform(0.2, 0.8)
    D = 10 ** rng.uniform(2.0, 3.5)

    H = alpha / 2.0
    H = np.clip(H, 0.01, 0.99)
    sigma_dim = np.sqrt(2 * D * DT ** alpha)

    displacements = np.zeros((n_steps, 2))
    for d in range(2):
        path = generate_fbm(n=int(n_steps), hurst=float(H), length=1, method='cholesky')
        increments = np.diff(path)
        std_actual = np.std(increments)
        if std_actual > 0:
            increments = increments / std_actual * sigma_dim
        else:
            increments = rng.normal(0, sigma_dim, size=n_steps)
        displacements[:, d] = increments

    params = {"D": D, "alpha": alpha}
    return displacements, params


GENERATORS = {
    0: generate_normal_diffusion,
    1: generate_confined_diffusion,
    2: generate_directed_motion,
    3: generate_anomalous_subdiffusion,
}


def add_localization_noise(displacements: np.ndarray, sigma_noise: float) -> np.ndarray:
    """Add realistic localization noise to displacement sequence.

    Noise on positions becomes correlated noise on displacements:
    dx_obs = dx_true + eps(t+1) - eps(t)
    """
    rng = np.random.default_rng()
    n_steps = len(displacements)
    noise_positions = rng.normal(0, sigma_noise, size=(n_steps + 1, 2))
    noise_displacements = np.diff(noise_positions, axis=0)
    return displacements + noise_displacements


def generate_dataset(
    n_per_class: int,
    fixed_length: int = None,
    noise_range: Tuple[float, float] = (15, 40),
    seed: int = None
) -> Dict:
    """Generate a balanced classification dataset.

    Args:
        n_per_class: number of samples per class
        fixed_length: if set, all trajectories have this many steps.
                      If None, sample variable lengths from real distribution.
        noise_range: (min, max) localization noise in nm
        seed: random seed

    Returns:
        dict with keys:
            'displacements': list of (n_steps, 2) arrays
            'labels': (N,) int array of class labels
            'params': list of parameter dicts
            'noise_levels': (N,) float array
            'lengths': (N,) int array of trajectory lengths
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_class * N_CLASSES

    all_displacements = []
    all_labels = []
    all_params = []
    all_noise = []
    all_lengths = []

    for class_id in range(N_CLASSES):
        gen_fn = GENERATORS[class_id]
        for _ in range(n_per_class):
            n_steps = fixed_length if fixed_length else sample_trajectory_length(rng)
            displacements, params = gen_fn(n_steps, rng)

            # Add localization noise
            sigma_noise = rng.uniform(*noise_range)
            displacements = add_localization_noise(displacements, sigma_noise)

            all_displacements.append(displacements)
            all_labels.append(class_id)
            all_params.append(params)
            all_noise.append(sigma_noise)
            all_lengths.append(n_steps)

    # Shuffle
    order = rng.permutation(n_total)
    all_displacements = [all_displacements[i] for i in order]
    all_labels = np.array(all_labels)[order]
    all_params = [all_params[i] for i in order]
    all_noise = np.array(all_noise)[order]
    all_lengths = np.array(all_lengths)[order]

    return {
        'displacements': all_displacements,
        'labels': all_labels,
        'params': all_params,
        'noise_levels': all_noise,
        'lengths': all_lengths,
    }


def pad_sequences(sequences, max_len=None):
    """Pad variable-length displacement sequences to uniform length.

    Returns:
        X: (N, max_len, 2) padded array
        mask: (N, max_len) boolean mask (True = valid)
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    N = len(sequences)
    X = np.zeros((N, max_len, 2))
    mask = np.zeros((N, max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        L = min(len(seq), max_len)
        X[i, :L] = seq[:L]
        mask[i, :L] = True
    return X, mask


def compute_handcrafted_features(displacements: np.ndarray) -> np.ndarray:
    """Extract hand-crafted features from a single displacement sequence.

    Features (13 total):
        0: mean step size (nm)
        1: std step size
        2: max step size
        3: mean/std ratio (regularity)
        4: net displacement / total path length (straightness)
        5: MSD ratio (MSD at lag 2) / (2 * MSD at lag 1)  — detects confinement
        6: displacement autocorrelation lag 1 (detects subdiffusion/confinement)
        7: displacement autocorrelation lag 2
        8: mean turning angle (radians)
        9: std turning angle
        10: estimated alpha from log-log MSD fit
        11: fraction of steps in dominant quadrant (detects directed)
        12: trajectory length (n_steps)
    """
    steps = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    n = len(steps)

    # Basic step statistics
    mean_step = np.mean(steps)
    std_step = np.std(steps)
    max_step = np.max(steps)
    regularity = mean_step / std_step if std_step > 0 else 0

    # Straightness index
    positions = np.concatenate([np.zeros((1, 2)), np.cumsum(displacements, axis=0)], axis=0)
    net_disp = np.linalg.norm(positions[-1] - positions[0])
    total_path = np.sum(steps)
    straightness = net_disp / total_path if total_path > 0 else 0

    # MSD at lags 1 and 2
    def msd_at_lag(positions, lag):
        if lag >= len(positions):
            return np.nan
        diffs = positions[lag:] - positions[:-lag]
        return np.mean(np.sum(diffs**2, axis=1))

    msd1 = msd_at_lag(positions, 1)
    msd2 = msd_at_lag(positions, 2)
    msd_ratio = msd2 / (2 * msd1) if msd1 > 0 and not np.isnan(msd2) else 1.0

    # Displacement autocorrelation
    dx = displacements[:, 0]
    dy = displacements[:, 1]
    step_magnitudes = steps

    def displacement_autocorr(dx, dy, lag):
        if lag >= len(dx):
            return 0
        dot = dx[:-lag] * dx[lag:] + dy[:-lag] * dy[lag:]
        norm = np.sqrt((dx[:-lag]**2 + dy[:-lag]**2) * (dx[lag:]**2 + dy[lag:]**2))
        valid = norm > 0
        if valid.sum() == 0:
            return 0
        return np.mean(dot[valid] / norm[valid])

    autocorr1 = displacement_autocorr(dx, dy, 1)
    autocorr2 = displacement_autocorr(dx, dy, 2) if n > 2 else 0

    # Turning angles
    angles = []
    for t in range(len(displacements) - 1):
        v1 = displacements[t]
        v2 = displacements[t + 1]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_theta = np.clip(cos_theta, -1, 1)
        angles.append(np.arccos(cos_theta))
    mean_angle = np.mean(angles) if angles else np.pi / 2
    std_angle = np.std(angles) if angles else 0

    # Alpha from log-log MSD
    max_lag = min(n // 3, 10)
    if max_lag >= 3:
        lags = np.arange(1, max_lag + 1)
        msds = np.array([msd_at_lag(positions, l) for l in lags])
        valid = (msds > 0) & ~np.isnan(msds)
        if valid.sum() >= 2:
            log_lags = np.log(lags[valid] * DT)
            log_msds = np.log(msds[valid])
            alpha_fit = np.polyfit(log_lags, log_msds, 1)[0]
        else:
            alpha_fit = 1.0
    else:
        alpha_fit = 1.0

    # Quadrant dominance (detects directional bias)
    quadrant_counts = np.zeros(4)
    for d in displacements:
        q = int(d[0] > 0) + 2 * int(d[1] > 0)
        quadrant_counts[q] += 1
    dominant_fraction = np.max(quadrant_counts) / n

    return np.array([
        mean_step, std_step, max_step, regularity, straightness,
        msd_ratio, autocorr1, autocorr2, mean_angle, std_angle,
        alpha_fit, dominant_fraction, n
    ])


FEATURE_NAMES = [
    "mean_step_nm", "std_step_nm", "max_step_nm", "step_regularity",
    "straightness", "msd_ratio_lag2_lag1", "autocorr_lag1", "autocorr_lag2",
    "mean_turning_angle", "std_turning_angle", "alpha_msd_fit",
    "dominant_quadrant_frac", "n_steps"
]


def extract_features_batch(dataset: Dict) -> np.ndarray:
    """Extract hand-crafted features for all trajectories in a dataset.

    Returns: (N, 13) feature matrix
    """
    features = []
    for disp in dataset['displacements']:
        features.append(compute_handcrafted_features(disp))
    return np.array(features)


# ── Verification ────────────────────────────────────────────────────────────

def verify_dataset(dataset: Dict):
    """Print summary statistics for each class."""
    labels = dataset['labels']
    lengths = dataset['lengths']

    print(f"\nDataset summary: {len(labels)} trajectories, {N_CLASSES} classes")
    print(f"  Length range: {lengths.min()}-{lengths.max()} steps (median {np.median(lengths):.0f})")
    print()

    for c in range(N_CLASSES):
        mask = labels == c
        n = mask.sum()
        cls_lengths = lengths[mask]
        cls_disps = [dataset['displacements'][i] for i in range(len(labels)) if labels[i] == c]

        # Compute mean step size per trajectory
        mean_steps = [np.mean(np.sqrt(d[:, 0]**2 + d[:, 1]**2)) for d in cls_disps]

        print(f"  Class {c} ({CLASS_NAMES[c]}): n={n}")
        print(f"    lengths: {cls_lengths.min()}-{cls_lengths.max()} (median {np.median(cls_lengths):.0f})")
        print(f"    mean step: {np.mean(mean_steps):.1f} ± {np.std(mean_steps):.1f} nm")

        # Quick MSD check on first few
        positions = np.concatenate([np.zeros((1, 2)), np.cumsum(cls_disps[0], axis=0)], axis=0)
        msd1 = np.mean(np.sum(np.diff(positions, axis=0)**2, axis=1))
        print(f"    sample MSD(lag=1): {msd1:.0f} nm²")
    print()


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    full_mode = '--full' in sys.argv

    if full_mode:
        print("Generating FULL classification training dataset...")
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_data')
        os.makedirs(out_dir, exist_ok=True)

        splits = [
            ('train', 5000, 42),    # 5000 per class = 20,000 total
            ('val',   1000, 123),   # 1000 per class = 4,000 total
            ('test',  1000, 456),   # 1000 per class = 4,000 total
        ]

        for split_name, n_per_class, seed in splits:
            print(f"\n--- Generating {split_name} set ({n_per_class} per class) ---")
            dataset = generate_dataset(n_per_class=n_per_class, seed=seed)
            verify_dataset(dataset)

            # Pad sequences and save
            X_pad, X_mask = pad_sequences(dataset['displacements'], max_len=MAX_STEPS)
            np.savez_compressed(
                os.path.join(out_dir, f'{split_name}_classification.npz'),
                X=X_pad,
                mask=X_mask,
                labels=dataset['labels'],
                lengths=dataset['lengths'],
                noise_levels=dataset['noise_levels'],
            )
            print(f"  Saved: {out_dir}/{split_name}_classification.npz")
            print(f"  X shape: {X_pad.shape}, labels shape: {dataset['labels'].shape}")

            # Also extract and save hand-crafted features
            features = extract_features_batch(dataset)
            np.savez_compressed(
                os.path.join(out_dir, f'{split_name}_features.npz'),
                features=features,
                labels=dataset['labels'],
                feature_names=FEATURE_NAMES,
            )
            print(f"  Features: {features.shape}")

    else:
        print("Generating small test batch (200 per class = 800 total)...")
        dataset = generate_dataset(n_per_class=200, seed=42)
        verify_dataset(dataset)

        # Test feature extraction
        print("Extracting hand-crafted features...")
        features = extract_features_batch(dataset)
        print(f"  Feature matrix: {features.shape}")
        print(f"  Feature names: {FEATURE_NAMES}")

        # Quick sklearn baseline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        print("\nBaseline: Random Forest on hand-crafted features (5-fold CV)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, features, dataset['labels'], cv=5, scoring='accuracy')
        print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        # Per-class
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report
        y_pred = cross_val_predict(rf, features, dataset['labels'], cv=5)
        print(f"\n{classification_report(dataset['labels'], y_pred, target_names=list(CLASS_NAMES.values()))}")
