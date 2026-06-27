#!/usr/bin/env python3
"""
compute_msd_alpha.py — Compute MSD curves and anomalous diffusion exponent (alpha)
for all real trajectories from the master_locus_features.csv.

MSD model (2D with localization error):
    MSD(τ) = 4D·τ^α + 2σ²

Where:
    D = apparent diffusion coefficient
    α = anomalous exponent (α<1: subdiffusion, α=1: normal, α>1: superdiffusion)
    σ = localization error (nm)
    τ = lag time (seconds)

Two fitting approaches:
    1. "raw_alpha": log-log slope of MSD vs τ (ignores localization error)
    2. "corrected_alpha": nonlinear fit of MSD = A·τ^α + B (accounts for 2σ² offset)

Outputs:
    msd_alpha_results.csv       — per-trajectory alpha, D, sigma, trajectory stats
    msd_curves.npz              — raw MSD curves for plotting

Usage:
    python3 compute_msd_alpha.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PIXEL_SIZE_NM = 183.3
FRAME_INTERVAL_S = 24.5      # seconds between frames (approximate, from metadata)
MIN_FRAMES_FOR_MSD = 5       # minimum trajectory length for MSD computation
MAX_LAG_FRACTION = 0.33      # use lags up to 1/3 of trajectory length
INPUT_CSV = Path('master_locus_features.csv')
OUTPUT_DIR = Path('.')


# ══════════════════════════════════════════════════════════════════════════════
# MSD COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_msd_curve(x_nm, y_nm, frames, max_lag=None):
    """
    Compute the MSD curve from a single trajectory.

    Args:
        x_nm, y_nm: position arrays in nm
        frames: frame indices (1-indexed)
        max_lag: maximum lag in frames (default: len/3)

    Returns:
        lags: array of lag values (in frames)
        msd: array of MSD values (in nm²)
        n_pairs: number of pairs used for each lag
    """
    n = len(x_nm)
    if max_lag is None:
        max_lag = max(1, int(n * MAX_LAG_FRACTION))

    lags = []
    msd_vals = []
    n_pairs_list = []

    for lag in range(1, max_lag + 1):
        # For irregularly-spaced frames, match pairs with the right frame gap
        sq_displacements = []
        for i in range(n):
            for j in range(i + 1, n):
                if frames[j] - frames[i] == lag:
                    dx = x_nm[j] - x_nm[i]
                    dy = y_nm[j] - y_nm[i]
                    sq_displacements.append(dx**2 + dy**2)

        if len(sq_displacements) >= 1:
            lags.append(lag)
            msd_vals.append(np.mean(sq_displacements))
            n_pairs_list.append(len(sq_displacements))

    return np.array(lags), np.array(msd_vals), np.array(n_pairs_list)


def fit_alpha_loglog(lags_s, msd_nm2):
    """
    Fit alpha from log-log slope of MSD vs lag time.
    Simple but biased by localization error at short lags.

    MSD = C * τ^α  →  log(MSD) = log(C) + α*log(τ)
    """
    if len(lags_s) < 2:
        return np.nan, np.nan

    log_tau = np.log(lags_s)
    log_msd = np.log(msd_nm2)

    # Weighted linear fit (weight by 1/lag to emphasize early lags)
    weights = 1.0 / np.arange(1, len(lags_s) + 1)

    try:
        coeffs = np.polyfit(log_tau, log_msd, 1, w=weights)
        alpha = coeffs[0]
        log_C = coeffs[1]
        D_apparent = np.exp(log_C) / 4.0  # from C = 4D
        return alpha, D_apparent
    except Exception:
        return np.nan, np.nan


def msd_model(tau, A, alpha, B):
    """MSD(τ) = A * τ^α + B, where B ≈ 2σ²."""
    return A * np.power(tau, alpha) + B


def fit_alpha_corrected(lags_s, msd_nm2):
    """
    Fit MSD = A·τ^α + B (accounts for localization error offset).

    Returns: alpha, D_apparent (nm²/s), sigma_nm (localization error)
    """
    if len(lags_s) < 3:
        return np.nan, np.nan, np.nan, np.nan

    try:
        # Initial guesses
        # A ~ 4D ~ slope of MSD at first lag
        A0 = msd_nm2[0] / lags_s[0]
        alpha0 = 0.5  # expect subdiffusive
        B0 = max(0, msd_nm2[0] - A0 * lags_s[0])  # offset from first point

        # Bounds: A>0, 0<alpha<2, B>=0
        popt, pcov = curve_fit(
            msd_model, lags_s, msd_nm2,
            p0=[A0, alpha0, B0],
            bounds=([0, 0.01, 0], [np.inf, 2.5, np.inf]),
            maxfev=5000,
        )
        A, alpha, B = popt

        D_apparent = A / 4.0  # nm²/s
        sigma_nm = np.sqrt(max(B / 2.0, 0))  # from B = 2σ²

        # Check fit quality
        residuals = msd_nm2 - msd_model(lags_s, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_nm2 - np.mean(msd_nm2))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return alpha, D_apparent, sigma_nm, r_squared
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL TRAJECTORY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_trajectory_stats(x_nm, y_nm, frames, dt=FRAME_INTERVAL_S):
    """Compute basic trajectory statistics."""
    n = len(x_nm)
    if n < 2:
        return {}

    # Step displacements
    dx = np.diff(x_nm)
    dy = np.diff(y_nm)
    step_sizes = np.sqrt(dx**2 + dy**2)

    # Confinement ratio: end-to-end distance / total path length
    end_to_end = np.sqrt((x_nm[-1] - x_nm[0])**2 + (y_nm[-1] - y_nm[0])**2)
    total_path = np.sum(step_sizes)
    confinement_ratio = end_to_end / total_path if total_path > 0 else 0

    # Displacement autocorrelation at lag 1
    if len(step_sizes) >= 3:
        steps_centered = step_sizes - np.mean(step_sizes)
        var = np.var(step_sizes)
        if var > 0:
            autocorr_1 = np.mean(steps_centered[:-1] * steps_centered[1:]) / var
        else:
            autocorr_1 = 0
    else:
        autocorr_1 = np.nan

    return {
        'n_steps': n - 1,
        'mean_step_nm': round(np.mean(step_sizes), 2),
        'std_step_nm': round(np.std(step_sizes), 2),
        'max_step_nm': round(np.max(step_sizes), 2),
        'end_to_end_nm': round(end_to_end, 2),
        'total_path_nm': round(total_path, 2),
        'confinement_ratio': round(confinement_ratio, 4),
        'displacement_autocorr_1': round(autocorr_1, 4),
        'duration_s': round((frames[-1] - frames[0]) * dt, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MSD ALPHA ANALYSIS — Real Trajectory Data")
    print("=" * 70)

    # ── Load trajectory data ──────────────────────────────────────────────
    print("\n1. Loading trajectories from master_locus_features.csv...")
    with open(str(INPUT_CSV), newline='') as f:
        rows = list(csv.DictReader(f))

    # Group by (nucleus_id, locus_id)
    trajectories = defaultdict(list)
    for r in rows:
        key = (r['nucleus_id'], r['locus_id'])
        trajectories[key].append(r)

    print(f"   Total trajectories: {len(trajectories)}")

    # Filter by minimum length
    valid_trajs = {k: v for k, v in trajectories.items() if len(v) >= MIN_FRAMES_FOR_MSD}
    print(f"   With >= {MIN_FRAMES_FOR_MSD} frames: {len(valid_trajs)}")

    # ── Compute MSD and alpha for each trajectory ─────────────────────────
    print(f"\n2. Computing MSD curves and fitting alpha...")

    results = []
    msd_curves_data = {}  # for saving raw curves

    for i, ((nuc_id, locus_id), traj_rows) in enumerate(sorted(valid_trajs.items())):
        # Sort by frame
        traj_rows.sort(key=lambda r: int(r['frame']))

        frames = np.array([int(r['frame']) for r in traj_rows])
        x_nm = np.array([float(r['x_nm']) for r in traj_rows])
        y_nm = np.array([float(r['y_nm']) for r in traj_rows])

        experiment = traj_rows[0]['experiment']
        channel = locus_id.split('_')[0]  # G, R, or P
        n_frames = len(frames)

        # Compute MSD curve
        lags, msd, n_pairs = compute_msd_curve(x_nm, y_nm, frames)

        if len(lags) < 2:
            continue

        # Convert lags to seconds
        lags_s = lags * FRAME_INTERVAL_S

        # Fit 1: simple log-log (no localization error correction)
        alpha_raw, D_raw = fit_alpha_loglog(lags_s, msd)

        # Fit 2: corrected with offset (MSD = A*τ^α + B)
        fit_result = fit_alpha_corrected(lags_s, msd)
        alpha_corr, D_corr, sigma_est, r_squared = fit_result

        # Trajectory statistics
        stats = compute_trajectory_stats(x_nm, y_nm, frames)

        # Spatial context from the features
        radial_positions = [float(r['norm_radial_pos']) for r in traj_rows
                           if r.get('norm_radial_pos') and r['norm_radial_pos'] != '']
        mean_radial = np.mean(radial_positions) if radial_positions else np.nan

        membrane_dists = [float(r['dist_to_membrane_nm']) for r in traj_rows
                         if r.get('dist_to_membrane_nm') and r['dist_to_membrane_nm'] != '']
        mean_membrane_dist = np.mean(membrane_dists) if membrane_dists else np.nan

        result = {
            'nucleus_id': nuc_id,
            'locus_id': locus_id,
            'experiment': experiment,
            'channel': channel,
            'n_frames': n_frames,
            'alpha_raw': round(alpha_raw, 4) if not np.isnan(alpha_raw) else None,
            'D_raw_nm2s': round(D_raw, 2) if not np.isnan(D_raw) else None,
            'alpha_corrected': round(alpha_corr, 4) if not np.isnan(alpha_corr) else None,
            'D_corrected_nm2s': round(D_corr, 2) if not np.isnan(D_corr) else None,
            'sigma_est_nm': round(sigma_est, 2) if not np.isnan(sigma_est) else None,
            'fit_r_squared': round(r_squared, 4) if not np.isnan(r_squared) else None,
            'mean_radial_pos': round(mean_radial, 4) if not np.isnan(mean_radial) else None,
            'mean_membrane_dist_nm': round(mean_membrane_dist, 2) if not np.isnan(mean_membrane_dist) else None,
        }
        result.update(stats)
        results.append(result)

        # Save MSD curve for plotting
        msd_curves_data[f"{nuc_id}_{locus_id}"] = {
            'lags_s': lags_s,
            'msd_nm2': msd,
            'n_pairs': n_pairs,
        }

        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1}/{len(valid_trajs)} trajectories...")

    print(f"\n   Computed alpha for {len(results)} trajectories")

    # ── Save results ──────────────────────────────────────────────────────
    print("\n3. Saving results...")

    # CSV
    csv_path = OUTPUT_DIR / 'msd_alpha_results.csv'
    if results:
        all_keys = []
        for r in results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(str(csv_path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"   Saved: {csv_path.name} ({len(results)} rows)")

    # NPZ with MSD curves (for figure generation)
    npz_path = OUTPUT_DIR / 'msd_curves.npz'
    # Save a subset for plotting (save all lags/msd as ragged arrays)
    curve_keys = list(msd_curves_data.keys())
    np.savez(str(npz_path),
             keys=curve_keys,
             **{f'lags_{i}': msd_curves_data[k]['lags_s'] for i, k in enumerate(curve_keys)},
             **{f'msd_{i}': msd_curves_data[k]['msd_nm2'] for i, k in enumerate(curve_keys)})
    print(f"   Saved: {npz_path.name} ({len(curve_keys)} curves)")

    # ── Analysis summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    alphas_raw = [r['alpha_raw'] for r in results if r['alpha_raw'] is not None]
    alphas_corr = [r['alpha_corrected'] for r in results if r['alpha_corrected'] is not None]
    sigmas = [r['sigma_est_nm'] for r in results if r['sigma_est_nm'] is not None]
    D_vals = [r['D_corrected_nm2s'] for r in results if r['D_corrected_nm2s'] is not None]

    print(f"\n   Trajectories analyzed: {len(results)}")

    if alphas_raw:
        print(f"\n   Raw alpha (no localization correction):")
        print(f"     mean = {np.mean(alphas_raw):.3f} ± {np.std(alphas_raw):.3f}")
        print(f"     median = {np.median(alphas_raw):.3f}")
        print(f"     range = [{np.min(alphas_raw):.3f}, {np.max(alphas_raw):.3f}]")
        n_sub = sum(1 for a in alphas_raw if a < 0.9)
        n_norm = sum(1 for a in alphas_raw if 0.9 <= a <= 1.1)
        n_super = sum(1 for a in alphas_raw if a > 1.1)
        print(f"     subdiffusive (α<0.9): {n_sub} ({100*n_sub/len(alphas_raw):.1f}%)")
        print(f"     normal (0.9≤α≤1.1):  {n_norm} ({100*n_norm/len(alphas_raw):.1f}%)")
        print(f"     superdiffusive (α>1.1): {n_super} ({100*n_super/len(alphas_raw):.1f}%)")

    if alphas_corr:
        print(f"\n   Corrected alpha (with localization error offset):")
        print(f"     mean = {np.mean(alphas_corr):.3f} ± {np.std(alphas_corr):.3f}")
        print(f"     median = {np.median(alphas_corr):.3f}")
        print(f"     range = [{np.min(alphas_corr):.3f}, {np.max(alphas_corr):.3f}]")
        n_sub = sum(1 for a in alphas_corr if a < 0.9)
        n_norm = sum(1 for a in alphas_corr if 0.9 <= a <= 1.1)
        n_super = sum(1 for a in alphas_corr if a > 1.1)
        print(f"     subdiffusive (α<0.9): {n_sub} ({100*n_sub/len(alphas_corr):.1f}%)")
        print(f"     normal (0.9≤α≤1.1):  {n_norm} ({100*n_norm/len(alphas_corr):.1f}%)")
        print(f"     superdiffusive (α>1.1): {n_super} ({100*n_super/len(alphas_corr):.1f}%)")

    if sigmas:
        print(f"\n   Estimated localization error (σ):")
        print(f"     mean = {np.mean(sigmas):.1f} ± {np.std(sigmas):.1f} nm")
        print(f"     median = {np.median(sigmas):.1f} nm")

    if D_vals:
        D_um2s = [d / 1e6 for d in D_vals]  # convert nm²/s to µm²/s
        print(f"\n   Apparent diffusion coefficient (D):")
        print(f"     mean = {np.mean(D_um2s):.4f} ± {np.std(D_um2s):.4f} µm²/s")
        print(f"     median = {np.median(D_um2s):.4f} µm²/s")

    # Per-channel breakdown
    print(f"\n   Per-channel alpha (corrected):")
    for ch in ['G', 'R', 'P']:
        ch_alphas = [r['alpha_corrected'] for r in results
                     if r['channel'] == ch and r['alpha_corrected'] is not None]
        if ch_alphas:
            locus_names = {'G': '195M (488nm)', 'R': '195.7M (565nm)', 'P': '198M (647nm)'}
            print(f"     {ch} [{locus_names.get(ch, '')}]: "
                  f"α = {np.mean(ch_alphas):.3f} ± {np.std(ch_alphas):.3f} "
                  f"(n={len(ch_alphas)})")

    print("\nDone.")


if __name__ == '__main__':
    main()
