#!/usr/bin/env python3
"""
extract_nuclear_features.py — Extract nuclear morphology features from
Kenaj's nuclear masks across all "all nd2 files" folders.

Mask format: <stem>_mask_<number>.tif
  - Multi-frame TIF (one frame per time point), binary (0=background, 255=nucleus)
  - Located alongside nucleus TIFs and metadata JSONs in experiment subfolders

Outputs:
    nuclear_features_kenaj.csv — per-nucleus, per-frame morphology features
    nuclear_features_kenaj_summary.csv — per-nucleus summary (averaged over frames)

Usage:
    python3 extract_nuclear_features.py [--data-root <path>]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PIXEL_SIZE_NM = 183.3  # nm per pixel (from ND2 metadata)


# ══════════════════════════════════════════════════════════════════════════════
# MASK LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_mask_frames(path: Path) -> list:
    """
    Load a multi-frame mask TIF. Returns list of boolean arrays (or None for blank frames).
    """
    from PIL import Image
    img = Image.open(str(path))
    frames = []
    i = 0
    try:
        while True:
            img.seek(i)
            arr = np.array(img)
            # Handle RGB or multi-channel masks by taking first channel or max
            if arr.ndim == 3:
                arr = arr.max(axis=-1)
            if arr.max() == 0:
                frames.append(None)
            else:
                frames.append(arr > 0)
            i += 1
    except EOFError:
        pass
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# NUCLEUS MORPHOLOGY (reused from extract_features.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_nucleus_morphology(mask: np.ndarray, pixel_size_nm: float = PIXEL_SIZE_NM) -> dict:
    """
    Extract morphological features from a single binary nucleus mask.
    """
    # Ensure 2D
    if mask.ndim != 2:
        return None

    area = int(mask.sum())
    if area == 0:
        return None

    # Centroid
    com = ndimage.center_of_mass(mask)
    cy, cx = float(com[0]), float(com[1])

    # Perimeter via boundary transitions (4-connectivity)
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    h_edges = np.sum(padded[1:-1, 1:] != padded[1:-1, :-1])
    v_edges = np.sum(padded[1:, 1:-1] != padded[:-1, 1:-1])
    perimeter = int(h_edges + v_edges)

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0

    # Bounding box
    rows, cols = np.where(mask)
    bbox_h = int(rows.max() - rows.min() + 1)
    bbox_w = int(cols.max() - cols.min() + 1)

    # Second-order central moments for ellipse fitting
    y_centered = rows - cy
    x_centered = cols - cx
    mu20 = np.sum(y_centered ** 2) / area
    mu02 = np.sum(x_centered ** 2) / area
    mu11 = np.sum(y_centered * x_centered) / area

    common = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    lambda1 = (mu20 + mu02 + common) / 2
    lambda2 = max((mu20 + mu02 - common) / 2, 0)

    major_axis = 4 * np.sqrt(lambda1)
    minor_axis = 4 * np.sqrt(lambda2)
    eccentricity = np.sqrt(1 - lambda2 / lambda1) if lambda1 > 0 else 0.0
    orientation = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

    # Solidity
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack((cols, rows))
        hull = ConvexHull(points)
        hull_area = hull.volume
        solidity = min(area / hull_area, 1.0) if hull_area > 0 else 1.0
    except Exception:
        solidity = 1.0

    # Convert to physical units
    area_um2 = area * (pixel_size_nm / 1000) ** 2
    perimeter_um = perimeter * (pixel_size_nm / 1000)
    major_axis_um = major_axis * (pixel_size_nm / 1000)
    minor_axis_um = minor_axis * (pixel_size_nm / 1000)

    return {
        'area_px': area,
        'area_um2': round(area_um2, 2),
        'perimeter_px': perimeter,
        'perimeter_um': round(perimeter_um, 2),
        'circularity': round(circularity, 4),
        'centroid_y': round(cy, 2),
        'centroid_x': round(cx, 2),
        'bbox_h': bbox_h,
        'bbox_w': bbox_w,
        'eccentricity': round(eccentricity, 4),
        'major_axis_px': round(major_axis, 2),
        'minor_axis_px': round(minor_axis, 2),
        'major_axis_um': round(major_axis_um, 2),
        'minor_axis_um': round(minor_axis_um, 2),
        'orientation_deg': round(orientation, 2),
        'solidity': round(solidity, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_masks(data_root: Path) -> list:
    """
    Find all Kenaj's mask files across all "all nd2 files" folders.

    Returns list of dicts:
        {mask_path, experiment_folder, stem, nucleus_idx, metadata_path (or None)}
    """
    mask_pattern = re.compile(r'^(.+)_mask_(\d+)\.tif$')
    results = []

    for nd2_dir in sorted(data_root.iterdir()):
        if not nd2_dir.is_dir() or 'all nd2' not in nd2_dir.name:
            continue

        for exp_dir in sorted(nd2_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            for f in sorted(exp_dir.iterdir()):
                m = mask_pattern.match(f.name)
                if m:
                    stem = m.group(1)
                    nucleus_idx = int(m.group(2))

                    # Look for corresponding metadata
                    meta_path = exp_dir / f'{stem}_{nucleus_idx}_metadata.json'
                    if not meta_path.exists():
                        meta_path = None

                    results.append({
                        'mask_path': f,
                        'experiment_folder': exp_dir.name,
                        'nd2_set': nd2_dir.name,
                        'stem': stem,
                        'nucleus_idx': nucleus_idx,
                        'metadata_path': meta_path,
                    })

    return results


def load_metadata(meta_path: Path) -> dict:
    """Load and return metadata JSON."""
    with open(str(meta_path)) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract nuclear features from Kenaj's masks")
    parser.add_argument('--data-root', type=str,
                        default='data for analysis',
                        help='Root folder containing "all nd2 files" subfolders')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output CSVs (default: current dir)')
    parser.add_argument('--pixel-size', type=float, default=PIXEL_SIZE_NM,
                        help=f'Pixel size in nm (default: {PIXEL_SIZE_NM})')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    pixel_size = args.pixel_size

    print(f"Data root  : {data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Pixel size : {pixel_size} nm/px")

    # ── Discover masks ─────────────────────────────────────────────────────
    print("\nDiscovering mask files...")
    mask_entries = discover_masks(data_root)
    print(f"  Found {len(mask_entries)} mask files across "
          f"{len(set(e['experiment_folder'] for e in mask_entries))} experiment folders")

    if not mask_entries:
        print("ERROR: No mask files found. Check data-root path.")
        sys.exit(1)

    # ── Extract features ───────────────────────────────────────────────────
    print("\nExtracting nuclear morphology features...")

    per_frame_rows = []
    summary_rows = []
    n_processed = 0
    n_failed = 0

    for entry in mask_entries:
        mask_path = entry['mask_path']
        exp = entry['experiment_folder']
        stem = entry['stem']
        nuc_idx = entry['nucleus_idx']
        nucleus_id = f"{stem}_{nuc_idx}"

        # Load metadata if available
        meta = {}
        if entry['metadata_path']:
            try:
                meta = load_metadata(entry['metadata_path'])
            except Exception:
                pass

        # Extract metadata fields
        bbox = meta.get('bbox', {})
        time_info = meta.get('time', {})
        n_timepoints = time_info.get('n_frames', None)
        frame_interval = time_info.get('finterval_s', None)

        # Use metadata pixel size if available, otherwise default
        px_meta = meta.get('pixel_size', {})
        px_size = px_meta.get('x_um', pixel_size / 1000) * 1000  # convert um -> nm

        # Load mask frames
        try:
            mask_frames = load_mask_frames(mask_path)
        except Exception as e:
            print(f"  ERROR loading {mask_path.name}: {e}")
            n_failed += 1
            continue

        n_frames = len(mask_frames)
        n_valid = sum(1 for m in mask_frames if m is not None)

        if n_valid == 0:
            print(f"  SKIP {nucleus_id}: all {n_frames} frames blank")
            n_failed += 1
            continue

        # Extract per-frame morphology
        frame_features = []
        for t, mask in enumerate(mask_frames):
            if mask is None:
                continue

            morph = extract_nucleus_morphology(mask, px_size)
            if morph is None:
                continue

            row = {
                'nucleus_id': nucleus_id,
                'experiment': exp,
                'nd2_set': entry['nd2_set'],
                'nucleus_idx': nuc_idx,
                'frame': t,
                'time_s': round(t * frame_interval, 2) if frame_interval else None,
                'bbox_r0': bbox.get('r0'),
                'bbox_c0': bbox.get('c0'),
                'bbox_r1': bbox.get('r1'),
                'bbox_c1': bbox.get('c1'),
            }
            row.update(morph)
            per_frame_rows.append(row)
            frame_features.append(morph)

        # Compute summary across frames
        if frame_features:
            summary = {
                'nucleus_id': nucleus_id,
                'experiment': exp,
                'nd2_set': entry['nd2_set'],
                'nucleus_idx': nuc_idx,
                'n_frames': n_frames,
                'n_valid_frames': len(frame_features),
                'bbox_r0': bbox.get('r0'),
                'bbox_c0': bbox.get('c0'),
                'bbox_r1': bbox.get('r1'),
                'bbox_c1': bbox.get('c1'),
                'frame_interval_s': frame_interval,
            }

            # Average morphology features
            for key in ['area_px', 'area_um2', 'perimeter_px', 'perimeter_um',
                        'circularity', 'eccentricity', 'major_axis_px', 'minor_axis_px',
                        'major_axis_um', 'minor_axis_um', 'solidity']:
                vals = [f[key] for f in frame_features if key in f and f[key] is not None]
                if vals:
                    summary[f'{key}_mean'] = round(np.mean(vals), 4)
                    summary[f'{key}_std'] = round(np.std(vals), 4)

            summary_rows.append(summary)

        n_processed += 1

        if n_processed % 50 == 0:
            print(f"  Processed {n_processed}/{len(mask_entries)} nuclei...")

    print(f"\nProcessed: {n_processed} nuclei ({n_failed} failed/skipped)")
    print(f"  Per-frame rows: {len(per_frame_rows)}")
    print(f"  Summary rows:   {len(summary_rows)}")

    # ── Save outputs ───────────────────────────────────────────────────────
    # Per-frame CSV
    perframe_path = output_dir / 'nuclear_features_kenaj.csv'
    if per_frame_rows:
        all_keys = []
        for r in per_frame_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(str(perframe_path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(per_frame_rows)
        print(f"\nSaved: {perframe_path.name}  ({len(per_frame_rows)} rows, {len(all_keys)} cols)")

    # Summary CSV
    summary_path = output_dir / 'nuclear_features_kenaj_summary.csv'
    if summary_rows:
        all_keys = []
        for r in summary_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(str(summary_path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved: {summary_path.name}  ({len(summary_rows)} rows, {len(all_keys)} cols)")

    # Print summary stats
    if summary_rows:
        areas = [r['area_um2_mean'] for r in summary_rows if 'area_um2_mean' in r]
        circs = [r['circularity_mean'] for r in summary_rows if 'circularity_mean' in r]
        eccs = [r['eccentricity_mean'] for r in summary_rows if 'eccentricity_mean' in r]
        sols = [r['solidity_mean'] for r in summary_rows if 'solidity_mean' in r]

        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS (across {len(summary_rows)} nuclei)")
        print(f"{'='*60}")
        if areas:
            print(f"  Area (µm²):      {np.mean(areas):.1f} ± {np.std(areas):.1f}  "
                  f"[{np.min(areas):.1f} – {np.max(areas):.1f}]")
        if circs:
            print(f"  Circularity:     {np.mean(circs):.3f} ± {np.std(circs):.3f}  "
                  f"[{np.min(circs):.3f} – {np.max(circs):.3f}]")
        if eccs:
            print(f"  Eccentricity:    {np.mean(eccs):.3f} ± {np.std(eccs):.3f}  "
                  f"[{np.min(eccs):.3f} – {np.max(eccs):.3f}]")
        if sols:
            print(f"  Solidity:        {np.mean(sols):.3f} ± {np.std(sols):.3f}  "
                  f"[{np.min(sols):.3f} – {np.max(sols):.3f}]")

    print("\nDone.")


if __name__ == '__main__':
    main()
