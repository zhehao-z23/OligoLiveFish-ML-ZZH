#!/usr/bin/env python3
"""
extract_features.py — Extract spatial and morphological features from
Oligo-LiveFISH imaging data.

Usage:
    python3 extract_features.py /path/to/output_folder [--pixel-size 108.33]

Expects the output folder produced by the existing pipeline (Stages 2-4):
    Nucleus_masks.tif          — per-frame binary nucleus masks (from auto_roi_v2.2.py)
    *_Nucleus.tif              — nucleus channel intensity images
    *_green.tif                — green channel intensity images
    RoiSet_all.zip or *.roi    — ROI rectangle(s) around detected loci
    *_G-loci*.csv              — trajectory CSV(s) from ThunderSTORM (optional)

Outputs (saved to the same folder):
    locus_features.csv         — per-locus, per-frame spatial features
    nucleus_features.csv       — per-frame and summary nucleus morphology features

Modules:
    Module 1: Nucleus morphology (area, perimeter, circularity, eccentricity, solidity, intensity)
    Module 2: Locus-to-membrane distance, distance to centroid, normalized radial position
    Module 3: Local chromatin environment around each locus
"""

import sys
import struct
import zipfile
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PIXEL_SIZE_NM = 183.3        # nm per pixel (from ND2 metadata: 5.4545 px/µm)
LOCAL_WINDOW  = 11           # side length (px) for local chromatin intensity sampling


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILITIES                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_tiff_stack(path: Path) -> np.ndarray:
    """Load a multi-frame TIFF as float32 array (frames, H, W)."""
    from PIL import Image
    img = Image.open(str(path))
    frames = []
    i = 0
    try:
        while True:
            img.seek(i)
            frames.append(np.array(img, dtype=np.float32))
            i += 1
    except EOFError:
        pass
    return np.stack(frames)


def load_mask_stack(path: Path) -> list:
    """
    Load Nucleus_masks.tif as a list of boolean masks (or None for failed frames).
    Frames that are all-zero are treated as None (nucleus undetectable in that frame).
    """
    from PIL import Image
    img = Image.open(str(path))
    masks = []
    i = 0
    try:
        while True:
            img.seek(i)
            frame = np.array(img)
            if frame.max() == 0:
                masks.append(None)
            else:
                masks.append(frame > 0)
            i += 1
    except EOFError:
        pass
    return masks


def parse_roi_file(data: bytes) -> dict:
    """
    Parse an ImageJ binary rectangle ROI file.
    Returns dict with top, left, bottom, right in pixels.
    """
    if data[:4] != b'Iout':
        raise ValueError("Not a valid ImageJ ROI file")
    top    = struct.unpack('>H', data[8:10])[0]
    left   = struct.unpack('>H', data[10:12])[0]
    bottom = struct.unpack('>H', data[12:14])[0]
    right  = struct.unpack('>H', data[14:16])[0]
    return {'top': top, 'left': left, 'bottom': bottom, 'right': right}


def discover_rois(output_dir: Path) -> list:
    """
    Find all ROI rectangles in the output folder.
    Checks RoiSet_all.zip, RoiSet_green.zip, then standalone .roi files.
    Returns list of dicts: [{name, top, left, bottom, right}, ...]
    """
    rois = []

    # Try known RoiSet zip names
    for zip_name in ['RoiSet_all.zip', 'RoiSet_green.zip']:
        zip_path = output_dir / zip_name
        if zip_path.exists():
            with zipfile.ZipFile(str(zip_path)) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith('.roi'):
                        data = zf.read(name)
                        roi = parse_roi_file(data)
                        roi['name'] = Path(name).stem
                        rois.append(roi)
            if rois:
                return rois

    # Fall back to standalone .roi files
    for roi_path in sorted(output_dir.glob('*.roi')):
        data = roi_path.read_bytes()
        roi = parse_roi_file(data)
        roi['name'] = roi_path.stem
        rois.append(roi)

    return rois


def discover_trajectory_csvs(output_dir: Path) -> dict:
    """
    Find trajectory CSVs for all channels (G, R, P).
    Supports both old format (*_G-loci*.csv) and Kevin's pipeline format
    (G_loci*_traj_rela2wholeimg.csv, R_loci*_traj_rela2wholeimg.csv, etc.)

    Returns dict mapping locus name -> Path, e.g. {'G_loci1': Path(...), 'P_loci1': Path(...)}
    """
    csvs = {}

    # Kevin's pipeline format: G_loci1_traj_rela2wholeimg.csv
    for csv_path in sorted(output_dir.glob('*_loci*_traj_rela2wholeimg.csv')):
        stem = csv_path.stem  # e.g. "G_loci1_traj_rela2wholeimg"
        # Extract channel_locus part: "G_loci1"
        parts = stem.split('_traj_')[0]  # "G_loci1"
        csvs[parts] = csv_path

    # Also check for m2DGaussian_cleaned versions (prefer these if available)
    for csv_path in sorted(output_dir.glob('*_loci*_traj_m2DGaussian_cleaned.csv')):
        stem = csv_path.stem
        parts = stem.split('_traj_')[0]  # "G_loci1"
        key = parts + '_cleaned'
        csvs[key] = csv_path

    # Legacy format fallback: *_G-loci*.csv
    if not csvs:
        for csv_path in sorted(output_dir.glob('*_G-loci*.csv')):
            stem = csv_path.stem
            idx = stem.find('G-loci')
            if idx >= 0:
                locus_name = stem[idx:]
                csvs[locus_name] = csv_path

    return csvs


def load_trajectory(csv_path: Path) -> dict:
    """
    Load a trajectory CSV. Handles both formats:
      - Kevin's pipeline: frame, x_nm, y_nm (3 columns, coords relative to whole image)
      - ThunderSTORM legacy: id, frame, x_nm, y_nm, sigma, intensity, ... (9 columns)

    Returns dict with keys:
        frames, x_nm, y_nm, and optionally sigma_nm, intensity, uncertainty_nm
    Also includes 'format': 'pipeline' or 'thunderstorm' to signal coordinate handling.
    """
    data = np.genfromtxt(str(csv_path), delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    n_cols = data.shape[1]

    if n_cols == 3:
        # Kevin's pipeline format: frame, x_nm, y_nm
        return {
            'format':      'pipeline',
            'frames':      data[:, 0].astype(int),
            'x_nm':        data[:, 1],
            'y_nm':        data[:, 2],
            'sigma_nm':    np.full(len(data), np.nan),
            'intensity':   np.full(len(data), np.nan),
            'uncertainty_nm': np.full(len(data), np.nan),
        }
    else:
        # ThunderSTORM format: id, frame, x_nm, y_nm, sigma, intensity, offset, bkgstd, uncertainty
        return {
            'format':      'thunderstorm',
            'frames':      data[:, 1].astype(int),
            'x_nm':        data[:, 2],
            'y_nm':        data[:, 3],
            'sigma_nm':    data[:, 4] if n_cols > 4 else np.full(len(data), np.nan),
            'intensity':   data[:, 5] if n_cols > 5 else np.full(len(data), np.nan),
            'uncertainty_nm': data[:, 8] if n_cols > 8 else np.full(len(data), np.nan),
        }


def discover_nucleus_channel(output_dir: Path) -> Path | None:
    """Find the *_Nucleus.tif file in the output folder."""
    candidates = list(output_dir.glob('*_Nucleus.tif'))
    return candidates[0] if candidates else None


def discover_green_channel(output_dir: Path) -> Path | None:
    """Find the *_green.tif file in the output folder."""
    candidates = list(output_dir.glob('*_green.tif'))
    return candidates[0] if candidates else None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1: NUCLEUS MORPHOLOGY                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_nucleus_morphology(mask: np.ndarray) -> dict:
    """
    Extract morphological features from a single binary nucleus mask.

    Returns dict with:
        area_px          — number of pixels in the nucleus
        perimeter_px     — border pixels (mask minus eroded mask)
        circularity      — 4*pi*area / perimeter^2
        centroid_y/x     — center of mass (pixel coords)
        bbox_h/w         — bounding box height and width
        eccentricity     — from second-order central moments (0=circle, 1=line)
        major_axis_px    — length of major axis in pixels
        minor_axis_px    — length of minor axis in pixels
        orientation_deg  — angle of major axis relative to horizontal
        solidity         — area / convex_hull_area
    """
    area = int(mask.sum())
    if area == 0:
        return None

    # Centroid
    cy, cx = ndimage.center_of_mass(mask)

    # Perimeter: count boundary transitions using 4-connectivity.
    # A pixel contributes to perimeter for each of its 4 cardinal neighbors
    # that is outside the mask (or at the image edge). This gives the true
    # boundary length in pixel units, unlike erosion which undercounts.
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    # Count edges between mask and non-mask in horizontal and vertical directions
    h_edges = np.sum(padded[1:-1, 1:] != padded[1:-1, :-1])  # vertical boundaries
    v_edges = np.sum(padded[1:, 1:-1] != padded[:-1, 1:-1])  # horizontal boundaries
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
    mu20 = np.sum(y_centered ** 2) / area   # moment about y
    mu02 = np.sum(x_centered ** 2) / area   # moment about x
    mu11 = np.sum(y_centered * x_centered) / area

    # Eigenvalues of the inertia tensor → semi-axes
    common = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    lambda1 = (mu20 + mu02 + common) / 2   # larger eigenvalue
    lambda2 = (mu20 + mu02 - common) / 2   # smaller eigenvalue
    lambda2 = max(lambda2, 0)               # numerical safety

    major_axis = 4 * np.sqrt(lambda1)       # ~length in pixels
    minor_axis = 4 * np.sqrt(lambda2)

    eccentricity = np.sqrt(1 - lambda2 / lambda1) if lambda1 > 0 else 0.0

    # Orientation (angle of major axis from horizontal, in degrees)
    orientation = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

    # Solidity = area / convex_hull_area
    # Use scipy's convex hull via filling
    from scipy.ndimage import binary_fill_holes
    # Approximate convex hull: compute convex hull of boundary points
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack((cols, rows))     # (x, y) for ConvexHull
        hull = ConvexHull(points)
        hull_area = hull.volume   # In 2D, .volume gives area
        solidity = area / hull_area if hull_area > 0 else 1.0
    except Exception:
        solidity = 1.0   # fallback if hull computation fails

    return {
        'area_px':        area,
        'perimeter_px':   perimeter,
        'circularity':    round(circularity, 4),
        'centroid_y':     round(cy, 2),
        'centroid_x':     round(cx, 2),
        'bbox_h':         bbox_h,
        'bbox_w':         bbox_w,
        'eccentricity':   round(eccentricity, 4),
        'major_axis_px':  round(major_axis, 2),
        'minor_axis_px':  round(minor_axis, 2),
        'orientation_deg': round(orientation, 2),
        'solidity':       round(solidity, 4),
    }


def extract_nucleus_intensity_stats(nucleus_channel_frame: np.ndarray,
                                     mask: np.ndarray) -> dict:
    """
    Extract intensity statistics of the nucleus channel within the mask.
    """
    values = nucleus_channel_frame[mask]
    if len(values) == 0:
        return {}

    from scipy.stats import skew, kurtosis
    return {
        'nuc_intensity_mean': round(float(values.mean()), 2),
        'nuc_intensity_std':  round(float(values.std()), 2),
        'nuc_intensity_med':  round(float(np.median(values)), 2),
        'nuc_intensity_skew': round(float(skew(values)), 4),
        'nuc_intensity_kurt': round(float(kurtosis(values)), 4),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2: LOCUS-TO-MEMBRANE DISTANCE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_distance_features(mask: np.ndarray, locus_y: float, locus_x: float,
                               pixel_size_nm: float) -> dict | None:
    """
    Compute spatial features for one locus position in one frame.

    Args:
        mask:           boolean nucleus mask for this frame
        locus_y/x:      locus position in full-image pixel coordinates
        pixel_size_nm:  nm per pixel for unit conversion

    Returns dict with:
        dist_to_membrane_px   — Euclidean distance from locus to nearest mask edge (px)
        dist_to_membrane_nm   — same in nanometers
        dist_to_centroid_px   — distance from locus to nucleus centroid (px)
        dist_to_centroid_nm   — same in nanometers
        norm_radial_pos       — normalized radial position (0=periphery, 1=center)
        in_nucleus            — True if the locus falls inside the mask
    """
    if mask is None:
        return None

    H, W = mask.shape
    yi, xi = int(round(locus_y)), int(round(locus_x))

    # Check if locus is inside the nucleus
    in_nucleus = (0 <= yi < H and 0 <= xi < W and bool(mask[yi, xi]))

    # Distance transform: distance of every pixel inside the mask to the boundary
    dist_map = ndimage.distance_transform_edt(mask)

    # For pixels outside the mask, compute distance to nearest mask pixel
    # (negative of distance transform of the inverted mask)
    outside_dist_map = ndimage.distance_transform_edt(~mask)

    # Locus distance to membrane
    if 0 <= yi < H and 0 <= xi < W:
        if in_nucleus:
            dist_to_membrane_px = float(dist_map[yi, xi])
        else:
            dist_to_membrane_px = -float(outside_dist_map[yi, xi])  # negative = outside
    else:
        # Locus is outside the image bounds entirely
        return None

    # Nucleus centroid
    cy, cx = ndimage.center_of_mass(mask)
    dist_to_centroid_px = float(np.hypot(locus_y - cy, locus_x - cx))

    # Normalized radial position
    # 0 = at the membrane, 1 = at the centroid
    # Use the distance from centroid to membrane along the locus direction as reference
    if dist_to_membrane_px > 0 and (dist_to_membrane_px + dist_to_centroid_px) > 0:
        norm_radial = dist_to_membrane_px / (dist_to_membrane_px + dist_to_centroid_px)
    else:
        norm_radial = 0.0

    return {
        'dist_to_membrane_px': round(dist_to_membrane_px, 2),
        'dist_to_membrane_nm': round(dist_to_membrane_px * pixel_size_nm, 2),
        'dist_to_centroid_px': round(dist_to_centroid_px, 2),
        'dist_to_centroid_nm': round(dist_to_centroid_px * pixel_size_nm, 2),
        'norm_radial_pos':     round(norm_radial, 4),
        'in_nucleus':          in_nucleus,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 3: LOCAL CHROMATIN ENVIRONMENT                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_local_chromatin(nucleus_frame: np.ndarray, mask: np.ndarray,
                             locus_y: float, locus_x: float,
                             window: int = LOCAL_WINDOW) -> dict | None:
    """
    Sample chromatin intensity around the locus and compare to whole-nucleus.

    Args:
        nucleus_frame:  2D float array of nucleus channel for this frame
        mask:           boolean nucleus mask for this frame
        locus_y/x:      locus position in full-image pixel coords
        window:         side length of sampling window (pixels)

    Returns dict with:
        local_intensity_mean  — mean DAPI intensity in window around locus
        local_intensity_std   — std of same
        nuc_mean_intensity    — whole-nucleus mean for this frame
        local_to_nuc_ratio    — local / whole-nucleus (>1 = denser chromatin)
    """
    if mask is None:
        return None

    H, W = nucleus_frame.shape
    yi, xi = int(round(locus_y)), int(round(locus_x))
    half = window // 2

    # Clamp window to image bounds
    y0 = max(0, yi - half)
    y1 = min(H, yi + half + 1)
    x0 = max(0, xi - half)
    x1 = min(W, xi + half + 1)

    # Sample only pixels that are inside the nucleus mask
    local_mask = mask[y0:y1, x0:x1]
    local_values = nucleus_frame[y0:y1, x0:x1][local_mask]

    if len(local_values) == 0:
        return None

    nuc_values = nucleus_frame[mask]
    nuc_mean = float(nuc_values.mean()) if len(nuc_values) > 0 else 1.0

    local_mean = float(local_values.mean())

    return {
        'local_intensity_mean': round(local_mean, 2),
        'local_intensity_std':  round(float(local_values.std()), 2),
        'nuc_mean_intensity':   round(nuc_mean, 2),
        'local_to_nuc_ratio':   round(local_mean / nuc_mean, 4) if nuc_mean > 0 else 0.0,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Extract spatial and morphological features from Oligo-LiveFISH data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir",
                        help="Path to the output folder from the imaging pipeline")
    parser.add_argument("--pixel-size", type=float, default=PIXEL_SIZE_NM,
                        help=f"Pixel size in nm (default: {PIXEL_SIZE_NM})")
    parser.add_argument("--local-window", type=int, default=LOCAL_WINDOW,
                        help=f"Window size for local chromatin sampling (default: {LOCAL_WINDOW})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    pixel_size = args.pixel_size
    local_window = args.local_window

    if not output_dir.is_dir():
        print(f"ERROR: {output_dir} is not a directory")
        sys.exit(1)

    print(f"Output dir  : {output_dir}")
    print(f"Pixel size  : {pixel_size} nm/px")

    # ── Discover files ─────────────────────────────────────────────────────────
    masks_path = output_dir / 'Nucleus_masks.tif'
    if not masks_path.exists():
        print(f"ERROR: Nucleus_masks.tif not found in {output_dir}")
        print("  Run auto_roi_for_published_v2.2.py first to generate nucleus masks.")
        sys.exit(1)

    nucleus_ch_path = discover_nucleus_channel(output_dir)
    green_ch_path   = discover_green_channel(output_dir)
    rois            = discover_rois(output_dir)
    traj_csvs       = discover_trajectory_csvs(output_dir)

    print(f"Masks       : {masks_path.name}")
    print(f"Nucleus ch  : {nucleus_ch_path.name if nucleus_ch_path else 'NOT FOUND'}")
    print(f"Green ch    : {green_ch_path.name if green_ch_path else 'NOT FOUND'}")
    print(f"ROIs        : {len(rois)} found  {[r['name'] for r in rois]}")
    print(f"Trajectories: {len(traj_csvs)} found  {list(traj_csvs.keys())}")

    if not rois and not traj_csvs:
        print("\nERROR: No ROI files or trajectory CSVs found.")
        print("  Run the processing pipeline first.")
        sys.exit(1)
    elif not rois:
        print("  (No ROIs found, but trajectory CSVs are available — proceeding)")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading masks...")
    masks = load_mask_stack(masks_path)
    n_frames = len(masks)
    n_valid = sum(1 for m in masks if m is not None)
    print(f"  {n_frames} frames, {n_valid} with valid nucleus masks")

    nucleus_stack = None
    if nucleus_ch_path:
        print("Loading nucleus channel...")
        nucleus_stack = load_tiff_stack(nucleus_ch_path)
        print(f"  {nucleus_stack.shape[0]} frames, {nucleus_stack.shape[1]}x{nucleus_stack.shape[2]} px")

    # ── MODULE 1: Nucleus morphology ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("MODULE 1: Nucleus morphology")
    print(f"{'═'*60}")

    nucleus_rows = []
    for t in range(n_frames):
        mask = masks[t]
        row = {'frame': t}

        if mask is not None:
            morph = extract_nucleus_morphology(mask)
            if morph:
                row.update(morph)

            if nucleus_stack is not None and t < nucleus_stack.shape[0]:
                intensity = extract_nucleus_intensity_stats(nucleus_stack[t], mask)
                row.update(intensity)
        else:
            row['area_px'] = None   # signal that this frame had no valid mask

        nucleus_rows.append(row)

    # Compute summary statistics across valid frames
    valid_morph = [r for r in nucleus_rows if r.get('area_px') is not None]
    summary = {}
    if valid_morph:
        for key in ['area_px', 'perimeter_px', 'circularity', 'eccentricity',
                     'major_axis_px', 'minor_axis_px', 'solidity',
                     'nuc_intensity_mean', 'nuc_intensity_std']:
            vals = [r[key] for r in valid_morph if key in r and r[key] is not None]
            if vals:
                summary[f'{key}_avg'] = round(np.mean(vals), 2)
                summary[f'{key}_sd']  = round(np.std(vals), 2)

        print(f"  Valid frames: {len(valid_morph)}/{n_frames}")
        print(f"  Area (px)  : {summary.get('area_px_avg', 'N/A')} "
              f"± {summary.get('area_px_sd', 'N/A')}")
        print(f"  Circularity: {summary.get('circularity_avg', 'N/A')} "
              f"± {summary.get('circularity_sd', 'N/A')}")
        print(f"  Eccentricity: {summary.get('eccentricity_avg', 'N/A')} "
              f"± {summary.get('eccentricity_sd', 'N/A')}")
        print(f"  Solidity   : {summary.get('solidity_avg', 'N/A')} "
              f"± {summary.get('solidity_sd', 'N/A')}")

    # ── MODULE 2 & 3: Per-locus features ───────────────────────────────────────
    print(f"\n{'═'*60}")
    print("MODULE 2: Locus-to-membrane distance")
    print("MODULE 3: Local chromatin environment")
    print(f"{'═'*60}")

    locus_rows = []

    # Process each discovered trajectory CSV directly (not via ROI matching)
    # In Kevin's pipeline, trajectories are already in full-image nm coordinates
    for traj_key, traj_path in sorted(traj_csvs.items()):
        # Skip "_cleaned" duplicates for now — use the rela2wholeimg versions
        if '_cleaned' in traj_key:
            continue

        print(f"\n  Processing {traj_key}  ({traj_path.name})")

        traj_data = load_trajectory(traj_path)
        n_pts = len(traj_data['frames'])
        print(f"    {n_pts} frames in trajectory")

        if n_pts == 0:
            continue

        # Convert nm -> full-image pixel coordinates
        if traj_data['format'] == 'pipeline':
            # Kevin's pipeline: coordinates already relative to whole image
            traj_data['x_px_full'] = traj_data['x_nm'] / pixel_size
            traj_data['y_px_full'] = traj_data['y_nm'] / pixel_size
        else:
            # ThunderSTORM legacy: coordinates relative to ROI crop
            # Try to find matching ROI for offset
            roi_match = None
            for roi in rois:
                # Match locus number in ROI name to trajectory key
                if roi['name'] in traj_key or traj_key.replace('G-', '') == roi['name']:
                    roi_match = roi
                    break
            if roi_match:
                traj_data['x_px_full'] = traj_data['x_nm'] / pixel_size + roi_match['left']
                traj_data['y_px_full'] = traj_data['y_nm'] / pixel_size + roi_match['top']
            else:
                traj_data['x_px_full'] = traj_data['x_nm'] / pixel_size
                traj_data['y_px_full'] = traj_data['y_nm'] / pixel_size

        print(f"    Locus position (full-image px): "
              f"x={traj_data['x_px_full'].mean():.1f} "
              f"y={traj_data['y_px_full'].mean():.1f}")

        # ── Compute features per frame ─────────────────────────────────────────
        n_computed = 0
        n_skipped  = 0

        for i in range(n_pts):
            frame_idx = traj_data['frames'][i]

            # Frames are 1-indexed; masks are 0-indexed
            mask_idx = frame_idx - 1
            if mask_idx < 0 or mask_idx >= n_frames:
                n_skipped += 1
                continue

            mask = masks[mask_idx]
            if mask is None:
                n_skipped += 1
                continue

            locus_x = traj_data['x_px_full'][i]
            locus_y = traj_data['y_px_full'][i]

            row = {
                'locus_id':     traj_key,
                'frame':        frame_idx,
                'frame_0idx':   mask_idx,
                'x_nm':         round(traj_data['x_nm'][i], 2),
                'y_nm':         round(traj_data['y_nm'][i], 2),
                'x_px_full':    round(locus_x, 2),
                'y_px_full':    round(locus_y, 2),
                'sigma_nm':     round(float(traj_data['sigma_nm'][i]), 2) if not np.isnan(traj_data['sigma_nm'][i]) else None,
                'intensity':    round(float(traj_data['intensity'][i]), 2) if not np.isnan(traj_data['intensity'][i]) else None,
                'uncertainty_nm': round(float(traj_data['uncertainty_nm'][i]), 4) if not np.isnan(traj_data['uncertainty_nm'][i]) else None,
            }

            # Module 2: distance features
            dist = compute_distance_features(mask, locus_y, locus_x, pixel_size)
            if dist:
                row.update(dist)

            # Module 3: local chromatin
            if nucleus_stack is not None and mask_idx < nucleus_stack.shape[0]:
                local = compute_local_chromatin(
                    nucleus_stack[mask_idx], mask, locus_y, locus_x, local_window)
                if local:
                    row.update(local)

            locus_rows.append(row)
            n_computed += 1

        print(f"    Computed features for {n_computed} frames "
              f"({n_skipped} skipped — no valid mask)")

        # Print summary for this locus
        if n_computed > 0:
            dists = [r['dist_to_membrane_nm'] for r in locus_rows[-n_computed:]
                     if 'dist_to_membrane_nm' in r]
            radials = [r['norm_radial_pos'] for r in locus_rows[-n_computed:]
                       if 'norm_radial_pos' in r]
            if dists:
                print(f"    Membrane distance (nm): "
                      f"mean={np.mean(dists):.1f} ± {np.std(dists):.1f}  "
                      f"min={np.min(dists):.1f}  max={np.max(dists):.1f}")
            if radials:
                print(f"    Normalized radial pos:  "
                      f"mean={np.mean(radials):.3f} ± {np.std(radials):.3f}  "
                      f"(0=periphery, 1=center)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("SAVING OUTPUTS")
    print(f"{'═'*60}")

    # Save nucleus_features.csv
    nuc_csv_path = output_dir / 'nucleus_features.csv'
    if nucleus_rows:
        # Collect all possible keys
        all_keys = []
        for r in nucleus_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(str(nuc_csv_path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(nucleus_rows)
        print(f"  Saved: {nuc_csv_path.name}  ({len(nucleus_rows)} rows, "
              f"{len(all_keys)} columns)")

    # Save locus_features.csv
    locus_csv_path = output_dir / 'locus_features.csv'
    if locus_rows:
        all_keys = []
        for r in locus_rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)

        with open(str(locus_csv_path), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(locus_rows)
        print(f"  Saved: {locus_csv_path.name}  ({len(locus_rows)} rows, "
              f"{len(all_keys)} columns)")

    # Save summary to a separate file for quick reference
    summary_path = output_dir / 'feature_summary.txt'
    with open(str(summary_path), 'w') as f:
        f.write("Oligo-LiveFISH Feature Extraction Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Source: {output_dir}\n")
        f.write(f"Frames: {n_frames} total, {n_valid} with valid masks\n")
        f.write(f"Pixel size: {pixel_size} nm/px\n\n")

        f.write("Nucleus Morphology (averages across valid frames):\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Locus Features:\n")
        # Group by locus_id
        locus_ids = sorted(set(r['locus_id'] for r in locus_rows))
        for lid in locus_ids:
            matching = [r for r in locus_rows if r.get('locus_id') == lid]
            if matching:
                dists = [r['dist_to_membrane_nm'] for r in matching
                         if 'dist_to_membrane_nm' in r]
                radials = [r['norm_radial_pos'] for r in matching
                           if 'norm_radial_pos' in r]
                f.write(f"\n  {lid}:\n")
                f.write(f"    Frames analyzed: {len(matching)}\n")
                if dists:
                    f.write(f"    Membrane distance (nm): "
                            f"{np.mean(dists):.1f} ± {np.std(dists):.1f}\n")
                if radials:
                    f.write(f"    Normalized radial position: "
                            f"{np.mean(radials):.3f} ± {np.std(radials):.3f}\n")

    print(f"  Saved: {summary_path.name}")

    if not locus_rows:
        print("\nWARNING: No locus features were computed.")
        print("  This may be because no trajectory CSVs were found, or all loci")
        print("  fell outside the nucleus mask in every frame.")

    print("\nDone.")


if __name__ == '__main__':
    main()
