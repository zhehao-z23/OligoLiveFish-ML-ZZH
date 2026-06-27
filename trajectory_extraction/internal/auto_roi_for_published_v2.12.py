#!/usr/bin/env python3
"""
auto_roi_for_published_v2.12.py — Per-channel inter-frame limit; red relaxed to 750 nm.

Usage:
    python3 auto_roi_for_published_v2.12.py /path/to/image_Nucleus.tif

Changes from v2.11:
  • INTER_FRAME_MAX_NM_RED added (default 750 nm, vs 500 nm for purple).
    Analysis of MATLAB red trajectories shows the 90th-percentile frame-to-frame
    displacement is ~580 nm; the old 500 nm limit rejected ~16% of real steps.
  • track_channel_in_roi, find_seed_trajectories_in_mask, and propagate_from_seed
    now accept an inter_frame_max_px argument; callers pass the channel-appropriate
    value.  Purple tracking is unchanged (still 500 nm).
  • K_SIGNAL['red'] = 0.5 (same as v2.11).
  • Both track_channel_in_roi (singletons) and propagate_from_seed (overlap
    groups) gain a green-proximity fallback: when no candidate passes the
    inter-frame constraint, the candidate closest to the green spot (within
    _GREEN_PROX_MAX_PX) is accepted instead.  This recovers highly mobile loci
    whose frame-to-frame displacement exceeds inter_frame_max_px but whose
    signal stays near the green anchor.
  • Filtering of red tracks missing the first 3 frames is applied in
    match_m2DGaussian_to_reference.py (on the final MATLAB trajectory), not
    here.

Outputs (same naming as v2.8 / v2.10 / v2.11):
    RoiSet_green.zip
    Nucleus_masks.tif
    G_loci{N}_traj_rela2wholeimg.csv
    P_loci{N}_traj_rela2wholeimg.csv   (omitted if no unique match found)
    R_loci{N}_traj_rela2wholeimg.csv   (omitted if no unique match found)
"""

import sys
import csv
import struct
import zipfile
from collections import defaultdict
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS  —  edit these to tune the analysis                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

K_SIGNAL = {
    'green':  2.0,
    'red':    0.5,
    'purple': 0.5,
}

# Nucleus boundary detection: Gaussian pre-blur sigma (px) before Otsu thresholding.
NUCLEUS_SIGMA = 2.0

# Maximum fraction of tracked frames allowed to be outside the nucleus mask
# before a locus is rejected.  Default 0.10 = up to 10 % of frames may be outside.
NUCLEUS_OUTSIDE_FRAC = 0.10

# Drift-correction fill detection: a pixel is a fill candidate if its value is
# below FILL_RATIO × (75th-percentile of the image).  Only border-connected
# candidates are excluded.
FILL_RATIO = 0.85

MIN_SPOT_PX = 10

N_MAX        = 5
PADDING      = 20   # pixels added around the trajectory bounding box on all sides
ADJACENCY_PX = 30

# ── Physical-unit parameters for purple/red tracking ─────────────────────────
# Pixel size in pixels per µm (e.g. 5.45 px/µm → 1 px ≈ 183 nm).
PIXEL_SIZE_UM = 5.45

# Maximum frame-to-frame drift allowed when tracking purple/red (nm).
INTER_FRAME_MAX_NM     = 500
INTER_FRAME_MAX_NM_RED = 750   # relaxed for red: 90th-percentile observed jump ~580 nm

# Maximum distance from the green spot in the same frame for purple/red (µm).
GREEN_PROX_MAX_UM = 3.0

# Give up seeding a purple/red trajectory if no match is found in first N frames.
SEED_MAX_FRAME = 5

# Maximum component area (px) in the union-mask seed step before adaptive
# k-elevation is triggered.  Components larger than this are treated as merged
# spots and re-examined at higher k values.
MAX_BLOB_PX = 120

# ── Derived pixel-space limits (computed once at module load) ─────────────────
_INTER_FRAME_MAX_PX     = INTER_FRAME_MAX_NM     * PIXEL_SIZE_UM / 1000.0
_INTER_FRAME_MAX_PX_RED = INTER_FRAME_MAX_NM_RED * PIXEL_SIZE_UM / 1000.0
_GREEN_PROX_MAX_PX      = GREEN_PROX_MAX_UM      * PIXEL_SIZE_UM

# k values tried (in order) when a large blob is found; must be > base k.
_ADAPTIVE_K_STEPS = [1.0, 1.5, 2.0]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMPLEMENTATION — helpers unchanged from v2.8                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_stack(path: Path) -> np.ndarray:
    """Load a multi-frame TIFF → float32 array (frames, height, width)."""
    from PIL import Image
    img = Image.open(str(path))
    frames = []
    try:
        i = 0
        while True:
            img.seek(i)
            frames.append(np.array(img, dtype=np.float32))
            i += 1
    except EOFError:
        pass
    return np.stack(frames)


def compute_valid_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect drift-correction fill pixels in a single 2D image (one frame or average).

    Fill values vary per frame (0, 13, 52, 76 … ADU) and are always well below
    true background.  Algorithm: threshold at 75th-percentile × FILL_RATIO, label
    connected components, keep only border-connected ones, return complement.
    """
    H, W = image.shape
    bg_ref     = float(np.percentile(image, 75))
    low_thresh = bg_ref * FILL_RATIO
    is_candidate = image < low_thresh
    labeled, _ = ndimage.label(is_candidate)
    border_labels = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(int(v) for v in edge if v > 0)
    if not border_labels:
        return np.ones((H, W), dtype=bool)
    fill_mask = np.zeros((H, W), dtype=bool)
    for bl in border_labels:
        fill_mask |= (labeled == bl)
    return ~fill_mask


def otsu_threshold(image: np.ndarray) -> float:
    """Compute Otsu's optimal binary threshold (maximises between-class variance)."""
    hist, bin_edges = np.histogram(image.flatten(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total   = float(hist.sum())
    sum_all = float(np.dot(hist, bin_centers))
    w0, sum0, best_thresh, best_var = 0.0, 0.0, bin_centers[0], 0.0
    for i in range(len(hist)):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mu0   = sum0 / w0
        mu1   = (sum_all - sum0) / w1
        var_between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var    = var_between
            best_thresh = bin_centers[i]
    return best_thresh


def detect_nucleus_frames(nucleus_stack: np.ndarray) -> list:
    """
    Per-frame nucleus mask: Gaussian blur + Otsu + hole filling.
    Returns list of 2D bool arrays or None where detection failed.
    None frames are skipped in nucleus check (benefit of the doubt).
    """
    masks, n_skipped = [], 0
    for t, frame in enumerate(nucleus_stack):
        valid    = compute_valid_mask(frame)
        smoothed = ndimage.gaussian_filter(frame.astype(np.float32), sigma=NUCLEUS_SIGMA)
        thresh   = otsu_threshold(smoothed[valid])
        p5 = float(np.percentile(smoothed[valid], 5))
        if thresh < p5:
            n_skipped += 1
            masks.append(None)
            continue
        binary  = ndimage.binary_fill_holes((smoothed > thresh) & valid)
        labeled, n_comp = ndimage.label(binary)
        if n_comp == 0:
            n_skipped += 1
            masks.append(None)
            continue
        sizes = [(int(np.sum(labeled == i)), i) for i in range(1, n_comp + 1)]
        largest_idx = max(sizes, key=lambda x: x[0])[1]
        masks.append(labeled == largest_idx)
    if n_skipped > 0:
        print(f"    Note: {n_skipped} frame(s) near-unimodal → nucleus undetectable, "
              f"skipped in signal check")
    return masks


def save_nucleus_masks_tiff(masks: list, H: int, W: int, path: Path):
    """Save per-frame nucleus masks as multi-frame binary TIFF (uint8, 0/255)."""
    from PIL import Image
    imgs = []
    for m in masks:
        frame = (m.astype(np.uint8) * 255) if m is not None else np.zeros((H, W), dtype=np.uint8)
        imgs.append(Image.fromarray(frame))
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:])


def point_in_mask(y: float, x: float, mask: np.ndarray) -> bool:
    """Return True if pixel (y, x) falls inside the binary nucleus mask."""
    H, W = mask.shape
    yi, xi = int(round(y)), int(round(x))
    if 0 <= yi < H and 0 <= xi < W:
        return bool(mask[yi, xi])
    return False


def detect_signal_clusters(avg: np.ndarray, k: float, min_px: int, n_max: int) -> list:
    """
    Threshold the time-averaged image at mean + k×std, find connected components
    >= min_px pixels.  Returns up to n_max cluster dicts sorted by mean intensity
    (brightest first).  Each dict: centroid (y,x), bbox, intensity, size.
    """
    valid  = compute_valid_mask(avg)
    m, s   = avg[valid].mean(), avg[valid].std()
    labeled, n = ndimage.label((avg > (m + k * s)) & valid)
    clusters = []
    for idx in range(1, n + 1):
        comp = labeled == idx
        size = int(comp.sum())
        if size < min_px:
            continue
        cy, cx = ndimage.center_of_mass(avg * comp)
        rows, cols = np.where(comp)
        clusters.append({
            'centroid':  (cy, cx),
            'bbox':      (int(rows.min()), int(cols.min()),
                          int(rows.max()), int(cols.max())),
            'intensity': float(avg[comp].mean()),
            'size':      size,
        })
    clusters.sort(key=lambda c: c['intensity'], reverse=True)
    return clusters[:n_max]


def track_spot_indexed(stack: np.ndarray, centroid_avg: tuple, k: float) -> list:
    """
    Track a signal spot across all frames.
    Returns (frame_index, y, x) triples; frames with no match within ADJACENCY_PX
    are omitted.
    """
    cy0, cx0 = centroid_avg
    positions = []
    for t, frame in enumerate(stack):
        valid  = compute_valid_mask(frame)
        m, s   = frame[valid].mean(), frame[valid].std()
        labeled, n = ndimage.label((frame > (m + k * s)) & valid)
        best, best_d = None, ADJACENCY_PX
        for idx in range(1, n + 1):
            comp = labeled == idx
            cy, cx = ndimage.center_of_mass(frame * comp)
            d = np.hypot(cy - cy0, cx - cx0)
            if d < best_d:
                best_d, best = d, (cy, cx)
        if best:
            positions.append((t, best[0], best[1]))
    return positions


def compute_roi_from_trajectory(indexed_pos: list, H: int, W: int) -> tuple:
    """
    Compute ROI as the trajectory bounding box expanded by PADDING on all sides,
    clipped to image bounds.  Returns (top, left, bottom, right).
    """
    ys = [p[1] for p in indexed_pos]
    xs = [p[2] for p in indexed_pos]
    top    = max(0, int(min(ys)) - PADDING)
    left   = max(0, int(min(xs)) - PADDING)
    bottom = min(H, int(max(ys)) + PADDING + 1)
    right  = min(W, int(max(xs)) + PADDING + 1)
    return top, left, bottom, right


def track_channel_in_roi(stack: np.ndarray, roi: tuple,
                         green_indexed_pos: list, green_centroid_avg: tuple,
                         k: float, inter_frame_max_px: float = _INTER_FRAME_MAX_PX) -> list:
    """
    Track a purple or red spot within a fixed ROI across all frames.
    Used for non-overlapping (singleton) loci.

    Primary linking: candidate must be within inter_frame_max_px of last position
    AND within _GREEN_PROX_MAX_PX of the green spot.

    Fallback (when primary yields nothing): accept the candidate closest to the
    green spot, as long as it is within _GREEN_PROX_MAX_PX.  This handles highly
    mobile loci whose frame-to-frame displacement exceeds inter_frame_max_px but
    whose signal stays near the green anchor.
    """
    top, left, bottom, right = roi
    n_frames = stack.shape[0]
    green_tracked = {t: (fy, fx) for t, fy, fx in green_indexed_pos}

    def candidates_in_roi(frame_arr):
        valid  = compute_valid_mask(frame_arr)
        m, s   = frame_arr[valid].mean(), frame_arr[valid].std()
        labeled, n = ndimage.label((frame_arr > (m + k * s)) & valid)
        found = []
        for idx in range(1, n + 1):
            comp = labeled == idx
            if int(comp.sum()) < MIN_SPOT_PX:
                continue
            cy, cx = ndimage.center_of_mass(comp)
            if top <= cy < bottom and left <= cx < right:
                found.append((cy, cx))
        return found

    seed_frame = None
    last_pos   = None
    positions  = []

    for f in range(min(SEED_MAX_FRAME, n_frames)):
        cands = candidates_in_roi(stack[f])
        if not cands:
            continue
        ref_y, ref_x = green_tracked.get(f, green_centroid_avg)
        best = min(cands, key=lambda p: np.hypot(p[0] - ref_y, p[1] - ref_x))
        seed_frame = f
        last_pos   = best
        positions.append((f, best[0], best[1]))
        break

    if seed_frame is None:
        return []

    for f in range(seed_frame + 1, n_frames):
        cands = candidates_in_roi(stack[f])
        gy, gx = green_tracked.get(f, green_centroid_avg)

        # Primary: inter-frame constraint + green proximity
        cands_primary = [(cy, cx) for cy, cx in cands
                         if np.hypot(cy - last_pos[0], cx - last_pos[1]) <= inter_frame_max_px
                         and np.hypot(cy - gy, cx - gx) <= _GREEN_PROX_MAX_PX]
        if cands_primary:
            best = min(cands_primary,
                       key=lambda p: np.hypot(p[0] - last_pos[0], p[1] - last_pos[1]))
            last_pos = best
            positions.append((f, best[0], best[1]))
            continue

        # Fallback: green proximity only (handles highly mobile loci)
        cands_fallback = [(cy, cx) for cy, cx in cands
                          if np.hypot(cy - gy, cx - gx) <= _GREEN_PROX_MAX_PX]
        if cands_fallback:
            best = min(cands_fallback, key=lambda p: np.hypot(p[0] - gy, p[1] - gx))
            last_pos = best
            positions.append((f, best[0], best[1]))

    return positions


def save_trajectory_csv(indexed_pos: list, path: Path):
    """
    Save a trajectory as a CSV file with columns frame, x_nm, y_nm.
    - frame: 1-based
    - x_nm, y_nm: position in nanometres, 1-indexed pixel convention
    """
    with open(str(path), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x_nm', 'y_nm'])
        for frame_idx, y, x in indexed_pos:
            x_nm = (x + 1) / PIXEL_SIZE_UM * 1000.0
            y_nm = (y + 1) / PIXEL_SIZE_UM * 1000.0
            writer.writerow([frame_idx + 1, f'{x_nm:.2f}', f'{y_nm:.2f}'])


def write_roi_file(path: Path, top: int, left: int, bottom: int, right: int):
    """Write a minimal ImageJ binary rectangle .roi file (version 228)."""
    header = bytearray(64)
    header[0:4] = b'Iout'
    struct.pack_into('>H', header, 4,  228)
    header[6] = 1
    struct.pack_into('>H', header, 8,  max(0, top))
    struct.pack_into('>H', header, 10, max(0, left))
    struct.pack_into('>H', header, 12, max(0, bottom))
    struct.pack_into('>H', header, 14, max(0, right))
    path.write_bytes(bytes(header))


def save_roi_zip(rois: list, zip_path: Path, out_dir: Path):
    """Write a list of ROI dicts to a zip file. Does nothing if rois is empty."""
    if not rois:
        return
    tmp_files = []
    for roi in rois:
        name = f"loci{roi['label']}.roi"
        p    = out_dir / name
        write_roi_file(p, roi['top'], roi['left'], roi['bottom'], roi['right'])
        tmp_files.append((name, p))
    with zipfile.ZipFile(str(zip_path), 'w') as zf:
        for name, p in tmp_files:
            zf.write(str(p), name)
    for _, p in tmp_files:
        p.unlink()


def roi_dict(t, l, b, r, label):
    return {'top': t, 'left': l, 'bottom': b, 'right': r, 'label': label}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS — v2.10 joint seeding (unchanged)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _rois_overlap(roi1: tuple, roi2: tuple) -> bool:
    """Return True if two (top, left, bottom, right) rectangles share any pixel."""
    t1, l1, b1, r1 = roi1
    t2, l2, b2, r2 = roi2
    return t1 < b2 and t2 < b1 and l1 < r2 and l2 < r1


def find_overlap_groups(accepted: list) -> list:
    """
    Union-find connected components of overlapping ROIs.
    Returns list of groups (each a list of indices into accepted).
    """
    n = len(accepted)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _rois_overlap(accepted[i][2], accepted[j][2]):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def compute_union_mask(rois: list, H: int, W: int) -> np.ndarray:
    """Boolean (H×W) mask = union of all (top, left, bottom, right) ROI rectangles."""
    mask = np.zeros((H, W), dtype=bool)
    for (t, l, b, r) in rois:
        mask[t:b, l:r] = True
    return mask


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  NEW HELPERS — v2.11 adaptive k seed detection                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _find_candidates_adaptive(frame_arr: np.ndarray, union_mask: np.ndarray,
                               base_k: float, frame_idx: int) -> list:
    """
    Detect signal candidates in frame_arr whose centroid falls inside union_mask.

    Each component detected at base_k is classified:
      • area <= MAX_BLOB_PX — kept directly; detection k = base_k.
      • area >  MAX_BLOB_PX — likely a merged blob; try to split by re-examining
        the blob's own pixels at progressively higher thresholds (k in
        _ADAPTIVE_K_STEPS).  Only pixels belonging to this particular blob are
        affected; all other components are unchanged.

    Splitting logic (for each k_try in _ADAPTIVE_K_STEPS):
      1. Apply k_try threshold to the whole frame; label connected components.
      2. Identify sub-components whose pixels overlap this blob's mask.
      3. If no sub-components survive (blob fully disappeared or all fragments
         below MIN_SPOT_PX) → immediate fallback: use original centroid at base_k.
      4. If all valid sub-components have area <= MAX_BLOB_PX → success: replace
         the blob with those sub-components, each carrying k_try.
      5. Otherwise → some sub-components still too large; try next k_try.
      6. If k_try exhausted at 2.0 without success → fallback to original centroid.

    Returns list of (cy, cx, k_detected).
    """
    H, W = union_mask.shape
    valid = compute_valid_mask(frame_arr)
    m = float(frame_arr[valid].mean())
    s = float(frame_arr[valid].std())

    binary_base = (frame_arr > m + base_k * s) & valid
    labeled_base, n_base = ndimage.label(binary_base)

    result = []

    for idx in range(1, n_base + 1):
        comp_mask = (labeled_base == idx)
        area = int(comp_mask.sum())
        if area < MIN_SPOT_PX:
            continue
        cy, cx = ndimage.center_of_mass(comp_mask)
        yi, xi = int(round(cy)), int(round(cx))
        if not (0 <= yi < H and 0 <= xi < W and union_mask[yi, xi]):
            continue

        if area <= MAX_BLOB_PX:
            result.append((cy, cx, base_k))
            continue

        # ── Large blob: try adaptive splitting ───────────────────────────────
        split_placed = False
        for k_try in _ADAPTIVE_K_STEPS:
            binary_high = (frame_arr > m + k_try * s) & valid
            labeled_high, _ = ndimage.label(binary_high)

            # Sub-component labels from the higher threshold that overlap this blob
            sub_indices = set(int(v) for v in labeled_high[comp_mask] if v > 0)

            if not sub_indices:
                # Blob fully disappeared at k_try — no point trying larger k
                print(f"      [fr{frame_idx}] Large blob (area={area} px) disappeared "
                      f"at k={k_try} — fallback (k={base_k})")
                result.append((cy, cx, base_k))
                split_placed = True
                break

            # Collect valid sub-candidates
            sub_cands = []
            for sub_idx in sub_indices:
                sub_mask = (labeled_high == sub_idx)
                sub_area = int(sub_mask.sum())
                if sub_area < MIN_SPOT_PX:
                    continue
                sub_cy, sub_cx = ndimage.center_of_mass(sub_mask)
                sub_yi = int(round(sub_cy))
                sub_xi = int(round(sub_cx))
                if not (0 <= sub_yi < H and 0 <= sub_xi < W
                        and union_mask[sub_yi, sub_xi]):
                    continue
                sub_cands.append((sub_cy, sub_cx, sub_area))

            if not sub_cands:
                # All fragments below MIN_SPOT_PX or outside mask
                print(f"      [fr{frame_idx}] Large blob (area={area} px) fragmented "
                      f"below MIN_SPOT_PX at k={k_try} — fallback (k={base_k})")
                result.append((cy, cx, base_k))
                split_placed = True
                break

            if all(sc[2] <= MAX_BLOB_PX for sc in sub_cands):
                # Successfully split
                print(f"      [fr{frame_idx}] Large blob (area={area} px) → "
                      f"{len(sub_cands)} sub-component(s) at k={k_try} "
                      f"(areas: {[sc[2] for sc in sub_cands]})")
                for sub_cy, sub_cx, _ in sub_cands:
                    result.append((sub_cy, sub_cx, k_try))
                split_placed = True
                break
            # Some sub-components still too large → try next k_try

        if not split_placed:
            # All k steps tried; sub-components still exceed MAX_BLOB_PX
            print(f"      [fr{frame_idx}] Large blob (area={area} px) unsplittable "
                  f"at k={_ADAPTIVE_K_STEPS[-1]} — fallback (k={base_k})")
            result.append((cy, cx, base_k))

    return result


def find_seed_trajectories_in_mask(stack: np.ndarray, union_mask: np.ndarray,
                                    n_seed_frames: int, k: float,
                                    inter_frame_max_px: float = _INTER_FRAME_MAX_PX) -> tuple:
    """
    Find all distinct signal spot trajectories within union_mask across the first
    n_seed_frames frames, using adaptive k detection for large blobs.

    Per frame: _find_candidates_adaptive is called, which returns (cy, cx, k_detected)
    for each candidate.  Normal spots use k=base_k; sub-components from blob splitting
    use k=k_try (the k at which the split succeeded).

    Candidates are linked across consecutive frames with greedy nearest-neighbour
    matching (spatial constraint: _INTER_FRAME_MAX_PX; temporal: gap ≤ 1 frame).

    Returns (seed_trajs, seed_ks):
      seed_trajs : list of seed trajectories, each a list of (frame, cy, cx).
      seed_ks    : list of floats, one per trajectory.  seed_ks[i] = k_detected
                   of the last candidate in trajectory i.  Used as the propagation
                   k for that trajectory (so the same threshold applied during
                   seeding is used during propagation).
    """
    n_frames = min(n_seed_frames, stack.shape[0])

    track_points = []   # track_points[i] = list of (f, cy, cx)
    track_k_last = []   # track_k_last[i] = k_detected of last appended candidate

    for f in range(n_frames):
        cands = _find_candidates_adaptive(stack[f], union_mask, k, f)
        if not cands:
            continue

        # Build scored list: (distance, track_index, candidate_index)
        scored = []
        for ti, tpts in enumerate(track_points):
            last_f, last_y, last_x = tpts[-1]
            if f - last_f > 2:   # allow at most 1-frame gap
                continue
            for ci, (cy, cx, _) in enumerate(cands):
                d = np.hypot(cy - last_y, cx - last_x)
                if d <= inter_frame_max_px:
                    scored.append((d, ti, ci))

        scored.sort()
        matched_tracks, matched_cands = set(), set()
        for d, ti, ci in scored:
            if ti in matched_tracks or ci in matched_cands:
                continue
            cy, cx, k_det = cands[ci]
            track_points[ti].append((f, cy, cx))
            track_k_last[ti] = k_det
            matched_tracks.add(ti)
            matched_cands.add(ci)

        for ci, (cy, cx, k_det) in enumerate(cands):
            if ci not in matched_cands:
                track_points.append([(f, cy, cx)])
                track_k_last.append(k_det)

    return track_points, track_k_last


def match_seeds_to_greens(seed_trajs: list, seed_ks: list,
                           green_trajs_dict: dict) -> dict:
    """
    Optimal one-to-one matching between seed trajectories and green loci using
    the Linear Sum Assignment algorithm (minimises total pairwise distance).

    Cost matrix entry [i, j]:
      - Average Euclidean distance (px) between seed_trajs[i] and green locus j
        across their shared seed frames.
      - Set to a large sentinel (1e9) when there are no shared frames.

    After solving, pairs whose average distance exceeds _GREEN_PROX_MAX_PX are
    rejected.

    Returns dict mapping label → (seed_traj, k_propagate), where k_propagate is
    seed_ks[i] for the matched seed trajectory i.
    """
    labels = list(green_trajs_dict.keys())
    green_by_frame = {
        label: {f: (y, x) for f, y, x in traj}
        for label, traj in green_trajs_dict.items()
    }

    n_seeds = len(seed_trajs)
    n_loci  = len(labels)
    if n_seeds == 0 or n_loci == 0:
        return {}

    _INF = 1e9
    cost = np.full((n_seeds, n_loci), _INF)

    for si, seed_traj in enumerate(seed_trajs):
        seed_by_frame = {f: (y, x) for f, y, x in seed_traj}
        for li, label in enumerate(labels):
            shared = [f for f in seed_by_frame if f in green_by_frame[label]]
            if not shared:
                continue
            cost[si, li] = float(np.mean([
                np.hypot(seed_by_frame[f][0] - green_by_frame[label][f][0],
                         seed_by_frame[f][1] - green_by_frame[label][f][1])
                for f in shared
            ]))

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}
    for si, li in zip(row_ind, col_ind):
        if cost[si, li] >= _INF:
            continue
        if cost[si, li] > _GREEN_PROX_MAX_PX:
            continue
        assignments[labels[li]] = (seed_trajs[si], seed_ks[si])

    return assignments


def propagate_from_seed(stack: np.ndarray, roi: tuple, seed_traj: list,
                         green_indexed_pos: list, green_centroid_avg: tuple,
                         k: float, inter_frame_max_px: float = _INTER_FRAME_MAX_PX) -> list:
    """
    Propagate tracking from a pre-computed seed trajectory within roi.

    Starts from the last frame in seed_traj and continues to the end of the
    stack.  k is the detection threshold (may differ from channel base k when
    the seed came from adaptive blob splitting).

    Primary linking: candidate within inter_frame_max_px of last position AND
    within _GREEN_PROX_MAX_PX of the green spot.

    Fallback (same as track_channel_in_roi): when primary yields nothing,
    accept the candidate closest to the green spot within _GREEN_PROX_MAX_PX.
    This recovers highly mobile loci whose frame-to-frame displacement exceeds
    inter_frame_max_px but whose signal stays near the green anchor.

    Returns the complete trajectory: seed_traj positions + propagated positions.
    """
    if not seed_traj:
        return []

    top, left, bottom, right = roi
    n_frames = stack.shape[0]
    green_tracked = {t: (fy, fx) for t, fy, fx in green_indexed_pos}

    def candidates_in_roi(frame_arr):
        valid  = compute_valid_mask(frame_arr)
        m, s   = frame_arr[valid].mean(), frame_arr[valid].std()
        labeled, n = ndimage.label((frame_arr > (m + k * s)) & valid)
        found = []
        for idx in range(1, n + 1):
            comp = labeled == idx
            if int(comp.sum()) < MIN_SPOT_PX:
                continue
            cy, cx = ndimage.center_of_mass(comp)
            if top <= cy < bottom and left <= cx < right:
                found.append((cy, cx))
        return found

    positions  = list(seed_traj)
    seed_frame = seed_traj[-1][0]
    last_pos   = (seed_traj[-1][1], seed_traj[-1][2])

    for f in range(seed_frame + 1, n_frames):
        cands = candidates_in_roi(stack[f])
        gy, gx = green_tracked.get(f, green_centroid_avg)

        # Primary: inter-frame constraint + green proximity
        cands_primary = [(cy, cx) for cy, cx in cands
                         if np.hypot(cy - last_pos[0], cx - last_pos[1]) <= inter_frame_max_px
                         and np.hypot(cy - gy, cx - gx) <= _GREEN_PROX_MAX_PX]
        if cands_primary:
            best = min(cands_primary,
                       key=lambda p: np.hypot(p[0] - last_pos[0], p[1] - last_pos[1]))
            last_pos = best
            positions.append((f, best[0], best[1]))
            continue

        # Fallback: green proximity only (handles highly mobile loci)
        cands_fallback = [(cy, cx) for cy, cx in cands
                          if np.hypot(cy - gy, cx - gx) <= _GREEN_PROX_MAX_PX]
        if cands_fallback:
            best = min(cands_fallback, key=lambda p: np.hypot(p[0] - gy, p[1] - gx))
            last_pos = best
            positions.append((f, best[0], best[1]))

    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 auto_roi_for_published_v2.11.py <path_to_*_Nucleus.tif>")
        sys.exit(1)

    nucleus_path = Path(sys.argv[1]).resolve()
    if not nucleus_path.name.endswith('_Nucleus.tif'):
        print("ERROR: input filename must end with '_Nucleus.tif'")
        sys.exit(1)

    stem = str(nucleus_path)[:-len('_Nucleus.tif')]
    paths = {
        'nucleus': nucleus_path,
        'green':   Path(stem + '_green.tif'),
        'red':     Path(stem + '_red.tif'),
        'purple':  Path(stem + '_purple.tif'),
    }
    for ch, p in paths.items():
        if not p.exists():
            print(f"ERROR: {ch} file not found: {p}")
            sys.exit(1)

    out_dir = nucleus_path.parent
    print(f"Input   : {nucleus_path.name}")
    print(f"Green   : {paths['green'].name}")
    print(f"Red     : {paths['red'].name}")
    print(f"Purple  : {paths['purple'].name}")
    print(f"Output  : {out_dir}")
    print(f"Pixel   : {PIXEL_SIZE_UM} px/µm  "
          f"inter-frame limit purple={INTER_FRAME_MAX_NM} nm ({_INTER_FRAME_MAX_PX:.1f} px)  "
          f"red={INTER_FRAME_MAX_NM_RED} nm ({_INTER_FRAME_MAX_PX_RED:.1f} px)  "
          f"green-proximity limit {GREEN_PROX_MAX_UM} µm ({_GREEN_PROX_MAX_PX:.1f} px)  "
          f"padding {PADDING} px")
    print(f"Blob    : MAX_BLOB_PX={MAX_BLOB_PX} px  "
          f"adaptive k steps={_ADAPTIVE_K_STEPS} (overlap groups only)")

    # ── Load stacks ───────────────────────────────────────────────────────────
    print("\nLoading stacks...")
    stacks   = {ch: load_stack(p) for ch, p in paths.items()}
    H, W     = stacks['green'].shape[1], stacks['green'].shape[2]
    n_frames = stacks['green'].shape[0]
    avgs     = {ch: stacks[ch].mean(axis=0) for ch in ('green', 'red', 'purple')}
    print(f"  {n_frames} frames, {H}×{W} px")

    # ── Per-frame nucleus masks ───────────────────────────────────────────────
    print(f"\nBuilding per-frame nucleus masks (Otsu + Gaussian σ={NUCLEUS_SIGMA}, "
          f"FILL_RATIO={FILL_RATIO})...")
    nucleus_masks = detect_nucleus_frames(stacks['nucleus'])
    n_detected = sum(1 for m in nucleus_masks if m is not None)
    print(f"  Nucleus detected in {n_detected}/{n_frames} frames")
    masks_path = out_dir / 'Nucleus_masks.tif'
    save_nucleus_masks_tiff(nucleus_masks, H, W, masks_path)
    print(f"  Saved : {masks_path.name}")

    # ── Signal detection on green averaged image ──────────────────────────────
    print("\nDetecting green signal clusters (time-averaged image)...")
    _valid = compute_valid_mask(avgs['green'])
    m_g = float(avgs['green'][_valid].mean())
    s_g = float(avgs['green'][_valid].std())
    green_clusters = detect_signal_clusters(avgs['green'], K_SIGNAL['green'], MIN_SPOT_PX, N_MAX)
    print(f"  green   : {len(green_clusters)} cluster(s)  "
          f"threshold = {m_g:.1f} + {K_SIGNAL['green']}×{s_g:.1f} = "
          f"{m_g + K_SIGNAL['green']*s_g:.1f}  "
          f"sizes = {[c['size'] for c in green_clusters]}")

    if not green_clusters:
        print("\nNo green signal clusters found. "
              "Try lowering K_SIGNAL['green'] or MIN_SPOT_PX.")
        sys.exit(0)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  PASS 1 — Green tracking and ROI computation                        ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print("\n" + "═"*60)
    print("Pass 1 — Green tracking and ROI computation")
    print("═"*60)

    accepted    = []
    accepted_rois = []

    for idx, spot in enumerate(green_clusters):
        cy, cx = spot['centroid']
        print(f"\n  Spot {idx+1}: centroid=({cy:.1f}, {cx:.1f})  "
              f"size={spot['size']} px  intensity={spot['intensity']:.1f}")

        indexed_pos = track_spot_indexed(stacks['green'], (cy, cx), K_SIGNAL['green'])
        print(f"    Green tracked in {len(indexed_pos)}/{n_frames} frames")
        if not indexed_pos:
            print(f"    ✗ Skipped: green spot not trackable")
            continue

        checkable = [(t, fy, fx) for t, fy, fx in indexed_pos
                     if nucleus_masks[t] is not None]
        outside   = [(t, fy, fx) for t, fy, fx in checkable
                     if not point_in_mask(fy, fx, nucleus_masks[t])]
        frac_outside = len(outside) / len(checkable) if checkable else 0.0
        if frac_outside > NUCLEUS_OUTSIDE_FRAC:
            print(f"    ✗ Skipped: {len(outside)}/{len(checkable)} checkable frames "
                  f"({100*frac_outside:.1f}%) outside nucleus mask "
                  f"(limit {100*NUCLEUS_OUTSIDE_FRAC:.0f}%)")
            continue
        if outside:
            print(f"    ~ {len(outside)}/{len(checkable)} frame(s) outside nucleus "
                  f"({100*frac_outside:.1f}%) — within tolerance, accepted")
        else:
            print(f"    ✓ All green positions within nucleus")

        ys = [p[1] for p in indexed_pos]
        xs = [p[2] for p in indexed_pos]
        print(f"    Motion: Δy={max(ys)-min(ys):.1f} px  Δx={max(xs)-min(xs):.1f} px")

        t_roi, l, b, r = compute_roi_from_trajectory(indexed_pos, H, W)
        print(f"    ROI: top={t_roi} left={l} bottom={b} right={r}  "
              f"({r-l}×{b-t_roi} px)")

        label = idx + 1
        accepted.append((label, indexed_pos, (t_roi, l, b, r), (cy, cx)))
        accepted_rois.append(roi_dict(t_roi, l, b, r, label))

        g_csv = out_dir / f'G_loci{label}_traj_rela2wholeimg.csv'
        save_trajectory_csv(indexed_pos, g_csv)
        print(f"    Saved: {g_csv.name}  ({len(indexed_pos)} points)")

        if len(accepted) >= N_MAX:
            break

    if not accepted:
        print("\nNo green loci accepted. Exiting.")
        sys.exit(0)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  Overlap detection                                                  ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print(f"\n{'═'*60}")
    print("Overlap detection")
    print(f"{'═'*60}")

    overlap_groups = find_overlap_groups(accepted)

    for group_indices in overlap_groups:
        labels = [accepted[i][0] for i in group_indices]
        if len(group_indices) > 1:
            print(f"  ⚠ Overlapping group detected: loci {labels} — will use joint seeding")
        else:
            print(f"  Singleton: loci{labels[0]} (no ROI overlap)")

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  PASS 2 — Purple and Red tracking                                   ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print(f"\n{'═'*60}")
    print("Pass 2 — Purple and Red tracking")
    print(f"{'═'*60}")

    for group_indices in overlap_groups:
        group = [accepted[i] for i in group_indices]

        if len(group) == 1:
            # ── No overlap: identical to v2.8 / v2.10 ────────────────────────
            label, green_traj, roi, centroid_avg = group[0]
            print(f"\n  Loci {label} (singleton — standard tracking):")

            for ch, prefix in (('purple', 'P'), ('red', 'R')):
                ifpx = _INTER_FRAME_MAX_PX_RED if ch == 'red' else _INTER_FRAME_MAX_PX
                print(f"    ── {ch.capitalize()} tracking ──")
                pos = track_channel_in_roi(
                    stacks[ch], roi, green_traj, centroid_avg, K_SIGNAL[ch],
                    inter_frame_max_px=ifpx)
                if pos:
                    csv_path = out_dir / f'{prefix}_loci{label}_traj_rela2wholeimg.csv'
                    save_trajectory_csv(pos, csv_path)
                    print(f"    Saved: {csv_path.name}  ({len(pos)} points)")
                else:
                    print(f"    {ch.capitalize()}: no trajectory found within ROI "
                          f"(seed not found in first {SEED_MAX_FRAME} frames)")

        else:
            # ── Overlapping group: joint seeding with adaptive k ──────────────
            loci_labels = [acc[0] for acc in group]
            print(f"\n  Overlap group: loci {loci_labels} — joint seeding")

            group_rois = [acc[2] for acc in group]
            union_mask = compute_union_mask(group_rois, H, W)

            for ch, prefix in (('purple', 'P'), ('red', 'R')):
                ifpx = _INTER_FRAME_MAX_PX_RED if ch == 'red' else _INTER_FRAME_MAX_PX
                print(f"    ── {ch.capitalize()} tracking (joint) ──")

                seed_trajs, seed_ks = find_seed_trajectories_in_mask(
                    stacks[ch], union_mask, SEED_MAX_FRAME, K_SIGNAL[ch],
                    inter_frame_max_px=ifpx)
                print(f"    Seed trajectories found in union ROI: {len(seed_trajs)}")

                green_trajs_dict = {acc[0]: acc[1] for acc in group}
                assignments = match_seeds_to_greens(
                    seed_trajs, seed_ks, green_trajs_dict)

                for label, green_traj, roi, centroid_avg in group:
                    if label in assignments:
                        seed_traj, k_propagate = assignments[label]
                        _sbf = {f: (y, x) for f, y, x in seed_traj}
                        _gbf = {f: (y, x) for f, y, x in green_traj}
                        _shared = [f for f in _sbf if f in _gbf]
                        avg_d = float(np.mean([
                            np.hypot(_sbf[f][0] - _gbf[f][0],
                                     _sbf[f][1] - _gbf[f][1])
                            for f in _shared
                        ])) if _shared else float('nan')
                        print(f"    Loci{label}: matched seed "
                              f"(seed frames {[s[0] for s in seed_traj]}, "
                              f"avg dist to green = {avg_d:.2f} px, "
                              f"propagation k={k_propagate})")
                        pos = propagate_from_seed(
                            stacks[ch], roi, seed_traj,
                            green_traj, centroid_avg, k_propagate,
                            inter_frame_max_px=ifpx)
                        if pos:
                            csv_path = out_dir / (
                                f'{prefix}_loci{label}_traj_rela2wholeimg.csv')
                            save_trajectory_csv(pos, csv_path)
                            print(f"    Saved: {csv_path.name}  ({len(pos)} points)")
                        else:
                            print(f"    Loci{label} {ch}: seed matched but propagation "
                                  f"yielded 0 points within individual ROI")
                    else:
                        print(f"    Loci{label} {ch}: no unique seed match found "
                              f"(seed candidates={len(seed_trajs)}, "
                              f"competing loci={len(group)}) — no trajectory written")

    # ── Save ROI zip ──────────────────────────────────────────────────────────
    green_zip = out_dir / 'RoiSet_green.zip'
    save_roi_zip(accepted_rois, green_zip, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Summary")
    print(f"{'═'*60}")
    print(f"  Green ROIs ({len(accepted_rois)}): "
          f"{['loci'+str(r['label']) for r in accepted_rois]}")
    n_overlap = sum(1 for g in overlap_groups if len(g) > 1)
    if n_overlap:
        print(f"  Overlap groups: {n_overlap} (joint seeding + adaptive k applied)")
    print(f"\n  Nucleus masks → {masks_path.name}")
    print("\nDone.")


if __name__ == '__main__':
    main()
