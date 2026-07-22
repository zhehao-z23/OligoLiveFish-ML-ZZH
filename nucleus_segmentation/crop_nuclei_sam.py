"""
crop_nuclei_sam.py — LiveFISH nucleus auto-cropping via µSAM (vit_b_lm, APG mode)

Usage:
    python nucleus_segmentation/crop_nuclei_sam.py INPUT_ND2_OR_DIRECTORY \
        --nucleus-channel 0 --margin 30 \
        --min-area 1000 --max-area 200000 \
        --segmentation-mode apg --model-type vit_b_lm --device auto
"""

import argparse
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # matplotlib will not plot to the screen, only to the file
import matplotlib.pyplot as plt
import numpy as np
from nd2 import ND2File
import dask.array as da
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter # pre and post processing of masks
from skimage.feature import peak_local_max # peak detection for watershed split
from skimage.measure import regionprops # region properties for mask filtering
from skimage.segmentation import watershed # watershed segmentation for mask (adjacent nuclei) splitting
import tifffile


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def resolve_device(device_arg: str) -> str:
    """Resolve auto/cuda/mps/cpu into a torch device string for this machine."""
    import torch

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but this PyTorch installation cannot "
                "access CUDA. Use --device cpu, or install a CUDA-enabled PyTorch "
                "build that matches the NVIDIA driver."
            )
        return "cuda"
    if device_arg == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is not available.")
        return "mps"
    if device_arg == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def describe_device(device: str) -> str:
    """Return a concise, auditable description of the selected backend."""
    if device != "cuda":
        return device
    import torch

    index = torch.cuda.current_device()
    total_gib = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    return f"cuda:{index} ({torch.cuda.get_device_name(index)}, {total_gib:.1f} GiB)"

# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_BORDER_MARGIN_PX = 5 # minimum distance from the border of the image to the nucleus center
MIN_SOLIDITY     = 0.70 # minimum solidity of the nucleus to not be passed to watershed split (by raising this, we allow more fragmented nuclei to be passed to watershed split)
SPLIT_SIGMA      = 5 # sigma for gaussian filtering of the distance transform for peak detection
SPLIT_MIN_DIST   = 20 # minimum distance between peaks of nuclei to be considered as separate nuclei (and therefore passed to watershed split)
MERGE_PROXIMITY  = 2 # maximum distance between nuclei to be considered as part of the same nucleus (and therefore passed to merging)
MERGE_MIN_SOLID  = 0.60 # minimum solidity of the nucleus to not be passed to merging (by raising this, we allow more fragmented nuclei to be passed to merging)
MIN_CIRC         = 0.3 # minimum circularity 4*pi*area/perimeter^2 of the nucleus to not be passed to merging (by raising this, we allow more elongated nuclei to be passed to merging)
IOU_THRESH       = 0.3 # maximum allowed overlap between nuclei to be considered as part of the same nucleus and discarded since it's redundant
CONTAIN_THRESH   = 0.5 # minimum containment of the nucleus to not be passed to merging (raising this allows more elongated nuclei to be passed to merging)
BAD_QC_CH0_CONTRAST_MAX = 0.085
BAD_QC_CH0_BOUNDARY_GRAD_MAX = 1.70


# ── helpers ────────────────────────────────────────────────────────────────────
"""
load_fov() loads an .nd2 FOV file into a fully-axed numpy array.
Inputs:  path — path to the .nd2 file
Outputs: numpy array of shape (T, Z, C, Y, X) uint16

The nd2 library squeezes out size-1 axes, so the raw array can have fewer
than 5 dims depending on acquisition mode. We reinsert any missing T/Z/C axes
and transpose to a canonical TZCYX order so downstream code can assume a fixed
shape regardless of how the file was acquired.

1. open the .nd2 file and read the array + axis order
2. reinsert any size-1 T/Z/C axes the nd2 library squeezed out
3. transpose to canonical TZCYX order
4. return as uint16
"""
def load_fov(path: Path) -> np.ndarray:
    with ND2File(path) as f:
        arr = f.asarray()
        keys = list(f.sizes.keys())

    axes = [k.upper() for k in keys]
    print(f"  nd2 axes string: {''.join(axes)}, raw shape: {arr.shape}")

    # Reinsert any size-1 axes the nd2 library squeezed out
    for ax in ('T', 'Z', 'C'):
        if ax not in axes:
            arr = np.expand_dims(arr, axis=0)
            axes.insert(0, ax)
            print(f"  Reinserted size-1 '{ax}' axis (squeezed out by nd2 library)")

    target = ['T', 'Z', 'C', 'Y', 'X']
    order = [axes.index(a) for a in target]
    arr = np.transpose(arr, order).astype(np.uint16, copy=False)
    print(f"  Final shape TZCYX: {arr.shape}")
    return arr


def load_nucleus_average(path: Path, nucleus_channel: int):
    """Load only the projected channel needed for segmentation."""
    with ND2File(path) as f:
        arr = f.to_dask()
        keys = list(f.sizes.keys())

        axes = [k.upper() for k in keys]
        print(f"  nd2 axes string: {''.join(axes)}, raw shape: {arr.shape}")

        for ax in ('T', 'Z', 'C'):
            if ax not in axes:
                arr = da.expand_dims(arr, axis=0)
                axes.insert(0, ax)
                print(f"  Reinserted size-1 '{ax}' axis (squeezed out by nd2 library)")

        target = ['T', 'Z', 'C', 'Y', 'X']
        order = [axes.index(a) for a in target]
        arr = arr.transpose(order)
        T, Z, C, H, W = [int(v) for v in arr.shape]
        print(f"  Final shape TZCYX: {(T, Z, C, H, W)}")

        nuc_sum = np.zeros((H, W), dtype=np.float32)
        for t in range(T):
            frame = (
                arr[t, :, nucleus_channel, :, :]
                .max(axis=0)
                .compute()
                .astype(np.float32, copy=False)
            )
            nuc_sum += frame
        nuc_avg = nuc_sum / max(T, 1)
    return nuc_avg, (T, Z, C, H, W)

"""
norm_u8() percentile-stretches an image into the uint8 range for µSAM input.
Inputs:  img — numpy array (float or uint); lo_pct/hi_pct — clip percentiles
Outputs: numpy array in the uint8 range (0-255)

1. compute the lo/hi percentile values of the image
2. clip the image to those values (guards against outlier-driven contrast loss)
3. rescale the clipped range to 0-255
4. return as uint8
"""
def norm_u8(img: np.ndarray, lo_pct=1, hi_pct=99) -> np.ndarray:
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    if hi == lo:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def segment_2d_with_cuda_memory_guard(segment_fn, predictor, segmenter, nuc_u8, args):
    """Run micro-SAM; on CUDA, precompute embeddings with decoder offloaded."""
    use_guard = str(getattr(predictor, "device", "")).startswith("cuda") and hasattr(segmenter, "_decoder")
    if not use_guard:
        return segment_fn(
            predictor=predictor,
            segmenter=segmenter,
            input_path=nuc_u8,
            ndim=2,
        )

    import torch
    from micro_sam import util as micro_sam_util

    decoder = getattr(segmenter, "_decoder", None)
    decoder_param = next(decoder.parameters(), None) if decoder is not None else None
    decoder_device = decoder_param.device if decoder_param is not None else None

    print("  CUDA memory guard: precomputing embeddings with APG decoder offloaded to CPU")
    if decoder is not None:
        decoder.to("cpu")
    torch.cuda.empty_cache()

    try:
        with tempfile.TemporaryDirectory(prefix="micro_sam_embeddings_") as tmp:
            embedding_path = str(Path(tmp) / "embeddings.zarr")
            micro_sam_util.precompute_image_embeddings(
                predictor=predictor,
                input_=nuc_u8,
                save_path=embedding_path,
                ndim=2,
                verbose=True,
                batch_size=1,
            )
            predictor.reset_image()
            torch.cuda.empty_cache()
            if decoder is not None and decoder_device is not None:
                decoder.to(decoder_device)
            return segment_fn(
                predictor=predictor,
                segmenter=segmenter,
                input_path=nuc_u8,
                embedding_path=embedding_path,
                ndim=2,
            )
    finally:
        if decoder is not None and decoder_device is not None:
            decoder.to(decoder_device)


"""
circularity() computes the circularity of a region (1.0 = perfect circle).
Inputs:  area, perimeter — region area and perimeter in pixels
Outputs: circularity 4*pi*area/perimeter^2 (0.0 if perimeter is 0)
"""
def circularity(area, perimeter):
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)



"""
try_split_mask() attempts to split a low-solidity mask into two nuclei via watershed.
Inputs:  mask — (H, W) bool mask; min_area — minimum area for a valid piece
Outputs: list of two (H, W) bool masks if the split succeeds, else None

1. calculate the distance transform of the mask
2. smooth the distance transform with a gaussian filter
3. find the peaks in the distance transform
4. create markers for the watershed segmentation
5. perform the watershed segmentation
6. get the properties of the nucleus
7. if the area is too small, we discard the nucleus
8. if the circularity is low, (too elongated) we discard the nucleus
9. return the pieces
"""
def try_split_mask(mask: np.ndarray, min_area: int):
    dist = distance_transform_edt(mask) # this modifies the mask by creating a distance transform of the mask
    dist_smooth = gaussian_filter(dist, sigma=SPLIT_SIGMA)
    coords = peak_local_max(dist_smooth, min_distance=SPLIT_MIN_DIST,
                            num_peaks=2, labels=mask)
    if len(coords) < 2:
        return None
    markers = np.zeros_like(mask, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    labels_ws = watershed(-dist_smooth, markers, mask=mask)
    pieces = []
    for lbl in [1, 2]:
        piece = labels_ws == lbl # get the mask for the nucleus
        props = regionprops(piece.astype(np.uint8)) # get the properties of the nucleus
        if not props: # if no properties, the nucleus isn't valid
            continue
        p = props[0]
        if p.area < min_area: # if the area is too small, we discard the nucleus
            continue
        if circularity(p.area, p.perimeter) < MIN_CIRC: # if the circularity is low, (too elongated) we discard the nucleus
            continue 
        pieces.append(piece) # add the piece to the list
    if len(pieces) == 2: # if there are 2 pieces, we return the pieces
        return pieces
    return None # if there are not 2 pieces, we return None (keeping the original nucleus, probably malformed)

"""
filter_masks() drops debris/border masks and watershed-splits low-solidity ones.
Inputs:  masks — list of (H, W) bool masks; img_shape — (H, W);
         min_area, max_area — area bounds in px; border_margin — min px from edge to centroid
Outputs: (kept_masks, stats) where stats is a dict of per-reason discard counts

1. get the height and width of the image
2. create a list to keep the masks
3. create a dictionary to keep the statistics
4. for each mask:
5. get the properties of the mask
6. if the area is too small, we discard the mask
7. if the area is too large, we discard the mask
8. if the centroid is too close to the border, we discard the mask
9. if the solidity is too low, we attempt to split the mask
10. if the split is successful, we keep the pieces and add the statistics
11. if the split is unsuccessful, we keep the original mask and add the statistics
12. return the kept masks and the statistics
"""
def filter_masks(masks, img_shape, min_area, max_area, border_margin):
    H, W = img_shape 
    kept = [] # list to keep the masks
    stats = {"too_small": 0, "too_large": 0, "border": 0, # dictionary to keep the statistics
             "mask_border": 0,
             "split_ok": 0, "split_fail_kept": 0, "split_fail_dropped": 0, # you can see what global variables might need to be changed based on the respective statistics
             "kept": 0}  
    for mask in masks:
        props = regionprops(mask.astype(np.uint8)) # get the properties of the mask
        if not props:
            continue # if no properties, the mask isn't valid
        p = props[0] # get the first property
        area = p.area # get the area of the mask
        cy, cx = p.centroid # get the centroid of the mask

        # check if the area is too small, too large, or if centroid is too close to the border
        if area < min_area:
            stats["too_small"] += 1
            continue
        if area > max_area * 2: # if the area is too large, we discard the mask (2x allows leeway for merged pairs)
            stats["too_large"] += 1
            continue
        if cy < border_margin or cy > H - border_margin:
            stats["border"] += 1
            continue
        if cx < border_margin or cx > W - border_margin:
            stats["border"] += 1
            continue
        if p.solidity < MIN_SOLIDITY: # if the solidity is too low, we attempt to split the mask
            pieces = try_split_mask(mask, min_area)
            if pieces: # if the split is successful, we keep the pieces and add the statistics
                kept.extend(pieces)
                stats["split_ok"] += len(pieces)
                continue
            else: # if the split is unsuccessful, we keep the original mask and add the statistics
                if area <= max_area:
                    kept.append(mask)
                    stats["split_fail_kept"] += 1
                else:
                    stats["split_fail_dropped"] += 1
                continue

        kept.append(mask)
        stats["kept"] += 1

    return kept, stats

"""
deduplicate_masks() removes masks that substantially overlap a larger accepted mask.
Inputs:  masks — list of (H, W) bool masks
Outputs: list of surviving (H, W) bool masks

1. Sort masks largest-first so that when two masks overlap, the larger one is
   accepted and the smaller one is rejected (not the reverse).
2. For each candidate (largest to smallest), compute IoU and containment against
   every already-accepted mask. Discard if either exceeds its threshold.
3. Return the surviving masks.
"""
def deduplicate_masks(masks):
    if not masks:
        return masks
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]
    accepted = []
    accepted_meta = []
    for i in order:
        m = masks[i]
        rows, cols = np.where(m)
        if rows.size == 0:
            continue
        bbox = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
        area_m = int(areas[i])
        discard = False
        for a, area_a, bbox_a in accepted_meta:
            r0 = max(bbox[0], bbox_a[0])
            r1 = min(bbox[1], bbox_a[1])
            c0 = max(bbox[2], bbox_a[2])
            c1 = min(bbox[3], bbox_a[3])
            if r0 >= r1 or c0 >= c1:
                inter = 0
            else:
                inter = np.logical_and(m[r0:r1, c0:c1], a[r0:r1, c0:c1]).sum()
            union = area_m + area_a - inter
            iou = inter / union if union else 0
            containment = inter / area_m if area_m else 0  # fraction of m's pixels covered by already-accepted a
            if iou > IOU_THRESH or containment > CONTAIN_THRESH:
                discard = True  # m is a duplicate or subset of a — keep a, drop m
                break
        if not discard:
            accepted.append(m)
            accepted_meta.append((m, area_m, bbox))
    return accepted


"""
merge_adjacent_masks() fuses µSAM fragment masks that are adjacent and whose
union passes nucleus validity checks (µSAM occasionally splits one nucleus in two).
Inputs:  masks — list of (H, W) bool masks; max_area — max area for a valid union
Outputs: (merged_masks, total_merges) — updated mask list and merge count

1. Build a square dilation kernel of radius MERGE_PROXIMITY to use as a
   proximity detector.
2. Repeat until a full pass produces no new merges (handles chains: if
   A+B merge, then B+C can merge in the next pass):
   a. For each unmerged mask i, dilate it and check whether any other
      unmerged mask j overlaps the dilation (i.e. is within MERGE_PROXIMITY
      px of i).
   b. If adjacent, form the union and validate: area ≤ max_area AND
      solidity ≥ MERGE_MIN_SOLID. Low solidity means the union is still
      two-blob shaped — not a real fused nucleus — so reject it.
   c. Accept the first valid partner found (greedy). Mark both used; add
      union to result.
   d. Unmerged masks carry through to result unchanged.
3. Return the updated mask list and total merge count.
"""
def merge_adjacent_masks(masks, max_area):
    if len(masks) < 2:
        return masks, 0

    # square kernel: dilating by struct expands a mask by MERGE_PROXIMITY px in all 8 directions
    struct = np.ones((MERGE_PROXIMITY * 2 + 1, MERGE_PROXIMITY * 2 + 1), dtype=bool)
    total_merges = 0

    changed = True
    while changed:  # repeat until a full pass produces no merges; needed so chains of ≥3 fragments can fully fuse
        changed = False
        used = [False] * len(masks)
        result = []
        for i in range(len(masks)):
            if used[i]:
                continue
            dil_i = binary_dilation(masks[i], structure=struct)
            merged_with = None
            for j in range(i + 1, len(masks)):
                if used[j]:
                    continue
                if not (dil_i & masks[j]).any():
                    continue  # j has no pixel within MERGE_PROXIMITY px of i — not adjacent
                union = masks[i] | masks[j]
                props = regionprops(union.astype(np.uint8))
                if not props:
                    continue
                p = props[0]
                if p.area > max_area:
                    continue
                if p.solidity < MERGE_MIN_SOLID:
                    continue  # union is still two-blob shaped; rejecting avoids fusing unrelated nearby nuclei
                merged_with = j
                result.append(union)
                used[i] = True
                used[j] = True
                total_merges += 1
                changed = True
                break  # greedy: take first valid partner; remaining masks re-evaluated next pass
            if merged_with is None and not used[i]:  # not used[i] guards against i already consumed as j in an earlier merge this pass
                result.append(masks[i])
                used[i] = True
        masks = result

    return masks, total_merges


def filter_mask_border_after_merge(masks, img_shape, mask_border_margin):
    """Drop masks touching the image border after fragment merging is complete."""
    if mask_border_margin < 0:
        return masks, 0

    H, W = img_shape
    kept = []
    removed = 0
    for mask in masks:
        rows, cols = np.where(mask)
        if rows.size == 0:
            continue
        mask_edge_dist = min(rows.min(), cols.min(), H - 1 - rows.max(), W - 1 - cols.max())
        if mask_edge_dist <= mask_border_margin:
            removed += 1
            continue
        kept.append(mask)
    return kept, removed


def compute_ch0_quality_metrics(mask, nuc_avg):
    """Compute final channel-0 quality metrics for one candidate mask."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {
            "ch0_median_inside": float("nan"),
            "ch0_background_ring_median": float("nan"),
            "ch0_contrast": float("nan"),
            "ch0_boundary_grad": float("nan"),
        }

    H, W = nuc_avg.shape
    pad = 8
    r0, r1 = max(0, rows.min() - pad), min(H, rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(W, cols.max() + pad + 1)
    local_img = nuc_avg[r0:r1, c0:c1].astype(np.float32, copy=False)
    local_mask = mask[r0:r1, c0:c1]

    dilated = binary_dilation(local_mask, iterations=6)
    bg = dilated & (~local_mask)
    if int(bg.sum()) < 25:
        bg = ~local_mask

    eroded = binary_erosion(local_mask, iterations=1, border_value=0)
    boundary = local_mask ^ eroded
    gy, gx = np.gradient(local_img)
    grad_mag = np.hypot(gx, gy)

    inside_med = float(np.median(local_img[local_mask])) if local_mask.any() else float("nan")
    bg_med = float(np.median(local_img[bg])) if bg.any() else float("nan")
    contrast = float((inside_med - bg_med) / (bg_med + 1e-6)) if np.isfinite(bg_med) else float("nan")
    boundary_grad = float(np.median(grad_mag[boundary])) if boundary.any() else float("nan")
    return {
        "ch0_median_inside": inside_med,
        "ch0_background_ring_median": bg_med,
        "ch0_contrast": contrast,
        "ch0_boundary_grad": boundary_grad,
    }


def filter_low_quality_nuclei(masks, nuc_avg):
    """Apply the locked low-quality nucleus rule as the final ordered filter."""
    kept = []
    kept_metrics = []
    removed = []
    for mask in masks:
        metrics = compute_ch0_quality_metrics(mask, nuc_avg)
        bad_qc = (
            metrics["ch0_contrast"] < BAD_QC_CH0_CONTRAST_MAX
            and metrics["ch0_boundary_grad"] < BAD_QC_CH0_BOUNDARY_GRAD_MAX
        )
        metrics["bad_qc"] = bool(bad_qc)
        if bad_qc:
            removed.append((mask, metrics))
        else:
            kept.append(mask)
            kept_metrics.append(metrics)
    return kept, kept_metrics, removed


"""
crop_with_suppression() crops one nucleus from the FOV and zeros all other nuclei.
Inputs:  fov — (T, Z, C, H, W) uint16; mask — (H, W) bool target nucleus;
         all_masks — list of all accepted masks; margin — px padding around the bbox
Outputs: (crop, bbox, suppress) where crop is (T, Z, C, cropH, cropW) uint16 with
         neighbour pixels zeroed, bbox is (r0, r1, c0, c1), suppress is the (H, W)
         union of all other masks

1. compute the padded bounding box of the target mask, clipped to the image
2. build a suppression mask = union of every other accepted nucleus
3. copy the bbox region and zero the suppression pixels across all T, Z, C
"""
def crop_with_suppression(fov: np.ndarray, mask: np.ndarray,
                          all_masks: list, margin: int):
    T, Z, C, H, W = fov.shape
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - margin)
    r1 = min(H, rows.max() + margin + 1)
    c0 = max(0, cols.min() - margin)
    c1 = min(W, cols.max() + margin + 1)

    suppress = np.zeros((H, W), dtype=bool)
    for other in all_masks:
        if other is mask:
            continue
        suppress |= other

    crop = fov[:, :, :, r0:r1, c0:c1].copy()
    crop[:, :, :, suppress[r0:r1, c0:c1]] = 0
    return crop, (r0, r1, c0, c1), suppress


def crop_metadata_with_suppression(mask: np.ndarray, all_masks: list, margin: int, img_shape):
    """Return bbox and suppression mask without loading the full multi-channel FOV."""
    H, W = img_shape
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - margin)
    r1 = min(H, rows.max() + margin + 1)
    c0 = max(0, cols.min() - margin)
    c1 = min(W, cols.max() + margin + 1)

    suppress = np.zeros((H, W), dtype=bool)
    for other in all_masks:
        if other is mask:
            continue
        suppress |= other

    return (r0, r1, c0, c1), suppress


def sort_masks_spatially(masks):
    """Return masks in a stable top-to-bottom, left-to-right cell order."""
    keyed = []
    for original_idx, mask in enumerate(masks):
        props = regionprops(mask.astype(np.uint8))
        if props:
            cy, cx = props[0].centroid
        else:
            cy, cx = float("inf"), float("inf")
        keyed.append(((cy, cx, original_idx), mask))
    return [mask for _, mask in sorted(keyed, key=lambda item: item[0])]


def sort_raw_candidates_spatially(raw_masks):
    """Return ``(raw_label, mask)`` records in stable spatial order."""
    keyed = []
    for raw_label, mask in enumerate(raw_masks, start=1):
        props = regionprops(mask.astype(np.uint8))
        if props:
            cy, cx = props[0].centroid
        else:
            cy, cx = float("inf"), float("inf")
        keyed.append(((cy, cx, raw_label), (raw_label, mask)))
    return [record for _, record in sorted(keyed, key=lambda item: item[0])]


def raw_candidate_qc_record(mask, raw_label, candidate_idx, nuc_avg, args):
    """Annotate one unmodified micro-SAM instance without discarding it.

    These direct, per-mask rules are intentionally annotations only. The locked
    analysis path may subsequently split, merge, or deduplicate masks, so
    ``default_gate_pass`` is not claimed to be a one-to-one reconstruction of
    membership in ``final_masks``.
    """
    H, W = nuc_avg.shape
    props = regionprops(mask.astype(np.uint8))
    if not props:
        raise ValueError(f"micro-SAM label {raw_label} produced an empty mask")

    p = props[0]
    cy, cx = p.centroid
    rows, cols = np.where(mask)
    mask_edge_dist = int(min(
        rows.min(), cols.min(), H - 1 - rows.max(), W - 1 - cols.max()
    ))
    metrics = compute_ch0_quality_metrics(mask, nuc_avg)
    bad_qc = bool(
        metrics["ch0_contrast"] < BAD_QC_CH0_CONTRAST_MAX
        and metrics["ch0_boundary_grad"] < BAD_QC_CH0_BOUNDARY_GRAD_MAX
    )

    reasons = []
    if p.area < args.min_area:
        reasons.append("too_small")
    if p.area > args.max_area * 2:
        reasons.append("too_large")
    if (
        cy < args.border_margin or cy > H - args.border_margin
        or cx < args.border_margin or cx > W - args.border_margin
    ):
        reasons.append("centroid_border")
    if args.mask_border_margin >= 0 and mask_edge_dist <= args.mask_border_margin:
        reasons.append("mask_border")
    if bad_qc:
        reasons.append("bad_qc")

    return {
        "candidate_id": f"candidate_{candidate_idx:03d}",
        "raw_usam_label": int(raw_label),
        "crop_index": int(candidate_idx),
        "default_gate_pass": not reasons,
        "manual_decision": "",
        "exclusion_reasons": ";".join(reasons),
        "area": int(p.area),
        "centroid_y": float(cy),
        "centroid_x": float(cx),
        "solidity": float(p.solidity),
        "circularity": float(circularity(p.area, p.perimeter)),
        "edge_dist": float(min(cy, cx, H - 1 - cy, W - 1 - cx)),
        "mask_edge_dist": mask_edge_dist,
        "low_solidity_split_candidate": bool(p.solidity < MIN_SOLIDITY),
        "bad_qc": bad_qc,
        **metrics,
    }


def write_raw_candidate_archive(
    nd2_path: Path,
    raw_masks: list,
    nuc_avg: np.ndarray,
    fov_shape: tuple,
    args,
):
    """Write every original micro-SAM instance to an isolated crop archive.

    The archive contains cropped mask TIFFs immediately and a crops JSON that
    ``save_crops.py`` can use to export the corresponding multi-channel TIFFs.
    It never changes the locked ``final_masks`` analysis output.
    """
    if not getattr(args, "preserve_all_candidates", True):
        return None

    T, _Z, _C, H, W = fov_shape
    source_stem = nd2_path.stem
    archive_stem = f"{source_stem}_candidate"
    if args.candidate_output_root:
        candidate_root = Path(args.candidate_output_root)
    elif getattr(args, "output_root", ""):
        analysis_root = Path(args.output_root)
        candidate_root = analysis_root.with_name(
            analysis_root.name + "_all_usam_candidates"
        )
    else:
        candidate_root = nd2_path.parent / "all_usam_candidates"
    archive_dir = candidate_root / source_stem
    archive_dir.mkdir(parents=True, exist_ok=True)

    records = []
    crops_json = []
    spatial_candidates = sort_raw_candidates_spatially(raw_masks)
    all_raw_masks = [mask for _label, mask in spatial_candidates]

    for candidate_idx, (raw_label, mask) in enumerate(spatial_candidates, start=1):
        record = raw_candidate_qc_record(
            mask, raw_label, candidate_idx, nuc_avg, args
        )
        bbox, suppress = crop_metadata_with_suppression(
            mask, all_raw_masks, args.margin, (H, W)
        )
        # Overlapping raw instances must never suppress the target candidate.
        suppress &= ~mask
        r0, r1, c0, c1 = bbox
        suppress_crop = suppress[r0:r1, c0:c1]
        sup_rows, sup_cols = np.where(suppress_crop)

        crop_tiff = f"{archive_stem}_{candidate_idx}.tif"
        mask_tiff = f"{archive_stem}_mask_{candidate_idx}.tif"
        record["crop_tiff"] = crop_tiff
        record["mask_tiff"] = mask_tiff
        record["bbox_r0"] = int(r0)
        record["bbox_r1"] = int(r1)
        record["bbox_c0"] = int(c0)
        record["bbox_c1"] = int(c1)
        records.append(record)

        crops_json.append({
            "idx": candidate_idx,
            "candidate_id": record["candidate_id"],
            "raw_usam_label": raw_label,
            "bbox": [int(r0), int(r1), int(c0), int(c1)],
            "microsam_mask": mask_tiff,
            "suppression_rows": sup_rows.tolist(),
            "suppression_cols": sup_cols.tolist(),
            "default_gate_pass": record["default_gate_pass"],
            "exclusion_reasons": record["exclusion_reasons"],
            "ch0_median_inside": record["ch0_median_inside"],
            "ch0_background_ring_median": record["ch0_background_ring_median"],
            "ch0_contrast": record["ch0_contrast"],
            "ch0_boundary_grad": record["ch0_boundary_grad"],
            "bad_qc": record["bad_qc"],
        })

        mask_crop = mask[r0:r1, c0:c1].astype(np.uint8) * 255
        mask_stack = np.repeat(mask_crop[np.newaxis], T, axis=0)
        tifffile.imwrite(
            archive_dir / mask_tiff,
            mask_stack,
            imagej=True,
            metadata={"axes": "TYX"},
        )

    manifest_fields = [
        "candidate_id", "raw_usam_label", "crop_index", "crop_tiff",
        "mask_tiff", "default_gate_pass", "manual_decision",
        "exclusion_reasons", "area", "centroid_y", "centroid_x",
        "solidity", "circularity", "edge_dist", "mask_edge_dist",
        "low_solidity_split_candidate", "bad_qc", "ch0_median_inside",
        "ch0_background_ring_median", "ch0_contrast", "ch0_boundary_grad",
        "bbox_r0", "bbox_r1", "bbox_c0", "bbox_c1",
    ]
    manifest_path = archive_dir / "candidate_selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(records)

    json_path = archive_dir / f"{archive_stem}_crops.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "nd2_path": str(nd2_path.resolve()),
            "stem": archive_stem,
            "archive_kind": "raw_usam_candidates",
            "source_instance_count": len(raw_masks),
            "crops": crops_json,
        }, f)

    gate_pass = sum(bool(record["default_gate_pass"]) for record in records)
    print(
        f"  Preserved {len(records)} raw micro-SAM candidates -> {archive_dir}/ "
        f"(direct QC pass={gate_pass}, annotated only)"
    )
    return archive_dir


def write_cell_id_mapping(out_dir: Path, crops_info: list, image_shape: tuple[int, int]):
    """Write the stable cell ID to crop-file index mapping used by batch tracking."""
    H, W = image_shape
    fields = [
        "cell_id", "original_idx", "centroid_y", "centroid_x", "area",
        "solidity", "circularity", "edge_dist", "mask_edge_dist",
        "crop_h", "crop_w", "bbox_r0", "bbox_r1", "bbox_c0", "bbox_c1",
    ]
    mapping_path = out_dir / "cell_id_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (_crop, bbox, mask, _suppress, _metrics) in enumerate(crops_info, start=1):
            r0, r1, c0, c1 = bbox
            props = regionprops(mask.astype(np.uint8))
            if not props:
                continue
            p = props[0]
            cy, cx = p.centroid
            rows, cols = np.where(mask)
            mask_edge_dist = min(
                rows.min(), cols.min(), H - 1 - rows.max(), W - 1 - cols.max()
            )
            writer.writerow({
                "cell_id": f"cell_{idx:03d}",
                # New runs already write crops in stable spatial order, so the
                # TIFF suffix and batch lookup index are intentionally identical.
                "original_idx": idx,
                "centroid_y": float(cy),
                "centroid_x": float(cx),
                "area": int(p.area),
                "solidity": float(p.solidity),
                "circularity": float(circularity(p.area, p.perimeter)),
                "edge_dist": float(min(cy, cx, H - 1 - cy, W - 1 - cx)),
                "mask_edge_dist": int(mask_edge_dist),
                "crop_h": int(r1 - r0),
                "crop_w": int(c1 - c0),
                "bbox_r0": int(r0),
                "bbox_r1": int(r1),
                "bbox_c0": int(c0),
                "bbox_c1": int(c1),
            })
    print(f"  Saved cell ID mapping -> {mapping_path.name}")


# ── visualisation ──────────────────────────────────────────────────────────────

"""
_draw_masks_numbered() overlays numbered, colour-coded masks onto an axis.
Inputs:  ax — matplotlib axis; nuc_u8 — (H, W) uint8 background image;
         masks — list of (H, W) bool masks; title — axis title
Outputs: none (draws onto ax in place)
"""
def _draw_masks_numbered(ax, nuc_u8, masks, title):
    ax.imshow(nuc_u8, cmap='gray', alpha=0.5)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(masks), 1))
    label_overlay = np.zeros(nuc_u8.shape, dtype=np.uint16)
    for idx, m in enumerate(masks):
        label_overlay[m] = idx + 1
        props = regionprops(m.astype(np.uint8))
        if props:
            cy, cx = props[0].centroid
            ax.text(cx, cy, str(idx + 1), color='white',
                    fontsize=7, ha='center', va='center', fontweight='bold')
    masked_overlay = np.ma.masked_where(label_overlay == 0, label_overlay)
    ax.imshow(masked_overlay, cmap=cmap, alpha=0.5, interpolation='none')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


"""
save_seg_overview() saves a 4-panel segmentation diagnostic figure.
Inputs:  nuc_u8 — (H, W) uint8 image; segmentation — (H, W) µSAM label map;
         filtered_masks, merged_masks, deduped_masks — mask lists at each stage;
         out_path — where to write the PNG
Outputs: none (writes raw image | µSAM raw | after filter+merge | final deduped)
"""
def save_seg_overview(nuc_u8, segmentation, filtered_masks, merged_masks,
                      deduped_masks, out_path: Path):
    stride = max(1, int(np.ceil(max(nuc_u8.shape) / 1400)))
    if stride > 1:
        nuc_u8 = nuc_u8[::stride, ::stride]
        segmentation = segmentation[::stride, ::stride]
        filtered_masks = [m[::stride, ::stride] for m in filtered_masks]
        merged_masks = [m[::stride, ::stride] for m in merged_masks]
        deduped_masks = [m[::stride, ::stride] for m in deduped_masks]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(nuc_u8, cmap='gray')
    axes[0].set_title("Time-averaged nucleus channel (uint8)", fontsize=10)
    axes[0].axis('off')

    overlay_raw = np.zeros((*nuc_u8.shape, 3), dtype=float)
    n_inst = int(segmentation.max())
    cmap_inst = matplotlib.colormaps.get_cmap('tab20').resampled(max(n_inst, 1))
    for i in range(1, n_inst + 1):
        color = cmap_inst(i % cmap_inst.N)[:3]
        for ch in range(3):
            overlay_raw[:, :, ch] += (segmentation == i) * color[ch]
    overlay_raw = np.clip(overlay_raw, 0, 1)  # adjacent instances can additively exceed 1.0
    axes[1].imshow(nuc_u8, cmap='gray', alpha=0.5)
    axes[1].imshow(overlay_raw, alpha=0.5)
    axes[1].set_title(f"µSAM raw: {n_inst} instances", fontsize=10)
    axes[1].axis('off')

    _draw_masks_numbered(axes[2], nuc_u8, merged_masks,
                         f"After ordered structural filters: {len(merged_masks)}")
    _draw_masks_numbered(axes[3], nuc_u8, deduped_masks,
                         f"Final after all filters: {len(deduped_masks)}")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved segmentation overview -> {out_path.name}")


"""
save_crop_grid() saves a thumbnail grid of every accepted nucleus crop.
Inputs:  crops_info — list of (crop, bbox, mask, ...) tuples; nuc_u8 — unused ref image;
         out_path — where to write the PNG
Outputs: none (writes a grid of nucleus-channel max-T thumbnails)
"""
def save_crop_grid(crops_info, nuc_u8, out_path: Path):
    n = len(crops_info)
    if n == 0:
        return
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols  # ceiling division

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.5))
    axes = np.array(axes).reshape(-1)  # flatten 2D axes grid → 1D for uniform indexing

    for idx, (crop, bbox, mask, *_) in enumerate(crops_info):
        nuc_crop = crop[:, :, 0].max(axis=1).max(axis=0).astype(float)
        ax = axes[idx]
        ax.imshow(nuc_crop, cmap='gray',
                  vmin=np.percentile(nuc_crop, 1),
                  vmax=np.percentile(nuc_crop, 99))
        ax.set_title(f"#{idx+1}\n{crop.shape[3]}×{crop.shape[4]}px", fontsize=7)
        ax.axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"All {n} nucleus crops — nucleus channel (max-T projection)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved crop grid -> {out_path.name}")


"""
save_suppression_demo() saves before/after neighbour-suppression panels.
Inputs:  crops_info — list of (crop, bbox, mask, ...) tuples; fov — (T,Z,C,H,W) source;
         deduped_masks — unused ref; out_path — PNG path; max_show — max nuclei to show
Outputs: none (writes side-by-side raw vs suppressed nucleus-channel crops)
"""
def save_suppression_demo(crops_info, fov, deduped_masks, out_path: Path,
                          max_show=5):
    n = min(len(crops_info), max_show)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]  # plt.subplots(1,2) returns 1D; add dim for consistent [row,col] indexing

    for row, (crop, bbox, mask, *_) in enumerate(crops_info[:n]):
        r0, r1, c0, c1 = bbox
        raw_region  = fov[:, :, 0, r0:r1, c0:c1].max(axis=1).max(axis=0).astype(float)
        supp_region = crop[:, :, 0].max(axis=1).max(axis=0).astype(float)

        vmin = np.percentile(raw_region[raw_region > 0], 1) if raw_region.max() > 0 else 0 # exclude zeroed suppression pixels from contrast floor 
        vmax = np.percentile(raw_region, 99)

        axes[row, 0].imshow(raw_region,  cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"#{row+1} Before suppression", fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(supp_region, cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"#{row+1} After suppression", fontsize=8)
        axes[row, 1].axis('off')

    plt.suptitle("Neighbour suppression demo (nucleus channel, max-T)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved suppression demo -> {out_path.name}")


"""
save_all_channels_demo() saves an all-channels view of the first few crops.
Inputs:  crops_info — list of (crop, bbox, mask, ...) tuples; out_path — PNG path;
         max_show — max crops to display
Outputs: none (writes one row per crop, one column per channel, max-T projected)
"""
def save_all_channels_demo(crops_info, out_path: Path, max_show=5):
    n = min(len(crops_info), max_show)
    if n == 0:
        return
    n_chan = crops_info[0][0].shape[2]
    chan_labels = ['C0 Nucleus (DAPI)', 'C1 640nm', 'C2 488nm', 'C3 561nm']

    fig, axes = plt.subplots(n, n_chan, figsize=(n_chan * 2.8, n * 2.8))
    axes = np.atleast_2d(np.array(axes).reshape(n, n_chan))  # subplots squeezes when n==1 or n_chan==1; force 2D

    for row, (crop, bbox, mask, *_) in enumerate(crops_info[:n]):
        for ch in range(n_chan):
            img = crop[:, :, ch].max(axis=1).max(axis=0).astype(float)
            axes[row, ch].imshow(img, cmap='gray',
                                 vmin=np.percentile(img, 1),
                                 vmax=np.percentile(img, 99))
            if row == 0:
                axes[row, ch].set_title(
                    chan_labels[ch] if ch < len(chan_labels) else f"C{ch}",
                    fontsize=8)
            axes[row, ch].set_ylabel(f"#{row+1}", fontsize=8)
            axes[row, ch].axis('off')

    plt.suptitle("All channels — first crops (max-T projection)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all-channels demo -> {out_path.name}")


# ── main processing ────────────────────────────────────────────────────────────

"""
process_file() runs the full nucleus-cropping pipeline on one .nd2 FOV file.
Inputs:  nd2_path — path to the .nd2 file; predictor, segmenter — preloaded µSAM
         objects; segment_fn — µSAM automatic_instance_segmentation; args — CLI args
Outputs: number of crops written (also writes JSON, mask TIFFs, and visualizations
         into a folder next to the input file); returns the crop count

1. Load and Z-max-project the .nd2 → (T, C, H, W) uint16.
2. Time-average the nucleus channel → (H, W) float, normalize to uint8.
   Time-averaging accumulates stationary locus signal while washing out
   diffuse unbound probes — gives µSAM a clean nucleus silhouette.
3. Run µSAM automatic instance segmentation → (H, W) int label map.
4. Convert labels → bool masks; label 0 is background, so start at 1.
5. filter_masks: drop debris/border nuclei; watershed-split low-solidity
   (merged-pair) masks.
6. merge_adjacent_masks: fuse µSAM fragments back into whole nuclei.
7. deduplicate_masks: remove masks that substantially overlap a larger one.
8. Crop each nucleus with neighbour suppression; emit visualizations.
9. Write <stem>_crops.json encoding bboxes and suppression pixel coords
   for save_crops.py to consume without re-running µSAM.
"""
def process_file(nd2_path: Path, predictor, segmenter, segment_fn, args):
    print(f"\n-- {nd2_path.name} --")

    nuc_avg, fov_shape = load_nucleus_average(nd2_path, args.nucleus_channel)
    T, Z, C, H, W = fov_shape
    print(f"  Loaded nucleus projection source: T={T} Z={Z} C={C} Y={H} X={W}")

    # Max-project Z only for segmentation input — µSAM needs 2D
    # The full Z info is preserved in fov for cropping and TIFF export
    nuc_u8  = norm_u8(nuc_avg)

    segmentation = segment_2d_with_cuda_memory_guard(
        segment_fn,
        predictor,
        segmenter,
        nuc_u8,
        args,
    )

    n_inst = int(segmentation.max())
    print(f"  micro-SAM found {n_inst} instances")

    raw_masks = [segmentation == i for i in range(1, n_inst + 1)]  # label 0 is background — skip it

    # Optional, isolated archive of every unmodified micro-SAM instance. This
    # is written before any scientific filtering and does not affect the locked
    # analysis path below.
    write_raw_candidate_archive(nd2_path, raw_masks, nuc_avg, fov_shape, args)

    filtered_masks, stats = filter_masks(
        raw_masks,
        (H, W),
        args.min_area,
        args.max_area,
        args.border_margin,
    )

    merged_masks, n_merges = merge_adjacent_masks(filtered_masks, args.max_area)
    if n_merges:
        print(f"  {n_merges} fragment pair(s) merged -> {len(merged_masks)} masks")

    merged_masks, post_merge_mask_border = filter_mask_border_after_merge(
        merged_masks,
        (H, W),
        args.mask_border_margin,
    )
    stats["mask_border"] = post_merge_mask_border
    print(f"  {len(filtered_masks)} masks after ordered area/centroid/split filters "
          f"(small={stats['too_small']}, large={stats['too_large']}, "
          f"border={stats['border']}, mask_border={stats['mask_border']}, "
          f"split={stats['split_ok']}, "
          f"split_kept={stats['split_fail_kept']}, "
          f"split_dropped={stats['split_fail_dropped']})")
    if args.mask_border_margin >= 0:
        print(f"  {len(merged_masks)} masks after post-merge mask_border filter "
              f"(removed={post_merge_mask_border})")

    deduped_masks = deduplicate_masks(merged_masks)
    merged_masks = sort_masks_spatially(merged_masks)
    deduped_masks = sort_masks_spatially(deduped_masks)
    print(f"  {len(deduped_masks)} masks after deduplication")

    final_masks, quality_metrics, bad_qc_removed = filter_low_quality_nuclei(deduped_masks, nuc_avg)
    final_masks = sort_masks_spatially(final_masks)
    quality_metrics = [compute_ch0_quality_metrics(mask, nuc_avg) | {"bad_qc": False} for mask in final_masks]
    stats["bad_qc"] = len(bad_qc_removed)
    print(f"  {len(final_masks)} masks after final low-quality filter "
          f"(bad_qc={stats['bad_qc']})")
    print(f"  ordered filter summary: raw={n_inst}, small={stats['too_small']}, "
          f"large={stats['too_large']}, centroid_border={stats['border']}, "
          f"split_ok={stats['split_ok']}, split_fail_kept={stats['split_fail_kept']}, "
          f"split_fail_dropped={stats['split_fail_dropped']}, merge_pairs={n_merges}, "
          f"mask_border={stats['mask_border']}, bad_qc={stats['bad_qc']}, "
          f"final={len(final_masks)}")

    stem    = nd2_path.stem
    out_dir = (Path(args.output_root) if args.output_root else nd2_path.parent) / stem
    out_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    with open(out_dir / "cell_crop_quality_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "cell_id", "ch0_median_inside", "ch0_background_ring_median",
            "ch0_contrast", "ch0_boundary_grad", "bad_qc",
        ]
        import csv
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, metrics in enumerate(quality_metrics, start=1):
            row = dict(metrics)
            row["cell_id"] = f"cell_{idx:03d}"
            writer.writerow({key: row.get(key, "") for key in fields})

    with open(out_dir / "filtered_bad_qc_candidates.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "candidate_idx", "area", "centroid_y", "centroid_x",
            "ch0_median_inside", "ch0_background_ring_median",
            "ch0_contrast", "ch0_boundary_grad", "bad_qc",
        ]
        import csv
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (mask, metrics) in enumerate(bad_qc_removed, start=1):
            props = regionprops(mask.astype(np.uint8))
            row = dict(metrics)
            row["candidate_idx"] = idx
            if props:
                row["area"] = int(props[0].area)
                row["centroid_y"] = float(props[0].centroid[0])
                row["centroid_x"] = float(props[0].centroid[1])
            writer.writerow({key: row.get(key, "") for key in fields})

    save_seg_overview(nuc_u8, segmentation, filtered_masks, merged_masks,
                      final_masks, viz_dir / "seg_overview.png")

    crops_info = []
    for mask, metrics in zip(final_masks, quality_metrics):
        bbox, suppress = crop_metadata_with_suppression(mask, final_masks, args.margin, (H, W))
        r0, r1, c0, c1 = bbox
        crop = nuc_avg[r0:r1, c0:c1][np.newaxis, np.newaxis, np.newaxis, :, :]
        crops_info.append((crop, bbox, mask, suppress, metrics))

    write_cell_id_mapping(out_dir, crops_info, (H, W))

    save_crop_grid(crops_info, nuc_u8, viz_dir / "crop_grid.png")
    preview_fov = nuc_avg[np.newaxis, np.newaxis, np.newaxis, :, :]
    save_suppression_demo(crops_info, preview_fov, final_masks, viz_dir / "suppression_demo.png")
    save_all_channels_demo(crops_info, viz_dir / "all_channels_demo.png")

    # Save bboxes + suppression masks to JSON for save_crops.py to consume
    crops_json = []
    for idx, (crop, bbox, mask, suppress, metrics) in enumerate(crops_info, start=1):
        r0, r1, c0, c1 = bbox
        suppress_crop = suppress[r0:r1, c0:c1]
        rows, cols = np.where(suppress_crop)  # JSON can't serialize numpy arrays; coordinate lists are compact and load-friendly
        crops_json.append({
            'idx':              idx,
            'bbox':             [int(r0), int(r1), int(c0), int(c1)],
            'microsam_mask':    f"{stem}_mask_{idx}.tif",
            'suppression_rows': rows.tolist(),
            'suppression_cols': cols.tolist(),
            'ch0_median_inside': metrics["ch0_median_inside"],
            'ch0_background_ring_median': metrics["ch0_background_ring_median"],
            'ch0_contrast': metrics["ch0_contrast"],
            'ch0_boundary_grad': metrics["ch0_boundary_grad"],
            'bad_qc': metrics["bad_qc"],
        })

    json_path = out_dir / f"{stem}_crops.json"
    with open(json_path, 'w') as f:
        json.dump({'nd2_path': str(nd2_path.resolve()), 'stem': stem,
                   'crops': crops_json}, f)

    # Save binary mask TIFFs — one per nucleus, cropped to bbox, (T, cropH, cropW),
    # nucleus=255 background=0, same mask repeated across T to match the
    # downstream trajectory-extraction pipeline's expected mask format
    for idx, (crop, bbox, mask, suppress, metrics) in enumerate(crops_info, start=1):
        r0, r1, c0, c1 = bbox
        mask_crop = mask[r0:r1, c0:c1].astype(np.uint8) * 255
        mask_stack = np.repeat(mask_crop[np.newaxis], T, axis=0)
        tifffile.imwrite(
            out_dir / f"{stem}_mask_{idx}.tif",
            mask_stack,
            imagej=True,
            metadata={"axes": "TYX"},
        )
    print(f"  OK {len(crops_info)} nuclear masks -> {out_dir}/")

    print(f"  OK {len(crops_info)} crops -> {out_dir}/")
    print(f"  OK Visualizations -> {viz_dir}/")
    return len(crops_info)


"""
main() parses CLI args, loads µSAM once, then batch-processes all .nd2 files.
Inputs:  none (reads command-line arguments)
Outputs: none (processes each file via process_file and prints a summary)

The predictor and segmenter are shared across files — model load (~30s) happens once.
"""
def main():
    parser = argparse.ArgumentParser(
        description="Crop nuclei from LiveFISH .nd2 FOV files using µSAM")
    parser.add_argument("input", help=".nd2 file or directory of .nd2 files")
    parser.add_argument("--nucleus-channel",   type=int, default=0)
    parser.add_argument("--margin",            type=int, default=30)
    parser.add_argument("--min-area",          type=int, default=1000)
    parser.add_argument("--max-area",          type=int, default=200000)
    parser.add_argument("--border-margin",     type=int, default=DEFAULT_BORDER_MARGIN_PX, help="Min distance (px) from image border to nucleus centroid")
    parser.add_argument("--mask-border-margin", type=int, default=-1, help="Drop masks whose pixels are <= this many px from image border; -1 disables")
    parser.add_argument("--segmentation-mode", default="apg", choices=["apg", "amg"])
    parser.add_argument("--model-type",        default="vit_b_lm")
    parser.add_argument("--device",            default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-root",       default="", help="Optional output parent; defaults to the ND2 parent")
    parser.add_argument(
        "--preserve-all-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve every raw micro-SAM instance in a separate archive "
            "(default: enabled; use --no-preserve-all-candidates to disable)."
        ),
    )
    parser.add_argument(
        "--candidate-output-root",
        default="",
        help=(
            "Optional separate output parent for all raw micro-SAM candidates. "
            "By default, <output-root>_all_usam_candidates is used. Keep this "
            "outside --output-root so analysis cannot discover candidates twice."
        ),
    )
    args = parser.parse_args()

    if args.candidate_output_root and args.output_root:
        analysis_root = Path(args.output_root).resolve()
        candidate_root = Path(args.candidate_output_root).resolve()
        if candidate_root == analysis_root or analysis_root in candidate_root.parents:
            parser.error(
                "--candidate-output-root must not equal or be nested under "
                "--output-root"
            )

    input_path = Path(args.input)
    if input_path.is_dir():
        nd2_files = sorted(input_path.rglob("*.nd2"))
        if not nd2_files:
            print(f"No .nd2 files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        nd2_files = [input_path]

    print(f"Files to process: {len(nd2_files)}")
    for f in nd2_files:
        print(f"  {f.name}")

    device = resolve_device(args.device)
    print(f"\nLoading micro-SAM model: {args.model_type}, "
          f"mode={args.segmentation_mode}, device={describe_device(device)}")
    if device == "cpu" and len(nd2_files) > 1:
        print(
            "TIP: CPU mode is supported, but directory batches are much faster on "
            "an NVIDIA GPU. If CUDA is available, rerun with --device cuda."
        )
    from micro_sam.automatic_segmentation import (  # deferred — µSAM has heavy import-time side effects
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        device=device,
        segmentation_mode=args.segmentation_mode,
    )
    print("  micro-SAM loaded.")

    total_crops = 0
    failed = []
    for nd2_path in nd2_files:
        try:
            n = process_file(nd2_path, predictor, segmenter, automatic_instance_segmentation, args)
            total_crops += n
        except Exception as e:
            print(f"  ERROR on {nd2_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append(nd2_path.name)

    print(f"\n{'='*60}")
    print(f"Done. {len(nd2_files) - len(failed)}/{len(nd2_files)} files OK, "
          f"{total_crops} total crops.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
