#!/usr/bin/env python3
"""Align a crop's saved micro-SAM instance mask to Fiji-corrected frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def discover_microsam_mask(crop_tiff: Path) -> Path:
    """Resolve the exact mask associated with a crop TIFF."""
    crop_tiff = crop_tiff.resolve()
    sidecar = crop_tiff.with_name(crop_tiff.stem + "_metadata.json")
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        entry = data.get("microsam_mask") or {}
        for candidate in (entry.get("path"), entry.get("relative_path")):
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = crop_tiff.parent / path
            if path.is_file():
                return path.resolve()

        stem = data.get("stem")
        index = data.get("crop_index")
        if stem is not None and index is not None:
            candidate = crop_tiff.parent / f"{stem}_mask_{index}.tif"
            if candidate.is_file():
                return candidate.resolve()

    match = re.match(r"^(?P<stem>.+)_(?P<index>\d+)$", crop_tiff.stem)
    if match:
        candidate = crop_tiff.parent / (
            f"{match.group('stem')}_mask_{match.group('index')}.tif"
        )
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No micro-SAM mask is associated with {crop_tiff}. "
        "Use --microsam-mask or rerun nucleus_segmentation/save_crops.py with v4."
    )


def _series_array(path: Path) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return np.asarray(series.asarray()), series.axes


def load_crop_channel_tyx(crop_tiff: Path, channel_index: int) -> np.ndarray:
    """Load one channel from a crop TIFF and max-project Z to TYX."""
    array, axes = _series_array(crop_tiff)
    if "C" not in axes:
        if channel_index != 0:
            raise ValueError(f"No C axis in {crop_tiff}; cannot select channel {channel_index}")
    else:
        array = np.take(array, channel_index, axis=axes.index("C"))
        axes = axes.replace("C", "")
    if "Z" in axes:
        array = np.max(array, axis=axes.index("Z"))
        axes = axes.replace("Z", "")
    for axis_name in tuple(axes):
        if axis_name not in "TYX":
            index = axes.index(axis_name)
            if array.shape[index] != 1:
                raise ValueError(
                    f"Unsupported non-singleton axis {axis_name} in {crop_tiff}: "
                    f"{axes=} {array.shape=}"
                )
            array = np.take(array, 0, axis=index)
            axes = axes.replace(axis_name, "")
    if "T" not in axes:
        array = np.expand_dims(array, axis=0)
        axes = "T" + axes
    if set(axes) != set("TYX") or len(axes) != 3:
        raise ValueError(f"Expected TYX after channel/Z selection: {axes=} {array.shape=}")
    order = [axes.index(axis) for axis in "TYX"]
    return np.transpose(array, order).astype(np.float32, copy=False)


def load_tyx(path: Path) -> np.ndarray:
    array, axes = _series_array(path)
    for axis_name in tuple(axes):
        if axis_name not in "TYX":
            index = axes.index(axis_name)
            if array.shape[index] != 1:
                raise ValueError(f"Unsupported axis {axis_name}: {axes=} {array.shape=}")
            array = np.take(array, 0, axis=index)
            axes = axes.replace(axis_name, "")
    if "T" not in axes:
        array = np.expand_dims(array, axis=0)
        axes = "T" + axes
    if set(axes) != set("TYX") or len(axes) != 3:
        raise ValueError(f"Expected TYX stack: {path}, {axes=}, {array.shape=}")
    return np.transpose(array, [axes.index(axis) for axis in "TYX"])


def load_static_mask(mask_path: Path) -> np.ndarray:
    array, axes = _series_array(mask_path)
    while array.ndim > 2:
        array = np.take(array, 0, axis=0)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D or repeated 2-D mask: {mask_path}, {axes=}")
    mask = array > 0
    if not mask.any():
        raise ValueError(f"Micro-SAM mask is empty: {mask_path}")
    return mask


def corrected_channel_path(analysis_dir: Path, channel: str) -> Path:
    suffix = "Nucleus" if channel == "nucleus" else channel
    matches = sorted(analysis_dir.glob(f"*_{suffix}.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_{suffix}.tif in {analysis_dir}, found {len(matches)}"
        )
    return matches[0]


def infer_alignment_channel(crop_tiff: Path) -> tuple[str, int]:
    with tifffile.TiffFile(crop_tiff) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
    channel_count = int(shape[axes.index("C")]) if "C" in axes else 1
    if channel_count == 4:
        return "nucleus", 0
    if channel_count == 3:
        return "green", 0
    raise ValueError(
        f"Expected a 3- or 4-channel crop for automatic alignment, found {channel_count}"
    )


def _normalized(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-12), 0, 1)


def align_mask(
    crop_tiff: Path,
    microsam_mask: Path,
    corrected_tiff: Path,
    *,
    raw_channel_index: int,
    dilation_px: int,
    output_dir: Path,
) -> dict:
    """Estimate Fiji translations and apply them to the saved static mask."""
    if dilation_px < 0:
        raise ValueError("dilation_px cannot be negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_crop_channel_tyx(crop_tiff, raw_channel_index)
    corrected = load_tyx(corrected_tiff).astype(np.float32, copy=False)
    static_mask = load_static_mask(microsam_mask)
    if raw.shape[0] != corrected.shape[0]:
        raise ValueError(f"Frame mismatch: raw={raw.shape}, corrected={corrected.shape}")
    if static_mask.shape != raw.shape[1:]:
        raise ValueError(f"Mask/crop mismatch: {static_mask.shape} vs {raw.shape[1:]}")
    if raw.shape[1] > corrected.shape[1] or raw.shape[2] > corrected.shape[2]:
        raise ValueError(f"Corrected canvas is smaller than raw: {corrected.shape} vs {raw.shape}")

    padded_raw = np.zeros(corrected.shape[1:], dtype=np.float32)
    padded_mask = np.zeros(corrected.shape[1:], dtype=np.uint8)
    padded_mask[: static_mask.shape[0], : static_mask.shape[1]] = static_mask
    aligned_raw = np.zeros(corrected.shape, dtype=bool)
    audit_rows = []
    for frame in range(raw.shape[0]):
        padded_raw.fill(0)
        padded_raw[: raw.shape[1], : raw.shape[2]] = raw[frame]
        shift, error, phase = phase_cross_correlation(
            corrected[frame],
            padded_raw,
            upsample_factor=20,
            normalization=None,
        )
        if not np.all(np.isfinite(shift)):
            raise RuntimeError(f"Non-finite drift estimate in frame {frame + 1}: {shift}")
        shifted_image = ndimage.shift(
            padded_raw,
            shift=shift,
            order=3,
            mode="constant",
            cval=0,
            prefilter=True,
        )
        aligned_raw[frame] = ndimage.shift(
            padded_mask,
            shift=shift,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ) > 0
        valid = (corrected[frame] > 0) & (shifted_image > 0)
        correlation = (
            float(np.corrcoef(corrected[frame][valid], shifted_image[valid])[0, 1])
            if valid.sum() > 2
            else float("nan")
        )
        audit_rows.append(
            {
                "frame": frame + 1,
                "shift_y_px": float(shift[0]),
                "shift_x_px": float(shift[1]),
                "registration_error": float(error),
                "phase_difference": float(phase),
                "registered_pixel_correlation": correlation,
            }
        )

    aligned_dilated = (
        aligned_raw.copy()
        if dilation_px == 0
        else np.stack(
            [ndimage.binary_dilation(mask, iterations=dilation_px) for mask in aligned_raw]
        )
    )
    raw_path = output_dir / "microsam_mask_aligned_raw.tif"
    dilated_path = output_dir / f"microsam_mask_aligned_dilated_{dilation_px}px.tif"
    tifffile.imwrite(raw_path, aligned_raw.astype(np.uint8) * 255, imagej=True, metadata={"axes": "TYX"})
    tifffile.imwrite(
        dilated_path,
        aligned_dilated.astype(np.uint8) * 255,
        imagej=True,
        metadata={"axes": "TYX"},
    )
    with (output_dir / "drift_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    snapshots = sorted({0, corrected.shape[0] // 2, corrected.shape[0] - 1})
    fig, axes_plot = plt.subplots(1, len(snapshots), figsize=(5 * len(snapshots), 4.5), squeeze=False)
    for axis, frame in zip(axes_plot[0], snapshots):
        axis.imshow(_normalized(corrected[frame]), cmap="gray")
        axis.contour(aligned_raw[frame], [0.5], colors=["#00A6D6"], linewidths=1.2)
        axis.contour(aligned_dilated[frame], [0.5], colors=["#E84A5F"], linewidths=1.0)
        axis.set_title(
            f"frame {frame + 1}\nshift x/y="
            f"{audit_rows[frame]['shift_x_px']:.2f}/"
            f"{audit_rows[frame]['shift_y_px']:.2f} px"
        )
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")
    fig.tight_layout()
    qc_path = output_dir / "microsam_mask_alignment_qc.png"
    fig.savefig(qc_path, dpi=220, facecolor="white")
    plt.close(fig)

    summary = {
        "bug_fix": "saved micro-SAM instance mask is drift-aligned and supplied to Stage 1 instead of being replaced by intensity Otsu",
        "crop_tiff": str(crop_tiff.resolve()),
        "microsam_mask_source": str(microsam_mask.resolve()),
        "corrected_alignment_tiff": str(corrected_tiff.resolve()),
        "raw_channel_index": raw_channel_index,
        "raw_shape_tyx": list(raw.shape),
        "corrected_shape_tyx": list(corrected.shape),
        "dilation_px": dilation_px,
        "aligned_raw_mask": str(raw_path),
        "aligned_dilated_mask": str(dilated_path),
        "shift_y_range_px": [float(min(row["shift_y_px"] for row in audit_rows)), float(max(row["shift_y_px"] for row in audit_rows))],
        "shift_x_range_px": [float(min(row["shift_x_px"] for row in audit_rows)), float(max(row["shift_x_px"] for row in audit_rows))],
        "registered_pixel_correlation_median": float(np.nanmedian([row["registered_pixel_correlation"] for row in audit_rows])),
        "aligned_mask_area_px_median": float(np.median(aligned_raw.sum(axis=(1, 2)))),
        "aligned_dilated_area_px_median": float(np.median(aligned_dilated.sum(axis=(1, 2)))),
        "qc_figure": str(qc_path),
    }
    (output_dir / "mask_alignment_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop_tiff", type=Path)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--microsam-mask", type=Path)
    parser.add_argument("--alignment-channel", choices=("auto", "nucleus", "green"), default="auto")
    parser.add_argument("--raw-channel-index", type=int)
    parser.add_argument("--dilation-px", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    crop_tiff = args.crop_tiff.resolve()
    analysis_dir = args.analysis_dir.resolve()
    mask = args.microsam_mask.resolve() if args.microsam_mask else discover_microsam_mask(crop_tiff)
    if args.alignment_channel == "auto":
        channel, inferred_index = infer_alignment_channel(crop_tiff)
    else:
        channel = args.alignment_channel
        inferred_index = 0 if channel in {"nucleus", "green"} else None
    raw_channel_index = args.raw_channel_index if args.raw_channel_index is not None else inferred_index
    if raw_channel_index is None:
        raise ValueError("--raw-channel-index is required for this alignment channel")
    corrected = corrected_channel_path(analysis_dir, channel)
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_dir / "anchor_roi_v4" / "mask_alignment"
    result = align_mask(
        crop_tiff,
        mask,
        corrected,
        raw_channel_index=raw_channel_index,
        dilation_px=args.dilation_px,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
