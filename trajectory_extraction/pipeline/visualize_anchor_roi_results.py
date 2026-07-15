#!/usr/bin/env python3
"""Create reproducible Python spatial QC for v4 anchor-ROI SPT outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import LinearSegmentedColormap

import max_step_model
from run_anchor_roi_spt import PREFIX, locate_channel_tiffs, read_candidate_nm, read_track_px


CHANNEL_COLORS = {"G": "#00A65A", "R": "#F4B400", "P": "#7A3DB8"}
CHANNEL_NAMES = {"G": "53BP1 (Green)", "R": "Site 1 (Yellow)", "P": "Site 2 (Purple)"}


def read_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def time_cmap(prefix: str) -> LinearSegmentedColormap:
    color = CHANNEL_COLORS[prefix]
    return LinearSegmentedColormap.from_list(f"{prefix}_time", ["#E8E8E8", color])


def normalized(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99.7))
    return np.clip((image - low) / max(high - low, 1e-12), 0, 1)


def load_projection(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        array = np.asarray(tif.series[0].asarray())
        axes = tif.series[0].axes
    for axis_name in tuple(axes):
        if axis_name not in "TYX":
            index = axes.index(axis_name)
            if array.shape[index] != 1:
                raise ValueError(f"Unsupported axis {axis_name} in {path}: {axes} {array.shape}")
            array = np.take(array, 0, axis=index)
            axes = axes.replace(axis_name, "")
    if "T" in axes:
        array = np.max(array, axis=axes.index("T"))
        axes = axes.replace("T", "")
    if axes != "YX":
        array = np.transpose(array, [axes.index(axis) for axis in "YX"])
    return normalized(array.astype(float, copy=False))


def load_mask_2d(path: Path) -> np.ndarray:
    array = tifffile.imread(path) > 0
    while array.ndim > 2:
        array = array[0]
    return array


def plot_gradient_line(axis, x, y, frames, *, cmap, linewidth=1.4, alpha=1.0, linestyle="-"):
    if len(x) == 1:
        axis.plot(x, y, color=cmap(1.0), marker=".", markersize=2, alpha=alpha)
        return
    first, last = float(np.min(frames)), float(np.max(frames))
    denominator = max(last - first, 1.0)
    for index in range(len(x) - 1):
        fraction = ((frames[index] + frames[index + 1]) / 2.0 - first) / denominator
        axis.plot(
            x[index : index + 2],
            y[index : index + 2],
            color=cmap(fraction),
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
        )


def spatial_overview(
    output: Path,
    roi_rows: list[dict],
    baseline_rows: list[dict],
    projection: np.ndarray,
    aligned_mask: np.ndarray,
    pixel_size_nm: float,
) -> None:
    if not roi_rows:
        return
    fig, axes = plt.subplots(
        len(roi_rows), 2, figsize=(11, max(4.5, 4.2 * len(roi_rows))), squeeze=False
    )
    for row_index, roi_row in enumerate(roi_rows):
        allele = int(roi_row["allele_index"])
        roi = load_mask_2d(Path(roi_row["roi_tiff"]))
        anchor = read_track_px(Path(roi_row["anchor_csv"]), pixel_size_nm)
        anchor_frames = np.asarray([point[0] for point in anchor])
        anchor_x = np.asarray([point[1] for point in anchor])
        anchor_y = np.asarray([point[2] for point in anchor])

        left = axes[row_index, 0]
        left.imshow(projection, cmap="gray", origin="upper")
        left.contour(aligned_mask, [0.5], colors=["#00A6D6"], linewidths=0.8)
        left.contour(roi, [0.5], colors=["#FF4E50"], linewidths=1.3)
        plot_gradient_line(
            left, anchor_x, anchor_y, anchor_frames, cmap=plt.get_cmap("Greys"),
            linewidth=1.6, linestyle="--"
        )
        left.set_title(f"Allele {allele}: anchor + static irregular ROI")
        left.set_xlabel("corrected x (px)")
        left.set_ylabel("corrected y (px)")

        right = axes[row_index, 1]
        right.imshow(projection, cmap="gray", origin="upper", alpha=0.35)
        right.contour(roi, [0.5], colors=["#555555"], linewidths=0.8)
        plot_gradient_line(
            right, anchor_x, anchor_y, anchor_frames, cmap=plt.get_cmap("Greys"),
            linewidth=1.2, linestyle="--"
        )
        for baseline in baseline_rows:
            if int(baseline["allele_index"]) != allele:
                continue
            prefix = baseline["channel"]
            points = read_candidate_nm(Path(baseline["baseline_csv"]))
            frames = np.asarray([point[0] for point in points])
            x = np.asarray([point[1] for point in points]) / pixel_size_nm - 1.0
            y = np.asarray([point[2] for point in points]) / pixel_size_nm - 1.0
            plot_gradient_line(right, x, y, frames, cmap=time_cmap(prefix), linewidth=1.8)
        right.set_title(f"Allele {allele}: longest SPT baselines")
        right.set_xlabel("corrected x (px)")
        right.set_ylabel("corrected y (px)")
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def all_candidates_figure(
    output: Path,
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    projections: dict[str, np.ndarray],
    pixel_size_nm: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    channel_by_prefix = {prefix: channel for channel, prefix in PREFIX.items()}
    selected_paths = {Path(row["candidate_csv"]).resolve() for row in baseline_rows}
    for axis, prefix in zip(axes, ("G", "R", "P")):
        axis.imshow(projections[channel_by_prefix[prefix]], cmap="gray", origin="upper", alpha=0.32)
        rows = [row for row in candidate_rows if row["channel"] == prefix]
        for row in rows:
            points = read_candidate_nm(Path(row["candidate_csv"]))
            frames = np.asarray([point[0] for point in points])
            x = np.asarray([point[1] for point in points]) / pixel_size_nm - 1.0
            y = np.asarray([point[2] for point in points]) / pixel_size_nm - 1.0
            selected = Path(row["candidate_csv"]).resolve() in selected_paths
            plot_gradient_line(
                axis,
                x,
                y,
                frames,
                cmap=time_cmap(prefix),
                linewidth=2.0 if selected else 0.65,
                alpha=1.0 if selected else 0.30,
            )
        axis.set_title(f"{CHANNEL_NAMES[prefix]}\n{len(rows)} candidates; longest emphasized")
        axis.set_xlabel("corrected x (px)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("corrected y (px)")
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def length_figure(output: Path, candidate_rows: list[dict], baseline_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(3)
    counts = [sum(row["channel"] == prefix for row in candidate_rows) for prefix in ("G", "R", "P")]
    axes[0].bar(positions, counts, color=[CHANNEL_COLORS[prefix] for prefix in ("G", "R", "P")])
    axes[0].set_xticks(positions, ["53BP1", "Site 1", "Site 2"])
    axes[0].set_ylabel("candidate trajectories")
    axes[0].set_title("SPT yield inside anchor-defined ROIs")
    for index, value in enumerate(counts):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    for prefix in ("G", "R", "P"):
        lengths = [int(row["points"]) for row in candidate_rows if row["channel"] == prefix]
        if lengths:
            axes[1].hist(lengths, bins=min(20, max(5, len(set(lengths)))), histtype="step", linewidth=1.8,
                         color=CHANNEL_COLORS[prefix], label=CHANNEL_NAMES[prefix])
        selected = [int(row["points"]) for row in baseline_rows if row["channel"] == prefix]
        for length in selected:
            axes[1].axvline(length, color=CHANNEL_COLORS[prefix], linewidth=1.2, alpha=0.7)
    axes[1].set_xlabel("trajectory length (points)")
    axes[1].set_ylabel("candidate count")
    axes[1].set_title("Candidate lengths; selected longest shown by vertical lines")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--aligned-microsam-mask", type=Path)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else analysis_dir / "anchor_roi_v4"
    output_dir = args.output_dir.resolve() if args.output_dir else results_dir / "figures" / "python_spatial_qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    audit_dir = results_dir / "audit"
    roi_rows = read_rows(audit_dir / "static_anchor_roi_geometry.csv")
    candidate_rows = read_rows(audit_dir / "all_candidate_trajectories.csv")
    baseline_rows = read_rows(results_dir / "baseline_longest" / "baseline_manifest.csv")
    tiffs = locate_channel_tiffs(analysis_dir)
    metadata = max_step_model.read_tracking_metadata(tiffs["green"])
    pixel_size_nm = float(metadata["pixel_size_nm_per_px"])
    projections = {channel: load_projection(path) for channel, path in tiffs.items()}
    summary = json.loads(
        (results_dir / "anchor_roi_v4_summary.json").read_text(encoding="utf-8")
    )
    anchor_channel = summary["anchor_channel"]
    if args.aligned_microsam_mask:
        aligned_mask_path = args.aligned_microsam_mask.resolve()
    else:
        mask_matches = sorted(
            (results_dir / "mask_alignment").glob("microsam_mask_aligned_dilated_*px.tif")
        )
        if len(mask_matches) != 1:
            raise FileNotFoundError(
                "Expected one aligned dilated micro-SAM mask under results/mask_alignment; "
                "pass --aligned-microsam-mask explicitly."
            )
        aligned_mask_path = mask_matches[0]
    aligned_mask = load_mask_2d(aligned_mask_path)

    outputs = {
        "anchor_mask_roi_overview": output_dir / "01_anchor_mask_roi_overview.png",
        "all_candidates_fixed_coordinates": output_dir / "02_all_candidates_fixed_coordinates.png",
        "candidate_length_and_baseline": output_dir / "03_candidate_length_and_baseline.png",
    }
    spatial_overview(outputs["anchor_mask_roi_overview"], roi_rows, baseline_rows, projections[anchor_channel], aligned_mask, pixel_size_nm)
    all_candidates_figure(outputs["all_candidates_fixed_coordinates"], candidate_rows, baseline_rows, projections, pixel_size_nm)
    length_figure(outputs["candidate_length_and_baseline"], candidate_rows, baseline_rows)
    manifest = {
        "purpose": "read-only QC; these figures do not change candidate generation or baseline selection",
        "source_tables": {
            "roi": str((audit_dir / "static_anchor_roi_geometry.csv").resolve()),
            "candidates": str((audit_dir / "all_candidate_trajectories.csv").resolve()),
            "baselines": str((results_dir / "baseline_longest" / "baseline_manifest.csv").resolve()),
        },
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
