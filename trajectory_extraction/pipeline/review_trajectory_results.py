#!/usr/bin/env python3
"""Create compact, read-only QC artifacts for one completed trajectory run.

The script deliberately performs no tracking and never changes ``analysis_dir``.
It must be run *before* the temporary Stage-1 and Stage-2 products are removed,
because ``match_filter_summary.csv`` is reconstructed from those products.

Outputs are written to ``--output-dir``:

* ``match_filter_summary.csv`` and ``match_filter_avg_distance.png``: Stage-3
  one-to-one assignment and its 2,000 nm decision threshold;
* ``cleaned_trajectory_summary.csv``: basic frame and displacement statistics;
* ``cleaned_trajectories_absolute.png``: positions in whole-image nm;
* ``cleaned_trajectories_start_aligned.png``: motion with each track at (0, 0);
* ``cleaned_tracks_on_channel_mip.png``: each channel's maximum projection
  with its retained tracks overlaid in pixel coordinates;
* ``review_manifest.json``: calibration, fixed filter parameters, and counts.

All trajectory CSV positions are whole-image nanometres.  The MIP panel is
solely a visual check against the original channel signal; it does not change
or re-fit any localisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from summarize_match_filter_qc import (
    MAX_AVG_DIST_NM,
    MIN_OVERLAP_FRAMES,
    read_track,
    summarize_filtering,
)


CHANNELS = (("G", "green", "#159947"), ("P", "purple", "#7436a8"), ("R", "red", "#d63b3b"))


def rational_to_float(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def inspect_channel(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "TYX":
            raise ValueError(f"Expected TYX TIFF, got {series.axes} at {path}")
        x_resolution = rational_to_float(tif.pages[0].tags["XResolution"].value)
        y_resolution = rational_to_float(tif.pages[0].tags["YResolution"].value)
        imagej = dict(tif.imagej_metadata or {})
        return {
            "path": path.name,
            "shape_tyx": [int(value) for value in series.shape],
            "x_resolution_px_per_um": x_resolution,
            "y_resolution_px_per_um": y_resolution,
            "pixel_size_nm": 1000.0 / x_resolution,
            "frame_interval_s": imagej.get("finterval"),
        }


def find_channel_tiffs(analysis_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for _, name, _ in CHANNELS:
        paths = sorted(analysis_dir.glob(f"*_{name}.tif"))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *_{name}.tif in {analysis_dir}; found {len(paths)}."
            )
        result[name] = paths[0]
    return result


def validate_metadata(tiffs: dict[str, Path]) -> dict[str, dict]:
    metadata = {name: inspect_channel(path) for name, path in tiffs.items()}
    baseline = metadata["green"]
    for name, item in metadata.items():
        if item["shape_tyx"] != baseline["shape_tyx"]:
            raise ValueError(f"Channel shape mismatch: green={baseline}, {name}={item}")
        for key in ("x_resolution_px_per_um", "y_resolution_px_per_um"):
            if not math.isclose(item[key], baseline[key], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Channel calibration mismatch: green={baseline}, {name}={item}")
        if item["frame_interval_s"] != baseline["frame_interval_s"]:
            raise ValueError(f"Channel frame interval mismatch: green={baseline}, {name}={item}")
    return metadata


def cleaned_paths(analysis_dir: Path) -> list[Path]:
    paths = sorted(analysis_dir.glob("*_traj_m2DGaussian_cleaned.csv"))
    if not paths:
        raise FileNotFoundError(f"No cleaned trajectory CSVs found in {analysis_dir}")
    return paths


def trajectory_label(path: Path) -> str:
    return path.stem.removesuffix("_traj_m2DGaussian_cleaned")


def track_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track = read_track(path)
    frames = np.asarray(sorted(track), dtype=int)
    x = np.asarray([track[frame][0] for frame in frames], dtype=float)
    y = np.asarray([track[frame][1] for frame in frames], dtype=float)
    return frames, x, y


def write_trajectory_summary(paths: list[Path], output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        frames, x, y = track_arrays(path)
        displacement = np.hypot(x - x[0], y - y[0])
        rows.append(
            {
                "trajectory": trajectory_label(path),
                "channel": path.name[0],
                "points": int(len(frames)),
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "start_x_nm": f"{x[0]:.2f}",
                "start_y_nm": f"{y[0]:.2f}",
                "end_x_nm": f"{x[-1]:.2f}",
                "end_y_nm": f"{y[-1]:.2f}",
                "net_displacement_nm": f"{displacement[-1]:.2f}",
                "max_displacement_nm": f"{np.max(displacement):.2f}",
            }
        )
    with (output_dir / "cleaned_trajectory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def channel_color(code: str) -> str:
    return next(color for key, _, color in CHANNELS if key == code)


def plot_cleaned_trajectories(paths: list[Path], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    titles = ("Whole-image positions", "Start-aligned motion")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")

    for path in paths:
        _, x, y = track_arrays(path)
        color = channel_color(path.name[0])
        label = trajectory_label(path)
        axes[0].plot(x, y, color=color, linewidth=1.6, label=label)
        axes[0].scatter(x[0], y[0], color=color, s=20, marker="o", zorder=3)
        axes[0].scatter(x[-1], y[-1], color=color, s=27, marker="s", zorder=3)
        axes[1].plot(x - x[0], y - y[0], color=color, linewidth=1.6, label=label)
        axes[1].scatter(0, 0, color=color, s=20, marker="o", zorder=3)
        axes[1].scatter(x[-1] - x[0], y[-1] - y[0], color=color, s=27, marker="s", zorder=3)

    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Stage-3 cleaned m2DGaussian trajectories (circle=start, square=end)")
    fig.savefig(output_dir / "cleaned_trajectories_absolute_and_start_aligned.png", dpi=180)
    plt.close(fig)


def display_mip(stack: np.ndarray) -> np.ndarray:
    mip = np.max(stack, axis=0).astype(np.float32)
    low, high = np.percentile(mip, (1.0, 99.8))
    if high <= low:
        return np.zeros_like(mip)
    return np.clip((mip - low) / (high - low), 0.0, 1.0)


def plot_tracks_on_mips(paths: list[Path], tiffs: dict[str, Path], pixel_size_nm: float, output_dir: Path) -> None:
    paths_by_channel = {code: [path for path in paths if path.name.startswith(f"{code}_")] for code, _, _ in CHANNELS}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, (code, name, color) in zip(axes, CHANNELS):
        stack = tifffile.imread(tiffs[name])
        axis.imshow(display_mip(stack), cmap="gray", origin="upper")
        for path in paths_by_channel[code]:
            _, x_nm, y_nm = track_arrays(path)
            x_px, y_px = x_nm / pixel_size_nm, y_nm / pixel_size_nm
            axis.plot(x_px, y_px, color=color, linewidth=1.8, label=trajectory_label(path))
            axis.scatter(x_px[0], y_px[0], color=color, edgecolor="white", linewidth=0.35, s=25, marker="o", zorder=3)
            axis.scatter(x_px[-1], y_px[-1], color=color, edgecolor="white", linewidth=0.35, s=30, marker="s", zorder=3)
        axis.set_title(f"{name} max projection")
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")
        if paths_by_channel[code]:
            axis.legend(fontsize=8, loc="best")
    fig.suptitle("Retained tracks on channel MIPs (circle=start, square=end)")
    fig.savefig(output_dir / "cleaned_tracks_on_channel_mip.png", dpi=180)
    plt.close(fig)


def plot_match_filter(rows: list[dict], output_dir: Path) -> None:
    labels = [f"{row['channel']}_{row['locus']}" for row in rows]
    values = [float(row["avg_dist_nm"]) if row["avg_dist_nm"] else np.nan for row in rows]
    colors = ["#2a9d8f" if row["status"] == "saved" else "#d1495b" for row in rows]
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    axis.bar(labels, values, color=colors)
    axis.axhline(MAX_AVG_DIST_NM, color="#202020", linestyle="--", linewidth=1.2, label=f"{MAX_AVG_DIST_NM:.0f} nm filter")
    axis.set_xlabel("Stage-1 reference locus")
    axis.set_ylabel("Average distance to assigned Stage-2 candidate (nm)")
    axis.set_title("Stage-3 matching decision (green=saved; red=not retained)")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    fig.savefig(output_dir / "match_filter_avg_distance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review artifacts for one completed trajectory analysis directory.")
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not analysis_dir.is_dir():
        raise NotADirectoryError(analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiffs = find_channel_tiffs(analysis_dir)
    metadata = validate_metadata(tiffs)
    paths = cleaned_paths(analysis_dir)
    filter_rows = summarize_filtering(analysis_dir, output_dir)
    summary_rows = write_trajectory_summary(paths, output_dir)
    plot_match_filter(filter_rows, output_dir)
    plot_cleaned_trajectories(paths, output_dir)
    plot_tracks_on_mips(paths, tiffs, metadata["green"]["pixel_size_nm"], output_dir)

    manifest = {
        "status": "PASS",
        "coordinate_system": "trajectory CSVs are whole-image nanometres; MIP overlay converts nm to px using TIFF calibration",
        "stage3_parameters": {
            "minimum_overlap_frames": MIN_OVERLAP_FRAMES,
            "maximum_average_distance_nm": MAX_AVG_DIST_NM,
            "red_candidate_must_include_frame_1_to_3": True,
        },
        "input_channel_metadata": metadata,
        "reference_counts": {code: len(list(analysis_dir.glob(f"{code}_loci*_traj_rela2wholeimg.csv"))) for code, _, _ in CHANNELS},
        "candidate_counts": {code: len(list((analysis_dir / "matlab_result" / "matlab_trajectory").glob(f"{code}_m2DGaussian_traj*.csv"))) for code, _, _ in CHANNELS},
        "cleaned_counts": {code: len([path for path in paths if path.name.startswith(f"{code}_")]) for code, _, _ in CHANNELS},
        "cleaned_trajectory_count": len(summary_rows),
        "artifacts": [
            "match_filter_summary.csv",
            "match_filter_avg_distance.png",
            "cleaned_trajectory_summary.csv",
            "cleaned_trajectories_absolute_and_start_aligned.png",
            "cleaned_tracks_on_channel_mip.png",
        ],
    }
    (output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote trajectory review artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
