#!/usr/bin/env python3
"""Run v4 static anchor-ROI SPT and select a deterministic longest baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile
from scipy import ndimage
from skimage.draw import line

import max_step_model
import experiment_profiles


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


VERSION = "v4.1.4-experiment-profiles"
HERE = Path(__file__).resolve().parent
MATLAB_DEPS = HERE / "matlab_deps"
CHANNELS = ("green", "red", "purple")
PREFIX = {"green": "G", "red": "R", "purple": "P"}
GAUSSIAN_FIT_BOX_SIZE_PX = 9
MIN_ROI_DILATION_PX = math.ceil(GAUSSIAN_FIT_BOX_SIZE_PX / 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def locate_channel_tiffs(analysis_dir: Path) -> dict[str, Path]:
    green = sorted(analysis_dir.glob("*_green.tif"))
    if len(green) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_green.tif in {analysis_dir}, found {len(green)}"
        )
    stem = green[0].name[: -len("_green.tif")]
    paths = {channel: analysis_dir / f"{stem}_{channel}.tif" for channel in CHANNELS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing channel TIFF(s): " + ", ".join(missing))
    return paths


def read_track_px(path: Path, pixel_size_nm: float) -> list[tuple[int, float, float]]:
    points = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            points.append(
                (
                    int(float(row["frame"])),
                    float(row["x_nm"]) / pixel_size_nm - 1.0,
                    float(row["y_nm"]) / pixel_size_nm - 1.0,
                )
            )
    return sorted(points)


def read_candidate_nm(path: Path) -> list[tuple[int, float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            (int(float(row["frame"])), float(row["x_nm"]), float(row["y_nm"]))
            for row in csv.DictReader(handle)
        ]


def locus_number(path: Path) -> int:
    match = re.search(r"loci(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse locus number: {path}")
    return int(match.group(1))


def static_anchor_union(
    anchor: list[tuple[int, float, float]],
    nucleus_support_2d: np.ndarray,
    dilation_px: int,
) -> np.ndarray:
    """Rasterize the complete anchor path into one static connected ROI."""
    if dilation_px < MIN_ROI_DILATION_PX:
        raise ValueError(
            f"ROI dilation must be >= {MIN_ROI_DILATION_PX} px so the "
            f"{GAUSSIAN_FIT_BOX_SIZE_PX}px-wide Gaussian fit box is not under-supported"
        )
    centerline = np.zeros_like(nucleus_support_2d, dtype=bool)
    points = [(int(round(y)), int(round(x))) for _frame, x, y in anchor]
    for y, x in points:
        if 0 <= y < centerline.shape[0] and 0 <= x < centerline.shape[1]:
            centerline[y, x] = True
    for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
        rows, columns = line(y0, x0, y1, x1)
        valid = (
            (rows >= 0)
            & (rows < centerline.shape[0])
            & (columns >= 0)
            & (columns < centerline.shape[1])
        )
        centerline[rows[valid], columns[valid]] = True
    return ndimage.binary_dilation(centerline, iterations=dilation_px) & nucleus_support_2d


def partition_channels(worker_count: int) -> list[list[str]]:
    groups = [[] for _ in range(min(worker_count, len(CHANNELS)))]
    for index, channel in enumerate(CHANNELS):
        groups[index % len(groups)].append(channel)
    return groups


def matlab_group_command(
    channels: list[str],
    tiffs: dict[str, Path],
    roi_path: Path,
    frame_rate: float,
    pixel_um: float,
    output_dir: Path,
    save_filter_images: bool,
    max_step_px: float,
) -> str:
    keep = "true" if save_filter_images else "false"
    calls = "; ".join(
        f"spt_batch_anchor_roi('{matlab_quote(tiffs[channel])}', "
        f"'{matlab_quote(roi_path)}', {frame_rate:.17g}, {pixel_um:.17g}, "
        f"'{matlab_quote(output_dir)}', {keep}, {max_step_px:.17g})"
        for channel in channels
    )
    return (
        f"addpath('{matlab_quote(MATLAB_DEPS)}','-begin'); "
        f"addpath('{matlab_quote(HERE)}','-begin'); {calls}"
    )


def run_matlab_group(
    channels: list[str],
    *,
    matlab_bin: str,
    **kwargs,
) -> tuple[list[str], int, str, float]:
    started = time.perf_counter()
    command = matlab_group_command(channels, **kwargs)
    result = subprocess.run(
        [matlab_bin, "-batch", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = "\n".join(
        part for part in (result.stdout.rstrip(), result.stderr.rstrip()) if part
    )
    return channels, result.returncode, output, time.perf_counter() - started


def run_locus_spt(
    *,
    tiffs: dict[str, Path],
    roi_path: Path,
    frame_rate: float,
    pixel_um: float,
    output_dir: Path,
    matlab_bin: str,
    matlab_workers: int,
    save_filter_images: bool,
    max_step_px: float,
    log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = partition_channels(matlab_workers)
    common = {
        "tiffs": tiffs,
        "roi_path": roi_path,
        "frame_rate": frame_rate,
        "pixel_um": pixel_um,
        "output_dir": output_dir,
        "save_filter_images": save_filter_images,
        "max_step_px": max_step_px,
    }
    results = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        pending = {
            executor.submit(
                run_matlab_group,
                group,
                matlab_bin=matlab_bin,
                **common,
            )
            for group in groups
        }
        while pending:
            completed, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not completed:
                print(
                    f"  MATLAB still running: {len(pending)} process(es), "
                    f"{time.perf_counter() - started:.0f} s elapsed",
                    flush=True,
                )
                continue
            for future in completed:
                results.append(future.result())

    failures = []
    with log_path.open("w", encoding="utf-8") as log:
        for channels, returncode, output, elapsed in results:
            label = ",".join(channels)
            block = f"--- MATLAB {label} ({elapsed:.1f} s) ---\n{output}\n"
            print(block)
            log.write(block)
            if returncode:
                failures.append((label, returncode))
    if failures:
        raise RuntimeError(
            "MATLAB SPT failed: "
            + ", ".join(f"{label}=exit {code}" for label, code in failures)
        )


def export_candidates(tiffs: dict[str, Path], matlab_dir: Path) -> list[Path]:
    csv_dir = matlab_dir / "matlab_trajectory"
    csv_dir.mkdir(exist_ok=True)
    for prefix in PREFIX.values():
        for stale in csv_dir.glob(f"{prefix}_m2DGaussian_traj*.csv"):
            stale.unlink()

    outputs = []
    for channel, tiff in tiffs.items():
        mat_path = matlab_dir / f"{tiff.stem}.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        mat = sio.loadmat(str(mat_path))
        trajectory_struct = mat["traj"]
        pixel_nm = float(mat["sptpara"][0, 0]["pixl"].flat[0]) * 1000.0
        exported = 0
        for item in trajectory_struct.flat:
            try:
                positions = item["pos"]
            except (IndexError, TypeError, ValueError):
                continue
            if positions.shape == (1, 1):
                positions = positions[0, 0]
            if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] < 3:
                continue
            exported += 1
            path = csv_dir / f"{PREFIX[channel]}_m2DGaussian_traj{exported}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame", "x_nm", "y_nm"])
                for x, y, frame in positions[:, :3]:
                    writer.writerow([int(frame), f"{x * pixel_nm:.2f}", f"{y * pixel_nm:.2f}"])
            outputs.append(path)
    return outputs


def candidate_audit(
    path: Path,
    *,
    allele_index: int,
    anchor_locus: int,
    channel: str,
    roi: np.ndarray,
    pixel_size_nm: float,
    profile: experiment_profiles.ExperimentProfile,
) -> dict:
    points = read_candidate_nm(path)
    frames = np.asarray([point[0] for point in points], dtype=int)
    xy_nm = np.asarray([[point[1], point[2]] for point in points], dtype=float)
    xy_px = xy_nm / pixel_size_nm - 1.0
    frame_diffs = np.diff(frames)
    step_px = np.linalg.norm(np.diff(xy_px, axis=0), axis=1) if len(points) > 1 else np.array([])
    inside = []
    for x, y in xy_px:
        xi, yi = int(round(x)), int(round(y))
        inside.append(0 <= yi < roi.shape[0] and 0 <= xi < roi.shape[1] and bool(roi[yi, xi]))
    span = int(frames[-1] - frames[0] + 1)
    trajectory_match = re.search(r"traj(\d+)", path.stem)
    channel_spec = profile.channel_from_prefix(channel)
    return {
        "experiment_profile": profile.name,
        "allele_index": allele_index,
        "anchor_locus": anchor_locus,
        "channel": channel,
        "corrected_channel": channel_spec.corrected_channel,
        "raw_channel_index": channel_spec.raw_index,
        "marker": channel_spec.marker,
        "marker_slug": channel_spec.marker_slug,
        "site_id": channel_spec.site_id,
        "genomic_locus": channel_spec.genomic_locus,
        "fluorophore": channel_spec.fluorophore,
        "candidate_number": int(trajectory_match.group(1)) if trajectory_match else 0,
        "candidate_csv": str(path.resolve()),
        "points": len(points),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "frame_span": span,
        "temporal_coverage_fraction": len(points) / span,
        "maximum_missing_frames_between_points": int(np.max(frame_diffs - 1)) if len(frame_diffs) else 0,
        "median_step_px": float(np.median(step_px)) if len(step_px) else 0.0,
        "p95_step_px": float(np.percentile(step_px, 95)) if len(step_px) else 0.0,
        "inside_static_roi_fraction": float(np.mean(inside)),
    }


def longest_sort_key(row: dict) -> tuple:
    """Deterministic baseline: points, span, earliest start, candidate number."""
    return (-row["points"], -row["frame_span"], row["first_frame"], row["candidate_number"])


def select_longest_baselines(
    candidate_rows: list[dict],
    baseline_dir: Path,
    allele_anchors: list[tuple[int, int]],
    profile: experiment_profiles.ExperimentProfile,
) -> tuple[list[dict], list[dict]]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for stale in baseline_dir.glob("*.csv"):
        stale.unlink()
    selected_rows = []
    audit_rows = []
    for allele_index, anchor_locus in allele_anchors:
        for channel in PREFIX.values():
            channel_spec = profile.channel_from_prefix(channel)
            pool = [
                row
                for row in candidate_rows
                if row["allele_index"] == allele_index and row["channel"] == channel
            ]
            if not pool:
                audit_rows.append(
                    {
                        "experiment_profile": profile.name,
                        "allele_index": allele_index,
                        "anchor_locus": anchor_locus,
                        "channel": channel,
                        "corrected_channel": channel_spec.corrected_channel,
                        "raw_channel_index": channel_spec.raw_index,
                        "marker": channel_spec.marker,
                        "marker_slug": channel_spec.marker_slug,
                        "site_id": channel_spec.site_id,
                        "genomic_locus": channel_spec.genomic_locus,
                        "fluorophore": channel_spec.fluorophore,
                        "candidate_count": 0,
                        "selection_status": "no candidate; no baseline output",
                        "selected_candidate_csv": "",
                        "selected_points": 0,
                        "selected_frame_span": 0,
                        "baseline_csv": "",
                    }
                )
                continue
            selected = sorted(pool, key=longest_sort_key)[0]
            destination = baseline_dir / (
                f"allele_{allele_index:03d}_{channel_spec.marker_slug}_longest_spt_cleaned.csv"
            )
            shutil.copy2(selected["candidate_csv"], destination)
            baseline = {
                **selected,
                "selection_rule": "maximum points; tie: maximum frame span, earliest first frame, lowest candidate number",
                "cleaned_definition": "automatic ROI-restricted SPT baseline selected by the declared longest-track rule; no reference-distance matching",
                "baseline_csv": str(destination.resolve()),
            }
            selected_rows.append(baseline)
            audit_rows.append(
                {
                    "experiment_profile": profile.name,
                    "allele_index": allele_index,
                    "anchor_locus": anchor_locus,
                    "channel": channel,
                    "corrected_channel": channel_spec.corrected_channel,
                    "raw_channel_index": channel_spec.raw_index,
                    "marker": channel_spec.marker,
                    "marker_slug": channel_spec.marker_slug,
                    "site_id": channel_spec.site_id,
                    "genomic_locus": channel_spec.genomic_locus,
                    "fluorophore": channel_spec.fluorophore,
                    "candidate_count": len(pool),
                    "selection_status": "selected longest baseline",
                    "selected_candidate_csv": selected["candidate_csv"],
                    "selected_points": selected["points"],
                    "selected_frame_span": selected["frame_span"],
                    "baseline_csv": str(destination.resolve()),
                }
            )
    return selected_rows, audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--aligned-microsam-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--experiment-profile",
        choices=experiment_profiles.profile_choices(),
        required=True,
    )
    parser.add_argument("--roi-dilation-px", type=int, default=5)
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument("--matlab-workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--matlab-save-filter-images", action="store_true")
    parser.add_argument("--d-star", type=float, default=4.1e-3)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--coverage-probability", type=float, default=0.995)
    parser.add_argument("--localization-error-nm", type=float, default=0.0)
    parser.add_argument("--max-step-frame-gap", type=int, default=1)
    parser.add_argument("--max-step-rounding-px", type=float, default=0.05)
    parser.add_argument("--max-step-px", type=float, help="Explicit override; otherwise use the metadata/physical model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = experiment_profiles.get_profile(args.experiment_profile)
    anchor_channel = profile.anchor_channel
    analysis_dir = args.analysis_dir.resolve()
    anchor_dir = args.anchor_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else analysis_dir / "anchor_roi_v4"
    roi_dir = output_dir / "static_union_rois"
    spt_dir = output_dir / "roi_spt"
    baseline_dir = output_dir / "baseline_longest"
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (roi_dir, spt_dir, baseline_dir, audit_dir):
        if directory.exists():
            if directory.resolve().parent != output_dir:
                raise RuntimeError(f"Refusing to clean output outside {output_dir}: {directory}")
            shutil.rmtree(directory)
    for directory in (roi_dir, spt_dir, baseline_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.roi_dilation_px < MIN_ROI_DILATION_PX:
        raise ValueError(f"--roi-dilation-px must be >= {MIN_ROI_DILATION_PX}")
    tiffs = locate_channel_tiffs(analysis_dir)
    metadata_by_channel = max_step_model.validate_channel_metadata(tiffs)
    metadata = metadata_by_channel["green"]
    derivation = max_step_model.derive_from_metadata(
        metadata,
        diffusion_coefficient_um2_per_s_alpha=args.d_star,
        anomalous_exponent=args.alpha,
        coverage_probability=args.coverage_probability,
        localization_error_nm=args.localization_error_nm,
        frame_gap=args.max_step_frame_gap,
        rounding_increment_px=args.max_step_rounding_px,
        track_mem=3,
        explicit_max_step_px=args.max_step_px,
    )
    derivation["metadata_consistency_across_channel_tiffs"] = {
        "verified": True,
        "metadata_by_channel": metadata_by_channel,
    }
    (audit_dir / "max_step_model.json").write_text(
        json.dumps(derivation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    max_step_px = float(derivation["operational_max_step_px"])
    pixel_size_nm = float(metadata["pixel_size_nm_per_px"])

    aligned_mask = tifffile.imread(args.aligned_microsam_mask.resolve()) > 0
    if aligned_mask.ndim != 3 or aligned_mask.shape[0] != metadata["frame_count"]:
        raise ValueError(
            f"Expected aligned TYX mask with {metadata['frame_count']} frames, "
            f"found {aligned_mask.shape}"
        )
    with tifffile.TiffFile(tiffs["green"]) as tif:
        green_series = tif.series[0]
        yx_shape = tuple(
            int(green_series.shape[green_series.axes.index(axis)]) for axis in "YX"
        )
    if aligned_mask.shape[1:] != yx_shape:
        raise ValueError(
            f"Aligned mask YX shape {aligned_mask.shape[1:]} does not match "
            f"corrected channel TIFF YX shape {yx_shape}"
        )
    prefix = PREFIX[anchor_channel]
    anchor_paths = sorted(anchor_dir.glob(f"{prefix}_loci*_traj_rela2wholeimg.csv"), key=locus_number)
    if not anchor_paths:
        raise RuntimeError(f"No {prefix} anchor trajectories found in {anchor_dir}")

    candidate_rows = []
    roi_rows = []
    for allele_index, anchor_path in enumerate(anchor_paths, start=1):
        anchor_locus = locus_number(anchor_path)
        anchor = read_track_px(anchor_path, pixel_size_nm)
        roi = static_anchor_union(anchor, aligned_mask[0], args.roi_dilation_px)
        components = int(ndimage.label(roi)[1])
        if components != 1:
            raise RuntimeError(
                f"Allele {allele_index}/loci{anchor_locus} ROI has {components} components"
            )
        roi_path = roi_dir / f"allele_{allele_index:03d}_loci{anchor_locus}_static_anchor_roi.tif"
        tifffile.imwrite(roi_path, roi.astype(np.uint8) * 255)
        rows, columns = np.where(roi)
        roi_rows.append(
            {
                "allele_index": allele_index,
                "anchor_locus": anchor_locus,
                "experiment_profile": profile.name,
                "anchor_channel": anchor_channel,
                "anchor_raw_channel_index": profile.anchor.raw_index,
                "anchor_marker": profile.anchor.marker,
                "anchor_site_id": profile.anchor.site_id,
                "anchor_genomic_locus": profile.anchor.genomic_locus,
                "anchor_fluorophore": profile.anchor.fluorophore,
                "anchor_csv": str(anchor_path.resolve()),
                "anchor_points": len(anchor),
                "roi_dilation_px": args.roi_dilation_px,
                "gaussian_fit_box_size_px": GAUSSIAN_FIT_BOX_SIZE_PX,
                "nucleus_support": "frame 1 of drift-aligned dilated micro-SAM mask",
                "roi_area_px": int(roi.sum()),
                "connected_components": components,
                "bbox_top": int(rows.min()),
                "bbox_left": int(columns.min()),
                "bbox_bottom_exclusive": int(rows.max() + 1),
                "bbox_right_exclusive": int(columns.max() + 1),
                "roi_tiff": str(roi_path.resolve()),
            }
        )

        locus_output = spt_dir / f"allele_{allele_index:03d}_loci{anchor_locus}"
        matlab_output = locus_output / "matlab_result"
        print(
            f"Allele {allele_index} (anchor loci{anchor_locus}): ROI={int(roi.sum())} px; "
            f"SPT max_step={max_step_px:.3f} px",
            flush=True,
        )
        run_locus_spt(
            tiffs=tiffs,
            roi_path=roi_path,
            frame_rate=metadata["frame_rate_hz"],
            pixel_um=metadata["pixel_size_x_um_per_px"],
            output_dir=matlab_output,
            matlab_bin=args.matlab_bin,
            matlab_workers=args.matlab_workers,
            save_filter_images=args.matlab_save_filter_images,
            max_step_px=max_step_px,
            log_path=locus_output / "matlab_spt.log",
        )
        candidates = export_candidates(tiffs, matlab_output)
        for path in candidates:
            candidate_rows.append(
                candidate_audit(
                    path,
                    allele_index=allele_index,
                    anchor_locus=anchor_locus,
                    channel=path.name[0],
                    roi=roi,
                    pixel_size_nm=pixel_size_nm,
                    profile=profile,
                )
            )

    allele_anchors = [
        (row["allele_index"], row["anchor_locus"]) for row in roi_rows
    ]
    selected_rows, selection_audit = select_longest_baselines(
        candidate_rows, baseline_dir, allele_anchors, profile
    )
    for row in selected_rows:
        row["frame_interval_s"] = metadata["frame_interval_s"]
        row["movie_frame_count"] = metadata["frame_count"]
        row["pixel_size_nm_per_px"] = pixel_size_nm
    write_csv(audit_dir / "static_anchor_roi_geometry.csv", roi_rows)
    write_csv(audit_dir / "all_candidate_trajectories.csv", candidate_rows)
    write_csv(audit_dir / "baseline_selection.csv", selection_audit)
    write_csv(baseline_dir / "baseline_manifest.csv", selected_rows)

    summary = {
        "version": VERSION,
        "experiment_profile": profile.name,
        "profile_contract": profile.to_manifest(),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "anchor_channel": anchor_channel,
        "anchor_raw_channel_index": profile.anchor.raw_index,
        "anchor_marker": profile.anchor.marker,
        "anchor_count": len(anchor_paths),
        "static_irregular_roi_rule": "complete anchor path connected, dilated, intersected with frame-1 aligned micro-SAM support, reused for every frame",
        "roi_dilation_px": args.roi_dilation_px,
        "candidate_matching_to_reference_used": False,
        "baseline_rule": "longest candidate independently within each allele/channel; no cleaned/manual input",
        "candidate_count": len(candidate_rows),
        "selected_baseline_count": len(selected_rows),
        "max_step_model_audit": str((audit_dir / "max_step_model.json").resolve()),
        "operational_max_step_px": max_step_px,
        "outputs": {
            "roi_geometry": str((audit_dir / "static_anchor_roi_geometry.csv").resolve()),
            "all_candidates": str((audit_dir / "all_candidate_trajectories.csv").resolve()),
            "baseline_selection": str((audit_dir / "baseline_selection.csv").resolve()),
            "baseline_manifest": str((baseline_dir / "baseline_manifest.csv").resolve()),
        },
    }
    (output_dir / "anchor_roi_v4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
