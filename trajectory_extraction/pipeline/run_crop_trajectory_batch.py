#!/usr/bin/env python3
"""Run trajectory extraction on a cell-crop folder one cell at a time.

The batch runner is designed for overnight use on existing cell-crop outputs.
It prepares the four single-channel TIFFs expected by the trajectory pipeline,
runs the three trajectory stages, writes compact per-cell summaries, and removes
large intermediate files after each cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from summarize_match_filter_qc import summarize_filtering


HERE = Path(__file__).parent.resolve()
STAGE1 = HERE / "auto_roi_for_published_v2.13.py"
STAGE2 = HERE / "run_pipeline_v3.py"
STAGE3 = HERE / "match_m2DGaussian_to_reference.py"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path.anchor or path))
    return usage.free / (1024**3)


def read_mapping(crop_dir: Path) -> list[dict]:
    mapping_path = crop_dir / "cell_id_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"cell_id_mapping.csv not found: {mapping_path}")
    with mapping_path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_crop_file(crop_dir: Path, original_idx: str) -> Path:
    matches = sorted(crop_dir.glob(f"*_{original_idx}.tif"))
    matches = [p for p in matches if "_mask_" not in p.name and p.name != "Nucleus_masks.tif"]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one crop TIFF for original_idx={original_idx}, found {len(matches)}")
    return matches[0]


def rational_to_float(value) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def synthetic_nucleus_stack(frames: int, height: int, width: int) -> np.ndarray:
    frame = np.full((height, width), 1000, dtype=np.uint16)
    ys = np.arange(8, max(8, height - 8), 4)
    xs = np.arange(8, max(8, width - 8), 4)
    if len(ys) and len(xs):
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        frame[yy.ravel(), xx.ravel()] = 0
    return np.repeat(frame[np.newaxis, :, :], frames, axis=0)


def prepare_analysis_dir(crop_path: Path, analysis_dir: Path, stem: str, manifest: dict) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for stale in analysis_dir.glob("*"):
        if stale.name == "status.json":
            continue
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()

    with tifffile.TiffFile(str(crop_path)) as tf:
        series = tf.series[0]
        axes = series.axes
        data = series.asarray()
        if axes != "TCYX":
            raise ValueError(f"Expected TCYX crop TIFF, got axes={axes} for {crop_path}")
        frames, channels, height, width = data.shape
        if channels < 3:
            raise ValueError(f"Expected at least 3 channels, got {channels} for {crop_path}")

        ij = dict(tf.imagej_metadata or {})
        finterval = float(ij.get("finterval", 1.0))
        page0 = tf.pages[0]
        xres = rational_to_float(page0.tags["XResolution"].value)
        yres = rational_to_float(page0.tags["YResolution"].value) if "YResolution" in page0.tags else xres

    metadata = {"axes": "TYX", "unit": "micron", "finterval": finterval, "loop": False}
    roles = {
        "green": data[:, 0, :, :],
        "red": data[:, 1, :, :],
        "purple": data[:, 2, :, :],
        "Nucleus": synthetic_nucleus_stack(frames, height, width),
    }
    for role, stack in roles.items():
        tifffile.imwrite(
            str(analysis_dir / f"{stem}_{role}.tif"),
            stack,
            imagej=True,
            metadata=metadata,
            resolution=(xres, yres),
        )

    manifest.update(
        {
            "prepared_at": now(),
            "source_crop": str(crop_path),
            "analysis_dir": str(analysis_dir),
            "stem": stem,
            "shape_tcyx": [int(frames), int(channels), int(height), int(width)],
            "finterval_s": finterval,
            "x_resolution_px_per_um": xres,
            "y_resolution_px_per_um": yres,
            "channel_mapping": {
                "green": "crop channel 0",
                "red": "crop channel 1",
                "purple": "crop channel 2",
                "Nucleus": "synthetic smoke-test full-frame surrogate",
            },
        }
    )
    (analysis_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2))


def run_command(cmd: list[str], log_path: Path, env: dict) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {now()}\n")
        log.write("Command: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        proc.wait()
        log.write(f"\nFinished: {now()}\nExit code: {proc.returncode}\n")
        return int(proc.returncode)


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def summarize_cell(analysis_dir: Path, review_dir: Path) -> dict:
    review_dir.mkdir(parents=True, exist_ok=True)
    filter_rows = summarize_filtering(analysis_dir, review_dir)
    matlab_traj = analysis_dir / "matlab_result" / "matlab_trajectory"
    return {
        "reference_counts": {
            ch: count_files(analysis_dir, f"{ch}_loci*_traj_rela2wholeimg.csv") for ch in ("G", "P", "R")
        },
        "candidate_counts": {
            ch: count_files(matlab_traj, f"{ch}_m2DGaussian_traj*.csv") for ch in ("G", "P", "R")
        },
        "cleaned_outputs": sorted(p.name for p in analysis_dir.glob("*_traj_m2DGaussian_cleaned.csv")),
        "filter_status_counts": status_counts(filter_rows),
    }


def status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def cleanup_intermediates(analysis_dir: Path, stem: str) -> None:
    for role in ("Nucleus", "green", "red", "purple"):
        p = analysis_dir / f"{stem}_{role}.tif"
        if p.exists():
            p.unlink()
    for p in (analysis_dir / "Nucleus_masks.tif", analysis_dir / "RoiSet_green.zip"):
        if p.exists():
            p.unlink()
    matlab_dir = analysis_dir / "matlab_result"
    if matlab_dir.exists():
        shutil.rmtree(matlab_dir)


def write_status(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_batch_summary(fov_out: Path) -> None:
    statuses = []
    for status_path in sorted((fov_out / "cells").glob("*/status.json")):
        data = json.loads(status_path.read_text(encoding="utf-8"))
        statuses.append(data)
    fieldnames = [
        "cell_id",
        "original_idx",
        "status",
        "started_at",
        "finished_at",
        "duration_s",
        "free_gb_after",
        "G_refs",
        "P_refs",
        "R_refs",
        "G_candidates",
        "P_candidates",
        "R_candidates",
        "cleaned_count",
        "cleaned_outputs",
        "error",
    ]
    with (fov_out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in statuses:
            refs = data.get("reference_counts", {})
            cands = data.get("candidate_counts", {})
            cleaned = data.get("cleaned_outputs", [])
            writer.writerow(
                {
                    "cell_id": data.get("cell_id", ""),
                    "original_idx": data.get("original_idx", ""),
                    "status": data.get("status", ""),
                    "started_at": data.get("started_at", ""),
                    "finished_at": data.get("finished_at", ""),
                    "duration_s": data.get("duration_s", ""),
                    "free_gb_after": f"{data.get('free_gb_after', 0):.2f}" if "free_gb_after" in data else "",
                    "G_refs": refs.get("G", ""),
                    "P_refs": refs.get("P", ""),
                    "R_refs": refs.get("R", ""),
                    "G_candidates": cands.get("G", ""),
                    "P_candidates": cands.get("P", ""),
                    "R_candidates": cands.get("R", ""),
                    "cleaned_count": len(cleaned),
                    "cleaned_outputs": ";".join(cleaned),
                    "error": data.get("error", ""),
                }
            )


def process_cell(row: dict, crop_dir: Path, fov_out: Path, args, env: dict) -> dict:
    cell_id = row["cell_id"]
    original_idx = row["original_idx"]
    cell_dir = fov_out / "cells" / cell_id
    status_path = cell_dir / "status.json"
    if args.resume and status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            return old

    started = time.time()
    stem = f"{args.fov_label}_{cell_id}"
    status = {
        "fov_label": args.fov_label,
        "cell_id": cell_id,
        "original_idx": original_idx,
        "reference_channel": args.reference_channel,
        "status": "running",
        "started_at": now(),
        "bad_qc": cell_id in args.bad_qc,
    }
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_status(status_path, status)

    try:
        crop_path = find_crop_file(crop_dir, original_idx)
        manifest = {
            "cell_id": cell_id,
            "original_idx": original_idx,
            "bad_qc": status["bad_qc"],
            "reference_channel": args.reference_channel,
        }
        prepare_analysis_dir(crop_path, cell_dir, stem, manifest)

        commands = [
            (
                "stage1_auto_roi.log",
                [
                    sys.executable,
                    str(STAGE1),
                    str(cell_dir / f"{stem}_Nucleus.tif"),
                    "--reference-channel",
                    args.reference_channel,
                ],
            ),
            ("stage2_matlab_spt.log", [sys.executable, str(STAGE2), str(cell_dir)]),
            ("stage3_match.log", [sys.executable, str(STAGE3), str(cell_dir)]),
        ]
        for log_name, cmd in commands:
            code = run_command(cmd, cell_dir / log_name, env)
            if code != 0:
                raise RuntimeError(f"{log_name} failed with exit code {code}")

        summary = summarize_cell(cell_dir, cell_dir / "review")
        status.update(summary)
        status["status"] = "complete"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
    finally:
        if args.clean_intermediates:
            cleanup_intermediates(cell_dir, stem)
        status["finished_at"] = now()
        status["duration_s"] = round(time.time() - started, 1)
        status["free_gb_after"] = round(disk_free_gb(args.output_root), 2)
        write_status(status_path, status)
        write_batch_summary(fov_out)

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fov-label", required=True)
    parser.add_argument(
        "--reference-channel",
        choices=("green", "red", "purple"),
        default="green",
        help="Stage 1 anchor channel; the other two channels are targets (default: green).",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--start-after-cell", default=None)
    parser.add_argument("--stop-free-gb", type=float, default=15.0)
    parser.add_argument("--bad-qc", default="", help="Comma-separated cell IDs to mark as bad_qc.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-clean-intermediates", dest="clean_intermediates", action="store_false")
    parser.set_defaults(clean_intermediates=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.bad_qc = {x.strip() for x in args.bad_qc.split(",") if x.strip()}

    fov_out = args.output_root / args.fov_label
    (fov_out / "cells").mkdir(parents=True, exist_ok=True)
    batch_log = fov_out / "batch_runner.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    rows = read_mapping(args.crop_dir)
    if args.start_after_cell:
        seen = False
        filtered = []
        for row in rows:
            if seen:
                filtered.append(row)
            if row["cell_id"] == args.start_after_cell:
                seen = True
        rows = filtered
    if args.max_cells is not None:
        rows = rows[: args.max_cells]

    with batch_log.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] Starting {args.fov_label}: {len(rows)} cell(s)\n")
        log.write(f"Crop dir: {args.crop_dir}\n")
        log.write(f"Output root: {args.output_root}\n")

    for row in rows:
        free_before = disk_free_gb(args.output_root)
        if free_before < args.stop_free_gb:
            with batch_log.open("a", encoding="utf-8") as log:
                log.write(f"[{now()}] Stopping before {row['cell_id']}: free disk {free_before:.2f} GB\n")
            break

        with batch_log.open("a", encoding="utf-8") as log:
            log.write(f"[{now()}] Processing {row['cell_id']} original_idx={row['original_idx']} free={free_before:.2f} GB\n")
        result = process_cell(row, args.crop_dir, fov_out, args, env)
        with batch_log.open("a", encoding="utf-8") as log:
            log.write(
                f"[{now()}] Finished {row['cell_id']} status={result.get('status')} "
                f"duration={result.get('duration_s')}s free={result.get('free_gb_after')} GB\n"
            )

    write_batch_summary(fov_out)
    with batch_log.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] Batch loop ended for {args.fov_label}\n")


if __name__ == "__main__":
    main()
