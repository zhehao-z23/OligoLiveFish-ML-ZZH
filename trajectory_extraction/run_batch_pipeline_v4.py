#!/usr/bin/env python3
"""Run v4.1 profile-locked anchor-ROI SPT for every valid cell crop in one FOV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pipeline"))
from align_microsam_mask import discover_microsam_mask
import experiment_profiles


VERSION = "v4.1.6-experiment-profiles"
SINGLE_CELL_RUNNER = HERE / "run_full_pipeline_v4.py"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def is_cell_crop(
    path: Path, profile: experiment_profiles.ExperimentProfile
) -> tuple[bool, str]:
    try:
        validation = profile.validate_crop(path)
        mask = discover_microsam_mask(path)
    except Exception as exc:
        return False, str(exc)
    return True, (
        f"profile={profile.name}, axes={validation['axes']}, "
        f"shape={validation['shape']}, anchor={profile.anchor.marker}, "
        f"micro-SAM={mask.name}"
    )


def scientific_options(args: argparse.Namespace) -> dict:
    return {
        "experiment_profile": args.experiment_profile,
        "mask_dilation_px": args.mask_dilation_px,
        "roi_dilation_px": args.roi_dilation_px,
        "d_star": args.d_star,
        "alpha": args.alpha,
        "coverage_probability": args.coverage_probability,
        "localization_error_nm": args.localization_error_nm,
        "max_step_frame_gap": args.max_step_frame_gap,
        "max_step_rounding_px": args.max_step_rounding_px,
        "max_step_px": args.max_step_px,
        "matlab_workers": args.matlab_workers,
        "matlab_save_filter_images": args.matlab_save_filter_images,
    }


def completion_matches(analysis_dir: Path, args: argparse.Namespace) -> bool:
    path = (
        analysis_dir
        / f"anchor_roi_v4_{args.experiment_profile}"
        / "run_manifest.json"
    )
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return (
            manifest.get("version") == VERSION
            and manifest.get("status") == "complete"
            and all(manifest.get("options", {}).get(key) == value for key, value in scientific_options(args).items())
        )
    except (OSError, ValueError, TypeError):
        return False


def tail_log(
    analysis_dir: Path, experiment_profile: str, line_count: int = 25
) -> str:
    path = (
        analysis_dir
        / f"anchor_roi_v4_{experiment_profile}"
        / "log_anchor_roi_v4.txt"
    )
    if not path.is_file():
        return "v4 log was not created"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def run_cell(crop: Path, args: argparse.Namespace) -> dict:
    analysis_dir = crop.with_suffix("")
    if args.resume and completion_matches(analysis_dir, args):
        return {
            "crop": str(crop), "analysis_dir": str(analysis_dir), "status": "skipped_complete",
            "experiment_profile": args.experiment_profile,
            "started_at": "", "finished_at": now(), "duration_s": 0.0, "exit_code": 0, "error_tail": "",
        }
    command = [
        sys.executable, str(SINGLE_CELL_RUNNER), str(crop),
        "--fiji-bin", args.fiji_bin,
        "--matlab-bin", args.matlab_bin,
        "--matlab-workers", str(args.matlab_workers),
        "--experiment-profile", args.experiment_profile,
        "--mask-dilation-px", str(args.mask_dilation_px),
        "--roi-dilation-px", str(args.roi_dilation_px),
        "--d-star", str(args.d_star),
        "--alpha", str(args.alpha),
        "--coverage-probability", str(args.coverage_probability),
        "--localization-error-nm", str(args.localization_error_nm),
        "--max-step-frame-gap", str(args.max_step_frame_gap),
        "--max-step-rounding-px", str(args.max_step_rounding_px),
    ]
    if args.max_step_px is not None:
        command.extend(["--max-step-px", str(args.max_step_px)])
    if args.matlab_save_filter_images:
        command.append("--matlab-save-filter-images")
    started_at, started = now(), time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment)
        exit_code = int(result.returncode)
        status = "complete" if exit_code == 0 else "failed"
        error_tail = "" if exit_code == 0 else tail_log(
            analysis_dir, args.experiment_profile
        )
    except Exception as exc:
        exit_code, status, error_tail = -1, "failed_to_start", repr(exc)
    return {
        "crop": str(crop), "analysis_dir": str(analysis_dir), "status": status,
        "experiment_profile": args.experiment_profile,
        "started_at": started_at, "finished_at": now(),
        "duration_s": round(time.perf_counter() - started, 1),
        "exit_code": exit_code, "error_tail": error_tail,
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "crop", "analysis_dir", "experiment_profile", "status", "started_at",
        "finished_at", "duration_s", "exit_code", "error_tail",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop_dir", type=Path)
    parser.add_argument("--crop-glob", default="*.tif")
    parser.add_argument("--fiji-bin", default="fiji")
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument(
        "--experiment-profile",
        choices=experiment_profiles.profile_choices(),
        required=True,
        help="Required locked biological channel contract; it determines the anchor automatically.",
    )
    parser.add_argument("--cell-workers", type=int, default=1)
    parser.add_argument("--matlab-workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--matlab-save-filter-images", action="store_true")
    parser.add_argument("--mask-dilation-px", type=int, default=5)
    parser.add_argument("--roi-dilation-px", type=int, default=5)
    parser.add_argument("--d-star", type=float, default=4.1e-3)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--coverage-probability", type=float, default=0.995)
    parser.add_argument("--localization-error-nm", type=float, default=0.0)
    parser.add_argument("--max-step-frame-gap", type=int, default=1)
    parser.add_argument("--max-step-rounding-px", type=float, default=0.05)
    parser.add_argument("--max-step-px", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = experiment_profiles.get_profile(args.experiment_profile)
    crop_dir = args.crop_dir.resolve()
    if not crop_dir.is_dir() or args.cell_workers < 1:
        raise SystemExit("ERROR: crop_dir must exist and --cell-workers must be >= 1")
    candidates = sorted(path for path in crop_dir.glob(args.crop_glob) if path.is_file())
    crops = []
    for path in candidates:
        accepted, reason = is_cell_crop(path, profile)
        print(f"  [{'ACCEPT' if accepted else 'SKIP'}] {path.name}: {reason}")
        if accepted:
            crops.append(path)
    if not crops:
        raise SystemExit("ERROR: no valid cell crop with an associated micro-SAM mask")
    print(
        f"Profile={profile.name}; locked anchor={profile.anchor.marker} "
        f"(raw C{profile.anchor.raw_index})"
    )
    print(f"Valid crops={len(crops)}; cell workers={args.cell_workers}; MATLAB workers/cell={args.matlab_workers}")
    print(f"Maximum simultaneous MATLAB processes={args.cell_workers * args.matlab_workers}")
    if args.dry_run:
        print("Dry run complete; no analysis started.")
        return

    summary_path = crop_dir / f"trajectory_batch_v4_{profile.name}_summary.csv"
    results, started = [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.cell_workers) as executor:
        pending = {executor.submit(run_cell, crop, args) for crop in crops}
        while pending:
            completed, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not completed:
                print(
                    f"Batch still running: {len(pending)} cell(s) pending, "
                    f"{time.perf_counter() - started:.0f} s elapsed",
                    flush=True,
                )
                continue
            for future in completed:
                row = future.result()
                results.append(row)
                print(f"[{row['status']}] {Path(row['crop']).name} ({row['duration_s']} s)")
                write_summary(summary_path, results)

    order = {str(path): index for index, path in enumerate(crops)}
    results.sort(key=lambda row: order[row["crop"]])
    write_summary(summary_path, results)
    failures = [row for row in results if row["status"].startswith("failed")]
    print(f"Batch elapsed={time.perf_counter() - started:.1f} s; summary={summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
