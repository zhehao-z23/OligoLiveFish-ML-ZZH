#!/usr/bin/env python3
"""Recover legacy-v3 reference matching after symlinked staging.

The historical Sherlock staging used symlinked channel TIFFs.  Stage 1 resolved
the nucleus symlink and wrote reference CSVs beside the original cell inputs,
while MATLAB candidate trajectories remained in the isolated task directory.
This tool copies the small reference CSVs into each task directory and reruns
only the Python matcher.  It never reruns localization, MATLAB, or SPT.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


CHANNELS = ("G", "P", "R")


def count_by_channel(root: Path, pattern: str) -> dict[str, int]:
    return {
        channel: len(list(root.glob(pattern.format(channel=channel))))
        for channel in CHANNELS
    }


def source_cell_dir(task_dir: Path) -> Path | None:
    green = sorted(task_dir.glob("*_green.tif"))
    if len(green) != 1:
        return None
    return green[0].resolve().parent


def recover_task(task_index: int, task_dir: Path, matcher: Path) -> dict:
    source = source_cell_dir(task_dir)
    row = {
        "task_index": task_index,
        "task_dir": str(task_dir.resolve()),
        "source_cell_dir": str(source) if source else "",
        "reference_count": 0,
        "candidate_count": 0,
        "cleaned_count": 0,
        "G_cleaned": 0,
        "P_cleaned": 0,
        "R_cleaned": 0,
        "matcher_exit_code": "",
        "status": "",
    }
    if source is None:
        row["status"] = "technical_source_not_found"
        return row
    references = sorted(source.glob("[GPR]_loci*_traj_rela2wholeimg.csv"))
    row["reference_count"] = len(references)
    matlab_dir = task_dir / "matlab_result" / "matlab_trajectory"
    candidates = (
        sorted(matlab_dir.glob("[GPR]_m2DGaussian_traj*.csv"))
        if matlab_dir.is_dir()
        else []
    )
    row["candidate_count"] = len(candidates)
    if not references:
        row["status"] = "scientific_no_reference"
        return row
    if not candidates:
        row["status"] = "scientific_no_candidate"
        return row
    for path in task_dir.glob("[GPR]_loci*_traj_rela2wholeimg.csv"):
        path.unlink()
    for path in task_dir.glob("[GPR]_loci*_traj_m2DGaussian_cleaned.csv"):
        path.unlink()
    for reference in references:
        shutil.copy2(reference, task_dir / reference.name)
    completed = subprocess.run(
        [sys.executable, str(matcher), str(task_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )
    row["matcher_exit_code"] = completed.returncode
    cleaned = count_by_channel(
        task_dir,
        "{channel}_loci*_traj_m2DGaussian_cleaned.csv",
    )
    row["cleaned_count"] = sum(cleaned.values())
    for channel in CHANNELS:
        row[f"{channel}_cleaned"] = cleaned[channel]
    if completed.returncode != 0:
        row["status"] = "technical_matcher_failed"
    elif row["cleaned_count"]:
        row["status"] = "recovered"
    else:
        row["status"] = "scientific_no_match"
    return row


def recover(task_list: Path, matcher: Path, audit_path: Path) -> list[dict]:
    tasks = [
        Path(line.strip())
        for line in task_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tasks:
        raise ValueError(f"task list is empty: {task_list}")
    if not matcher.is_file():
        raise FileNotFoundError(f"matcher not found: {matcher}")
    rows = [
        recover_task(index, task_dir, matcher)
        for index, task_dir in enumerate(tasks, 1)
    ]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-list", required=True, type=Path)
    parser.add_argument("--matcher", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = recover(
        args.task_list.resolve(),
        args.matcher.resolve(),
        args.audit.resolve(),
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    for status in sorted(counts):
        print(f"{status}={counts[status]}")
    technical = sum(
        count for status, count in counts.items()
        if status.startswith("technical_")
    )
    print(f"tasks={len(rows)}")
    print(f"audit={args.audit.resolve()}")
    if technical:
        raise SystemExit(
            f"ERROR: legacy reference recovery has {technical} "
            "technical failure(s)"
        )
    print("LEGACY_V3_REFERENCE_MATCHING_RECOVERY_OK")


if __name__ == "__main__":
    main()
