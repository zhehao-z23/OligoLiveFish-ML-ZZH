#!/usr/bin/env python3
"""Summarize crop-to-trajectory batch outputs.

The script is read-only with respect to per-cell outputs. It scans fov
directories produced by run_crop_trajectory_batch.py and writes compact
overview files at the output root for morning review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


CHANNELS = ("G", "P", "R")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(value) -> int:
    try:
        if value in ("", None):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def collect_cells(fov_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for status_path in sorted((fov_dir / "cells").glob("*/status.json")):
        status = read_json(status_path)
        refs = status.get("reference_counts", {}) or {}
        cands = status.get("candidate_counts", {}) or {}
        cleaned = status.get("cleaned_outputs", []) or []
        filters = status.get("filter_status_counts", {}) or {}
        row = {
            "fov": fov_dir.name,
            "cell_id": status.get("cell_id", status_path.parent.name),
            "original_idx": status.get("original_idx", ""),
            "bad_qc": bool(status.get("bad_qc", False)),
            "status": status.get("status", ""),
            "started_at": status.get("started_at", ""),
            "finished_at": status.get("finished_at", ""),
            "duration_s": safe_float(status.get("duration_s", 0)),
            "free_gb_after": safe_float(status.get("free_gb_after", 0)),
            "cleaned_count": len(cleaned),
            "cleaned_outputs": ";".join(cleaned),
            "filter_saved": safe_int(filters.get("saved", 0)),
            "filter_rejected_distance": safe_int(filters.get("rejected_distance", 0)),
            "filter_rejected_red_late_start": safe_int(filters.get("rejected_red_late_start", 0)),
            "error": status.get("error", ""),
        }
        for channel in CHANNELS:
            row[f"{channel}_refs"] = safe_int(refs.get(channel, 0))
            row[f"{channel}_candidates"] = safe_int(cands.get(channel, 0))
        rows.append(row)
    inferred = infer_running_cell_from_log(fov_dir, rows)
    if inferred:
        rows.append(inferred)
    return rows


def infer_running_cell_from_log(fov_dir: Path, rows: list[dict]) -> dict | None:
    log_path = fov_dir / "batch_runner.log"
    if not log_path.exists():
        return None
    processing_re = re.compile(
        r"^\[(?P<time>[^\]]+)\] Processing (?P<cell_id>cell_\d+) original_idx=(?P<original_idx>\S+)"
    )
    finished_re = re.compile(r"^\[[^\]]+\] Finished (?P<cell_id>cell_\d+) status=")
    last_processing: dict | None = None
    finished_after: set[str] = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = processing_re.match(line)
        if match:
            last_processing = match.groupdict()
            finished_after = set()
            continue
        match = finished_re.match(line)
        if match:
            finished_after.add(match.group("cell_id"))
    if not last_processing:
        return None
    cell_id = last_processing["cell_id"]
    if cell_id in finished_after:
        return None
    if any(row["cell_id"] == cell_id for row in rows):
        return None
    row = {
        "fov": fov_dir.name,
        "cell_id": cell_id,
        "original_idx": last_processing["original_idx"],
        "bad_qc": False,
        "status": "running",
        "started_at": last_processing["time"],
        "finished_at": "",
        "duration_s": 0.0,
        "free_gb_after": 0.0,
        "cleaned_count": 0,
        "cleaned_outputs": "",
        "filter_saved": 0,
        "filter_rejected_distance": 0,
        "filter_rejected_red_late_start": 0,
        "error": "inferred from batch_runner.log; status.json not yet finalized",
    }
    for channel in CHANNELS:
        row[f"{channel}_refs"] = 0
        row[f"{channel}_candidates"] = 0
    return row


def summarize_fov(fov: str, rows: list[dict]) -> dict:
    durations = [r["duration_s"] for r in rows if r["duration_s"] > 0]
    complete = [r for r in rows if r["status"] == "complete"]
    failed = [r for r in rows if r["status"] == "failed"]
    running = [r for r in rows if r["status"] == "running"]
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    return {
        "fov": fov,
        "cells_seen": len(rows),
        "status_counts": by_status,
        "complete_cells": len(complete),
        "failed_cells": len(failed),
        "running_cells": len(running),
        "bad_qc_cells_seen": sum(1 for r in rows if r["bad_qc"]),
        "total_cleaned_outputs": sum(r["cleaned_count"] for r in complete),
        "total_reference_tracks": {ch: sum(r[f"{ch}_refs"] for r in complete) for ch in CHANNELS},
        "total_candidate_tracks": {ch: sum(r[f"{ch}_candidates"] for r in complete) for ch in CHANNELS},
        "median_duration_s": round(statistics.median(durations), 1) if durations else 0,
        "total_duration_s": round(sum(durations), 1),
        "failed_cells_list": [r["cell_id"] for r in failed],
        "running_cells_list": [r["cell_id"] for r in running],
        "last_finished_at": max((r["finished_at"] for r in rows if r["finished_at"]), default=""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "fov",
        "cell_id",
        "original_idx",
        "bad_qc",
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
        "filter_saved",
        "filter_rejected_distance",
        "filter_rejected_red_late_start",
        "cleaned_outputs",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(output_root: Path, summaries: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Trajectory Batch Results Summary",
        "",
        f"Generated: {now()}",
        "",
        f"Output root: `{output_root}`",
        "",
        "This summary covers trajectory extraction batch outputs only. It does not include deep learning or downstream nuclear-feature analysis.",
        "",
        "## FOV Overview",
        "",
        "| FOV | Cells seen | Complete | Failed | Running | Bad QC seen | Cleaned CSVs | Median duration (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {fov} | {cells_seen} | {complete_cells} | {failed_cells} | {running_cells} | "
            "{bad_qc_cells_seen} | {total_cleaned_outputs} | {median_duration_s} |".format(**summary)
        )
    lines.extend(["", "## Channel Totals", ""])
    for summary in summaries:
        lines.append(f"### {summary['fov']}")
        refs = summary["total_reference_tracks"]
        cands = summary["total_candidate_tracks"]
        lines.append(
            f"- References: G={refs['G']}, P={refs['P']}, R={refs['R']}"
        )
        lines.append(
            f"- MATLAB candidates: G={cands['G']}, P={cands['P']}, R={cands['R']}"
        )
        if summary["failed_cells_list"]:
            lines.append(f"- Failed cells: {', '.join(summary['failed_cells_list'])}")
        if summary["running_cells_list"]:
            lines.append(f"- Running cells at summary time: {', '.join(summary['running_cells_list'])}")
        lines.append("")
    lines.extend(
        [
            "## Output Policy",
            "",
            "Per-cell CSV trajectories, logs, manifests, status files, and match-filter summaries are retained.",
            "Split TIFFs, nucleus masks, ROI zip files, and MATLAB intermediate folders are removed after completed or failed cells by the batch runner.",
            "",
            "## Per-Cell Table",
            "",
            "See `batch_overview_cells.csv` for per-cell status, counts, cleaned outputs, and errors.",
        ]
    )
    if any(r["status"] == "failed" for r in rows):
        lines.extend(["", "## Failed Cell Details", ""])
        for row in rows:
            if row["status"] == "failed":
                lines.append(f"- {row['fov']} {row['cell_id']}: {row['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--fovs",
        default="fov7,fov17,fov17_part2,fov17_part2_tail,fov17_part3,fov17_part3_tail",
        help="Comma-separated FOV directory names to scan.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    fovs = [x.strip() for x in args.fovs.split(",") if x.strip()]
    all_rows: list[dict] = []
    summaries: list[dict] = []

    for fov in fovs:
        fov_dir = output_root / fov
        if not fov_dir.exists():
            summaries.append(
                {
                    "fov": fov,
                    "cells_seen": 0,
                    "status_counts": {},
                    "complete_cells": 0,
                    "failed_cells": 0,
                    "running_cells": 0,
                    "bad_qc_cells_seen": 0,
                    "total_cleaned_outputs": 0,
                    "total_reference_tracks": {ch: 0 for ch in CHANNELS},
                    "total_candidate_tracks": {ch: 0 for ch in CHANNELS},
                    "median_duration_s": 0,
                    "total_duration_s": 0,
                    "failed_cells_list": [],
                    "running_cells_list": [],
                    "last_finished_at": "",
                }
            )
            continue
        rows = collect_cells(fov_dir)
        all_rows.extend(rows)
        summaries.append(summarize_fov(fov, rows))

    fov17_shards = {"fov17", "fov17_part2", "fov17_part2_tail", "fov17_part3", "fov17_part3_tail"}
    if len(fov17_shards.intersection(fovs)) > 1:
        shard_rows = [row for row in all_rows if row["fov"] in fov17_shards]
        summaries.append(summarize_fov("fov17_total", shard_rows))

    payload = {
        "generated_at": now(),
        "output_root": str(output_root),
        "summaries": summaries,
    }
    (output_root / "batch_overview.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_root / "batch_overview_cells.csv", all_rows)
    (output_root / "BATCH_RESULTS_SUMMARY.md").write_text(
        markdown_report(output_root, summaries, all_rows),
        encoding="utf-8",
    )
    print(f"Wrote {output_root / 'batch_overview.json'}")
    print(f"Wrote {output_root / 'batch_overview_cells.csv'}")
    print(f"Wrote {output_root / 'BATCH_RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
