#!/usr/bin/env python3
"""Update TRAJECTORY_TEST_PLAN.md with the current batch overview.

This helper is intentionally conservative: it reads only batch overview files
and replaces a marked section in the project-level trajectory notes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


BEGIN = "<!-- TRAJECTORY_BATCH_FINAL_SUMMARY_BEGIN -->"
END = "<!-- TRAJECTORY_BATCH_FINAL_SUMMARY_END -->"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def status_text(status: dict) -> str:
    if not status:
        return "missing"
    value = status.get("status", "unknown")
    current = status.get("current_fov", "")
    if current and current != "none":
        return f"{value} ({current})"
    return value


def effective_status_text(overview: dict, master_status: dict) -> str:
    summaries = overview.get("summaries", [])
    if summaries:
        total_running = sum(int(s.get("running_cells", 0) or 0) for s in summaries)
        total_failed = sum(int(s.get("failed_cells", 0) or 0) for s in summaries)
        total_complete = sum(int(s.get("complete_cells", 0) or 0) for s in summaries)
        if total_running == 0 and total_failed == 0 and total_complete > 0:
            return "complete (all batch shards finished)"
        if total_running == 0 and total_failed > 0:
            return "complete-with-failures (all batch shards finished)"
    return status_text(master_status)


def cleanup_audit_text(output_root: Path, overview: dict) -> str:
    leftovers = []
    for summary in overview.get("summaries", []):
        fov = summary.get("fov", "")
        if not fov or fov == "fov17_total":
            continue
        batch_summary = output_root / fov / "batch_summary.csv"
        cells_dir = output_root / fov / "cells"
        if not batch_summary.exists() or not cells_dir.exists():
            continue
        for cell_dir in cells_dir.glob("cell_*"):
            for item in cell_dir.iterdir():
                name = item.name
                if (
                    name in {"matlab_result", "Nucleus_masks.tif", "RoiSet_green.zip"}
                    or name.endswith("_green.tif")
                    or name.endswith("_red.tif")
                    or name.endswith("_purple.tif")
                    or name.endswith("_Nucleus.tif")
                ):
                    leftovers.append(f"{fov}/{cell_dir.name}/{name}")
    if leftovers:
        return f"found {len(leftovers)} intermediate leftovers; review required"
    return "passed; no split TIFFs, nucleus masks, ROI zip files, or matlab_result folders remain in completed cell directories"


def render_section(output_root: Path, overview: dict, master_status: dict) -> str:
    summaries = overview.get("summaries", [])
    lines = [
        BEGIN,
        "",
        "## fov7/fov17 Batch Current Summary",
        "",
        f"Updated: `{now()}`",
        "",
        f"Output root: `{output_root}`",
        "",
        f"Batch status: `{effective_status_text(overview, master_status)}`",
        "",
        f"Cleanup audit: `{cleanup_audit_text(output_root, overview)}`",
        "",
        "| FOV | Cells seen | Complete | Failed | Running | Bad QC seen | Cleaned CSVs | Median duration (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {fov} | {cells_seen} | {complete_cells} | {failed_cells} | {running_cells} | "
            "{bad_qc_cells_seen} | {total_cleaned_outputs} | {median_duration_s} |".format(**summary)
        )

    lines.extend(["", "Channel totals from completed cells:", ""])
    for summary in summaries:
        refs = summary.get("total_reference_tracks", {})
        cands = summary.get("total_candidate_tracks", {})
        lines.append(
            f"- `{summary['fov']}` references: G={refs.get('G', 0)}, "
            f"P={refs.get('P', 0)}, R={refs.get('R', 0)}"
        )
        lines.append(
            f"- `{summary['fov']}` MATLAB candidates: G={cands.get('G', 0)}, "
            f"P={cands.get('P', 0)}, R={cands.get('R', 0)}"
        )
        failed = summary.get("failed_cells_list", [])
        running = summary.get("running_cells_list", [])
        if failed:
            lines.append(f"- `{summary['fov']}` failed cells: {', '.join(failed)}")
        if running:
            lines.append(f"- `{summary['fov']}` running cells at summary time: {', '.join(running)}")

    lines.extend(
        [
            "",
            "Batch-level review files:",
            "",
            "```text",
            "2026-07-08_overnight_trajectory_fov7_fov17\\batch_overview.json",
            "2026-07-08_overnight_trajectory_fov7_fov17\\batch_overview_cells.csv",
            "2026-07-08_overnight_trajectory_fov7_fov17\\BATCH_RESULTS_SUMMARY.md",
            "```",
            "",
            "Interpretation boundary: these outputs are trajectory extraction CSVs and",
            "match-filter summaries only. They do not represent deep-learning or",
            "downstream nuclear-feature analysis.",
            "",
            END,
        ]
    )
    return "\n".join(lines) + "\n"


def replace_marked_section(text: str, section: str) -> str:
    if BEGIN in text and END in text:
        before = text.split(BEGIN, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        return before + "\n\n" + section.rstrip() + "\n\n" + after
    return text.rstrip() + "\n\n" + section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--doc", required=True, type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    overview = read_json_if_exists(output_root / "batch_overview.json")
    master_status = read_json_if_exists(output_root / "overnight_master_status.json")
    section = render_section(output_root, overview, master_status)

    doc = args.doc.resolve()
    original = doc.read_text(encoding="utf-8")
    updated = replace_marked_section(original, section)
    doc.write_text(updated, encoding="utf-8")
    print(f"Updated {doc}")


if __name__ == "__main__":
    main()
