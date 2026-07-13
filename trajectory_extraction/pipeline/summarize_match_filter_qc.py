"""Reusable Stage-3 match/filter QC summary used by the crop batch runner."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


MIN_OVERLAP_FRAMES = 5
MAX_AVG_DIST_NM = 2000.0


def read_track(path: Path) -> dict[int, tuple[float, float]]:
    track: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            track[int(row["frame"])] = (float(row["x_nm"]), float(row["y_nm"]))
    return track


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg_distance(
    reference: dict[int, tuple[float, float]], candidate: dict[int, tuple[float, float]]
) -> tuple[float, int]:
    shared = sorted(set(reference) & set(candidate))
    if len(shared) < MIN_OVERLAP_FRAMES:
        return math.inf, len(shared)
    total = sum(
        math.hypot(reference[frame][0] - candidate[frame][0], reference[frame][1] - candidate[frame][1])
        for frame in shared
    )
    return total / len(shared), len(shared)


def extract_locus(path: Path) -> str:
    match = re.search(r"_loci(\d+)_", path.name)
    return match.group(1) if match else path.stem


def summarize_filtering(analysis_dir: Path, out_dir: Path) -> list[dict]:
    """Reconstruct Stage-3 decisions and write a compact match_filter_summary.csv."""
    matlab_dir = analysis_dir / "matlab_result" / "matlab_trajectory"
    rows: list[dict] = []

    for channel in ("G", "P", "R"):
        ref_paths = sorted(analysis_dir.glob(f"{channel}_loci*_traj_rela2wholeimg.csv"))
        cand_paths = sorted(matlab_dir.glob(f"{channel}_m2DGaussian_traj*.csv"))
        refs = {path: read_track(path) for path in ref_paths}
        cands = {path: read_track(path) for path in cand_paths}

        scored: list[tuple[float, int, Path, Path]] = []
        for ref_path, reference in refs.items():
            for candidate_path, candidate in cands.items():
                distance, shared = avg_distance(reference, candidate)
                if math.isfinite(distance):
                    scored.append((distance, shared, ref_path, candidate_path))
        scored.sort(key=lambda item: item[0])

        assigned_refs: set[Path] = set()
        assigned_candidates: set[Path] = set()
        assignments: dict[Path, tuple[Path, float, int]] = {}
        for distance, shared, ref_path, candidate_path in scored:
            if ref_path in assigned_refs or candidate_path in assigned_candidates:
                continue
            assigned_refs.add(ref_path)
            assigned_candidates.add(candidate_path)
            assignments[ref_path] = (candidate_path, distance, shared)

        for ref_path in ref_paths:
            locus = extract_locus(ref_path)
            row = {
                "channel": channel,
                "locus": f"loci{locus}",
                "reference": ref_path.name,
                "candidate": "",
                "shared_frames": 0,
                "avg_dist_nm": "",
                "status": "no_match",
                "cleaned_output": "",
            }
            if ref_path in assignments:
                candidate_path, distance, shared = assignments[ref_path]
                row.update(
                    {
                        "candidate": candidate_path.name,
                        "shared_frames": shared,
                        "avg_dist_nm": f"{distance:.2f}",
                    }
                )
                if distance > MAX_AVG_DIST_NM:
                    row["status"] = "rejected_distance"
                elif channel == "R" and not any(frame <= 3 for frame in cands[candidate_path]):
                    row["status"] = "rejected_red_late_start"
                else:
                    out_name = f"{channel}_loci{locus}_traj_m2DGaussian_cleaned.csv"
                    row["cleaned_output"] = out_name if (analysis_dir / out_name).exists() else ""
                    row["status"] = "saved" if row["cleaned_output"] else "accepted_missing_file"
            rows.append(row)

    write_csv(
        out_dir / "match_filter_summary.csv",
        rows,
        [
            "channel",
            "locus",
            "reference",
            "candidate",
            "shared_frames",
            "avg_dist_nm",
            "status",
            "cleaned_output",
        ],
    )
    return rows
