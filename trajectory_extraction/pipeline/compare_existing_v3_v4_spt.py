#!/usr/bin/env python3
"""Compare completed legacy-v3 and current-v4 SPT results without rerunning SPT.

The primary comparison is restricted to pixel-identical source crop TIFFs.
Cells with the same FOV/crop identity but different TIFF hashes are retained as
a separate sensitivity set.  Cohort summaries use every entered cell and state
their denominators explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

import summarize_candidate_policy_spt as policy_summary


PROFILE_RESULTS_DIR = "anchor_roi_v4_chr3_sites_2_3_4"
CHANNELS = ("G", "P", "R")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def trajectory_points(path: Path) -> int:
    return len(read_csv_rows(path))


def channel_from_baseline(row: dict[str, str]) -> str:
    values = " ".join(
        str(row.get(field, ""))
        for field in ("channel", "marker", "anchor_locus")
    ).lower()
    if "site2" in values or "a488" in values or "green" in values:
        return "G"
    if "site4" in values or "a647" in values or "red" in values:
        return "R"
    if "site3" in values or "a565" in values or "purple" in values:
        return "P"
    raw = str(row.get("channel", "")).strip().upper()
    return raw if raw in CHANNELS else ""


def summarize_point_files(
    files: Iterable[tuple[str, Path]],
) -> tuple[list[int], dict[str, list[int]]]:
    by_channel = {channel: [] for channel in CHANNELS}
    all_points: list[int] = []
    for channel, path in files:
        points = trajectory_points(path)
        all_points.append(points)
        if channel in by_channel:
            by_channel[channel].append(points)
    return all_points, by_channel


def mean_or_none(values: list[int]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def median_or_none(values: list[int]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def source_crop_from_cell_dir(cell_dir: Path) -> Path:
    return cell_dir.parent / f"{cell_dir.name}.tif"


def legacy_source_crop(task_dir: Path) -> Path | None:
    channel_files = sorted(task_dir.glob("*_green.tif"))
    if len(channel_files) != 1:
        return None
    source_channel = channel_files[0].resolve()
    candidate = source_crop_from_cell_dir(source_channel.parent)
    return candidate if candidate.is_file() else None


def legacy_record(task_index: int, task_dir: Path) -> dict:
    crop = legacy_source_crop(task_dir)
    cleaned = []
    for channel in CHANNELS:
        cleaned.extend(
            (channel, path)
            for path in sorted(
                task_dir.glob(
                    f"{channel}_loci*_traj_m2DGaussian_cleaned.csv"
                )
            )
        )
    points, by_channel = summarize_point_files(cleaned)
    matlab_dir = task_dir / "matlab_result" / "matlab_trajectory"
    if points:
        outcome = "success"
    elif matlab_dir.is_dir():
        outcome = "scientific_no_cleaned_trajectory"
    else:
        outcome = "technical_output_missing"
    crop_name = crop.name if crop else task_dir.name.split("_", 1)[-1] + ".tif"
    fov = crop.parent.name if crop else ""
    return {
        "legacy_task_index": task_index,
        "legacy_analysis_dir": str(task_dir.resolve()),
        "fov": fov,
        "crop_name": crop_name,
        "source_crop": str(crop.resolve()) if crop else "",
        "source_crop_bytes": crop.stat().st_size if crop else "",
        "source_crop_sha256": sha256_file(crop) if crop else "",
        "outcome": outcome,
        "baseline_count": len(points),
        "mean_baseline_points": mean_or_none(points),
        "median_baseline_points": median_or_none(points),
        "G_baselines": len(by_channel["G"]),
        "P_baselines": len(by_channel["P"]),
        "R_baselines": len(by_channel["R"]),
        "complete_GPR": all(by_channel[channel] for channel in CHANNELS),
        "_points": points,
        **{
            f"_{channel}_points": by_channel[channel]
            for channel in CHANNELS
        },
    }


def load_legacy_inventory(task_list: Path) -> list[dict]:
    tasks = [
        Path(line.strip())
        for line in task_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tasks:
        raise ValueError(f"legacy task list is empty: {task_list}")
    return [
        legacy_record(index, task_dir)
        for index, task_dir in enumerate(tasks, 1)
    ]


def apply_legacy_recovery_audit(
    rows: list[dict],
    audit_path: Path | None,
) -> None:
    if audit_path is None:
        return
    audit_rows = read_csv_rows(audit_path)
    by_index = {
        int(row["task_index"]): row
        for row in audit_rows
    }
    if set(by_index) != {
        int(row["legacy_task_index"]) for row in rows
    }:
        raise ValueError(
            "legacy recovery audit and task inventory have different indices"
        )
    for row in rows:
        audit = by_index[int(row["legacy_task_index"])]
        row["recovery_status"] = audit["status"]
        if row["outcome"] == "success":
            continue
        if audit["status"].startswith("technical_"):
            row["outcome"] = audit["status"]
        elif audit["status"].startswith("scientific_"):
            row["outcome"] = audit["status"]


def v4_baselines(row: dict[str, str]) -> tuple[list[int], dict[str, list[int]]]:
    if row["final_class"] != "success":
        return [], {channel: [] for channel in CHANNELS}
    manifest = (
        Path(row["analysis_dir"])
        / PROFILE_RESULTS_DIR
        / "baseline_longest"
        / "baseline_manifest.csv"
    )
    baseline_rows = read_csv_rows(manifest)
    by_channel = {channel: [] for channel in CHANNELS}
    points = []
    for baseline in baseline_rows:
        value = str(baseline.get("points", "")).strip()
        if not value:
            continue
        count = int(float(value))
        points.append(count)
        channel = channel_from_baseline(baseline)
        if channel:
            by_channel[channel].append(count)
    return points, by_channel


def v4_record(row: dict[str, str]) -> dict:
    crop = Path(row["crop"])
    points, by_channel = v4_baselines(row)
    return {
        "cohort": row["cohort"],
        "fov": row["fov"],
        "crop_name": crop.name,
        "source_crop": str(crop.resolve()),
        "source_crop_bytes": crop.stat().st_size,
        "source_crop_sha256": sha256_file(crop),
        "analysis_dir": row["analysis_dir"],
        "outcome": row["final_class"],
        "baseline_count": len(points),
        "mean_baseline_points": mean_or_none(points),
        "median_baseline_points": median_or_none(points),
        "G_baselines": len(by_channel["G"]),
        "P_baselines": len(by_channel["P"]),
        "R_baselines": len(by_channel["R"]),
        "complete_GPR": all(by_channel[channel] for channel in CHANNELS),
        "_points": points,
        **{
            f"_{channel}_points": by_channel[channel]
            for channel in CHANNELS
        },
    }


def load_v4_inventory(classification_tsv: Path) -> list[dict]:
    rows = policy_summary.load_classification(classification_tsv)
    return [v4_record(row) for row in rows]


def identity_key(row: dict) -> tuple[str, str]:
    return str(row["fov"]), str(row["crop_name"])


def unique_lookup(
    rows: list[dict],
    key,
) -> tuple[dict[object, dict], set[object]]:
    grouped: dict[object, list[dict]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    ambiguous = {
        value for value, members in grouped.items() if len(members) != 1
    }
    return {
        value: members[0]
        for value, members in grouped.items()
        if len(members) == 1
    }, ambiguous


def build_crosswalk(
    legacy: list[dict],
    v4_all: list[dict],
) -> list[dict]:
    v4_by_hash, ambiguous_hashes = unique_lookup(
        v4_all, lambda row: row["source_crop_sha256"]
    )
    v4_by_identity, ambiguous_identities = unique_lookup(v4_all, identity_key)
    used_v4: set[tuple[str, str]] = set()
    crosswalk = []
    for old in legacy:
        digest = old["source_crop_sha256"]
        key = identity_key(old)
        match = None
        confidence = "legacy_only"
        identity_match = (
            v4_by_identity.get(key)
            if key not in ambiguous_identities
            else None
        )
        if (
            identity_match
            and digest
            and digest == identity_match["source_crop_sha256"]
        ):
            match = identity_match
            confidence = "exact_pixel_hash"
        elif digest and digest not in ambiguous_hashes and digest in v4_by_hash:
            match = v4_by_hash[digest]
            confidence = "exact_pixel_hash"
        elif identity_match:
            match = identity_match
            confidence = "same_identity_different_pixels"
        if match:
            used_v4.add(identity_key(match))
        crosswalk.append(crosswalk_row(old, match, confidence))
    for current in v4_all:
        if identity_key(current) not in used_v4:
            crosswalk.append(crosswalk_row(None, current, "v4_only"))
    return crosswalk


def crosswalk_row(
    legacy: dict | None,
    current: dict | None,
    confidence: str,
) -> dict:
    return {
        "mapping_confidence": confidence,
        "fov": (legacy or current or {}).get("fov", ""),
        "crop_name": (legacy or current or {}).get("crop_name", ""),
        "legacy_task_index": (legacy or {}).get("legacy_task_index", ""),
        "legacy_source_crop": (legacy or {}).get("source_crop", ""),
        "legacy_source_crop_sha256": (
            legacy or {}
        ).get("source_crop_sha256", ""),
        "v4_source_crop": (current or {}).get("source_crop", ""),
        "v4_source_crop_sha256": (
            current or {}
        ).get("source_crop_sha256", ""),
        "legacy_outcome": (legacy or {}).get("outcome", ""),
        "v4_outcome": (current or {}).get("outcome", ""),
        "legacy_baseline_count": (legacy or {}).get("baseline_count", ""),
        "v4_baseline_count": (current or {}).get("baseline_count", ""),
        "legacy_mean_points": (
            legacy or {}
        ).get("mean_baseline_points", ""),
        "v4_mean_points": (current or {}).get("mean_baseline_points", ""),
        "legacy_G_baselines": (legacy or {}).get("G_baselines", ""),
        "v4_G_baselines": (current or {}).get("G_baselines", ""),
        "legacy_P_baselines": (legacy or {}).get("P_baselines", ""),
        "v4_P_baselines": (current or {}).get("P_baselines", ""),
        "legacy_R_baselines": (legacy or {}).get("R_baselines", ""),
        "v4_R_baselines": (current or {}).get("R_baselines", ""),
        "legacy_G_mean_points": mean_or_none(
            (legacy or {}).get("_G_points", [])
        ),
        "v4_G_mean_points": mean_or_none(
            (current or {}).get("_G_points", [])
        ),
        "legacy_P_mean_points": mean_or_none(
            (legacy or {}).get("_P_points", [])
        ),
        "v4_P_mean_points": mean_or_none(
            (current or {}).get("_P_points", [])
        ),
        "legacy_R_mean_points": mean_or_none(
            (legacy or {}).get("_R_points", [])
        ),
        "v4_R_mean_points": mean_or_none(
            (current or {}).get("_R_points", [])
        ),
        "legacy_complete_GPR": (legacy or {}).get("complete_GPR", ""),
        "v4_complete_GPR": (current or {}).get("complete_GPR", ""),
        "v4_raw_strict": (current or {}).get("raw_strict", ""),
        "v4_raw_no_badqc": (current or {}).get("raw_no_badqc", ""),
        "v4_raw_publicationlike": (
            current or {}
        ).get("raw_publicationlike", ""),
        "v4_raw_all": (current or {}).get("raw_all", ""),
    }


def summarize_records(name: str, rows: list[dict]) -> dict:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    successful = [row for row in rows if row["outcome"] == "success"]
    points = [
        int(point)
        for row in successful
        for point in row["_points"]
    ]
    return {
        "cohort": name,
        "entered_cells": len(rows),
        "success_cells": len(successful),
        "success_percent": (
            round(100 * len(successful) / len(rows), 2) if rows else None
        ),
        "no_site2_anchor": outcomes["scientific_no_site2_anchor"],
        "no_selected_baseline": sum(
            count
            for outcome, count in outcomes.items()
            if outcome.startswith("scientific_")
            and outcome != "scientific_no_site2_anchor"
        ),
        "technical_failures": sum(
            count
            for outcome, count in outcomes.items()
            if outcome.startswith("technical_")
        ),
        "baseline_count": len(points),
        "mean_baseline_points": mean_or_none(points),
        "median_baseline_points": median_or_none(points),
        "G_baselines": sum(int(row["G_baselines"]) for row in successful),
        "P_baselines": sum(int(row["P_baselines"]) for row in successful),
        "R_baselines": sum(int(row["R_baselines"]) for row in successful),
        "complete_GPR_cells": sum(
            bool(row["complete_GPR"]) for row in successful
        ),
    }


def attach_points_to_legacy(rows: list[dict]) -> None:
    for row in rows:
        analysis = Path(row["legacy_analysis_dir"])
        files = []
        for channel in CHANNELS:
            files.extend(
                (channel, path)
                for path in sorted(
                    analysis.glob(
                        f"{channel}_loci*_traj_m2DGaussian_cleaned.csv"
                    )
                )
            )
        row["_points"], by_channel = summarize_point_files(files)
        for channel in CHANNELS:
            row[f"_{channel}_points"] = by_channel[channel]


def attach_points_to_v4(
    inventory: list[dict],
    classification_tsv: Path,
) -> None:
    raw_rows = policy_summary.load_classification(classification_tsv)
    by_key = {
        (row["cohort"], row["fov"], Path(row["crop"]).name): row
        for row in raw_rows
    }
    for record in inventory:
        raw = by_key[
            (record["cohort"], record["fov"], record["crop_name"])
        ]
        record["_points"], by_channel = v4_baselines(raw)
        for channel in CHANNELS:
            record[f"_{channel}_points"] = by_channel[channel]


def selected_v4_cohorts(
    batch_root: Path,
    v4_inventory: list[dict],
) -> dict[str, list[dict]]:
    strict = [row for row in v4_inventory if row["cohort"] == "strict"]
    all_candidates = [
        row for row in v4_inventory if row["cohort"] == "all_candidates"
    ]
    all_by_key = {identity_key(row): row for row in all_candidates}
    cohorts = {"exact_strict_postprocessed": strict}
    for name, relative in policy_summary.DEFAULT_POLICY_VIEWS.items():
        selected = policy_summary.load_policy_view(batch_root / relative)
        cohorts[name] = [
            all_by_key[key]
            for key, included in selected.items()
            if included
        ]
    return cohorts


def annotate_v4_policies(batch_root: Path, v4_inventory: list[dict]) -> None:
    selections = {
        name: policy_summary.load_policy_view(batch_root / relative)
        for name, relative in policy_summary.DEFAULT_POLICY_VIEWS.items()
    }
    for row in v4_inventory:
        for name in selections:
            row[name] = ""
        if row["cohort"] != "all_candidates":
            continue
        key = identity_key(row)
        for name, selected in selections.items():
            row[name] = bool(selected[key])


def paired_summary(crosswalk: list[dict], confidence: str) -> dict:
    rows = [
        row for row in crosswalk
        if row["mapping_confidence"] == confidence
    ]
    both_success = [
        row
        for row in rows
        if row["legacy_outcome"] == "success"
        and row["v4_outcome"] == "success"
    ]
    deltas = [
        float(row["v4_mean_points"]) - float(row["legacy_mean_points"])
        for row in both_success
        if row["legacy_mean_points"] not in ("", None)
        and row["v4_mean_points"] not in ("", None)
    ]
    return {
        "mapping_set": confidence,
        "matched_inputs": len(rows),
        "both_success": len(both_success),
        "legacy_only_success": sum(
            row["legacy_outcome"] == "success"
            and row["v4_outcome"] != "success"
            for row in rows
        ),
        "v4_only_success": sum(
            row["legacy_outcome"] != "success"
            and row["v4_outcome"] == "success"
            for row in rows
        ),
        "neither_success": sum(
            row["legacy_outcome"] != "success"
            and row["v4_outcome"] != "success"
            for row in rows
        ),
        "mean_cell_level_point_delta_v4_minus_v3": mean_or_none(deltas),
        "median_cell_level_point_delta_v4_minus_v3": median_or_none(deltas),
        "v4_longer_cells": sum(delta > 0 for delta in deltas),
        "equal_length_cells": sum(delta == 0 for delta in deltas),
        "v4_shorter_cells": sum(delta < 0 for delta in deltas),
    }


def paired_channel_summary(
    crosswalk: list[dict],
    confidence: str,
    channel: str,
) -> dict:
    legacy_mean_field = f"legacy_{channel}_mean_points"
    v4_mean_field = f"v4_{channel}_mean_points"
    rows = [
        row for row in crosswalk
        if row["mapping_confidence"] == confidence
        and row["legacy_outcome"] == "success"
        and row["v4_outcome"] == "success"
        and row[legacy_mean_field] not in ("", None)
        and row[v4_mean_field] not in ("", None)
    ]
    legacy_means = [float(row[legacy_mean_field]) for row in rows]
    v4_means = [float(row[v4_mean_field]) for row in rows]
    deltas = [
        current - old
        for old, current in zip(legacy_means, v4_means)
    ]
    return {
        "mapping_set": confidence,
        "channel": channel,
        "paired_cells_with_channel": len(rows),
        "legacy_mean_of_cell_means": mean_or_none(legacy_means),
        "v4_mean_of_cell_means": mean_or_none(v4_means),
        "mean_cell_delta_v4_minus_v3": mean_or_none(deltas),
        "median_cell_delta_v4_minus_v3": median_or_none(deltas),
        "v4_longer_cells": sum(delta > 0 for delta in deltas),
        "equal_length_cells": sum(delta == 0 for delta in deltas),
        "v4_shorter_cells": sum(delta < 0 for delta in deltas),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = [field for field in rows[0] if not field.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def compare(
    legacy_task_list: Path,
    batch_root: Path,
    classification_tsv: Path,
    output_dir: Path,
    legacy_recovery_audit: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    legacy = load_legacy_inventory(legacy_task_list)
    apply_legacy_recovery_audit(legacy, legacy_recovery_audit)
    v4 = load_v4_inventory(classification_tsv)
    attach_points_to_legacy(legacy)
    attach_points_to_v4(v4, classification_tsv)
    annotate_v4_policies(batch_root, v4)
    v4_all = [row for row in v4 if row["cohort"] == "all_candidates"]
    crosswalk = build_crosswalk(legacy, v4_all)
    cohorts = selected_v4_cohorts(batch_root, v4)
    cohort_rows = [summarize_records("legacy_v3_reproduction", legacy)]
    cohort_rows.extend(
        summarize_records(name, rows)
        for name, rows in cohorts.items()
    )
    paired_rows = [
        paired_summary(crosswalk, "exact_pixel_hash"),
        paired_summary(crosswalk, "same_identity_different_pixels"),
    ]
    paired_channel_rows = [
        paired_channel_summary(crosswalk, confidence, channel)
        for confidence in (
            "exact_pixel_hash",
            "same_identity_different_pixels",
        )
        for channel in CHANNELS
    ]
    write_tsv(output_dir / "legacy_v3_input_inventory.tsv", legacy)
    write_tsv(output_dir / "current_v4_input_inventory.tsv", v4)
    write_tsv(output_dir / "v3_v4_cell_crosswalk.tsv", crosswalk)
    write_tsv(output_dir / "cohort_summary.tsv", cohort_rows)
    write_tsv(output_dir / "paired_summary.tsv", paired_rows)
    write_tsv(
        output_dir / "paired_channel_summary.tsv",
        paired_channel_rows,
    )
    summary = {
        "legacy_entered": len(legacy),
        "v4_all_entered": len(v4_all),
        "mapping_counts": dict(
            sorted(
                Counter(
                    row["mapping_confidence"] for row in crosswalk
                ).items()
            )
        ),
        "paired": paired_rows,
        "paired_channels": paired_channel_rows,
        "cohorts": cohort_rows,
        "primary_paired_set": "exact_pixel_hash",
        "sensitivity_paired_set": "same_identity_different_pixels",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "EXISTING_V3_V4_COMPARISON_OK",
        f"legacy_entered={len(legacy)}",
        f"v4_all_entered={len(v4_all)}",
    ]
    lines.extend(
        f"{name}={count}"
        for name, count in summary["mapping_counts"].items()
    )
    for row in paired_rows:
        lines.append(
            f"{row['mapping_set']}: matched={row['matched_inputs']} "
            f"both_success={row['both_success']} "
            f"legacy_only_success={row['legacy_only_success']} "
            f"v4_only_success={row['v4_only_success']}"
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-task-list", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--v4-classification", required=True, type=Path)
    parser.add_argument("--legacy-recovery-audit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compare(
        args.legacy_task_list.resolve(),
        args.batch_root.resolve(),
        args.v4_classification.resolve(),
        args.output_dir.resolve(),
        (
            args.legacy_recovery_audit.resolve()
            if args.legacy_recovery_audit
            else None
        ),
    )
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"COMPARISON_OUTPUT={args.output_dir.resolve()}")
    print(
        "COHORTS="
        + ",".join(row["cohort"] for row in summary["cohorts"])
    )


if __name__ == "__main__":
    main()
