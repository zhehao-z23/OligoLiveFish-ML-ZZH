#!/usr/bin/env python3
"""Detailed, no-rerun comparison of completed legacy-v3 and current-v4 SPT.

This companion to ``compare_existing_v3_v4_spt.py`` adds trajectory-quality,
three-channel bundle, FOV/condition, deterministic bootstrap and published
benchmark tables.  It reads only completed CSV artifacts and never runs SPT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import compare_existing_v3_v4_spt as base


CHANNEL_LABELS = {
    "G": "Site2_A488_anchor",
    "P": "Site3_A565",
    "R": "Site4_A647",
}
QUALITY_METRICS = (
    "points",
    "frame_span",
    "temporal_coverage",
    "maximum_missing_frames",
    "valid_contiguous_steps",
    "median_step_nm",
    "p95_step_nm",
)


def read_table(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def summary_value(values: list[float], kind: str) -> float | None:
    if not values:
        return None
    if kind == "mean":
        return rounded(statistics.fmean(values))
    if kind == "median":
        return rounded(statistics.median(values))
    raise ValueError(kind)


def condition_from_fov(fov: str) -> str:
    match = re.search(r"(?P<hours>\d+)h_(?P<dose>\d+(?:\.\d+)?)", fov)
    return (
        f"{match.group('hours')}h_{match.group('dose')}"
        if match
        else "unclassified"
    )


def load_xy_frames(path: Path) -> tuple[list[int], list[float], list[float]]:
    rows = read_table(path)
    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        frame = number(normalized.get("frame", normalized.get("t")))
        x = number(normalized.get("x_nm", normalized.get("x")))
        y = number(normalized.get("y_nm", normalized.get("y")))
        if frame is None or x is None or y is None:
            continue
        frames.append(int(frame))
        xs.append(x)
        ys.append(y)
    ordered = sorted(zip(frames, xs, ys))
    return (
        [row[0] for row in ordered],
        [row[1] for row in ordered],
        [row[2] for row in ordered],
    )


def trajectory_metrics(
    *,
    cohort: str,
    source: str,
    fov: str,
    crop_name: str,
    analysis_dir: str,
    channel: str,
    bundle_id: str,
    csv_path: Path,
) -> dict:
    frames, xs, ys = load_xy_frames(csv_path)
    frame_differences = [
        right - left for left, right in zip(frames, frames[1:])
    ]
    steps = [
        math.hypot(right_x - left_x, right_y - left_y)
        for left_x, right_x, left_y, right_y in zip(
            xs, xs[1:], ys, ys[1:]
        )
    ]
    frame_span = frames[-1] - frames[0] + 1 if frames else 0
    return {
        "cohort": cohort,
        "source": source,
        "fov": fov,
        "condition": condition_from_fov(fov),
        "crop_name": crop_name,
        "cell_key": f"{fov}/{crop_name}",
        "analysis_dir": analysis_dir,
        "channel": channel,
        "biological_channel": CHANNEL_LABELS[channel],
        "bundle_id": bundle_id,
        "trajectory_csv": str(csv_path.resolve()),
        "points": len(frames),
        "first_frame": frames[0] if frames else "",
        "last_frame": frames[-1] if frames else "",
        "frame_span": frame_span,
        "temporal_coverage": rounded(len(frames) / frame_span)
        if frame_span
        else None,
        "maximum_missing_frames": max(
            (difference - 1 for difference in frame_differences), default=0
        ),
        "valid_contiguous_steps": sum(
            difference == 1 for difference in frame_differences
        ),
        "median_step_nm": rounded(statistics.median(steps)) if steps else None,
        "p95_step_nm": rounded(percentile(steps, 0.95)),
        "_frames": set(frames),
    }


def legacy_trajectories(cohort: str, record: dict) -> list[dict]:
    analysis = Path(record["legacy_analysis_dir"])
    trajectories = []
    for channel in base.CHANNELS:
        for path in sorted(
            analysis.glob(f"{channel}_loci*_traj_m2DGaussian_cleaned.csv")
        ):
            match = re.search(r"_loci(\d+)_", path.name)
            trajectories.append(
                trajectory_metrics(
                    cohort=cohort,
                    source="legacy_v3",
                    fov=str(record["fov"]),
                    crop_name=str(record["crop_name"]),
                    analysis_dir=str(analysis.resolve()),
                    channel=channel,
                    bundle_id=match.group(1) if match else path.stem,
                    csv_path=path,
                )
            )
    return trajectories


def resolve_baseline_csv(results: Path, row: dict[str, str]) -> Path:
    raw = str(row.get("baseline_csv", "")).strip()
    candidates = [
        Path(raw) if raw else Path("__missing__"),
        results / "baseline_longest" / Path(raw).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"baseline CSV not found for {raw!r} below {results}")


def v4_trajectories(cohort: str, record: dict) -> list[dict]:
    if record["outcome"] != "success":
        return []
    analysis = Path(record["analysis_dir"])
    results = analysis / base.PROFILE_RESULTS_DIR
    manifest = results / "baseline_longest" / "baseline_manifest.csv"
    trajectories = []
    for index, row in enumerate(read_table(manifest), 1):
        channel = base.channel_from_baseline(row)
        if channel not in base.CHANNELS:
            continue
        bundle_id = str(
            row.get("allele_index")
            or row.get("candidate_number")
            or index
        )
        trajectories.append(
            trajectory_metrics(
                cohort=cohort,
                source="current_v4",
                fov=str(record["fov"]),
                crop_name=str(record["crop_name"]),
                analysis_dir=str(analysis.resolve()),
                channel=channel,
                bundle_id=bundle_id,
                csv_path=resolve_baseline_csv(results, row),
            )
        )
    return trajectories


def all_trajectory_metrics(
    legacy: list[dict], cohorts: dict[str, list[dict]]
) -> list[dict]:
    rows = [
        metric
        for record in legacy
        for metric in legacy_trajectories("legacy_v3_reproduction", record)
    ]
    rows.extend(
        metric
        for cohort, records in cohorts.items()
        for record in records
        for metric in v4_trajectories(cohort, record)
    )
    return rows


def aggregate_quality(
    trajectories: list[dict], group_fields: tuple[str, ...]
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[tuple(trajectory[field] for field in group_fields)].append(
            trajectory
        )
    output = []
    for key, members in sorted(grouped.items()):
        row = dict(zip(group_fields, key))
        row["trajectory_count"] = len(members)
        row["cell_count"] = len({member["cell_key"] for member in members})
        for metric in QUALITY_METRICS:
            values = [
                float(member[metric])
                for member in members
                if member[metric] not in ("", None)
            ]
            row[f"mean_{metric}"] = summary_value(values, "mean")
            row[f"median_{metric}"] = summary_value(values, "median")
        output.append(row)
    return output


def gpr_bundles(trajectories: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for trajectory in trajectories:
        key = (
            trajectory["cohort"],
            trajectory["fov"],
            trajectory["condition"],
            trajectory["crop_name"],
            trajectory["bundle_id"],
        )
        existing = grouped[key].get(trajectory["channel"])
        if existing is None or trajectory["points"] > existing["points"]:
            grouped[key][trajectory["channel"]] = trajectory
    rows = []
    for key, channels in sorted(grouped.items()):
        if set(channels) != set(base.CHANNELS):
            continue
        common = set.intersection(
            *(channels[channel]["_frames"] for channel in base.CHANNELS)
        )
        ordered = sorted(common)
        span = ordered[-1] - ordered[0] + 1 if ordered else 0
        rows.append(
            {
                **dict(
                    zip(
                        ("cohort", "fov", "condition", "crop_name", "bundle_id"),
                        key,
                    )
                ),
                "common_points": len(ordered),
                "common_valid_steps": sum(
                    right - left == 1
                    for left, right in zip(ordered, ordered[1:])
                ),
                "common_frame_span": span,
                "common_temporal_coverage": rounded(len(ordered) / span)
                if span
                else None,
                "G_points": channels["G"]["points"],
                "P_points": channels["P"]["points"],
                "R_points": channels["R"]["points"],
            }
        )
    return rows


def gpr_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["cohort"]].append(row)
    output = []
    for cohort, members in sorted(grouped.items()):
        common = [float(row["common_points"]) for row in members]
        steps = [float(row["common_valid_steps"]) for row in members]
        coverage = [
            float(row["common_temporal_coverage"])
            for row in members
            if row["common_temporal_coverage"] is not None
        ]
        output.append(
            {
                "cohort": cohort,
                "complete_GPR_bundles": len(members),
                "complete_GPR_cells": len(
                    {(row["fov"], row["crop_name"]) for row in members}
                ),
                "mean_common_points": summary_value(common, "mean"),
                "median_common_points": summary_value(common, "median"),
                "mean_common_valid_steps": summary_value(steps, "mean"),
                "median_common_valid_steps": summary_value(steps, "median"),
                "mean_common_temporal_coverage": summary_value(coverage, "mean"),
                "common_ge_4": sum(value >= 4 for value in common),
                "common_ge_8": sum(value >= 8 for value in common),
                "common_ge_10": sum(value >= 10 for value in common),
            }
        )
    return output


def bootstrap_ci(differences: list[float], seed: int = 20260724) -> tuple:
    if not differences:
        return None, None
    rng = random.Random(seed)
    means = [
        statistics.fmean(
            rng.choice(differences) for _ in range(len(differences))
        )
        for _ in range(2000)
    ]
    return rounded(percentile(means, 0.025)), rounded(percentile(means, 0.975))


def cell_metric_lookup(trajectories: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in trajectories:
        grouped[(row["cohort"], row["cell_key"], row["channel"])].append(row)
    output = {}
    for key, rows in grouped.items():
        output[key] = {
            metric: statistics.fmean(
                float(row[metric])
                for row in rows
                if row[metric] not in ("", None)
            )
            for metric in QUALITY_METRICS
            if any(row[metric] not in ("", None) for row in rows)
        }
    return output


def paired_quality(crosswalk: list[dict], trajectories: list[dict]) -> list[dict]:
    lookup = cell_metric_lookup(trajectories)
    output = []
    for channel in base.CHANNELS:
        pairs = [
            row
            for row in crosswalk
            if row["mapping_confidence"] == "exact_pixel_hash"
            and row["legacy_outcome"] == "success"
            and row["v4_outcome"] == "success"
        ]
        for metric in QUALITY_METRICS:
            values = []
            for pair in pairs:
                key = f"{pair['fov']}/{pair['crop_name']}"
                old = lookup.get(("legacy_v3_reproduction", key, channel), {})
                new = lookup.get(("raw_all", key, channel), {})
                if metric in old and metric in new:
                    values.append((old[metric], new[metric]))
            differences = [new - old for old, new in values]
            low, high = bootstrap_ci(differences)
            output.append(
                {
                    "mapping_set": "exact_pixel_hash",
                    "channel": channel,
                    "biological_channel": CHANNEL_LABELS[channel],
                    "metric": metric,
                    "paired_cells": len(values),
                    "legacy_mean": summary_value(
                        [old for old, _ in values], "mean"
                    ),
                    "v4_mean": summary_value(
                        [new for _, new in values], "mean"
                    ),
                    "mean_delta_v4_minus_v3": summary_value(
                        differences, "mean"
                    ),
                    "median_delta_v4_minus_v3": summary_value(
                        differences, "median"
                    ),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "v4_higher": sum(value > 0 for value in differences),
                    "equal": sum(value == 0 for value in differences),
                    "v4_lower": sum(value < 0 for value in differences),
                }
            )
    return output


def paired_gpr(crosswalk: list[dict], bundles: list[dict]) -> list[dict]:
    lookup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in bundles:
        lookup[(row["cohort"], f"{row['fov']}/{row['crop_name']}")].append(row)
    output = []
    for metric in (
        "common_points",
        "common_valid_steps",
        "common_temporal_coverage",
    ):
        values = []
        for pair in crosswalk:
            if (
                pair["mapping_confidence"] != "exact_pixel_hash"
                or pair["legacy_outcome"] != "success"
                or pair["v4_outcome"] != "success"
            ):
                continue
            cell = f"{pair['fov']}/{pair['crop_name']}"
            old_rows = lookup.get(("legacy_v3_reproduction", cell), [])
            new_rows = lookup.get(("raw_all", cell), [])
            if old_rows and new_rows:
                old = statistics.fmean(float(row[metric]) for row in old_rows)
                new = statistics.fmean(float(row[metric]) for row in new_rows)
                values.append((old, new))
        differences = [new - old for old, new in values]
        low, high = bootstrap_ci(differences)
        output.append(
            {
                "mapping_set": "exact_pixel_hash",
                "metric": metric,
                "paired_cells_with_GPR": len(values),
                "legacy_mean": summary_value(
                    [old for old, _ in values], "mean"
                ),
                "v4_mean": summary_value(
                    [new for _, new in values], "mean"
                ),
                "mean_delta_v4_minus_v3": summary_value(differences, "mean"),
                "median_delta_v4_minus_v3": summary_value(
                    differences, "median"
                ),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
            }
        )
    return output


def published_metrics(paths: list[Path], output_dir: Path) -> list[dict]:
    rows = []
    snapshots = output_dir / "published_benchmark_snapshots"
    snapshots.mkdir()
    for path in paths:
        shutil.copy2(path, snapshots / path.name)
        section = ""
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("===") and line.endswith("==="):
                section = line.strip("= ")
            elif "=" in line and not line.endswith((".csv", ".tsv")):
                key, value = line.split("=", 1)
                rows.append(
                    {
                        "source_file": str(path.resolve()),
                        "section": section,
                        "metric": key.strip(),
                        "value": value.strip(),
                        "comparison_scope": (
                            "published curated, unmatched; do not interpret "
                            "as an exact-cell causal comparison"
                        ),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-task-list", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--v4-classification", required=True, type=Path)
    parser.add_argument("--legacy-recovery-audit", required=True, type=Path)
    parser.add_argument(
        "--published-summary", action="append", default=[], type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    legacy = base.load_legacy_inventory(args.legacy_task_list.resolve())
    base.apply_legacy_recovery_audit(
        legacy, args.legacy_recovery_audit.resolve()
    )
    v4 = base.load_v4_inventory(args.v4_classification.resolve())
    base.annotate_v4_policies(args.batch_root.resolve(), v4)
    cohorts = base.selected_v4_cohorts(args.batch_root.resolve(), v4)
    v4_all = [row for row in v4 if row["cohort"] == "all_candidates"]
    crosswalk = base.build_crosswalk(legacy, v4_all)

    trajectories = all_trajectory_metrics(legacy, cohorts)
    bundles = gpr_bundles(trajectories)
    quality = aggregate_quality(trajectories, ("cohort", "channel"))
    fov_quality = aggregate_quality(trajectories, ("cohort", "fov"))
    condition_quality = aggregate_quality(
        trajectories, ("cohort", "condition", "channel")
    )
    paired = paired_quality(crosswalk, trajectories)
    paired_bundle = paired_gpr(crosswalk, bundles)
    published = published_metrics(
        [path.resolve() for path in args.published_summary], output
    )

    public_trajectory_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in trajectories
    ]
    write_tsv(output / "trajectory_metrics.tsv", public_trajectory_rows)
    write_tsv(output / "cohort_channel_quality.tsv", quality)
    write_tsv(output / "fov_quality.tsv", fov_quality)
    write_tsv(output / "condition_channel_quality.tsv", condition_quality)
    write_tsv(output / "gpr_bundle_metrics.tsv", bundles)
    write_tsv(output / "gpr_bundle_summary.tsv", gpr_summary(bundles))
    write_tsv(output / "paired_quality_effects.tsv", paired)
    write_tsv(output / "paired_gpr_effects.tsv", paired_bundle)
    write_tsv(
        output / "published_benchmark_metrics.tsv",
        published,
        [
            "source_file",
            "section",
            "metric",
            "value",
            "comparison_scope",
        ],
    )

    summary = {
        "status": "DETAILED_V3_V4_SPT_COMPARISON_OK",
        "trajectory_rows": len(trajectories),
        "gpr_bundle_rows": len(bundles),
        "exact_pixel_inputs": sum(
            row["mapping_confidence"] == "exact_pixel_hash"
            for row in crosswalk
        ),
        "published_summary_files": len(args.published_summary),
        "cohorts": sorted(
            {"legacy_v3_reproduction", *cohorts}
        ),
        "primary_paired_comparison": (
            "legacy_v3 versus v4 raw_all on exact pixel-identical inputs; "
            "trajectory extraction logic differs, source TIFF is identical"
        ),
        "overall_comparison": (
            "all entered cells with explicit denominators; cohorts differ in "
            "segmentation/QC selection and are not paired"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "\n".join(
            [
                summary["status"],
                f"trajectory_rows={len(trajectories)}",
                f"gpr_bundle_rows={len(bundles)}",
                f"exact_pixel_inputs={summary['exact_pixel_inputs']}",
                "",
                "Primary inference:",
                summary["primary_paired_comparison"],
                "",
                "Overall/sensitivity inference:",
                summary["overall_comparison"],
                "",
                "Channel mapping:",
                "G=Site2/A488 anchor; P=Site3/A565; R=Site4/A647.",
                "",
                "Published summaries are curated unmatched references and are "
                "stored separately from the exact-cell comparison.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print((output / "README.txt").read_text(encoding="utf-8"))
    print(f"DETAILED_OUTPUT={output}")


if __name__ == "__main__":
    main()
