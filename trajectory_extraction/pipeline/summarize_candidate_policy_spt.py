#!/usr/bin/env python3
"""Compare candidate QC policies using one completed all-candidate SPT run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


PROFILE_RESULTS_DIR = "anchor_roi_v4_chr3_sites_2_3_4"
DEFAULT_POLICY_VIEWS = {
    "raw_strict": "selection_strict_candidates_v2",
    "raw_no_badqc": "selection_no_badqc_v2",
    "raw_publicationlike": "selection_publicationlike_v2",
    "raw_all": "selection_all_candidates_v2",
}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected True or False, found {value!r}")


def load_classification(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "cohort",
        "fov",
        "crop",
        "analysis_dir",
        "final_class",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"classification TSV lacks required fields: {path}")
    for row in rows:
        row["crop_name"] = Path(row["crop"]).name
    return rows


def load_policy_view(root: Path) -> dict[tuple[str, str], bool]:
    manifests = sorted(root.glob("manifests/*/selection_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"no selection manifests under {root}")
    selected: dict[tuple[str, str], bool] = {}
    for path in manifests:
        fov = path.parent.name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (fov, row["crop_tiff"])
                if key in selected:
                    raise ValueError(f"duplicate candidate key in {root}: {key}")
                selected[key] = parse_bool(row["effective_selected"])
    return selected


def baseline_points(row: dict[str, str]) -> list[float]:
    if row["final_class"] != "success":
        return []
    path = (
        Path(row["analysis_dir"])
        / PROFILE_RESULTS_DIR
        / "baseline_longest"
        / "baseline_manifest.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"successful cell lacks baseline manifest: {path}"
        )
    values = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for baseline in csv.DictReader(handle):
            value = baseline.get("points", "").strip()
            if value:
                values.append(float(value))
    return values


def summarize_group(name: str, rows: list[dict[str, str]]) -> dict:
    classes = Counter(row["final_class"] for row in rows)
    points = [
        point
        for row in rows
        if row["final_class"] == "success"
        for point in baseline_points(row)
    ]
    return {
        "cohort": name,
        "selected_cells": len(rows),
        "success_cells": classes["success"],
        "no_site2_anchor": classes["scientific_no_site2_anchor"],
        "no_selected_baseline": classes["scientific_no_selected_baseline"],
        "technical_failures": sum(
            count for key, count in classes.items() if key.startswith("technical_")
        ),
        "success_percent": round(100 * classes["success"] / len(rows), 2),
        "baseline_count": len(points),
        "mean_baseline_points": (
            round(statistics.fmean(points), 4) if points else None
        ),
        "median_baseline_points": (
            round(statistics.median(points), 2) if points else None
        ),
        "minimum_baseline_points": min(points) if points else None,
        "maximum_baseline_points": max(points) if points else None,
    }


def compare_policies(
    base: Path,
    classification_tsv: Path,
    output_dir: Path,
) -> list[dict]:
    records = load_classification(classification_tsv)
    strict_rows = [row for row in records if row["cohort"] == "strict"]
    all_rows = [row for row in records if row["cohort"] == "all_candidates"]
    all_by_key = {
        (row["fov"], row["crop_name"]): row
        for row in all_rows
    }
    if len(all_by_key) != len(all_rows):
        raise ValueError("all-candidate classification contains duplicate keys")

    policies = {
        name: load_policy_view(base / relative)
        for name, relative in DEFAULT_POLICY_VIEWS.items()
    }
    expected_keys = set(all_by_key)
    for name, selected in policies.items():
        if set(selected) != expected_keys:
            missing = len(expected_keys - set(selected))
            extra = len(set(selected) - expected_keys)
            raise ValueError(
                f"policy {name} and all-candidate SPT differ: "
                f"missing={missing}, extra={extra}"
            )

    groups = {"exact_strict_postprocessed": strict_rows}
    for name, selected in policies.items():
        groups[name] = [
            all_by_key[key]
            for key, include in selected.items()
            if include
        ]
    summaries = [
        summarize_group(name, rows)
        for name, rows in groups.items()
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "policy_comparison.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "policy_comparison.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    status_fields = [
        "fov",
        "crop_name",
        "final_class",
        "analysis_dir",
        *DEFAULT_POLICY_VIEWS,
    ]
    with (output_dir / "raw_candidate_policy_status.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=status_fields,
            delimiter="\t",
        )
        writer.writeheader()
        for key in sorted(all_by_key):
            row = all_by_key[key]
            writer.writerow({
                "fov": key[0],
                "crop_name": key[1],
                "final_class": row["final_class"],
                "analysis_dir": row["analysis_dir"],
                **{
                    name: selected[key]
                    for name, selected in policies.items()
                },
            })
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("classification_tsv", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = compare_policies(
        args.batch_root.resolve(),
        args.classification_tsv.resolve(),
        args.output_dir.resolve(),
    )
    fields = [
        "cohort",
        "selected_cells",
        "success_cells",
        "no_site2_anchor",
        "no_selected_baseline",
        "technical_failures",
        "success_percent",
        "baseline_count",
        "mean_baseline_points",
        "median_baseline_points",
    ]
    print("\t".join(fields))
    for row in summaries:
        print("\t".join(str(row[field]) for field in fields))
    print(f"POLICY_COMPARISON_OK={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
