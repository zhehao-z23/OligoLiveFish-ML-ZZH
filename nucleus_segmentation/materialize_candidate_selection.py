#!/usr/bin/env python3
"""Materialize an analysis-safe view from preserved micro-SAM candidates.

Edit ``manual_decision`` in each ``candidate_selection_manifest.csv`` using
``include`` or ``exclude``. A blank value is evaluated using the selected QC
policy. This command creates a new tree containing only selected crop/mask
symlinks; the candidate archive is never modified or moved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y", "include", "included", "accept", "accepted"}
FALSE_VALUES = {"0", "false", "no", "n", "exclude", "excluded", "reject", "rejected"}


def parse_bool(value: str, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"invalid {field} value: {value!r}")


POLICY_IGNORED_REASONS = {
    "strict": set(),
    "no_badqc": {"bad_qc"},
    "publicationlike": {"bad_qc", "mask_border"},
    "all": {"too_small", "too_large", "centroid_border", "mask_border", "bad_qc"},
}


def effective_selection(row: dict[str, str], policy: str = "strict") -> tuple[bool, str]:
    manual = row.get("manual_decision", "").strip()
    if manual:
        return parse_bool(manual, field="manual_decision"), "manual"
    if policy not in POLICY_IGNORED_REASONS:
        raise ValueError(f"unknown selection policy: {policy}")
    if policy == "strict":
        return (
            parse_bool(row["default_gate_pass"], field="default_gate_pass"),
            "policy:strict",
        )
    if policy == "all":
        return True, "policy:all"
    reasons = {
        reason.strip()
        for reason in row.get("exclusion_reasons", "").split(";")
        if reason.strip()
    }
    blocking = reasons - POLICY_IGNORED_REASONS[policy]
    return not blocking, f"policy:{policy}"


def relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def materialize_manifest(
    manifest_path: Path,
    archive_root: Path,
    output_root: Path,
    policy: str = "strict",
) -> dict:
    relative_fov = manifest_path.parent.relative_to(archive_root)
    included_dir = output_root / "spt_included" / relative_fov
    excluded_dir = output_root / "spt_excluded" / relative_fov
    manifest_dir = output_root / "manifests" / relative_fov
    included_dir.mkdir(parents=True, exist_ok=False)
    excluded_dir.mkdir(parents=True, exist_ok=False)
    manifest_dir.mkdir(parents=True, exist_ok=False)

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"candidate manifest has no rows: {manifest_path}")

    audited_rows = []
    selected_count = 0
    for row in rows:
        selected, source = effective_selection(row, policy)
        row = dict(row)
        row["effective_selected"] = str(selected)
        row["selection_source"] = source
        destination = included_dir if selected else excluded_dir

        crop = manifest_path.parent / row["crop_tiff"]
        mask = manifest_path.parent / row["mask_tiff"]
        if not crop.is_file():
            raise FileNotFoundError(
                f"candidate crop has not been exported; run save_crops.py first: {crop}"
            )
        if not mask.is_file():
            raise FileNotFoundError(f"candidate mask is missing: {mask}")

        relative_symlink(crop.resolve(), destination / crop.name)
        relative_symlink(mask.resolve(), destination / mask.name)
        sidecar = crop.with_name(crop.stem + "_metadata.json")
        if sidecar.is_file():
            relative_symlink(sidecar.resolve(), destination / sidecar.name)
        audited_rows.append(row)
        selected_count += int(selected)

    fields = list(rows[0]) + ["effective_selected", "selection_source"]
    with (manifest_dir / "selection_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audited_rows)

    summary = {
        "source_manifest": str(manifest_path.resolve()),
        "policy": policy,
        "source_candidates": len(rows),
        "selected_candidates": selected_count,
        "excluded_candidates": len(rows) - selected_count,
        "included_directory": str(included_dir.resolve()),
        "excluded_directory": str(excluded_dir.resolve()),
    }
    (manifest_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--policy",
        choices=sorted(POLICY_IGNORED_REASONS),
        default="strict",
        help=(
            "QC policy used when manual_decision is blank: strict, no_badqc, "
            "publicationlike, or all (default: strict)."
        ),
    )
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    output_root = args.output_root.resolve()
    if not candidate_root.is_dir():
        parser.error(f"candidate root does not exist: {candidate_root}")
    if output_root.exists():
        parser.error(
            f"output root already exists: {output_root}; use a new selection label"
        )
    if candidate_root == output_root or candidate_root in output_root.parents:
        parser.error("output root must be outside the candidate archive")

    manifests = sorted(candidate_root.rglob("candidate_selection_manifest.csv"))
    if not manifests:
        parser.error(f"no candidate_selection_manifest.csv under {candidate_root}")

    output_root.mkdir(parents=True)
    summaries = [
        materialize_manifest(path, candidate_root, output_root, args.policy)
        for path in manifests
    ]
    source_total = sum(item["source_candidates"] for item in summaries)
    selected_total = sum(item["selected_candidates"] for item in summaries)
    excluded_total = sum(item["excluded_candidates"] for item in summaries)
    batch_summary = {
        "candidate_root": str(candidate_root),
        "output_root": str(output_root),
        "policy": args.policy,
        "fov_count": len(summaries),
        "source_candidates": source_total,
        "selected_candidates": selected_total,
        "excluded_candidates": excluded_total,
    }
    (output_root / "selection_batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2), encoding="utf-8"
    )
    print(
        f"SELECTION_VIEW_OK fovs={len(summaries)} "
        f"policy={args.policy} "
        f"source={source_total} selected={selected_total} "
        f"excluded={excluded_total} output={output_root}"
    )


if __name__ == "__main__":
    main()
