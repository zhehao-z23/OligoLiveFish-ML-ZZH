#!/usr/bin/env python3
"""Create portable SPT analysis bundles with and without candidate TIFFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


CORE_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".txt",
    ".md",
    ".png",
    ".svg",
    ".pdf",
    ".xlsx",
    ".yaml",
    ".yml",
}
ALL_CANDIDATE_DIR = "strict_filtered_all_usam_candidates"


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def regular_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in suffixes
    )


def canonical_candidate_tiffs(batch_root: Path) -> list[Path]:
    root = batch_root / ALL_CANDIDATE_DIR
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        path
        for fov in root.iterdir()
        if fov.is_dir()
        for path in fov.glob("*.tif")
        if path.is_file() and not path.is_symlink()
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths}, key=str)


def total_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def write_file_list(
    path: Path,
    files: list[Path],
    project_root: Path,
) -> None:
    path.write_text(
        "".join(
            str(file.relative_to(project_root)) + "\n"
            for file in files
        ),
        encoding="utf-8",
    )


def add_files(
    archive: tarfile.TarFile,
    files: list[Path],
    project_root: Path,
) -> None:
    for path in files:
        archive.add(
            path,
            arcname=str(path.relative_to(project_root)),
            recursive=False,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(
    project_root: Path,
    comparison_dir: Path,
    recovery_audit: Path,
    batch_root: Path,
    policy_comparison_dir: Path,
    classification_dir: Path,
    legacy_root: Path,
    published_root: Path,
    output_dir: Path,
    code_commit: str,
    extras: list[Path],
) -> dict:
    inputs = [
        comparison_dir,
        recovery_audit,
        batch_root,
        policy_comparison_dir,
        classification_dir,
        legacy_root,
        published_root,
        *extras,
    ]
    for path in inputs:
        if not is_within(path, project_root):
            raise ValueError(f"path is outside project root: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=False)
    core_files = []
    for root in (
        comparison_dir,
        batch_root,
        policy_comparison_dir,
        classification_dir,
        legacy_root,
        published_root,
    ):
        core_files.extend(regular_files(root, CORE_SUFFIXES))
    core_files.append(recovery_audit)
    for path in extras:
        if path.is_dir():
            core_files.extend(regular_files(path, CORE_SUFFIXES))
        elif path.suffix.lower() in CORE_SUFFIXES:
            core_files.append(path)
    core_files = unique_paths(core_files)
    tiff_files = unique_paths(canonical_candidate_tiffs(batch_root))

    metadata = {
        "description": (
            "Completed legacy-v3/current-v4 SPT comparison and modeling inputs"
        ),
        "code_commit": code_commit,
        "comparison_dir": str(comparison_dir),
        "recovery_audit": str(recovery_audit),
        "batch_root": str(batch_root),
        "policy_comparison_dir": str(policy_comparison_dir),
        "classification_dir": str(classification_dir),
        "legacy_root": str(legacy_root),
        "published_root": str(published_root),
        "core_file_count": len(core_files),
        "core_uncompressed_bytes": total_bytes(core_files),
        "candidate_tiff_count": len(tiff_files),
        "candidate_tiff_uncompressed_bytes": total_bytes(tiff_files),
        "tiff_scope": (
            "Canonical top-level crop/mask TIFFs from the 340-candidate tree; "
            "raw ND2, strict duplicate TIFFs, derived single-channel TIFFs, "
            "and MATLAB MAT files are excluded."
        ),
    }
    metadata_path = output_dir / "package_manifest.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "SPT analysis download bundles\n\n"
        "analysis_core_no_tiff.tar.gz contains comparison tables, manifests, "
        "trajectory CSVs, published reference data, and QC figures.\n"
        "analysis_full_with_candidate_tiff.tar.gz contains the same core plus "
        "the canonical all-candidate crop/mask TIFFs.\n"
        "Neither archive contains raw ND2 or MATLAB MAT intermediates.\n",
        encoding="utf-8",
    )
    core_list_path = output_dir / "core_files.txt"
    tiff_list_path = output_dir / "candidate_tiff_files.txt"
    write_file_list(core_list_path, core_files, project_root)
    write_file_list(tiff_list_path, tiff_files, project_root)
    metadata_files = [metadata_path, readme_path, core_list_path, tiff_list_path]
    core_archive = output_dir / "analysis_core_no_tiff.tar.gz"
    full_archive = output_dir / "analysis_full_with_candidate_tiff.tar.gz"
    with tarfile.open(core_archive, "w:gz", compresslevel=6) as archive:
        add_files(archive, core_files + metadata_files, project_root)
    with tarfile.open(full_archive, "w:gz", compresslevel=6) as archive:
        add_files(
            archive,
            core_files + tiff_files + metadata_files,
            project_root,
        )
    checksums = output_dir / "SHA256SUMS.txt"
    checksums.write_text(
        f"{sha256_file(core_archive)}  {core_archive.name}\n"
        f"{sha256_file(full_archive)}  {full_archive.name}\n",
        encoding="utf-8",
    )
    metadata.update({
        "core_archive": str(core_archive),
        "core_archive_bytes": core_archive.stat().st_size,
        "full_archive": str(full_archive),
        "full_archive_bytes": full_archive.stat().st_size,
        "checksums": str(checksums),
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--recovery-audit", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--policy-comparison-dir", required=True, type=Path)
    parser.add_argument("--classification-dir", required=True, type=Path)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--published-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--extra", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = package(
        args.project_root.resolve(),
        args.comparison_dir.resolve(),
        args.recovery_audit.resolve(),
        args.batch_root.resolve(),
        args.policy_comparison_dir.resolve(),
        args.classification_dir.resolve(),
        args.legacy_root.resolve(),
        args.published_root.resolve(),
        args.output_dir.resolve(),
        args.code_commit,
        [path.resolve() for path in args.extra],
    )
    print("SPT_ANALYSIS_PACKAGES_OK")
    for key in (
        "core_file_count",
        "core_uncompressed_bytes",
        "candidate_tiff_count",
        "candidate_tiff_uncompressed_bytes",
        "core_archive",
        "core_archive_bytes",
        "full_archive",
        "full_archive_bytes",
        "checksums",
    ):
        print(f"{key}={metadata[key]}")


if __name__ == "__main__":
    main()
