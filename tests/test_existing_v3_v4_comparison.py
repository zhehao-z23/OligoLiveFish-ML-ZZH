import csv
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE = (
    Path(__file__).resolve().parents[1]
    / "trajectory_extraction"
    / "pipeline"
)
sys.path.insert(0, str(PIPELINE))

from compare_existing_v3_v4_spt import compare  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_track(path: Path, points: int) -> None:
    write_csv(
        path,
        ["frame", "x_nm", "y_nm"],
        [
            {"frame": index, "x_nm": index, "y_nm": index}
            for index in range(points)
        ],
    )


class ExistingV3V4ComparisonTests(unittest.TestCase):
    def test_exact_and_identity_only_matches_are_separated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_source = root / "legacy_source" / "FOV_A"
            current_source = root / "current_source" / "FOV_A"
            tasks_root = root / "legacy_tasks"
            legacy_source.mkdir(parents=True)
            current_source.mkdir(parents=True)
            tasks_root.mkdir()

            task_paths = []
            for index, name in enumerate(("cell_1", "cell_2"), 1):
                source_cell = legacy_source / name
                source_cell.mkdir()
                source_crop = legacy_source / f"{name}.tif"
                source_crop.write_bytes(
                    b"same pixels" if index == 1 else b"legacy pixels"
                )
                for channel in ("green", "red", "purple"):
                    (source_cell / f"{name}_{channel}.tif").write_bytes(
                        channel.encode()
                    )
                task = tasks_root / f"{index:04d}_{name}"
                task.mkdir()
                (task / f"{name}_green.tif").symlink_to(
                    source_cell / f"{name}_green.tif"
                )
                if index == 1:
                    for channel, points in zip(("G", "P", "R"), (8, 7, 6)):
                        write_track(
                            task
                            / (
                                f"{channel}_loci1_"
                                "traj_m2DGaussian_cleaned.csv"
                            ),
                            points,
                        )
                (task / "matlab_result" / "matlab_trajectory").mkdir(
                    parents=True
                )
                task_paths.append(task)

            task_list = root / "legacy_tasks.txt"
            task_list.write_text(
                "\n".join(str(path) for path in task_paths) + "\n",
                encoding="utf-8",
            )

            classifications = []
            for index, name in enumerate(("cell_1", "cell_2"), 1):
                crop = current_source / f"{name}.tif"
                crop.write_bytes(
                    b"same pixels" if index == 1 else b"current pixels"
                )
                analysis = current_source / name
                analysis.mkdir()
                outcome = (
                    "success"
                    if index == 1
                    else "scientific_no_site2_anchor"
                )
                row = {
                    "cohort": "all_candidates",
                    "fov": "FOV_A",
                    "crop": str(crop),
                    "analysis_dir": str(analysis),
                    "final_class": outcome,
                }
                classifications.append(row)
                if index == 1:
                    classifications.append({**row, "cohort": "strict"})
                    manifest = (
                        analysis
                        / "anchor_roi_v4_chr3_sites_2_3_4"
                        / "baseline_longest"
                        / "baseline_manifest.csv"
                    )
                    write_csv(
                        manifest,
                        ["channel", "marker", "points"],
                        [
                            {"channel": "G", "marker": "Site2", "points": 9},
                            {"channel": "P", "marker": "Site3", "points": 8},
                            {"channel": "R", "marker": "Site4", "points": 7},
                        ],
                    )

            classification = root / "classification.tsv"
            with classification.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(classifications[0]),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(classifications)

            batch_root = root / "batch"
            selections = {
                "selection_strict_candidates_v2": (True, False),
                "selection_no_badqc_v2": (True, True),
                "selection_publicationlike_v2": (True, True),
                "selection_all_candidates_v2": (True, True),
            }
            for view, included in selections.items():
                write_csv(
                    batch_root
                    / view
                    / "manifests"
                    / "FOV_A"
                    / "selection_manifest.csv",
                    ["crop_tiff", "effective_selected"],
                    [
                        {
                            "crop_tiff": f"cell_{index}.tif",
                            "effective_selected": value,
                        }
                        for index, value in enumerate(included, 1)
                    ],
                )

            output = root / "comparison"
            summary = compare(
                task_list,
                batch_root,
                classification,
                output,
            )

            self.assertEqual(summary["legacy_entered"], 2)
            self.assertEqual(summary["v4_all_entered"], 2)
            self.assertEqual(summary["mapping_counts"]["exact_pixel_hash"], 1)
            self.assertEqual(
                summary["mapping_counts"][
                    "same_identity_different_pixels"
                ],
                1,
            )
            exact = summary["paired"][0]
            self.assertEqual(exact["both_success"], 1)
            self.assertEqual(
                exact["mean_cell_level_point_delta_v4_minus_v3"],
                1.0,
            )
            with (output / "v3_v4_cell_crosswalk.tsv").open(
                newline="", encoding="utf-8"
            ) as handle:
                crosswalk = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(crosswalk[0]["v4_raw_strict"], "True")
            self.assertEqual(crosswalk[1]["v4_raw_strict"], "False")
            self.assertTrue((output / "cohort_summary.tsv").is_file())
            self.assertTrue((output / "v3_v4_cell_crosswalk.tsv").is_file())
            self.assertTrue(
                (output / "paired_channel_summary.tsv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
