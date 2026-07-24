import csv
import tempfile
import unittest
from pathlib import Path

from trajectory_extraction.pipeline.summarize_candidate_policy_spt import (
    DEFAULT_POLICY_VIEWS,
    compare_policies,
)


class CandidatePolicySummaryTests(unittest.TestCase):
    def test_policy_views_subset_one_all_candidate_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "batch"
            analysis = root / "analysis"
            classification = root / "classification.tsv"
            output = root / "output"

            candidates = [
                ("fov_a", "candidate_1.tif", "success", True),
                (
                    "fov_a",
                    "candidate_2.tif",
                    "scientific_no_site2_anchor",
                    False,
                ),
            ]
            for relative in DEFAULT_POLICY_VIEWS.values():
                manifest = (
                    base
                    / relative
                    / "manifests"
                    / "fov_a"
                    / "selection_manifest.csv"
                )
                manifest.parent.mkdir(parents=True)
                with manifest.open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["crop_tiff", "effective_selected"],
                    )
                    writer.writeheader()
                    for _, crop, _, selected in candidates:
                        writer.writerow({
                            "crop_tiff": crop,
                            "effective_selected": (
                                "True"
                                if relative == "selection_all_candidates_v2"
                                or selected
                                else "False"
                            ),
                        })

            successful_dir = analysis / "successful"
            baseline = (
                successful_dir
                / "anchor_roi_v4_chr3_sites_2_3_4"
                / "baseline_longest"
                / "baseline_manifest.csv"
            )
            baseline.parent.mkdir(parents=True)
            baseline.write_text("points\n10\n12\n", encoding="utf-8")

            strict_dir = analysis / "strict"
            strict_baseline = (
                strict_dir
                / "anchor_roi_v4_chr3_sites_2_3_4"
                / "baseline_longest"
                / "baseline_manifest.csv"
            )
            strict_baseline.parent.mkdir(parents=True)
            strict_baseline.write_text("points\n8\n", encoding="utf-8")

            with classification.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fields = [
                    "cohort",
                    "fov",
                    "crop",
                    "analysis_dir",
                    "final_class",
                ]
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields,
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({
                    "cohort": "strict",
                    "fov": "strict_fov",
                    "crop": "strict_1.tif",
                    "analysis_dir": strict_dir,
                    "final_class": "success",
                })
                for fov, crop, final_class, _ in candidates:
                    writer.writerow({
                        "cohort": "all_candidates",
                        "fov": fov,
                        "crop": crop,
                        "analysis_dir": successful_dir,
                        "final_class": final_class,
                    })

            summaries = compare_policies(base, classification, output)
            by_name = {
                row["cohort"]: row
                for row in summaries
            }
            self.assertEqual(
                by_name["exact_strict_postprocessed"]["baseline_count"],
                1,
            )
            self.assertEqual(
                by_name["raw_strict"]["selected_cells"],
                1,
            )
            self.assertEqual(
                by_name["raw_all"]["selected_cells"],
                2,
            )
            self.assertEqual(
                by_name["raw_all"]["no_site2_anchor"],
                1,
            )
            self.assertEqual(
                by_name["raw_all"]["mean_baseline_points"],
                11.0,
            )


if __name__ == "__main__":
    unittest.main()
