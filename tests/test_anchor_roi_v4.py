from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "trajectory_extraction" / "pipeline"
sys.path.insert(0, str(PIPELINE))

import align_microsam_mask
import max_step_model
import run_anchor_roi_spt


class MaxStepModelTests(unittest.TestCase):
    def test_fov15_metadata_reproduces_approved_radius(self):
        metadata = {
            "frame_interval_s": 1.0114487409591675,
            "pixel_size_x_um_per_px": 0.10833333604166673,
        }
        result = max_step_model.derive_from_metadata(metadata)
        self.assertTrue(
            math.isclose(
                result["calculation"]["theoretical_radius_px"],
                2.7268931949549495,
                rel_tol=1e-12,
            )
        )
        self.assertEqual(result["modeled_max_step_px"], 2.75)
        self.assertEqual(result["operational_source"], "metadata + physical prior + upward rounding")
        self.assertFalse(result["tracker_implementation"]["gap_scaled_radius_implemented"])

    def test_explicit_override_is_audited(self):
        metadata = {"frame_interval_s": 1.0, "pixel_size_x_um_per_px": 0.1}
        result = max_step_model.derive_from_metadata(metadata, explicit_max_step_px=3.0)
        self.assertEqual(result["operational_max_step_px"], 3.0)
        self.assertEqual(result["operational_source"], "explicit CLI override")


class StaticRoiTests(unittest.TestCase):
    def test_complete_anchor_path_becomes_one_static_mask(self):
        support = np.ones((40, 40), dtype=bool)
        anchor = [(1, 10.0, 10.0), (2, 12.0, 12.0), (3, 15.0, 12.0)]
        roi = run_anchor_roi_spt.static_anchor_union(anchor, support, 5)
        self.assertTrue(roi[10, 10])
        self.assertTrue(roi[12, 15])
        self.assertEqual(run_anchor_roi_spt.ndimage.label(roi)[1], 1)

    def test_dilation_smaller_than_gaussian_support_is_rejected(self):
        with self.assertRaises(ValueError):
            run_anchor_roi_spt.static_anchor_union(
                [(1, 5.0, 5.0)], np.ones((10, 10), dtype=bool), 4
            )


class MaskAssociationTests(unittest.TestCase):
    def test_sidecar_resolves_exact_relative_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "fov_1.tif"
            mask = root / "fov_mask_1.tif"
            crop.touch()
            mask.touch()
            crop.with_name("fov_1_metadata.json").write_text(
                json.dumps({"microsam_mask": {"relative_path": mask.name}}),
                encoding="utf-8",
            )
            self.assertEqual(align_microsam_mask.discover_microsam_mask(crop), mask.resolve())


class BaselineSelectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(path: Path, allele: int, locus: int, channel: str, number: int, points: int, span: int, first: int) -> dict:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "x_nm", "y_nm"])
            writer.writerow([first, 1.0, 2.0])
        return {
            "allele_index": allele,
            "anchor_locus": locus,
            "channel": channel,
            "marker": run_anchor_roi_spt.MARKER[channel],
            "candidate_number": number,
            "candidate_csv": str(path),
            "points": points,
            "first_frame": first,
            "last_frame": first + span - 1,
            "frame_span": span,
            "temporal_coverage_fraction": points / span,
            "maximum_missing_frames_between_points": 0,
            "median_step_px": 0.0,
            "p95_step_px": 0.0,
            "inside_static_roi_fraction": 1.0,
        }

    def test_longest_rule_and_no_candidate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self._candidate(root / "p1.csv", 1, 2, "P", 1, 10, 12, 2),
                self._candidate(root / "p2.csv", 1, 2, "P", 2, 10, 13, 3),
                self._candidate(root / "p3.csv", 1, 2, "P", 3, 9, 20, 1),
            ]
            selected, audit = run_anchor_roi_spt.select_longest_baselines(
                rows, root / "baseline", [(1, 2), (2, 5)]
            )
            self.assertEqual(len(selected), 1)
            self.assertTrue(selected[0]["candidate_csv"].endswith("p2.csv"))
            self.assertTrue(selected[0]["baseline_csv"].endswith("_cleaned.csv"))
            self.assertEqual(len(audit), 6)
            allele2 = [row for row in audit if row["allele_index"] == 2]
            self.assertEqual(len(allele2), 3)
            self.assertTrue(all(row["candidate_count"] == 0 for row in allele2))


if __name__ == "__main__":
    unittest.main()
