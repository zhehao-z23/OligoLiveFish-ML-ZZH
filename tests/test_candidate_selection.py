import csv
import tempfile
import unittest
from pathlib import Path

from nucleus_segmentation.materialize_candidate_selection import (
    effective_selection,
    materialize_manifest,
)


class CandidateSelectionTests(unittest.TestCase):
    def test_manual_decision_overrides_default_gate(self):
        self.assertEqual(
            effective_selection({"manual_decision": "include", "default_gate_pass": "False"}),
            (True, "manual"),
        )
        self.assertEqual(
            effective_selection({"manual_decision": "exclude", "default_gate_pass": "True"}),
            (False, "manual"),
        )
        self.assertEqual(
            effective_selection({"manual_decision": "", "default_gate_pass": "True"}),
            (True, "default_qc"),
        )

    def test_materialized_view_contains_only_effective_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive"
            fov = archive / "fov_a"
            fov.mkdir(parents=True)
            output = root / "selected"

            rows = []
            decisions = [
                ("True", ""),
                ("False", "include"),
                ("True", "exclude"),
            ]
            for idx, (default, manual) in enumerate(decisions, start=1):
                crop_name = f"fov_candidate_{idx}.tif"
                mask_name = f"fov_candidate_mask_{idx}.tif"
                (fov / crop_name).write_bytes(b"crop")
                (fov / mask_name).write_bytes(b"mask")
                rows.append({
                    "candidate_id": f"candidate_{idx:03d}",
                    "crop_tiff": crop_name,
                    "mask_tiff": mask_name,
                    "default_gate_pass": default,
                    "manual_decision": manual,
                })

            manifest = fov / "candidate_selection_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            summary = materialize_manifest(manifest, archive, output)
            self.assertEqual(summary["source_candidates"], 3)
            self.assertEqual(summary["selected_candidates"], 2)
            selected_fov = output / "spt_included" / "fov_a"
            excluded_fov = output / "spt_excluded" / "fov_a"
            self.assertTrue((selected_fov / "fov_candidate_1.tif").is_symlink())
            self.assertTrue((selected_fov / "fov_candidate_2.tif").is_symlink())
            self.assertFalse((selected_fov / "fov_candidate_3.tif").exists())
            self.assertTrue((excluded_fov / "fov_candidate_3.tif").is_symlink())
            self.assertFalse((excluded_fov / "fov_candidate_1.tif").exists())


if __name__ == "__main__":
    unittest.main()
