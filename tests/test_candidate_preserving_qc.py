import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from nucleus_segmentation.crop_nuclei_sam import write_raw_candidate_archive
from nucleus_segmentation.save_crops import discover_crop_json_files


class CandidateArchiveTests(unittest.TestCase):
    def test_normal_export_discovers_default_sibling_candidate_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "strict"
            candidate_root = Path(tmp) / "strict_all_usam_candidates"
            (root / "fov").mkdir(parents=True)
            (candidate_root / "fov").mkdir(parents=True)
            strict_json = root / "fov" / "strict_crops.json"
            candidate_json = candidate_root / "fov" / "candidate_crops.json"
            strict_json.write_text("{}", encoding="utf-8")
            candidate_json.write_text("{}", encoding="utf-8")

            self.assertEqual(
                discover_crop_json_files(root),
                sorted([strict_json, candidate_json]),
            )
            self.assertEqual(
                discover_crop_json_files(root, include_sibling_candidates=False),
                [strict_json],
            )

    def test_every_raw_usam_mask_is_preserved_and_annotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_root = root / "candidate_archive"
            nd2_path = root / "example.nd2"

            large = np.zeros((12, 12), dtype=bool)
            large[3:6, 3:6] = True
            small = np.zeros((12, 12), dtype=bool)
            small[9, 9] = True
            nucleus_average = np.full((12, 12), 10.0, dtype=np.float32)
            nucleus_average[large] = 100.0
            nucleus_average[small] = 100.0

            args = SimpleNamespace(
                candidate_output_root=str(candidate_root),
                min_area=5,
                max_area=100,
                border_margin=0,
                mask_border_margin=-1,
                margin=1,
            )
            archive_dir = write_raw_candidate_archive(
                nd2_path,
                [large, small],
                nucleus_average,
                (3, 1, 1, 12, 12),
                args,
            )

            manifest = archive_dir / "candidate_selection_manifest.csv"
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["raw_usam_label"] for row in rows}, {"1", "2"})
            self.assertEqual(sum(row["exclusion_reasons"] == "too_small" for row in rows), 1)

            crop_json = archive_dir / "example_candidate_crops.json"
            data = json.loads(crop_json.read_text(encoding="utf-8"))
            self.assertEqual(data["archive_kind"], "raw_usam_candidates")
            self.assertEqual(data["source_instance_count"], 2)
            self.assertEqual(len(data["crops"]), 2)

            masks = sorted(archive_dir.glob("example_candidate_mask_*.tif"))
            self.assertEqual(len(masks), 2)
            for path in masks:
                self.assertEqual(tifffile.imread(path).shape[0], 3)


if __name__ == "__main__":
    unittest.main()
