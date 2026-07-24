import importlib.util
import sys
import unittest
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[1] / "trajectory_extraction" / "pipeline"
sys.path.insert(0, str(PIPELINE))
SPEC = importlib.util.spec_from_file_location(
    "detailed", PIPELINE / "analyze_detailed_v3_v4_spt.py"
)
detailed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(detailed)


class DetailedComparisonTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(detailed.percentile([1, 2, 3], 0.5), 2)
        self.assertAlmostEqual(detailed.percentile([0, 10], 0.95), 9.5)

    def test_condition_from_fov(self):
        self.assertEqual(
            detailed.condition_from_fov("sample_Bright_7h_0.9_14t"),
            "7h_0.9",
        )
        self.assertEqual(detailed.condition_from_fov("unknown"), "unclassified")

    def test_gpr_common_frames(self):
        rows = []
        for channel, frames in {
            "G": {1, 2, 3, 5},
            "P": {1, 2, 3, 4},
            "R": {1, 2, 3},
        }.items():
            rows.append(
                {
                    "cohort": "test",
                    "fov": "7h_0.9",
                    "condition": "7h_0.9",
                    "crop_name": "cell.tif",
                    "bundle_id": "1",
                    "channel": channel,
                    "points": len(frames),
                    "_frames": frames,
                }
            )
        result = detailed.gpr_bundles(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["common_points"], 3)
        self.assertEqual(result[0]["common_valid_steps"], 2)
        self.assertEqual(result[0]["common_temporal_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
