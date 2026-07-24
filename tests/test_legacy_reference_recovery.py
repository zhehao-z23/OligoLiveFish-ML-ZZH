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

from recover_legacy_v3_reference_matching import recover  # noqa: E402


class LegacyReferenceRecoveryTests(unittest.TestCase):
    def test_references_are_copied_from_resolved_source_before_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "cell_1"
            source.mkdir(parents=True)
            (source / "cell_1_green.tif").write_bytes(b"green")
            (source / "G_loci1_traj_rela2wholeimg.csv").write_text(
                "frame,x_nm,y_nm\n1,1,1\n",
                encoding="utf-8",
            )
            task = root / "task"
            task.mkdir()
            (task / "cell_1_green.tif").symlink_to(
                source / "cell_1_green.tif"
            )
            candidates = task / "matlab_result" / "matlab_trajectory"
            candidates.mkdir(parents=True)
            (
                candidates / "G_m2DGaussian_traj1.csv"
            ).write_text("frame,x_nm,y_nm\n1,1,1\n", encoding="utf-8")
            matcher = root / "matcher.py"
            matcher.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "root = Path(sys.argv[1])\n"
                "assert (root / "
                "'G_loci1_traj_rela2wholeimg.csv').is_file()\n"
                "(root / "
                "'G_loci1_traj_m2DGaussian_cleaned.csv').write_text("
                "'frame,x_nm,y_nm\\\\n1,1,1\\\\n')\n",
                encoding="utf-8",
            )
            task_list = root / "tasks.txt"
            task_list.write_text(str(task) + "\n", encoding="utf-8")
            audit = root / "audit.csv"

            rows = recover(task_list, matcher, audit)

            self.assertEqual(rows[0]["status"], "recovered")
            self.assertEqual(rows[0]["reference_count"], 1)
            self.assertEqual(rows[0]["candidate_count"], 1)
            self.assertEqual(rows[0]["cleaned_count"], 1)
            with audit.open(newline="", encoding="utf-8") as handle:
                audit_rows = list(csv.DictReader(handle))
            self.assertEqual(audit_rows[0]["status"], "recovered")


if __name__ == "__main__":
    unittest.main()
