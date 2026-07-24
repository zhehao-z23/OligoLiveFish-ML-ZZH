import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PIPELINE = (
    Path(__file__).resolve().parents[1]
    / "trajectory_extraction"
    / "pipeline"
)
sys.path.insert(0, str(PIPELINE))

from package_spt_analysis_results import package  # noqa: E402


class SptAnalysisPackagingTests(unittest.TestCase):
    def test_core_excludes_tiffs_and_full_adds_only_canonical_tiffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            comparison = project / "manifests" / "comparison"
            policy = project / "manifests" / "policy"
            classification = project / "manifests" / "classification"
            batch = project / "work" / "batch"
            legacy = project / "work" / "legacy"
            published = project / "incoming" / "published"
            recovery = project / "manifests" / "recovery.csv"
            for path in (
                comparison,
                policy,
                classification,
                legacy,
                published,
            ):
                path.mkdir(parents=True)
                (path / "result.csv").write_text("x\n1\n", encoding="utf-8")
            recovery.write_text("status\nok\n", encoding="utf-8")
            candidate_fov = (
                batch / "strict_filtered_all_usam_candidates" / "FOV_A"
            )
            candidate_fov.mkdir(parents=True)
            crop = candidate_fov / "cell_1.tif"
            crop.write_bytes(b"crop")
            derived = candidate_fov / "cell_1"
            derived.mkdir()
            (derived / "cell_1_green.tif").write_bytes(b"derived")
            (derived / "baseline.csv").write_text(
                "points\n10\n",
                encoding="utf-8",
            )
            output = project / "packages" / "result"

            metadata = package(
                project,
                comparison,
                recovery,
                batch,
                policy,
                classification,
                legacy,
                published,
                output,
                "abc123",
                [],
            )

            self.assertEqual(metadata["candidate_tiff_count"], 1)
            with tarfile.open(
                metadata["core_archive"], "r:gz"
            ) as archive:
                core_names = set(archive.getnames())
            with tarfile.open(
                metadata["full_archive"], "r:gz"
            ) as archive:
                full_names = set(archive.getnames())
            crop_name = str(crop.relative_to(project))
            derived_name = str(
                (derived / "cell_1_green.tif").relative_to(project)
            )
            self.assertNotIn(crop_name, core_names)
            self.assertIn(crop_name, full_names)
            self.assertNotIn(derived_name, full_names)
            manifest = json.loads(
                (output / "package_manifest.json").read_text()
            )
            self.assertEqual(manifest["code_commit"], "abc123")


if __name__ == "__main__":
    unittest.main()
