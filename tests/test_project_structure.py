import csv
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTest(unittest.TestCase):
    def test_required_roots_exist(self):
        for name in ("acoustic", "baseline", "model", "experiments", "dataset", "tests", "server", "docs"):
            self.assertTrue((ROOT / name).is_dir(), name)

    def test_dataset_top_level_is_stable(self):
        children = {path.name for path in (ROOT / "dataset").iterdir() if path.is_dir()}
        self.assertEqual(children, {"raw", "processed", "script"})

    def test_retired_roots_are_not_tracked(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "codex", "tmp", "results"], cwd=ROOT, text=True
        ).splitlines()
        self.assertEqual(tracked, [])

    def test_experiment_index_is_minimal_and_resolves(self):
        index_path = ROOT / "experiments" / "index.csv"
        with index_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            list(rows[0]),
            [
                "experiment_id",
                "name",
                "owner",
                "status",
                "code_commit",
                "definition_path",
                "result_path",
                "depends_on",
                "created_at",
                "updated_at",
            ],
        )
        self.assertEqual(len({row["experiment_id"] for row in rows}), len(rows))
        for row in rows:
            self.assertTrue((ROOT / row["definition_path"]).is_file(), row["definition_path"])
            self.assertTrue(row["result_path"].startswith("result/"), row["result_path"])

    def test_model_is_reserved(self):
        self.assertTrue((ROOT / "model" / "README.md").is_file())
        self.assertFalse((ROOT / "model" / "strict_patient_v3").exists())
        self.assertFalse((ROOT / "model" / "strict_patient_v4").exists())

    def test_ignore_contract(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in ("result/", "results/", ".cache/", "tmp/", "codex/"):
            self.assertIn(entry, text)


if __name__ == "__main__":
    unittest.main()
