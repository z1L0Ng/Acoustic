import csv
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
GITHUB_REGULAR_FILE_LIMIT_BYTES = 100 * 1024 * 1024


class ProjectStructureTest(unittest.TestCase):
    def test_required_roots_exist(self):
        for name in (
            "acoustic",
            "baseline",
            "model",
            "experiments",
            "checkpoints",
            "dataset",
            "tests",
            "server",
            "docs",
        ):
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

    def test_checkpoint_publication_policy(self):
        with (ROOT / "experiments" / "index.csv").open(newline="", encoding="utf-8") as handle:
            experiment_ids = {row["experiment_id"] for row in csv.DictReader(handle)}

        checkpoint_root = ROOT / "checkpoints"
        self.assertTrue((checkpoint_root / "README.md").is_file())
        for experiment_dir in checkpoint_root.iterdir():
            if not experiment_dir.is_dir():
                continue
            self.assertIn(experiment_dir.name, experiment_ids)
            files = [path for path in experiment_dir.iterdir() if path.is_file()]
            self.assertEqual(len(files), 1, experiment_dir)
            checkpoint = files[0]
            self.assertEqual(checkpoint.stem, "best", checkpoint)
            self.assertIn(checkpoint.suffix, CHECKPOINT_SUFFIXES, checkpoint)
            self.assertLessEqual(
                checkpoint.stat().st_size,
                GITHUB_REGULAR_FILE_LIMIT_BYTES,
                checkpoint,
            )

        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
        for relative in tracked:
            path = Path(relative)
            if path.suffix not in CHECKPOINT_SUFFIXES:
                continue
            self.assertEqual(path.parts[0], "checkpoints", relative)
            self.assertEqual(path.name, f"best{path.suffix}", relative)

    def test_ignore_contract(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in ("result/", "results/", ".cache/", "tmp/", "codex/"):
            self.assertIn(entry, text)


if __name__ == "__main__":
    unittest.main()
