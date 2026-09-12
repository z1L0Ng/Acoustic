import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baseline.pafa.table2_clean_controls import (
    ACTIVE_DATASETS,
    CORE_UPDATES_PER_POINT,
    Table2Config,
    apply_variant_eligibility,
    epoch_batches,
    fit_attribute_thresholds,
    fixed_hierarchy_loss,
    native_attributes_loss,
    native_selection_scores,
    split_reference_payload,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baseline/pafa/table2_clean_split_manifest.json"


class Table2CleanControlsTest(unittest.TestCase):
    def test_manifest_group_disjoint_and_required_support(self) -> None:
        payload = json.loads(MANIFEST.read_text())
        self.assertEqual(payload["source_scope"], "official training records only; no official test records")
        for dataset in ("icbhi", "sprsound"):
            groups = payload["datasets"][dataset]["groups"]
            self.assertFalse(set(groups["subtrain"]) & set(groups["calibration"]))
            self.assertFalse(set(groups["subtrain"]) & set(groups["selection"]))
            self.assertFalse(set(groups["calibration"]) & set(groups["selection"]))
            for partition in ("subtrain", "calibration", "selection"):
                support = payload["datasets"][dataset]["support"][partition]
                self.assertGreater(len(support["native_classes"]), 1)
                for node in ("a", "c", "w"):
                    self.assertGreater(support["nodes"][node]["positive"], 0)
                    self.assertGreater(support["nodes"][node]["negative"], 0)

        icbhi = payload["datasets"]["icbhi"]
        self.assertEqual(
            icbhi["groups"]["calibration"], ["105", "108", "130", "166", "188"]
        )
        self.assertEqual(
            icbhi["groups"]["selection"],
            ["114", "125", "132", "137", "163", "181", "190", "206", "213", "218", "220"],
        )
        selection_native = icbhi["support"]["selection"]["native_classes"]
        self.assertGreaterEqual(selection_native["normal"], 100)
        self.assertGreaterEqual(
            selection_native["crackle"]
            + selection_native["wheeze"]
            + selection_native["both"],
            100,
        )
        self.assertTrue(all(selection_native[label] > 0 for label in ("normal", "crackle", "wheeze", "both")))

        for partition in ("calibration", "selection"):
            rows = [
                row for row in icbhi["records"] if row["partition"] == partition
            ]
            normal_groups = {
                row["group_id"] for row in rows if row["native_label"] == "normal"
            }
            crackle_groups = {
                row["group_id"]
                for row in rows
                if row["native_label"] in {"crackle", "both"}
            }
            wheeze_groups = {
                row["group_id"]
                for row in rows
                if row["native_label"] in {"wheeze", "both"}
            }
            self.assertGreaterEqual(len(icbhi["groups"][partition]), 4)
            self.assertGreaterEqual(len(normal_groups), 2)
            self.assertGreaterEqual(len(crackle_groups), 2)
            self.assertGreaterEqual(len(wheeze_groups), 2)

    def test_active_sources_and_common_budget(self) -> None:
        self.assertEqual(ACTIVE_DATASETS["icbhi_only"], ("icbhi",))
        self.assertEqual(ACTIVE_DATASETS["sprsound_only"], ("sprsound",))
        self.assertEqual(ACTIVE_DATASETS["full_hf_off"], ("icbhi", "sprsound"))
        single = epoch_batches(
            {"icbhi": 9},
            ("icbhi",),
            batch_size=2,
            model_seed=42,
            validation_point=1,
        )
        joint = epoch_batches(
            {"icbhi": 9, "sprsound": 11},
            ("icbhi", "sprsound"),
            batch_size=2,
            model_seed=42,
            validation_point=1,
        )
        self.assertEqual(len(single), CORE_UPDATES_PER_POINT)
        self.assertEqual(len(joint), CORE_UPDATES_PER_POINT)

        config = Table2Config(
            repo_root=ROOT,
            variant="icbhi_only",
            model_seed=42,
            output_dir=ROOT / "result/reproduce/pafa_joint_hierarchy/Table2_clean_controls/icbhi_only/seed_42",
            split_manifest=MANIFEST,
        )
        payload = split_reference_payload(config, json.loads(MANIFEST.read_text()))
        self.assertEqual(list(payload["datasets"]), ["icbhi"])
        self.assertIn("groups", payload["datasets"]["icbhi"])
        self.assertIn("records", payload["datasets"]["icbhi"])
        self.assertIn("support", payload["datasets"]["icbhi"])

    def test_fixed_node_weights_do_not_renormalize(self) -> None:
        logits = {
            "level1": torch.tensor([[0.0, 0.0]], requires_grad=True),
            "crackle": torch.tensor([0.0], requires_grad=True),
            "wheeze": torch.tensor([0.0], requires_grad=True),
        }
        targets = torch.tensor([[1.0, 1.0, 0.0]])
        full = torch.tensor([[True, True, True]])
        coarse = apply_variant_eligibility(full, "sprsound", "coarse_spr")
        full_loss = fixed_hierarchy_loss(logits, targets, full)
        coarse_loss = fixed_hierarchy_loss(logits, targets, coarse)
        self.assertAlmostEqual(float(full_loss.detach()), float(F.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([1.0]))), places=6)
        self.assertAlmostEqual(float(coarse_loss.detach()), float(F.cross_entropy(torch.tensor([[0.0, 0.0]]), torch.tensor([1]))) / 3.0, places=6)

    def test_native_attributes_keeps_auxiliary_weight(self) -> None:
        logits = {
            "icbhi_native": torch.zeros((2, 4), requires_grad=True),
            "crackle": torch.zeros(2, requires_grad=True),
            "wheeze": torch.zeros(2, requires_grad=True),
        }
        targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        eligible = torch.ones((2, 3), dtype=torch.bool)
        loss = native_attributes_loss(
            logits,
            "icbhi",
            torch.tensor([1, 2]),
            targets,
            eligible,
        )
        expected = (
            F.cross_entropy(logits["icbhi_native"], torch.tensor([1, 2]))
            + F.binary_cross_entropy_with_logits(logits["crackle"], targets[:, 1])
            + F.binary_cross_entropy_with_logits(logits["wheeze"], targets[:, 2])
        ) / 3.0
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=6)

    def test_threshold_and_active_source_selection(self) -> None:
        predictions = {
            "dataset_ids": np.asarray(["icbhi", "icbhi", "sprsound", "sprsound"]),
            "raw_ground_truth": np.asarray(["normal", "crackle", "Normal", "Wheeze"]),
            "level1_predictions": np.asarray([0, 1, 0, 1]),
            "attribute_probabilities": np.asarray(
                [[0.2, 0.1], [0.8, 0.2], [0.2, 0.1], [0.1, 0.9]], dtype=np.float32
            ),
            "targets": np.asarray(
                [[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=np.float32
            ),
            "eligible": np.ones((4, 3), dtype=bool),
            "native_predictions": np.full(4, -1, dtype=np.int64),
        }
        thresholds, details = fit_attribute_thresholds(predictions, ("icbhi", "sprsound"))
        self.assertEqual(details["crackle"]["tie_break"], "higher_threshold")
        joint = native_selection_scores(
            predictions,
            variant="full_hf_off",
            active_datasets=("icbhi", "sprsound"),
            thresholds=thresholds,
        )
        icbhi = native_selection_scores(
            predictions,
            variant="icbhi_only",
            active_datasets=("icbhi",),
            thresholds=thresholds,
        )
        self.assertAlmostEqual(
            joint["selection_utility"],
            np.mean(list(joint["per_source_scores"].values())),
        )
        self.assertEqual(
            icbhi["selection_utility"], icbhi["per_source_scores"]["icbhi"]
        )


if __name__ == "__main__":
    unittest.main()
