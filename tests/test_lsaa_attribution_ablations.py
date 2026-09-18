import json
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F

from baseline.pafa.jh2_main_multiseed_external import (
    hf_cas_recording_readout,
    kauh_patient_primary,
)
from baseline.pafa import joint_hierarchy_main_multiseed as main_runner
from baseline.pafa.lsaa_attribution_ablations import _stat, aggregate, config_payload
from baseline.pafa.table2_clean_controls import (
    NODE_WEIGHT,
    NativeAttributesHead,
    native_only_loss,
)


class LSAAAttributionAblationsTest(unittest.TestCase):
    def test_native_only_keeps_one_third_ce_and_no_attribute_gradient(self):
        head = NativeAttributesHead()
        logits = head(torch.randn(4, 768))
        target = torch.tensor([0, 1, 2, 3])
        loss = native_only_loss(logits, "icbhi", target)
        expected = NODE_WEIGHT * F.cross_entropy(logits["icbhi_native"], target)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(head.shared_projector.weight.grad)
        self.assertIsNotNone(head.icbhi.weight.grad)
        self.assertIsNone(head.crackle.weight.grad)
        self.assertIsNone(head.wheeze.weight.grad)

    def test_without_pafa_does_not_call_criterion(self):
        class ForbiddenCriterion(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("PAFA criterion must not be called")

        classification = torch.tensor(2.0, requires_grad=True)
        config = type(
            "Config",
            (),
            {
                "classification_weight": 1.0,
                "pafa_weight": 1.0,
                "lambda_pcsl": 50.0,
                "lambda_gpal": 0.0005,
            },
        )()
        total, pafa = main_runner._training_objective(
            classification,
            torch.zeros(2, 3),
            torch.tensor([0, 1]),
            ForbiddenCriterion(),
            config,
            main_runner.WITHOUT_PAFA_MODE,
        )
        self.assertIsNone(pafa)
        self.assertEqual(total.item(), 2.0)
        total.backward()
        self.assertEqual(classification.grad.item(), 1.0)

    def test_server_configs_keep_reference_contract_and_new_roots(self):
        repo_root = Path("/files1/Zilong/Acoustic")
        native = config_payload(repo_root, "native_only", 0, device="cuda")
        without = config_payload(repo_root, "lsaa_without_pafa", 42, device="cuda")
        self.assertEqual(native["node_weights"], {"native": 1 / 3, "c": 0.0, "w": 0.0})
        self.assertEqual(native["attribute_supervision"], False)
        self.assertEqual(native["lambda_pcsl"], 50.0)
        self.assertEqual(native["lambda_gpal"], 0.0005)
        self.assertEqual(native["learning_rate"], 5e-5)
        self.assertEqual(native["max_validation_points"], 50)
        self.assertTrue(native["source_split_contract"].endswith("lsaa_attribution_source_splits.json"))
        self.assertIn("LSAA_ATTRIBUTION_20260918/native_only/seed_0", native["output_dir"])
        self.assertEqual(without["pafa_enabled"], False)
        self.assertEqual(without["pcsl_gpal_criterion_called"], False)
        self.assertEqual(without["learning_rate"], 5e-5)
        self.assertEqual(without["epochs"], 50)
        self.assertEqual(without["precision"], "FP32")
        self.assertEqual(without["cuda_amp_enabled"], False)
        self.assertTrue(without["source_split_contract"].endswith("lsaa_attribution_source_splits.json"))
        self.assertFalse(
            main_runner._cuda_amp_enabled(
                torch.device("cuda"), main_runner.WITHOUT_PAFA_MODE
            )
        )
        self.assertIn("LSAA_ATTRIBUTION_20260918/lsaa_without_pafa/seed_42", without["output_dir"])

        contract = json.loads(
            Path("baseline/pafa/lsaa_attribution_source_splits.json").read_text()
        )
        self.assertEqual(
            contract["seeds"]["0"]["datasets"]["icbhi"]["expected_support"][
                "validation"
            ]["units"],
            506,
        )
        self.assertEqual(
            contract["seeds"]["42"]["datasets"]["sprsound"]["expected_support"][
                "subtrain"
            ]["units"],
            5219,
        )

    def test_builders_do_not_import_author_models_package(self):
        for path in (
            Path("baseline/pafa/joint_hierarchy.py"),
            Path("baseline/pafa/table2_clean_controls.py"),
        ):
            self.assertNotIn("from models.beats import", path.read_text())

    def test_aggregate_rejects_noncomplete_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            root = repo_root / "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only"
            for seed in (0, 1, 42):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / "run_summary.json").write_text(
                    json.dumps({"status": "failed" if seed == 1 else "complete_test_selected_benchmark_control"})
                )
            with self.assertRaisesRegex(RuntimeError, "all three seeds must complete"):
                aggregate(repo_root, "native_only")

    def test_hf_cas_and_kauh_primary_readouts_match_table_contract(self):
        recording_ids = np.repeat(np.asarray(["r1", "r2", "r3"]), 3)
        probabilities = np.zeros((9, 2), dtype=np.float32)
        probabilities[:, 1] = [0.1, 0.2, 0.1, 0.4, 0.9, 0.3, 0.8, 0.2, 0.1]
        rows, metrics = hf_cas_recording_readout(
            {
                "recording_ids": recording_ids,
                "window_indices": np.tile(np.arange(3), 3),
                "attribute_probabilities": probabilities,
            },
            {
                "r1": {"tokens": ["D"]},
                "r2": {"tokens": ["Wheeze"]},
                "r3": {"tokens": ["Rhonchi"]},
            },
        )
        self.assertEqual(metrics["support"], 3)
        self.assertEqual(metrics["positive"], 2)
        self.assertEqual(metrics["negative"], 1)
        self.assertEqual(metrics["hf_cas_auroc"], 1.0)
        self.assertEqual(rows["window_count"].tolist(), [3, 3, 3])
        kauh = kauh_patient_primary(
            {
                "patient_level_after_BDE_probability_mean": {
                    "level1_binary": {
                        "average_score": 0.7,
                        "rows": 86,
                        "confusion": [[30, 5], [20, 31]],
                    }
                }
            }
        )
        self.assertEqual(kauh["kauh_patient_ba"], 0.7)
        self.assertEqual(kauh["kauh_patient_support"], 86)

    def test_full_status_compatibility_and_nonfinite_rejection(self):
        self.assertEqual(
            main_runner._run_status(main_runner.FULL_MODE, True),
            "early_stopped_epochwise_icbhi_test_selected",
        )
        with self.assertRaisesRegex(FloatingPointError, "non-finite without-PAFA"):
            main_runner._require_finite_total(
                torch.tensor(float("nan")),
                main_runner.WITHOUT_PAFA_MODE,
                seed=0,
                epoch=1,
                batch=2,
                dataset="icbhi",
            )
        with self.assertRaisesRegex(FloatingPointError, "non-finite attribution metric"):
            _stat([0.5, float("nan"), 0.7])


if __name__ == "__main__":
    unittest.main()
