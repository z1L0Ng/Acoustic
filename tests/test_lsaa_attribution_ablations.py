import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from baseline.pafa import joint_hierarchy_main_multiseed as main_runner
from baseline.pafa.lsaa_attribution_ablations import aggregate, config_payload
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
        self.assertIn("LSAA_ATTRIBUTION_20260918/native_only/seed_0", native["output_dir"])
        self.assertEqual(without["pafa_enabled"], False)
        self.assertEqual(without["pcsl_gpal_criterion_called"], False)
        self.assertEqual(without["learning_rate"], 5e-5)
        self.assertEqual(without["epochs"], 50)
        self.assertIn("LSAA_ATTRIBUTION_20260918/lsaa_without_pafa/seed_42", without["output_dir"])

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


if __name__ == "__main__":
    unittest.main()
