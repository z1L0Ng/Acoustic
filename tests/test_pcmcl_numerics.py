import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from baseline.frozen_method_baselines import pcmcl_source_runner as runner
from baseline.frozen_method_baselines.pcmcl_numerics import backward_and_step
from baseline.frozen_method_baselines.pcmcl_source_transfer import (
    hf_maximum_window_p_w,
    icbhi_flat4_from_ncw,
    kauh_patient_from_ncw_views,
)
from baseline.frozen_method_baselines.source_transfer_queue import run_queue
from baseline.frozen_method_baselines.source_transfer_summary import METHOD_ROOTS, summarize


class PCMCLNumericsTest(unittest.TestCase):
    def test_finite_readout_keeps_existing_class_mapping(self):
        scores = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.6, 0.1],
                               [0.1, 0.1, 0.6], [0.1, 0.6, 0.6]])
        self.assertEqual(icbhi_flat4_from_ncw(scores).tolist(), [0, 1, 2, 3])

    def test_nonfinite_predictions_are_not_normal_or_transfer_scores(self):
        with self.assertRaises(FloatingPointError):
            icbhi_flat4_from_ncw(torch.full((1, 3), float("nan")))
        with self.assertRaises(FloatingPointError):
            hf_maximum_window_p_w(torch.full((1, 3, 3), float("nan")))
        with self.assertRaises(FloatingPointError):
            kauh_patient_from_ncw_views(torch.full((1, 3, 3), float("inf")))

    def test_nan_loss_stops_before_backward_or_optimizer_update(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        with mock.patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(FloatingPointError, "loss.*epoch=35"):
                backward_and_step(parameter * float("nan"), optimizer, "epoch=35")
            step.assert_not_called()
        self.assertIsNone(parameter.grad)
        self.assertEqual(parameter.item(), 2.0)

    def test_finite_loss_with_infinite_gradient_stops_before_update(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        with mock.patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(FloatingPointError, "gradient norms"):
                backward_and_step(parameter.sqrt(), optimizer, "scalar derivative")
            step.assert_not_called()
        self.assertEqual(parameter.item(), 0.0)

    def test_finite_update_is_not_clipped(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        backward_and_step(parameter.square(), optimizer, "finite scalar")
        self.assertAlmostEqual(parameter.item(), 1.6)

    def test_legacy_nan_run_is_excluded_and_cannot_resume_or_reenter_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / METHOD_ROOTS["pcmcl"]
            failed = result_root / "seed_0"
            finite = result_root / "seed_1"
            failed.mkdir(parents=True)
            finite.mkdir()
            complete = {"status": "complete_test_selected_source_transfer", "seed": 0,
                        "metrics": {"icbhi": {"icbhi_score": 0.5}}}
            original = json.dumps(complete)
            (failed / "run_summary.json").write_text(original)
            (failed / "train_log.jsonl").write_text(json.dumps({"epoch": 35, "train_loss": float("nan")}) + "\n")
            (finite / "run_summary.json").write_text(json.dumps({**complete, "seed": 1}))
            (finite / "train_log.jsonl").write_text(json.dumps({"epoch": 1, "train_loss": 0.7}) + "\n")
            aggregate = summarize(root, "pcmcl")
            self.assertEqual(aggregate["completed_seeds"], [1])
            self.assertEqual(aggregate["excluded_seeds"][0]["seed"], 0)
            config_path = root / "baseline/frozen_method_baselines/pcmcl_source_run.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"method": "test metadata", "output_root": METHOD_ROOTS["pcmcl"]}))
            with mock.patch.object(runner, "train_source") as train:
                with self.assertRaisesRegex(FloatingPointError, "refusing to resume"):
                    runner.run(root, config_path, 0, "cpu", failed / "last_checkpoint.pt")
                train.assert_not_called()
            with mock.patch("baseline.frozen_method_baselines.source_transfer_queue._call") as call:
                with self.assertRaisesRegex(FloatingPointError, "refusing to reuse"):
                    run_queue(root, ("pcmcl",), "cpu")
                call.assert_not_called()
            self.assertEqual((failed / "run_summary.json").read_text(), original)

    def test_failure_propagates_with_status_and_preserves_prior_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"method": "test error handling", "output_root": "runs"}))
            result_dir = root / "runs/seed_0"

            def interrupted_training(*args, **kwargs):
                (result_dir / "last_checkpoint.pt").write_bytes(b"prior completed epoch")
                raise FloatingPointError("non-finite loss (seed=0, epoch=35, batch=2)")

            with mock.patch.object(runner, "train_source", side_effect=interrupted_training):
                with mock.patch.object(runner, "evaluate_fixed_transfer") as evaluate:
                    with self.assertRaises(FloatingPointError):
                        runner.run(root, config_path, 0, "cpu", None)
                    evaluate.assert_not_called()
            status = json.loads((result_dir / "run_summary.json").read_text())
            self.assertEqual(status["status"], "failed_nonfinite")
            self.assertIn("epoch=35", status["error"])
            self.assertEqual((result_dir / "last_checkpoint.pt").read_bytes(), b"prior completed epoch")


if __name__ == "__main__":
    unittest.main()
