import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from baseline.frozen_method_baselines import pcmcl_source_runner as runner
from baseline.frozen_method_baselines.pcmcl_numerics import (
    NonFiniteTrainingError,
    backward_and_step,
)
from baseline.frozen_method_baselines.pcmcl_source_transfer import (
    hf_maximum_window_p_w,
    icbhi_flat4_from_ncw,
    kauh_patient_from_ncw_views,
)
from baseline.frozen_method_baselines.source_transfer_queue import run_queue
from baseline.frozen_method_baselines.source_transfer_summary import METHOD_ROOTS, summarize
from baseline.frozen_method_baselines.source_transfer_common import AudioUnit


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
            with self.assertRaisesRegex(FloatingPointError, "gradient") as raised:
                backward_and_step(parameter.sqrt(), optimizer, "scalar derivative")
            step.assert_not_called()
        self.assertEqual(raised.exception.diagnostics["stage"], "gradient")
        self.assertEqual(raised.exception.diagnostics["parameter"], "optimizer_parameter_0")
        self.assertEqual(parameter.item(), 0.0)

    def test_finite_update_is_not_clipped(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        backward_and_step(parameter.square(), optimizer, "finite scalar")
        self.assertAlmostEqual(parameter.item(), 1.6)

    def test_author_specaugment_gate_and_time_frequency_axes(self):
        value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        augment = runner.SourceSpecAugment(mask_value="zero")
        with mock.patch("torch.randn", return_value=torch.tensor(2.0)):
            self.assertTrue(torch.equal(augment(value), value))
        draws = [1, 0, 1, 2, 1, 0, 1, 2]
        with mock.patch("torch.randn", return_value=torch.tensor(-1.0)):
            with mock.patch("torch.randint", side_effect=[torch.tensor(v) for v in draws]):
                output = augment(value)
        self.assertTrue(torch.equal(output[:, :, :, 0], torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(output[:, :, :, 2], torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(output[:, :, 0, :], torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(output[:, :, 2, :], torch.zeros(1, 1, 4)))

    def test_patient_hard_negative_matches_aggregate_patient_profile(self):
        units = [
            AudioUnit("p1-c", "icbhi", Path("p1.wav"), "P1", target=1),
            AudioUnit("p1-n", "icbhi", Path("p1.wav"), "P1", target=0),
            AudioUnit("p2-c", "icbhi", Path("p2.wav"), "P2", target=1),
            AudioUnit("p2-n", "icbhi", Path("p2.wav"), "P2", target=0),
            AudioUnit("p3-c", "icbhi", Path("p3.wav"), "P3", target=1),
            AudioUnit("p3-w", "icbhi", Path("p3.wav"), "P3", target=2),
        ]
        dataset = runner.PCMCLTrainDataset(
            units, seed=42, mixing_probability=0.0, patient_probability=0.5
        )
        first, second = dataset._profile_matched_negative_pair(random.Random(42))
        self.assertNotEqual(units[first].group_id, units[second].group_id)
        self.assertEqual(
            dataset.patient_profiles[units[first].group_id],
            dataset.patient_profiles[units[second].group_id],
        )

    def test_legacy_nan_run_is_excluded_and_cannot_resume_or_reenter_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / METHOD_ROOTS["pcmcl"]
            failed = result_root / "seed_0"
            finite = result_root / "seed_1"
            invalid_summary = result_root / "seed_42"
            failed.mkdir(parents=True)
            finite.mkdir()
            invalid_summary.mkdir()
            complete = {"status": "complete_test_selected_source_transfer", "seed": 0,
                        "metrics": {"icbhi": {"icbhi_score": 0.5}}}
            original = json.dumps(complete)
            (failed / "run_summary.json").write_text(original)
            (failed / "train_log.jsonl").write_text(json.dumps({"epoch": 35, "train_loss": float("nan")}) + "\n")
            (finite / "run_summary.json").write_text(json.dumps({**complete, "seed": 1}))
            (finite / "train_log.jsonl").write_text(json.dumps({"epoch": 1, "train_loss": 0.7}) + "\n")
            (invalid_summary / "run_summary.json").write_text(
                json.dumps(
                    {
                        **complete,
                        "seed": 42,
                        "metrics": {"icbhi": {"icbhi_score": float("nan")}},
                    }
                )
            )
            aggregate = summarize(root, "pcmcl")
            self.assertEqual(aggregate["completed_seeds"], [1])
            self.assertEqual(
                [row["seed"] for row in aggregate["excluded_seeds"]], [0, 42]
            )
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
                raise NonFiniteTrainingError(
                    "non-finite loss (seed=0, epoch=35, batch=2)",
                    {"stage": "loss", "epoch": 35, "batch": 2},
                )

            with mock.patch.object(runner, "train_source", side_effect=interrupted_training):
                with mock.patch.object(runner, "evaluate_fixed_transfer") as evaluate:
                    with self.assertRaises(FloatingPointError):
                        runner.run(root, config_path, 0, "cpu", None)
                    evaluate.assert_not_called()
            status = json.loads((result_dir / "run_summary.json").read_text())
            self.assertEqual(status["status"], "failed_nonfinite")
            self.assertIn("epoch=35", status["error"])
            self.assertEqual(status["diagnostics"]["stage"], "loss")
            self.assertEqual((result_dir / "last_checkpoint.pt").read_bytes(), b"prior completed epoch")


if __name__ == "__main__":
    unittest.main()
