import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.pafa.native_only_external_posthoc import (
    aggregate,
    hf_cas_metrics,
    hf_recording_readout,
    kauh_patient_readout,
    native_probabilities,
    native_wheeze_marginal,
)


class NativeOnlyExternalPosthocTest(unittest.TestCase):
    def test_native_softmax_wheeze_marginal_and_finite_check(self):
        probability = native_probabilities(torch.zeros(2, 4)).numpy()
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(native_wheeze_marginal(probability), 0.5))
        with self.assertRaisesRegex(FloatingPointError, "non-finite ICBHI native"):
            native_probabilities(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))

    def test_hf_three_window_max_and_union_eligibility(self):
        prediction = {
            "recording_ids": np.repeat(np.asarray(["d", "w", "phase"]), 3),
            "window_indices": np.tile(np.arange(3), 3),
            "wheeze_marginal_scores": np.asarray(
                [0.1, 0.3, 0.2, 0.4, 0.9, 0.5, 0.8, 0.2, 0.1],
                dtype=np.float32,
            ),
        }
        label_free, scored = hf_recording_readout(
            prediction,
            {
                "d": {"tokens": ["D"]},
                "w": {"tokens": ["Wheeze"]},
                "phase": {"tokens": ["I"]},
            },
        )
        self.assertEqual(label_free["recording_ids"].tolist(), ["d", "phase", "w"])
        self.assertEqual(label_free["window_count"].tolist(), [3, 3, 3])
        self.assertEqual(scored["recording_ids"].tolist(), ["d", "w"])
        self.assertEqual(scored["cas_union_targets"].tolist(), [0, 1])
        self.assertTrue(np.allclose(scored["cas_ranking_scores"], [0.3, 0.9]))

    def test_kauh_bde_mean_and_half_tie_is_normal(self):
        views = {
            "sample_ids": np.asarray(
                ["kauh:P1:B", "kauh:P1:D", "kauh:P1:E", "kauh:P2:B", "kauh:P2:D", "kauh:P2:E"]
            ),
            "patient_ids": np.asarray(["P1", "P1", "P1", "P2", "P2", "P2"]),
            "abnormal_scores": np.asarray([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        }
        targets = {
            key: {
                "compatible": True,
                "level1_target": int("P2" in key),
                "raw_sound": "N" if "P1" in key else "E W",
            }
            for key in views["sample_ids"]
        }
        label_free, scored = kauh_patient_readout(views, targets)
        self.assertTrue(np.allclose(label_free["abnormal_scores"], [0.5, 0.8]))
        self.assertEqual(label_free["predictions"].tolist(), [0, 1])
        self.assertEqual(scored["targets"].tolist(), [0, 1])
        self.assertEqual(label_free["view_count"].tolist(), [3, 3])

    def test_missing_seed_is_not_three_seed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                (seed_dir / "run_summary.json").write_text(
                    json.dumps(
                        {"status": "complete_native_only_fixed_head_external_posthoc"}
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "missing Native-only external seed 42"):
                aggregate(root, root)

    def test_hf_cas_metrics_uses_standard_auc_and_rejects_nonfinite(self):
        scored = {
            "cas_union_targets": np.asarray([0, 1], dtype=np.int64),
            "cas_ranking_scores": np.asarray([0.1, 0.9], dtype=np.float32),
        }
        self.assertEqual(hf_cas_metrics(scored)["hf_cas_auroc"], 1.0)
        scored["cas_ranking_scores"][0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "non-finite Native-only HF CAS"):
            hf_cas_metrics(scored)


if __name__ == "__main__":
    unittest.main()
