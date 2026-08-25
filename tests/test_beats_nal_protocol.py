import unittest

import numpy as np
import torch

from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    WaveformAugmentationConfig,
    WaveformNormalizationConfig,
    augment_waveform,
    decode_icbhi_flat4,
    hierarchical_loss,
    hierarchical_loss_contribution,
    normalize_waveform,
)


class BEATsNALProtocolTest(unittest.TestCase):
    def test_normalization_is_native_waveform_level(self):
        waveform = torch.tensor([-0.5, 0.25, 0.5], dtype=torch.float32)
        peak = normalize_waveform(
            waveform,
            WaveformNormalizationConfig(mode="peak", peak_target=0.95),
        )
        self.assertAlmostEqual(float(peak.abs().max()), 0.95, places=6)
        rms = normalize_waveform(
            waveform,
            WaveformNormalizationConfig(mode="rms", rms_target_dbfs=-20.0),
        )
        self.assertAlmostEqual(float(rms.square().mean().sqrt()), 0.1, places=6)

    def test_augmentation_is_seeded_and_preserves_length(self):
        waveform = torch.linspace(-0.5, 0.5, 1600)
        config = WaveformAugmentationConfig(
            mode="gain_noise",
            probability=1.0,
        )
        first = augment_waveform(
            waveform,
            config,
            torch.Generator().manual_seed(42),
        )
        second = augment_waveform(
            waveform,
            config,
            torch.Generator().manual_seed(42),
        )
        self.assertEqual(first.shape, waveform.shape)
        self.assertTrue(torch.equal(first, second))

    def test_masked_hierarchical_loss_ignores_unknown_attributes(self):
        logits = {
            "level1": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
            "crackle": torch.tensor([-2.0, 2.0]),
            "wheeze": torch.tensor([50.0, -50.0]),
        }
        targets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        eligible = torch.tensor(
            [[True, True, False], [True, True, False]],
            dtype=torch.bool,
        )
        loss, named = hierarchical_loss(
            logits,
            targets,
            eligible,
            HierarchicalLossConfig(mode="focal"),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(named), {"level1", "crackle"})

    def test_unitwise_gradient_accumulation_matches_batch_loss(self):
        logits = {
            "level1": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
            "crackle": torch.tensor([-2.0, 2.0]),
            "wheeze": torch.tensor([0.5, -0.5]),
        }
        targets = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        eligible = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        )
        config = HierarchicalLossConfig(mode="ce_bce")
        batch_loss, _ = hierarchical_loss(logits, targets, eligible, config)
        denominators = eligible.sum(dim=0)
        accumulated = sum(
            hierarchical_loss_contribution(
                {node: value[row : row + 1] for node, value in logits.items()},
                targets[row : row + 1],
                eligible[row : row + 1],
                config,
                None,
                denominators,
            )
            for row in range(2)
        )
        self.assertAlmostEqual(float(batch_loss), float(accumulated), places=7)

    def test_icbhi_flat4_reconstruction_uses_atomic_bits(self):
        probabilities = np.asarray(
            [[0.1, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]
        )
        labels = decode_icbhi_flat4(
            probabilities,
            {"crackle": 0.5, "wheeze": 0.5},
        )
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
