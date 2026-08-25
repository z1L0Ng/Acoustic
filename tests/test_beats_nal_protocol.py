import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from baseline.multidataset_pipeline.beats_nal_protocol import (
    BEATsNALConfig,
    HierarchicalLossConfig,
    WaveformAugmentationConfig,
    WaveformNormalizationConfig,
    augment_waveform,
    decode_icbhi_flat4,
    hierarchical_loss,
    hierarchical_loss_contribution,
    normalize_waveform,
    _load_transformed_batch,
)
from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.multidataset_pipeline.beats_temporal import (
    BEATsGeometry,
    exact_patch_masks,
    temporalize_transformer_output,
)
from baseline.multidataset_pipeline.contracts import WaveformSample


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

    def test_batch_loss_can_skip_per_node_scalar_collection(self):
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
        expected, _ = hierarchical_loss(logits, targets, eligible, config)
        actual, named = hierarchical_loss(
            logits,
            targets,
            eligible,
            config,
            active_nodes=("level1", "crackle", "wheeze"),
            collect_named=False,
        )
        self.assertEqual(named, {})
        self.assertAlmostEqual(float(expected), float(actual), places=7)

    def test_transformed_batch_collates_units_and_reuses_decoded_cache(self):
        samples = [
            Sample(
                sample_id=f"icbhi:test-{index}",
                dataset="icbhi",
                partition="subtrain",
                group_id=str(index),
                audio_path=f"unused-{index}.wav",
                crop_start_s=0.0,
                crop_end_s=4.0 + index,
                targets={"icbhi_flat4": 0},
                metadata={},
            )
            for index in range(2)
        ]

        def decode(sample, lane, *, outer_test_accessed=False):
            length = 64_000 + int(sample.group_id) * 16_000
            waveform = WaveformSample(
                waveform=torch.zeros(length, dtype=torch.float32),
                sample_id=sample.sample_id,
                dataset_id=lane,
                prediction_unit="cycle",
                source_start_s=0.0,
                source_end_s=length / 16_000,
            )
            return waveform, {}

        config = BEATsNALConfig(
            repo_root=Path("."),
            source_repo=Path("."),
            checkpoint=Path("checkpoint.pt"),
            output_dir=Path("result"),
        )
        cache = {}
        generators = [
            torch.Generator().manual_seed(42),
            torch.Generator().manual_seed(43),
        ]
        with patch(
            "baseline.multidataset_pipeline.beats_nal_protocol.load_sample_waveform",
            side_effect=decode,
        ) as loader:
            first = _load_transformed_batch(
                samples,
                config,
                training=True,
                generators=generators,
                waveform_cache=cache,
            )
            second = _load_transformed_batch(
                samples,
                config,
                training=True,
                generators=generators,
                waveform_cache=cache,
            )
        self.assertEqual(first.waveform_windows.shape, (2, 2, 64_000))
        self.assertEqual(first.sample_ids, tuple(sample.sample_id for sample in samples))
        self.assertTrue(torch.equal(first.waveform_windows, second.waveform_windows))
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(len(cache), 2)

    def test_temporal_fast_path_reuses_precomputed_patch_mask(self):
        geometry = BEATsGeometry()
        valid_samples = torch.tensor([64_000, 48_000], dtype=torch.long)
        token_mask, padding_mask = exact_patch_masks(
            valid_samples,
            64_000,
            geometry,
        )
        frequency_patches = geometry.frequency_patches()
        flattened = torch.zeros(
            2,
            token_mask.shape[1] * frequency_patches,
            4,
            dtype=torch.float32,
        )
        output = temporalize_transformer_output(
            flattened,
            valid_samples,
            64_000,
            torch.zeros(2, dtype=torch.float64),
            geometry,
            transformer_padding_mask=padding_mask,
            token_mask=token_mask,
        )
        self.assertTrue(torch.equal(output.token_mask, token_mask))
        self.assertEqual(output.pooled.shape, (2, 4))

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
