from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

from baseline.multidataset_pipeline.beats_temporal import (
    BEATsGeometry,
    BEATsTemporalAdapter,
    HFInterval,
    HFRawInterval,
    HFTargetPolicy,
    TokenAlignmentPolicy,
    exact_patch_masks,
    intervals_to_token_supervision,
    raw_intervals_to_token_supervision,
    restore_patch_grid,
    temporalize_transformer_output,
    verify_non_hf_pooled_parity,
)
from baseline.multidataset_pipeline.adapter_factory import (
    AdapterFactoryConfig,
    build_production_adapter,
)
from baseline.multidataset_pipeline.ast_window_encoder import ASTWindowBackend
from baseline.multidataset_pipeline.hear_window_encoder import HeARWindowBackend
from baseline.multidataset_pipeline.panns_window_encoder import PANNsWindowBackend
from baseline.multidataset_pipeline.contracts import (
    ObservationState,
    WaveformSample,
    collate_waveforms,
)
from baseline.multidataset_pipeline.eligibility import (
    CompatibleRow,
    GuardrailSchema,
    P8ObjectiveConfig,
    ZeroEligibleDenominator,
    build_eligibility_targets,
    eligibility_masked_bce,
)
from baseline.multidataset_pipeline.joint_native import (
    CoreConfig,
    JointNativeProjector,
    assert_frozen_encoder,
    assert_p1_p2_matched,
    build_source_proportional_receipt,
)
from baseline.multidataset_pipeline.preflight import (
    CandidateDimensionAdapter,
    HF_NATIVE_METRICS,
    P1_P5_SELECTION_RULE,
    P1_P5_UPDATE_BUDGET,
    P1_P5_UPDATES_PER_REFERENCE_EPOCH,
    P1_P5_VALIDATION_INTERVAL_UPDATES,
    P6TokenTemporalHead,
    PIPELINE_ENCODERS,
    PRODUCTION_ADAPTER_STATUS,
    SharedWindowCoreConfig,
    SharedWindowEncoderOutput,
    assert_p1_p5_matched,
    freeze_receipt,
    hf_masked_channel_balanced_bce,
    select_validation_checkpoint,
    source_proportional_validation_selection_loss,
    validate_independent_verifier_receipt,
)
from baseline.multidataset_pipeline.sliding_window import (
    WINDOW_SAMPLES,
    WINDOW_STRIDE_SAMPLES,
    collate_sliding_windows,
    hf_window_supervision,
    masked_mean_window_embeddings,
    source_window_starts,
)
from baseline.multidataset_pipeline.window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    require_file_identity,
    sha256_file,
)


def sample(
    dataset_id: str,
    prediction_unit: str,
    samples: int,
    sample_id: str,
    start_s: float = 0.0,
) -> WaveformSample:
    waveform = torch.linspace(-0.5, 0.5, samples, dtype=torch.float32)
    return WaveformSample(
        waveform=waveform,
        sample_id=sample_id,
        dataset_id=dataset_id,
        prediction_unit=prediction_unit,
        source_start_s=start_s,
        source_end_s=start_s + samples / 16_000,
        lineage={"fixture": "synthetic"},
    )


class WaveformContractTest(unittest.TestCase):
    def test_zero_padding_and_batch_padding_invariance(self):
        short = sample("ICBHI", "cycle", 800, "short")
        long = sample("SPRSound", "event", 1_200, "long")
        alone = collate_waveforms([short])
        together = collate_waveforms([short, long])
        torch.testing.assert_close(alone.waveform[0], together.waveform[0, :800])
        self.assertTrue(together.waveform_padding_mask[0, 800:].all())
        self.assertEqual(torch.count_nonzero(together.waveform[0, 800:]).item(), 0)
        self.assertEqual(together.valid_samples.tolist(), [800, 1_200])
        self.assertEqual(together.dataset_ids, ("ICBHI", "SPRSound"))
        self.assertEqual(together.prediction_units, ("cycle", "event"))

    def test_native_units_remain_distinct(self):
        rows = [
            sample("ICBHI", "cycle", 800, "i"),
            sample("SPRSound", "event", 800, "s"),
            sample("HF", "recording_15s_with_intervals", 240_000, "h"),
            sample("KAUH", "recording", 800, "k"),
        ]
        batch = collate_waveforms(rows)
        self.assertEqual(
            batch.prediction_units,
            ("cycle", "event", "recording_15s_with_intervals", "recording"),
        )
        self.assertEqual(batch.sample_rate, 16_000)

    def test_batch_to_cpu_preserves_lineage_and_masks(self):
        batch = collate_waveforms(
            [
                sample("ICBHI", "cycle", 800, "i"),
                sample("KAUH", "recording", 1_000, "k"),
            ]
        )
        moved = batch.to("cpu")
        self.assertIsNot(moved, batch)
        self.assertEqual(moved.device, torch.device("cpu"))
        self.assertEqual(moved.lineage, batch.lineage)
        self.assertEqual(moved.sample_ids, batch.sample_ids)
        self.assertTrue(torch.equal(moved.waveform_padding_mask, batch.waveform_padding_mask))
        self.assertTrue(torch.equal(moved.valid_samples, batch.valid_samples))


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_padding_mask = None

    def forward(self, values, padding_mask=None):
        self.last_padding_mask = padding_mask.detach().clone()
        return values, []


class FakeBEATs(nn.Module):
    def __init__(self, geometry: BEATsGeometry) -> None:
        super().__init__()
        self.cfg = {"input_patch_size": 2}
        self.geometry = geometry
        self.patch_embedding = nn.Conv2d(
            1, 2, kernel_size=(2, 2), stride=(2, 2), bias=False
        )
        with torch.no_grad():
            self.patch_embedding.weight.copy_(
                torch.tensor(
                    [
                        [[[[1.0, 0.0], [0.0, 0.0]]]],
                        [[[[0.0, 0.0], [0.0, 1.0]]]],
                    ]
                ).reshape(2, 1, 2, 2)
            )
        self.layer_norm = nn.Identity()
        self.post_extract_proj = None
        self.dropout_input = nn.Identity()
        self.encoder = FakeTransformer()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def preprocess(self, waveform):
        frames = self.geometry.fbank_frames(waveform.shape[1])
        values = torch.arange(
            frames * self.geometry.mel_bins,
            dtype=waveform.dtype,
            device=waveform.device,
        ).reshape(1, frames, self.geometry.mel_bins)
        return values.expand(waveform.shape[0], -1, -1).clone()


class BEATsTemporalContractTest(unittest.TestCase):
    def setUp(self):
        self.small = BEATsGeometry(
            mel_bins=4,
            patch_kernel_time=2,
            patch_stride_time=2,
            patch_kernel_frequency=2,
            patch_stride_frequency=2,
        )

    def test_default_checkpoint_geometry(self):
        geometry = BEATsGeometry.from_checkpoint_config({"input_patch_size": 16})
        self.assertEqual(geometry.fbank_frames(80_000), 498)
        self.assertEqual(geometry.time_patches(80_000), 31)
        self.assertEqual(geometry.frequency_patches(), 8)
        self.assertAlmostEqual(geometry.temporal_stride_s, 0.160)
        self.assertAlmostEqual(geometry.receptive_interval_s, 0.175)

    def test_exact_flattened_mask_is_time_major_frequency_minor(self):
        valid_samples = torch.tensor([880, 1_520], dtype=torch.long)
        temporal_valid, flat_padding = exact_patch_masks(
            valid_samples, 1_520, self.small
        )
        self.assertEqual(
            temporal_valid.tolist(),
            [[True, True, False, False], [True, True, True, True]],
        )
        self.assertEqual(
            flat_padding[0].tolist(),
            [False, False, False, False, True, True, True, True],
        )

    def test_frequency_aggregation_and_time_map(self):
        flattened = torch.tensor(
            [[[0.0], [2.0], [10.0], [12.0], [20.0], [22.0], [30.0], [32.0]]]
        )
        output = temporalize_transformer_output(
            flattened,
            torch.tensor([880], dtype=torch.long),
            1_520,
            torch.tensor([1.0]),
            self.small,
        )
        self.assertEqual(tuple(output.tokens.shape), (1, 4, 1))
        torch.testing.assert_close(
            output.tokens[0, :, 0], torch.tensor([1.0, 11.0, 0.0, 0.0])
        )
        self.assertEqual(output.token_mask.tolist(), [[True, True, False, False]])
        torch.testing.assert_close(
            output.time_map[0, :2],
            torch.tensor([[1.000, 1.035], [1.020, 1.055]]),
        )
        torch.testing.assert_close(output.pooled, torch.tensor([[6.0]]))

    def test_restore_order(self):
        flat = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        grid = restore_patch_grid(flat, 4, 2)
        self.assertEqual(grid[0, 1, :, 0].tolist(), [2.0, 3.0])

    def test_deterministic_fake_beats_adapter(self):
        beats = FakeBEATs(self.small)
        adapter = BEATsTemporalAdapter(beats, self.small)
        short = sample("ICBHI", "cycle", 880, "short")
        batch = collate_waveforms(
            [
                short,
                sample("KAUH", "recording", 1_520, "long"),
            ]
        )
        output = adapter(batch)
        batch_padding_mask = beats.encoder.last_padding_mask.clone()
        alone = adapter(collate_waveforms([short]))
        self.assertEqual(tuple(output.tokens.shape), (2, 4, 2))
        self.assertEqual(output.tokens.dtype, torch.float32)
        self.assertEqual(output.time_map.dtype, torch.float32)
        self.assertEqual(
            batch_padding_mask[0].tolist(),
            [False, False, False, False, True, True, True, True],
        )
        self.assertEqual(output.token_mask[0].tolist(), [True, True, False, False])
        self.assertEqual(output.observation_mask.sum().item(), 0)
        torch.testing.assert_close(output.tokens[0, :2], alone.tokens[0])
        torch.testing.assert_close(output.pooled[0], alone.pooled[0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_adapter_requires_explicit_batch_device_move_on_cuda(self):
        beats = FakeBEATs(self.small).to("cuda")
        adapter = BEATsTemporalAdapter(beats, self.small)
        batch = collate_waveforms([sample("ICBHI", "cycle", 880, "short")])
        with self.assertRaisesRegex(RuntimeError, r"batch\.to"):
            adapter(batch)
        output = adapter(batch.to("cuda"))
        self.assertEqual(output.tokens.device.type, "cuda")

    def test_short_and_long_complete_patch_policy(self):
        with self.assertRaises(RuntimeError):
            exact_patch_masks(torch.tensor([500]), 1_520, self.small)
        temporal_valid, _ = exact_patch_masks(
            torch.tensor([880, 1_520]), 1_520, self.small
        )
        self.assertEqual(temporal_valid.sum(dim=1).tolist(), [2, 4])

    def test_interval_boundary_overlap_and_gap_regression(self):
        flat = torch.zeros(1, 8, 2)
        output = temporalize_transformer_output(
            flat,
            torch.tensor([880]),
            1_520,
            torch.tensor([0.0]),
            self.small,
        )
        supervision = intervals_to_token_supervision(
            output.time_map,
            output.token_mask,
            [
                [
                    HFInterval(
                        "I", 0.000, 0.035, ObservationState.OBSERVED, True
                    ),
                    HFInterval(
                        "E", 0.035, 0.055, ObservationState.OBSERVED, False
                    ),
                    HFInterval(
                        "CAS", 0.000, 0.055, ObservationState.MISSING
                    ),
                ]
            ],
        )
        self.assertTrue(supervision.valid_mask[0, 0, 0])
        self.assertFalse(supervision.valid_mask[0, 0, 1])
        self.assertTrue(supervision.valid_mask[0, 1, 1])
        self.assertEqual(supervision.valid_mask[:, :, 2:].sum().item(), 0)
        self.assertFalse(supervision.targets[0, 1, 1])

    def test_raw_conservative_positive_only_masks_all_gaps(self):
        time_map = torch.tensor(
            [[[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]], dtype=torch.float32
        )
        token_mask = torch.tensor([[True, True, True]])
        supervision = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[HFRawInterval("I", 0.0, 0.1)]],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        self.assertTrue(supervision.targets[0, 0, 0])
        self.assertEqual(supervision.valid_mask.sum().item(), 1)
        self.assertEqual(supervision.receipt["constructed_negative_values"], 0)
        self.assertEqual(
            supervision.receipt["negative_semantics"], "none_raw_positive_only"
        )
        self.assertFalse(supervision.receipt["detector_closed"])
        self.assertFalse(supervision.receipt["shared_label_eligible"])

    def test_paper_native_rasterized_negatives_have_bounded_receipt(self):
        time_map = torch.tensor(
            [[[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]], dtype=torch.float32
        )
        token_mask = torch.tensor([[True, True, True]])
        supervision = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[HFRawInterval("I", 0.0, 0.1)]],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        )
        self.assertTrue(supervision.valid_mask.all())
        self.assertEqual(supervision.targets.sum().item(), 1)
        self.assertEqual(supervision.receipt["constructed_negative_values"], 11)
        self.assertEqual(
            supervision.receipt["negative_semantics"],
            "source_task_constructed_not_raw_normal",
        )
        self.assertFalse(supervision.receipt["shared_label_eligible"])
        self.assertIn(
            "docs/datasets/four_dataset_task_contract_review",
            supervision.receipt["source_task_policy_reference"],
        )

    def test_cas_das_raw_mapping_and_overlap(self):
        time_map = torch.tensor(
            [[[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]], dtype=torch.float32
        )
        token_mask = torch.tensor([[True, True, True]])
        supervision = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [
                [
                    HFRawInterval("Wheeze", 0.0, 0.2),
                    HFRawInterval("Rhonchi", 0.1, 0.3),
                    HFRawInterval("D", 0.1, 0.2),
                ]
            ],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        )
        self.assertEqual(
            supervision.targets[0, :, 2].tolist(), [1.0, 1.0, 1.0]
        )
        self.assertEqual(
            supervision.targets[0, :, 3].tolist(), [0.0, 1.0, 0.0]
        )

    def test_token_center_half_open_boundary(self):
        time_map = torch.tensor([[[0.0, 0.1]]], dtype=torch.float32)
        token_mask = torch.tensor([[True]])
        at_start = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[HFRawInterval("I", 0.05, 0.08)]],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        at_end = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[HFRawInterval("I", 0.02, 0.05)]],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        self.assertTrue(at_start.targets[0, 0, 0])
        self.assertFalse(at_end.valid_mask.any())
        self.assertEqual(
            at_start.receipt["alignment"], "token_center_in_interval"
        )

    def test_empty_annotation_policies_and_missing_fail_closed(self):
        time_map = torch.tensor(
            [[[0.0, 0.1], [0.1, 0.2]]], dtype=torch.float32
        )
        token_mask = torch.tensor([[True, True]])
        conservative = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[]],
            [ObservationState.EMPTY],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        self.assertFalse(conservative.valid_mask.any())
        paper = raw_intervals_to_token_supervision(
            time_map,
            token_mask,
            [[]],
            [ObservationState.EMPTY],
            policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        )
        self.assertTrue(paper.valid_mask.all())
        self.assertEqual(paper.targets.sum().item(), 0)
        self.assertEqual(paper.receipt["constructed_negative_values"], 8)
        self.assertEqual(
            paper.receipt["negative_semantics"],
            "source_task_constructed_not_raw_normal",
        )
        with self.assertRaises(RuntimeError):
            raw_intervals_to_token_supervision(
                time_map,
                token_mask,
                [[]],
                [ObservationState.MISSING],
                policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
            )

    def test_conflicting_overlap_fails(self):
        time_map = torch.tensor([[[0.0, 0.1]]], dtype=torch.float32)
        token_mask = torch.tensor([[True]])
        with self.assertRaises(RuntimeError):
            intervals_to_token_supervision(
                time_map,
                token_mask,
                [
                    [
                        HFInterval(
                            "I", 0.0, 0.1, ObservationState.OBSERVED, True
                        ),
                        HFInterval(
                            "I", 0.0, 0.1, ObservationState.OBSERVED, False
                        ),
                    ]
                ],
            )

    def test_non_hf_pooled_parity_interface(self):
        reference = torch.tensor([[1.0, 2.0]])
        receipt = verify_non_hf_pooled_parity(reference.clone(), reference)
        self.assertEqual(receipt["status"], "pooled_parity_passed")
        with self.assertRaises(RuntimeError):
            verify_non_hf_pooled_parity(reference + 0.1, reference)


class JointNativeContractTest(unittest.TestCase):
    def test_native_head_gradient_routing_and_shared_projector(self):
        torch.manual_seed(20260728)
        model = JointNativeProjector()
        icbhi = model(torch.randn(3, 768), "ICBHI")["flat4"]
        F.cross_entropy(icbhi, torch.tensor([0, 1, 2])).backward()
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.heads["ICBHI:flat4"].weight.grad.abs().sum().item(), 0)
        for name, head in model.heads.items():
            if name != "ICBHI:flat4":
                self.assertIsNone(head.weight.grad)

        model.zero_grad(set_to_none=True)
        losses = []
        for lane, task_targets in {
            "ICBHI": {"flat4": torch.tensor([1, 2])},
            "SPRSound": {
                "binary": torch.tensor([0, 1]),
                "raw7": torch.tensor([2, 3]),
            },
            "HF": {"temporal4": torch.zeros(2, 3, 4)},
            "KAUH": {"raw9": torch.tensor([4, 5])},
        }.items():
            values = torch.randn(2, 3, 768) if lane == "HF" else torch.randn(2, 768)
            outputs = model(values, lane)
            for task, target in task_targets.items():
                losses.append(
                    F.binary_cross_entropy_with_logits(outputs[task], target)
                    if lane == "HF"
                    else F.cross_entropy(outputs[task], target)
                )
        sum(losses).backward()
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
        for head in model.heads.values():
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(
            tuple(model(torch.randn(2, 3, 768), "HF")["temporal4"].shape),
            (2, 3, 4),
        )

    def test_frozen_encoder_gate(self):
        encoder = nn.Linear(3, 3)
        with self.assertRaises(RuntimeError):
            assert_frozen_encoder(encoder)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        assert_frozen_encoder(encoder)

    def test_source_proportional_receipt_is_deterministic(self):
        counts = {"ICBHI": 10, "SPRSound": 20, "HF": 30, "KAUH": 40}
        first = build_source_proportional_receipt(counts, draws=20)
        second = build_source_proportional_receipt(counts, draws=20)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 20260728)
        self.assertEqual(first["probabilities"]["KAUH"], 0.4)
        self.assertEqual(first["projector_architecture"], "minimal_linear_projector")
        self.assertTrue(first["projector_bias"])
        with self.assertRaises(ValueError):
            build_source_proportional_receipt(counts, draws=20, seed=17)

    def test_p1_p2_matched_config_and_fail_closed(self):
        p1 = CoreConfig("P1", "AST", "frozen-split")
        p2 = CoreConfig("P2", "BEATs", "frozen-split")
        assert_p1_p2_matched(p1, p2)
        with self.assertRaises(RuntimeError):
            p1.require_execution_ready()
        with self.assertRaises(RuntimeError):
            assert_p1_p2_matched(p1, replace(p2, sampler="dataset_balanced"))
        with self.assertRaises(RuntimeError):
            assert_p1_p2_matched(p1, replace(p2, split_digest="changed-split"))
        with self.assertRaises(ValueError):
            CoreConfig("P1", "BEATs", "frozen-split").validate_static_contract()
        with self.assertRaises(ValueError):
            CoreConfig("P2", "AST", "frozen-split").validate_static_contract()
        with self.assertRaises(ValueError):
            replace(p1, seed=17).validate_static_contract()

    def test_minimal_linear_projector_receipt(self):
        model = JointNativeProjector()
        receipt = model.architecture_receipt()
        self.assertEqual(receipt["architecture"], "minimal_linear_projector")
        self.assertEqual(receipt["input_dim"], 768)
        self.assertEqual(receipt["output_dim"], 256)
        self.assertTrue(receipt["bias"])
        self.assertTrue(receipt["hf_uses_projector"])
        self.assertIsNotNone(model.projector.bias)


class SlidingWindowContractTest(unittest.TestCase):
    def test_boundary_short_long_tail_and_batch_invariance(self):
        self.assertEqual(source_window_starts(WINDOW_SAMPLES), (0,))
        self.assertEqual(
            source_window_starts(WINDOW_SAMPLES + WINDOW_STRIDE_SAMPLES),
            (0, WINDOW_STRIDE_SAMPLES),
        )
        short = sample("ICBHI", "cycle", 800, "short", start_s=5.0)
        long = sample("KAUH", "recording", 72_000, "long")
        alone = collate_sliding_windows([short])
        together = collate_sliding_windows([short, long])
        self.assertEqual(alone.receipt()["window_counts"], [1])
        receipt = together.receipt()
        self.assertEqual(receipt["window_counts"], [1, 4])
        self.assertFalse(receipt["repeat_pad"])
        self.assertFalse(receipt["truncate"])
        torch.testing.assert_close(
            alone.waveform_windows[0, 0], together.waveform_windows[0, 0]
        )
        self.assertEqual(together.valid_samples[0, 0].item(), 800)
        self.assertTrue(together.waveform_padding_mask[0, 0, 800:].all())
        self.assertEqual(
            torch.count_nonzero(together.waveform_windows[0, 0, 800:]).item(), 0
        )
        self.assertEqual(
            together.time_map[1, :4, 0].tolist(), [0.0, 1.0, 2.0, 2.5]
        )
        self.assertEqual(
            together.time_map[1, 0].tolist(), [0.0, 2.0]
        )
        self.assertEqual(
            together.time_map[1, 1].tolist(), [1.0, 3.0]
        )
        self.assertEqual(together.time_map[1, 3, 1].item(), 4.5)
        self.assertEqual(len(set(together.time_map[1, :4, 0].tolist())), 4)
        self.assertAlmostEqual(together.time_map[0, 0, 0].item(), 5.0)
        self.assertAlmostEqual(together.time_map[0, 0, 1].item(), 5.05)
        moved = together.to("cpu")
        self.assertEqual(moved.lineage, together.lineage)
        self.assertTrue(torch.equal(moved.window_mask, together.window_mask))

    def test_non_hf_masked_aggregation_ignores_invalid_slots(self):
        embeddings = torch.tensor(
            [
                [[1.0, 3.0], [99.0, 99.0], [99.0, 99.0]],
                [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
            ]
        )
        mask = torch.tensor([[True, False, False], [True, True, True]])
        pooled = masked_mean_window_embeddings(
            embeddings, mask, expected_dim=2
        )
        torch.testing.assert_close(
            pooled, torch.tensor([[1.0, 3.0], [4.0, 8.0]])
        )

    def test_hf_center_alignment_and_negative_semantics(self):
        batch = collate_sliding_windows(
            [sample("HF", "recording_15s_with_intervals", 240_000, "hf")]
        )
        rows = [
            HFRawInterval("I", 0.9, 1.1, ObservationState.OBSERVED),
            HFRawInterval("Wheeze", 1.9, 2.1, ObservationState.OBSERVED),
            HFRawInterval("D", 2.9, 3.1, ObservationState.OBSERVED),
        ]
        paper = hf_window_supervision(
            batch,
            [rows],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        )
        self.assertEqual(batch.window_mask.sum().item(), 14)
        self.assertEqual(paper.targets[0, 0, 0].item(), 1.0)
        self.assertEqual(paper.targets[0, 1, 2].item(), 1.0)
        self.assertEqual(paper.targets[0, 2, 3].item(), 1.0)
        self.assertEqual(
            paper.receipt["negative_semantics"],
            "source_task_constructed_not_raw_normal",
        )
        self.assertFalse(paper.receipt["shared_label_eligible"])
        conservative = hf_window_supervision(
            batch,
            [rows],
            [ObservationState.OBSERVED],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        self.assertEqual(conservative.receipt["constructed_negative_values"], 0)
        self.assertEqual(conservative.valid_mask.sum().item(), 3)
        self.assertFalse(conservative.valid_mask[0, 3:].any())

    def test_hf_empty_and_missing_are_not_raw_normal(self):
        batch = collate_sliding_windows(
            [sample("HF", "recording_15s_with_intervals", 240_000, "hf")]
        )
        empty = hf_window_supervision(
            batch,
            [[]],
            [ObservationState.EMPTY],
            policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        )
        self.assertEqual(empty.targets.sum().item(), 0)
        self.assertGreater(empty.receipt["constructed_negative_values"], 0)
        with self.assertRaises(RuntimeError):
            hf_window_supervision(
                batch,
                [[]],
                [ObservationState.MISSING],
                policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
            )
        conservative = hf_window_supervision(
            batch,
            [[]],
            [ObservationState.MISSING],
            policy=HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY,
        )
        self.assertFalse(conservative.valid_mask.any())


class FirstQueuePreflightTest(unittest.TestCase):
    def test_budget_selection_and_p1_p5_match_are_frozen(self):
        self.assertEqual(P1_P5_UPDATES_PER_REFERENCE_EPOCH, 1_725)
        self.assertEqual(P1_P5_UPDATE_BUDGET, 86_250)
        self.assertIn("outer_test_excluded", P1_P5_SELECTION_RULE)
        chosen = select_validation_checkpoint(
            [
                (P1_P5_VALIDATION_INTERVAL_UPDATES, 0.8),
                (2 * P1_P5_VALIDATION_INTERVAL_UPDATES, 0.7),
                (3 * P1_P5_VALIDATION_INTERVAL_UPDATES, 0.7),
            ]
        )
        self.assertEqual(chosen, (3_450, 0.7))
        scalar, receipt = source_proportional_validation_selection_loss(
            {
                "ICBHI_flat4": 1.0,
                "SPRSound_binary": 2.0,
                "SPRSound_raw7": 4.0,
                "HF_temporal4": 6.0,
                "KAUH_raw9": 5.0,
            }
        )
        expected = (
            3_055 * 1.0 + 5_219 * 3.0 + 5_322 * 6.0 + 198 * 5.0
        ) / 13_794
        self.assertAlmostEqual(scalar, expected)
        self.assertFalse(receipt["reported_as_cross_dataset_score"])
        configs = [
            SharedWindowCoreConfig(pid, encoder, "frozen-split")
            for pid, encoder in PIPELINE_ENCODERS.items()
        ]
        assert_p1_p5_matched(configs)
        with self.assertRaises(RuntimeError):
            changed = list(configs)
            changed[2] = replace(changed[2], sampler="dataset_balanced")
            assert_p1_p5_matched(changed)

    def test_shared_window_output_and_dimension_adapters(self):
        batch = collate_sliding_windows(
            [
                sample("ICBHI", "cycle", 800, "i"),
                sample("SPRSound", "event", 40_000, "s"),
                sample("KAUH", "recording", 72_000, "k"),
            ]
        )
        output = SharedWindowEncoderOutput(
            embeddings=torch.zeros(*batch.window_mask.shape, 768),
            window_mask=batch.window_mask,
            time_map=batch.time_map,
            encoder_identity="AST",
            sample_ids=batch.sample_ids,
            dataset_ids=batch.dataset_ids,
            prediction_units=batch.prediction_units,
        )
        output.validate_against(batch)
        invalid = output.embeddings.clone()
        invalid[~batch.window_mask] = 1.0
        with self.assertRaises(ValueError):
            replace(output, embeddings=invalid).validate_against(batch)
        self.assertEqual(
            tuple(CandidateDimensionAdapter("PANNs_Cnn14")(torch.zeros(2, 3, 2048)).shape),
            (2, 3, 768),
        )
        hear = CandidateDimensionAdapter("HeAR")
        self.assertTrue(hear.receipt()["package_level_comparison"])
        self.assertEqual(
            CandidateDimensionAdapter("AST").receipt()["trainable_parameters"], 0
        )

    def test_hf_window_head_loss_and_shared_projector(self):
        torch.manual_seed(20260728)
        model = JointNativeProjector()
        embeddings = torch.randn(2, 3, 768)
        logits = model(embeddings, "HF")["temporal4"]
        self.assertEqual(tuple(logits.shape), (2, 3, 4))
        targets = torch.zeros_like(logits)
        observation = torch.ones_like(logits, dtype=torch.bool)
        valid = observation.clone()
        valid[0, 0, 2] = False
        logits.retain_grad()
        loss, receipt = hf_masked_channel_balanced_bce(
            logits, targets, observation, valid
        )
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 2].item(), 0.0)
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0.0)
        self.assertEqual(receipt["denominators"]["CAS"], 5)
        self.assertFalse(receipt["shared_label_eligible"])
        self.assertIn("roc_auc", HF_NATIVE_METRICS["per_channel"])
        with self.assertRaises(RuntimeError):
            empty = torch.zeros_like(valid)
            hf_masked_channel_balanced_bce(logits.detach(), targets, empty, empty)


class FakeWindowBackend(FrozenWindowBackend):
    def __init__(self, native_dim: int) -> None:
        super().__init__()
        self.native_dim = native_dim
        self.register_buffer("basis", torch.linspace(0.25, 1.25, native_dim))

    def encode_valid_windows(self, waveform_windows, valid_samples):
        positions = torch.arange(
            waveform_windows.shape[1], device=waveform_windows.device
        ).unsqueeze(0)
        mask = positions < valid_samples.unsqueeze(1)
        means = (waveform_windows * mask).sum(dim=1) / valid_samples
        return means.unsqueeze(1) * self.basis.unsqueeze(0)


class FakePANNsModel(nn.Module):
    def forward(self, waveform, mixup_lambda=None):
        means = waveform.mean(dim=1, keepdim=True)
        return {"embedding": means.expand(-1, 2048)}


class FakeTensorFlow:
    @staticmethod
    def convert_to_tensor(value):
        return value


class FakeNumpyTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def fake_provenance(identity: str) -> AdapterProvenance:
    return AdapterProvenance(
        encoder_identity=identity,
        source_url="fixture://source",
        source_revision="0" * 40,
        source_license="test-only",
        checkpoint_name="fixture.pt",
        checkpoint_source="deterministic fixture",
        checkpoint_sha256="0" * 64,
        checkpoint_size_bytes=0,
        asset_status="synthetic_fixture_not_model_asset",
    )


class ProductionWindowAdapterTest(unittest.TestCase):
    def test_flatten_restore_lineage_padding_and_batch_invariance(self):
        short = sample("ICBHI", "cycle", 800, "short", start_s=3.0)
        long = sample("KAUH", "recording", 48_000, "long")
        adapter = ProductionWindowEncoder(
            "AST", FakeWindowBackend(768), fake_provenance("AST")
        )
        alone_batch = collate_sliding_windows([short])
        mixed_batch = collate_sliding_windows([short, long])
        alone = adapter(alone_batch)
        mixed = adapter(mixed_batch)
        torch.testing.assert_close(alone.embeddings[0, 0], mixed.embeddings[0, 0])
        self.assertEqual(mixed.embeddings.shape, (2, 2, 768))
        self.assertEqual(torch.count_nonzero(mixed.embeddings[0, 1]).item(), 0)
        self.assertEqual(mixed.sample_ids, ("short", "long"))
        self.assertTrue(torch.equal(mixed.window_mask, mixed_batch.window_mask))
        self.assertTrue(torch.equal(mixed.time_map, mixed_batch.time_map))
        self.assertTrue(torch.isfinite(mixed.embeddings).all())

    def test_dimension_adapter_is_trainable_and_invalid_slots_stay_zero(self):
        adapter = ProductionWindowEncoder(
            "PANNs_Cnn14",
            FakeWindowBackend(2048),
            fake_provenance("PANNs_Cnn14"),
            dimension_adapter=CandidateDimensionAdapter("PANNs_Cnn14"),
        )
        batch = collate_sliding_windows(
            [
                sample("SPRSound", "event", 1_000, "short"),
                sample("KAUH", "recording", 48_000, "long"),
            ]
        )
        output = adapter(batch)
        output.embeddings.sum().backward()
        gradients = [
            parameter.grad
            for parameter in adapter.dimension_adapter.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertEqual(torch.count_nonzero(output.embeddings[0, 1]).item(), 0)
        receipt = adapter.receipt()
        self.assertEqual(receipt["output_shape"], "[B,K,768]")
        self.assertEqual(receipt["candidate_encoder_scope"], "frozen_eval_no_grad")

    def test_checkpoint_identity_and_factory_fail_closed(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"adapter-asset")
            digest = sha256_file(path)
            receipt = require_file_identity(path, digest, expected_size_bytes=13)
            self.assertTrue(receipt["identity_verified"])
            with self.assertRaises(RuntimeError):
                require_file_identity(path, "0" * 64)
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(RuntimeError):
            build_production_adapter(AdapterFactoryConfig("P3", root))
        with self.assertRaises(RuntimeError):
            build_production_adapter(AdapterFactoryConfig("P4", root))

    def test_ast_frontend_zero_pads_fbank_without_repeat_or_truncate(self):
        waveform = torch.linspace(-0.1, 0.1, 32_000).repeat(2, 1)
        images = ASTWindowBackend.frontend(waveform)
        self.assertEqual(images.shape, (2, 1, 798, 128))
        torch.testing.assert_close(images[0], images[1])
        nonzero_frames = torch.count_nonzero(images[0, 0], dim=1).bool()
        self.assertTrue(nonzero_frames[:190].all())
        self.assertFalse(nonzero_frames[210:].any())

    def test_panns_and_hear_specific_backend_synthetic_cpu_smoke(self):
        waveforms = torch.linspace(-0.2, 0.2, 32_000).repeat(2, 1)
        valid = torch.tensor([32_000, 800], dtype=torch.long)
        panns = PANNsWindowBackend(FakePANNsModel())
        panns_values = panns.encode_valid_windows(waveforms, valid)
        self.assertEqual(panns_values.shape, (2, 2048))
        self.assertTrue(torch.isfinite(panns_values).all())

        def signature(*, x):
            means = x.mean(axis=1, keepdims=True)
            return {"output_0": FakeNumpyTensor(means.repeat(512, axis=1))}

        hear = HeARWindowBackend(
            signature, device="cpu", tensorflow_module=FakeTensorFlow()
        )
        hear_values = hear.encode_valid_windows(waveforms, valid)
        self.assertEqual(hear_values.shape, (2, 512))
        self.assertTrue(torch.isfinite(hear_values).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_p6_token_head_cuda_contract(self):
        device = torch.device("cuda")
        head = P6TokenTemporalHead().to(device)
        tokens = torch.randn(2, 3, 256, device=device)
        logits = head(tokens)
        targets = torch.zeros_like(logits)
        mask = torch.ones_like(logits, dtype=torch.bool)
        loss, receipt = hf_masked_channel_balanced_bce(logits, targets, mask, mask)
        loss.backward()
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(set(receipt["denominators"]), {"I", "E", "CAS", "DAS"})

    def test_preflight_separates_code_asset_and_scientific_holds(self):
        receipt = freeze_receipt()
        self.assertFalse(receipt["experiment_result"])
        self.assertEqual(receipt["p1_p5"]["update_budget"], 86_250)
        self.assertEqual(receipt["window_policy"]["status"], "proposed_benchmark_policy")
        self.assertIn("code_READY", PRODUCTION_ADAPTER_STATUS["AST"]["status"])
        self.assertIn("code_READY", PRODUCTION_ADAPTER_STATUS["BEATs"]["status"])
        self.assertIn("asset_HOLD", PRODUCTION_ADAPTER_STATUS["PANNs_Cnn14"]["status"])
        self.assertIn("HOLD", PRODUCTION_ADAPTER_STATUS["HeAR"]["status"])
        self.assertTrue(PRODUCTION_ADAPTER_STATUS["OPERA_CT"]["status"].startswith("HOLD"))
        self.assertFalse(receipt["p6"]["first_round_required"])

    def test_independent_verifier_schema_rejects_pooled_score_and_p6(self):
        base = {
            "schema_version": "shared_window_first_queue_verifier_v2",
            "pipeline_id": "P1",
            "verifier_identity": "new_independent_model_design_verifier",
            "verifier_code_commit": "pending",
            "subject_code_commit": "pending",
            "config_sha256": "pending",
            "provider_data_identity_sha256": "pending",
            "approval_receipt_sha256": "pending",
            "manifest_sha256_by_dataset": {},
            "split_sha256_by_dataset": {},
            "checkpoint_sha256_by_component": {},
            "checkpoint_artifact_receipts": [],
            "window_contract_receipt": {},
            "encoder_adapter_receipt": {},
            "adapter_asset_manifest_receipt": {},
            "embedding_cache_receipt": {},
            "runner_schema_version": "shared_window_training_v5",
            "phase_gate_receipt": {},
            "optimizer_receipt": {},
            "resume_receipt": {},
            "terminal_binding_receipt": {},
            "terminal_scorer_receipt": {},
            "seed": 20260728,
            "update_budget": 86250,
            "selection_receipt": {},
            "validation_selection_receipt": {},
            "validation_selection_receipt_sha256": "pending",
            "trainable_scope_receipt": {},
            "native_metrics_by_dataset_task": {"ICBHI": {}, "HF": {}},
            "outer_test_access_receipt": {"accessed": False},
            "artifact_sha256": {},
            "gate_results": {},
            "warnings": [],
            "status": "preflight_only",
        }
        validate_independent_verifier_receipt(base)
        with self.assertRaises(ValueError):
            validate_independent_verifier_receipt({**base, "pipeline_id": "P6"})
        with self.assertRaises(ValueError):
            validate_independent_verifier_receipt(
                {
                    **base,
                    "native_metrics_by_dataset_task": {"pooled_score": 0.5},
                }
            )


class EligibilityObjectiveTest(unittest.TestCase):
    def test_narrow_mapping_exclusion_and_masked_zero_gradient(self):
        rows = [
            CompatibleRow("ICBHI", "both", ObservationState.OBSERVED),
            CompatibleRow("SPRSound", "Wheeze", ObservationState.OBSERVED),
            CompatibleRow("HF", "I", ObservationState.OBSERVED),
            CompatibleRow("KAUH", "N", ObservationState.OBSERVED),
            CompatibleRow("ICBHI", "normal", ObservationState.MISSING),
        ]
        built = build_eligibility_targets(rows)
        self.assertEqual(built.receipt["eligible_denominator"], 4)
        self.assertTrue(built.eligible_mask[:2].all())
        self.assertFalse(built.eligible_mask[2:].any())
        logits = torch.zeros_like(built.targets, requires_grad=True)
        loss, receipt = eligibility_masked_bce(
            logits, built.targets, built.eligible_mask
        )
        loss.backward()
        self.assertEqual(receipt["eligible_denominator"], 4)
        self.assertGreater(logits.grad[:2].abs().sum().item(), 0)
        self.assertEqual(logits.grad[2:].abs().sum().item(), 0)

    def test_all_unobserved_states_remain_masked(self):
        rows = [
            CompatibleRow("ICBHI", "normal", state)
            for state in (
                ObservationState.MISSING,
                ObservationState.UNKNOWN,
                ObservationState.NOT_ANNOTATED,
                ObservationState.EMPTY,
            )
        ]
        built = build_eligibility_targets(rows)
        self.assertFalse(built.eligible_mask.any())
        self.assertEqual(
            built.receipt["state_counts"],
            {
                "empty": 1,
                "missing": 1,
                "not_annotated": 1,
                "unknown": 1,
            },
        )

    def test_zero_eligible_denominator_fails_safely(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        with self.assertRaises(ZeroEligibleDenominator):
            eligibility_masked_bce(
                logits, torch.zeros_like(logits), torch.zeros_like(logits).bool()
            )
        self.assertIsNone(logits.grad)

    def test_spr_narrow_mapping_boundary(self):
        rhonchi = build_eligibility_targets(
            [CompatibleRow("SPRSound", "Rhonchi", ObservationState.OBSERVED)]
        )
        self.assertFalse(rhonchi.eligible_mask.any())
        with self.assertRaises(ValueError):
            build_eligibility_targets(
                [CompatibleRow("SPRSound", "Unmapped", ObservationState.OBSERVED)]
            )

    def test_p8_schema_and_guardrails_fail_closed(self):
        P8ObjectiveConfig().validate()
        with self.assertRaises(ValueError):
            P8ObjectiveConfig(comparator="P2").validate()
        with self.assertRaises(RuntimeError):
            GuardrailSchema().require_execution_ready()


if __name__ == "__main__":
    unittest.main()
