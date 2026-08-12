from __future__ import annotations

import unittest
from dataclasses import replace

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
            "KAUH": {"raw9": torch.tensor([4, 5])},
        }.items():
            outputs = model(torch.randn(2, 768), lane)
            losses.extend(
                F.cross_entropy(outputs[task], target)
                for task, target in task_targets.items()
            )
        sum(losses).backward()
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
        for head in model.heads.values():
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 768), "HF")

    def test_frozen_encoder_gate(self):
        encoder = nn.Linear(3, 3)
        with self.assertRaises(RuntimeError):
            assert_frozen_encoder(encoder)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        assert_frozen_encoder(encoder)

    def test_source_proportional_receipt_is_deterministic(self):
        counts = {"ICBHI": 10, "SPRSound": 20, "KAUH": 30}
        first = build_source_proportional_receipt(counts, draws=20)
        second = build_source_proportional_receipt(counts, draws=20)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 20260728)
        self.assertEqual(first["probabilities"]["KAUH"], 0.5)
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
        self.assertIsNotNone(model.projector.bias)


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
