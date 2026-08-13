from __future__ import annotations

import json
import hashlib
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from baseline.multidataset_pipeline.asset_manifest import (
    ASSET_MANIFEST_SCHEMA_VERSION,
    load_adapter_asset_manifest,
    manifest_asset_paths,
)
from baseline.multidataset_pipeline.embedding_cache import (
    SCHEMA_IDENTITY_SHA256,
    EmbeddingCacheIdentity,
    EmbeddingCachePayload,
    FrozenEmbeddingCache,
    load_embedding_cache,
    payloads_equal,
)
from baseline.multidataset_pipeline.hf_thresholds import (
    HF_THRESHOLD_SELECTION_POLICY,
    VerifiedHFThresholdReceipt,
    load_and_verify_hf_threshold_receipt,
    threshold_receipt_payload,
    write_hf_threshold_receipt,
)
from baseline.multidataset_pipeline.runner_embedding_cache import (
    PIPELINE_CODE_DEPENDENCIES,
    _file_set_sha256,
)
from baseline.multidataset_pipeline.terminal_scoring import (
    HF_NEGATIVE_SEMANTICS,
    HFTemporalTerminalBatch,
    MulticlassTerminalBatch,
    NATIVE_TASKS,
    ProductionTerminalScorer,
    TerminalScoringInput,
    _reject_forbidden_keys,
    audit_terminal_provider_registration,
)


def _cache_identity(**updates: object) -> EmbeddingCacheIdentity:
    code_dependencies = {
        "baseline/multidataset_pipeline/fixture_encoder.py": "5" * 64
    }
    code_identity = hashlib.sha256(
        json.dumps(
            code_dependencies, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    values: dict[str, object] = {
        "dataset_id": "ICBHI",
        "dataset_release": "ICBHI_2017_manifest_fixture",
        "partition": "subtrain",
        "ordered_unit_ids": ("cycle-1", "cycle-2"),
        "data_identity_sha256": "1" * 64,
        "preprocessing": {
            "sample_rate": 16_000,
            "resample_policy": "torchaudio_pinned_fixture",
            "waveform_dtype": "float32",
        },
        "window_policy": {
            "window_length_s": 2.0,
            "window_stride_s": 1.0,
            "tail_policy": "append_unique_end_aligned_window_when_stride_misses_tail",
            "short_padding": "zero_pad_only",
            "repeat_pad": False,
            "truncate": False,
            "mask_semantics": "window_mask_true_is_valid;time_map_source_seconds_half_open",
        },
        "encoder_asset": {
            "encoder_identity": "AST",
            "source_url": "https://example.invalid/source",
            "source_revision": "revision-fixture",
            "checkpoint_sha256": "2" * 64,
            "checkpoint_size_bytes": 123,
            "license": "MIT-fixture",
            "asset_manifest_identity_sha256": "3" * 64,
        },
        "frontend_adapter_identity": {
            "frontend": "deterministic-fixture",
            "adapter": "identity-768",
            "identity_sha256": "4" * 64,
        },
        "output_dtype": "float32",
        "code_identity_sha256": code_identity,
        "code_dependency_sha256_by_path": code_dependencies,
        "config_identity_sha256": "6" * 64,
        "schema_identity_sha256": SCHEMA_IDENTITY_SHA256,
    }
    values.update(updates)
    return EmbeddingCacheIdentity(**values)


def _verified_threshold_fixture() -> VerifiedHFThresholdReceipt:
    return VerifiedHFThresholdReceipt(
        path=Path("threshold-fixture.json"),
        artifact_sha256="7" * 64,
        size_bytes=1,
        payload={"threshold_selection_policy": HF_THRESHOLD_SELECTION_POLICY},
        thresholds=(0.5, 0.5, 0.5, 0.5),
    )


def _cache_payload() -> EmbeddingCachePayload:
    return EmbeddingCachePayload(
        unit_ids=("cycle-1", "cycle-2"),
        embeddings=(
            torch.arange(12, dtype=torch.float32).reshape(2, 6),
            torch.arange(18, dtype=torch.float32).reshape(3, 6) / 7,
        ),
        time_maps=(
            torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float64),
            torch.tensor([[2.0, 4.0], [3.0, 5.0], [4.0, 5.5]], dtype=torch.float64),
        ),
        valid_samples=(
            torch.tensor([32_000, 32_000]),
            torch.tensor([32_000, 32_000, 24_000]),
        ),
        window_masks=(
            torch.ones(2, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool),
        ),
    )


class _CountingFrozenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return values * 1.25 - 0.5


def _terminal_input(
    *,
    ids_override: dict[str, tuple[str, ...]] | None = None,
    hf_override: HFTemporalTerminalBatch | None = None,
) -> TerminalScoringInput:
    identifiers = {
        "ICBHI_flat4": ("icbhi-0", "icbhi-1"),
        "SPRSound_binary": ("spr-0", "spr-1"),
        "SPRSound_raw7": ("spr-0", "spr-1"),
        "HF_temporal4": ("hf-0", "hf-1"),
        "KAUH_raw9": ("kauh-0", "kauh-1"),
    }
    if ids_override:
        identifiers.update(ids_override)
    batches = [
        MulticlassTerminalBatch(
            task=task,
            prediction_ids=("spr-0", "spr-1") if task.startswith("SPRSound") else identifiers[task],
            targets=torch.tensor([0, 1]),
            predicted_classes=torch.tensor([0, 1]),
        )
        for task in (
            "ICBHI_flat4",
            "SPRSound_binary",
            "SPRSound_raw7",
            "KAUH_raw9",
        )
    ]
    target = torch.tensor(
        [
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    hf = hf_override or HFTemporalTerminalBatch(
        prediction_ids=("hf-0", "hf-1"),
        probabilities=target * 0.8 + (1 - target) * 0.2,
        targets=target,
        window_mask=torch.tensor([[True, True, False], [True, True, False]]),
        annotation_mask=torch.ones(2, 3, 4, dtype=torch.bool),
        valid_mask=torch.tensor(
            [
                [[True] * 4, [True] * 4, [False] * 4],
                [[True] * 4, [True] * 4, [False] * 4],
            ]
        ),
        time_map=torch.tensor(
            [
                [[0.0, 2.0], [1.0, 3.0], [0.0, 0.0]],
                [[0.0, 2.0], [1.0, 3.0], [0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        thresholds=torch.full((4,), 0.5),
        threshold_receipt_sha256="7" * 64,
    )
    batches.append(hf)
    return TerminalScoringInput(
        batches=tuple(batches),
        expected_prediction_ids_by_task=identifiers,
        data_identity_sha256="8" * 64,
        provider_identity_sha256="9" * 64,
    )


class FrozenEmbeddingCacheTest(unittest.TestCase):
    def test_p1_p2_dependency_closure_is_explainable_and_candidate_sensitive(self):
        source_root = Path(__file__).resolve().parents[1]
        for pipeline_id, candidate in (
            ("P1", "baseline/multidataset_pipeline/ast_window_encoder.py"),
            ("P2", "baseline/multidataset_pipeline/beats_temporal.py"),
        ):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                for relative in PIPELINE_CODE_DEPENDENCIES[pipeline_id]:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_root / relative, destination)
                before = _file_set_sha256(root, pipeline_id)
                self.assertEqual(
                    set(before["files"]), set(PIPELINE_CODE_DEPENDENCIES[pipeline_id])
                )
                self.assertFalse(
                    any(path.startswith(("tests/", "docs/")) for path in before["files"])
                )
                target = root / candidate
                target.write_bytes(target.read_bytes() + b"\n# dependency-mutation-fixture\n")
                after = _file_set_sha256(root, pipeline_id)
                self.assertNotEqual(
                    before["aggregate_sha256"], after["aggregate_sha256"]
                )
                self.assertNotEqual(before["files"][candidate], after["files"][candidate])
                before_identity = _cache_identity(
                    code_identity_sha256=before["aggregate_sha256"],
                    code_dependency_sha256_by_path=before["files"],
                )
                after_identity = _cache_identity(
                    code_identity_sha256=after["aggregate_sha256"],
                    code_dependency_sha256_by_path=after["files"],
                )
                self.assertNotEqual(before_identity.cache_key(), after_identity.cache_key())
                (root / "baseline/multidataset_pipeline/preflight.py").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "identity file missing"):
                    _file_set_sha256(root, pipeline_id)

    def test_exact_hit_avoids_encoder_call_and_preserves_ragged_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "embedding_cache"
            identity = _cache_identity()
            uncached = _cache_payload()
            encoder = _CountingFrozenEncoder().eval()

            def compute() -> EmbeddingCachePayload:
                packed = torch.cat(uncached.embeddings)
                encoded = encoder(packed)
                return EmbeddingCachePayload(
                    unit_ids=uncached.unit_ids,
                    embeddings=(encoded[:2], encoded[2:]),
                    time_maps=uncached.time_maps,
                    valid_samples=uncached.valid_samples,
                    window_masks=uncached.window_masks,
                )

            cache = FrozenEmbeddingCache(root)
            first, first_receipt = cache.get_or_compute(
                identity,
                encoder,
                compute,
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            second, second_receipt = cache.get_or_compute(
                identity,
                encoder,
                compute,
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            self.assertEqual(encoder.calls, 1)
            self.assertTrue(payloads_equal(compute(), first))
            self.assertEqual(encoder.calls, 2)
            # The verified cache hit itself did not invoke the encoder again.
            self.assertTrue(payloads_equal(first, second))
            self.assertEqual(first_receipt["uncached_equivalence"], "exact")
            self.assertEqual(second_receipt["cache_status"], "hit_verified_existing")
            self.assertEqual(first_receipt["unit_count"], 2)
            self.assertEqual(first_receipt["total_valid_windows"], 5)

    def test_cache_rejects_forbidden_scope_identity_and_corruption(self):
        with self.assertRaises(PermissionError):
            _cache_identity(partition="test").validate()
        with self.assertRaises(ValueError):
            _cache_identity(ordered_unit_ids=("cycle-1", "cycle-1")).validate()
        with self.assertRaises(ValueError):
            _cache_identity(encoder_asset={}).validate()
        with TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                FrozenEmbeddingCache(Path(directory) / "outer_test")
            root = Path(directory) / "cache"
            identity = _cache_identity()
            payload = _cache_payload()
            trainable = nn.Linear(6, 6).eval()
            cache = FrozenEmbeddingCache(root)
            with self.assertRaisesRegex(RuntimeError, "frozen encoder"):
                cache.get_or_compute(
                    identity,
                    trainable,
                    lambda: payload,
                    frontend_deterministic=True,
                    augmentation_enabled=False,
                )
            for parameter in trainable.parameters():
                parameter.requires_grad_(False)
            trainable.train()
            with self.assertRaisesRegex(RuntimeError, "frozen encoder"):
                cache.get_or_compute(
                    identity,
                    trainable,
                    lambda: payload,
                    frontend_deterministic=True,
                    augmentation_enabled=False,
                )
            trainable.eval()
            with self.assertRaisesRegex(RuntimeError, "stochastic"):
                cache.get_or_compute(
                    identity,
                    trainable,
                    lambda: payload,
                    frontend_deterministic=False,
                    augmentation_enabled=True,
                )
            cache.get_or_compute(
                identity,
                trainable,
                lambda: payload,
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            artifact = root / identity.cache_key() / "embeddings.npy"
            with artifact.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                load_embedding_cache(root, identity)

    def test_identity_changes_key_and_partial_or_stale_manifest_fails(self):
        identity = _cache_identity()
        changed = _cache_identity(config_identity_sha256="a" * 64)
        self.assertNotEqual(identity.cache_key(), changed.cache_key())
        with TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            encoder = nn.Identity().eval()
            cache = FrozenEmbeddingCache(root)
            cache.get_or_compute(
                identity,
                encoder,
                _cache_payload,
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            manifest = root / identity.cache_key() / "cache_manifest.json"
            data = json.loads(manifest.read_text())
            data["identity"]["dataset_release"] = "stale"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity/schema"):
                load_embedding_cache(root, identity)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            identity = _cache_identity()
            encoder = nn.Identity().eval()
            FrozenEmbeddingCache(root).get_or_compute(
                identity,
                encoder,
                _cache_payload,
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            (root / identity.cache_key() / "window_mask.npy").unlink()
            with self.assertRaisesRegex(RuntimeError, "partial"):
                load_embedding_cache(root, identity)
        missing_unit = EmbeddingCachePayload(
            unit_ids=("cycle-1",),
            embeddings=_cache_payload().embeddings[:1],
            time_maps=_cache_payload().time_maps[:1],
            valid_samples=_cache_payload().valid_samples[:1],
            window_masks=_cache_payload().window_masks[:1],
        )
        with self.assertRaisesRegex(RuntimeError, "missing"):
            missing_unit.validate(identity)


class TerminalScoringTest(unittest.TestCase):
    def test_hf_threshold_receipt_byte_and_identity_chain_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hf_threshold_receipt.json"
            payload = threshold_receipt_payload(
                thresholds=(0.2, 0.3, 0.4, 0.5),
                validation_data_identity_sha256="1" * 64,
                hf_validation_manifest_identity_sha256="2" * 64,
                hf_validation_ordered_prediction_ids_sha256="3" * 64,
                full_approval_receipt_sha256="4" * 64,
                validation_selection_receipt_sha256="5" * 64,
                selected_checkpoint_sha256="6" * 64,
                validation_prediction_identity_sha256="7" * 64,
                per_channel_selection=[
                    {
                        "channel": channel,
                        "threshold": threshold,
                        "max_f1": 1.0,
                        "candidate_count": 1,
                        "valid_count": 2,
                        "positive_support": 1,
                        "negative_support": 1,
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                    }
                    for channel, threshold in zip(
                        ("I", "E", "CAS", "DAS"), (0.2, 0.3, 0.4, 0.5)
                    )
                ],
                scorer_schema_version="shared_window_terminal_scorer_v1",
            )
            artifact = write_hf_threshold_receipt(path, payload)
            verified = load_and_verify_hf_threshold_receipt(
                path,
                artifact["sha256"],
                expected_scorer_schema_version="shared_window_terminal_scorer_v1",
                expected_validation_data_identity_sha256="1" * 64,
                expected_hf_validation_manifest_identity_sha256="2" * 64,
                expected_hf_validation_ordered_prediction_ids_sha256="3" * 64,
                expected_full_approval_receipt_sha256="4" * 64,
                expected_validation_selection_receipt_sha256="5" * 64,
                expected_selected_checkpoint_sha256="6" * 64,
            )
            self.assertEqual(verified.thresholds, (0.2, 0.3, 0.4, 0.5))
            tampered = dict(payload)
            tampered["thresholds"] = [0.21, 0.3, 0.4, 0.5]
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "byte SHA256 mismatch"):
                load_and_verify_hf_threshold_receipt(
                    path,
                    artifact["sha256"],
                    expected_scorer_schema_version="shared_window_terminal_scorer_v1",
                    expected_validation_data_identity_sha256="1" * 64,
                    expected_hf_validation_manifest_identity_sha256="2" * 64,
                    expected_hf_validation_ordered_prediction_ids_sha256="3" * 64,
                    expected_full_approval_receipt_sha256="4" * 64,
                    expected_validation_selection_receipt_sha256="5" * 64,
                    expected_selected_checkpoint_sha256="6" * 64,
                )
            identity_fields = (
                "validation_data_identity_sha256",
                "hf_validation_manifest_identity_sha256",
                "hf_validation_ordered_prediction_ids_sha256",
                "full_approval_receipt_sha256",
                "validation_selection_receipt_sha256",
                "selected_checkpoint_sha256",
            )
            for index, field in enumerate(identity_fields):
                with self.subTest(field=field):
                    changed_identity = dict(payload)
                    changed_identity[field] = chr(ord("a") + index) * 64
                    changed_path = root / f"changed_{field}.json"
                    changed_artifact = write_hf_threshold_receipt(
                        changed_path, changed_identity
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "identity chain mismatch"
                    ):
                        load_and_verify_hf_threshold_receipt(
                            changed_path,
                            changed_artifact["sha256"],
                            expected_scorer_schema_version="shared_window_terminal_scorer_v1",
                            expected_validation_data_identity_sha256="1" * 64,
                            expected_hf_validation_manifest_identity_sha256="2" * 64,
                            expected_hf_validation_ordered_prediction_ids_sha256="3" * 64,
                            expected_full_approval_receipt_sha256="4" * 64,
                            expected_validation_selection_receipt_sha256="5" * 64,
                            expected_selected_checkpoint_sha256="6" * 64,
                        )

    def test_unregistered_production_provider_is_machine_readable_hold(self):
        with TemporaryDirectory() as directory:
            receipt = audit_terminal_provider_registration(Path(directory))
        self.assertEqual(
            receipt["status"], "HOLD_no_registered_production_terminal_provider"
        )
        self.assertFalse(receipt["terminal_score_ready"])
        self.assertFalse(receipt["provider_imported"])
        self.assertFalse(receipt["outer_test_accessed"])

    def test_all_native_tasks_exact_metrics_and_hf_mask_boundary(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            checkpoint.write_bytes(b"selected-checkpoint-fixture")
            scorer = ProductionTerminalScorer(
                lambda _: _terminal_input(),
                expected_provider_identity_sha256="9" * 64,
                provider_specification="fixture.module:provider",
            )
            receipt = scorer(
                checkpoint,
                verified_hf_threshold_receipt=_verified_threshold_fixture(),
            )
        self.assertEqual(receipt["native_task_names"], list(NATIVE_TASKS))
        json.dumps(receipt, allow_nan=False)
        self.assertEqual(set(receipt["native_tasks"]), set(NATIVE_TASKS))
        self.assertFalse(receipt["cross_dataset_pooling"])
        icbhi = receipt["native_tasks"]["ICBHI_flat4"]
        self.assertEqual(icbhi["metrics"]["confusion"][0][0], 1)
        self.assertEqual(icbhi["metrics"]["confusion"][1][1], 1)
        self.assertEqual(icbhi["denominator"], 2)
        hf = receipt["native_tasks"]["HF_temporal4"]
        self.assertEqual(hf["valid_source_windows"], 4)
        self.assertEqual(hf["negative_semantics"], HF_NEGATIVE_SEMANTICS)
        self.assertTrue(hf["raw_gap_missing_unknown_not_raw_negative"])
        self.assertFalse(hf["shared_label_eligible"])
        for channel in ("I", "E", "CAS", "DAS"):
            self.assertEqual(hf["per_channel"][channel]["denominator"], 4)
            self.assertEqual(hf["per_channel"][channel]["confusion"], [[2, 0], [0, 2]])
            self.assertAlmostEqual(hf["per_channel"][channel]["roc_auc"], 1.0)

    def test_missing_extra_duplicate_misaligned_and_semantics_fail_closed(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            checkpoint.write_bytes(b"fixture")
            missing = _terminal_input()
            missing = TerminalScoringInput(
                batches=missing.batches[:-1],
                expected_prediction_ids_by_task=missing.expected_prediction_ids_by_task,
                data_identity_sha256=missing.data_identity_sha256,
                provider_identity_sha256=missing.provider_identity_sha256,
            )
            with self.assertRaisesRegex(RuntimeError, "omitted"):
                ProductionTerminalScorer(
                    lambda _: missing,
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )
            original = _terminal_input()
            wrong_expected = dict(original.expected_prediction_ids_by_task)
            wrong_expected["ICBHI_flat4"] = ("icbhi-1", "icbhi-0")
            wrong_ids = TerminalScoringInput(
                batches=original.batches,
                expected_prediction_ids_by_task=wrong_expected,
                data_identity_sha256=original.data_identity_sha256,
                provider_identity_sha256=original.provider_identity_sha256,
            )
            with self.assertRaisesRegex(RuntimeError, "missing, duplicated, or out of order"):
                ProductionTerminalScorer(
                    lambda _: wrong_ids,
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )
            base = _terminal_input()
            hf = next(batch for batch in base.batches if batch.task == "HF_temporal4")
            bad_semantics = HFTemporalTerminalBatch(
                **{
                    **hf.__dict__,
                    "negative_semantics": "raw_gap_is_negative",
                }
            )
            bad = _terminal_input(hf_override=bad_semantics)
            with self.assertRaisesRegex(RuntimeError, "semantics"):
                ProductionTerminalScorer(
                    lambda _: bad,
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )
            arbitrary_sha = HFTemporalTerminalBatch(
                **{**hf.__dict__, "threshold_receipt_sha256": "a" * 64}
            )
            with self.assertRaisesRegex(RuntimeError, "verified threshold artifact"):
                ProductionTerminalScorer(
                    lambda _: _terminal_input(hf_override=arbitrary_sha),
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )
            arbitrary_thresholds = HFTemporalTerminalBatch(
                **{**hf.__dict__, "thresholds": torch.full((4,), 0.6)}
            )
            with self.assertRaisesRegex(RuntimeError, "verified threshold artifact"):
                ProductionTerminalScorer(
                    lambda _: _terminal_input(hf_override=arbitrary_thresholds),
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )
            duplicate = MulticlassTerminalBatch(
                task="ICBHI_flat4",
                prediction_ids=("same", "same"),
                targets=torch.tensor([0, 1]),
                predicted_classes=torch.tensor([0, 1]),
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                duplicate.validate()

    def test_hf_nonfinite_shape_empty_support_and_time_map_fail_closed(self):
        base = _terminal_input()
        hf = next(batch for batch in base.batches if batch.task == "HF_temporal4")
        with self.assertRaisesRegex(ValueError, r"\[B,Nw,4\]"):
            HFTemporalTerminalBatch(
                **{**hf.__dict__, "probabilities": torch.zeros(2, 3, 3)}
            ).validate()
        invalid_time = hf.time_map.clone()
        invalid_time[0, 1] = torch.tensor([0.0, 1.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            HFTemporalTerminalBatch(**{**hf.__dict__, "time_map": invalid_time}).validate()
        empty = hf.valid_mask.clone()
        empty[..., 0] = False
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.pt"
            checkpoint.write_bytes(b"fixture")
            inputs = _terminal_input(
                hf_override=HFTemporalTerminalBatch(**{**hf.__dict__, "valid_mask": empty})
            )
            with self.assertRaisesRegex(RuntimeError, "non-empty positive and negative"):
                ProductionTerminalScorer(
                    lambda _: inputs,
                    expected_provider_identity_sha256="9" * 64,
                    provider_specification="fixture.module:provider",
                )(
                    checkpoint,
                    verified_hf_threshold_receipt=_verified_threshold_fixture(),
                )

    def test_forbidden_pooled_or_ranking_keys_are_recursive(self):
        for key in ("pooled_score", "global_score", "cross_dataset_score", "ranking"):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                _reject_forbidden_keys({"native_tasks": {"bad": {key: 0.5}}})


class AdapterAssetManifestTest(unittest.TestCase):
    def test_tracked_manifest_has_exact_p1_p2_canonical_cache_paths(self):
        root = Path(__file__).resolve().parents[1]
        manifest = load_adapter_asset_manifest(root)
        self.assertEqual(manifest["schema_version"], ASSET_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(set(manifest["assets"]), {"P1", "P2"})
        _, p2_checkpoint, p2 = manifest_asset_paths(root, "P2")
        self.assertIn("/.cache/", str(p2_checkpoint))
        self.assertNotIn("result/pafa", str(p2_checkpoint))
        self.assertEqual(p2["checkpoint_sha256"], "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34")
        dependencies = {
            "baseline/multidataset_pipeline/fixture.py": "c" * 64
        }
        dependency_aggregate = hashlib.sha256(
            json.dumps(dependencies, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        bound = EmbeddingCacheIdentity.from_tracked_asset(
            repo_root=root,
            pipeline_id="P2",
            dataset_id="HF",
            dataset_release="hf-fixture",
            partition="validation",
            ordered_unit_ids=("hf-1",),
            data_identity_sha256="a" * 64,
            preprocessing={
                "sample_rate": 16_000,
                "resample_policy": "fixture",
                "waveform_dtype": "float32",
            },
            window_policy=_cache_identity().window_policy,
            frontend_adapter_identity={
                "frontend": "beats-fbank-fixture",
                "adapter": "identity-768",
                "identity_sha256": "b" * 64,
            },
            code_identity_sha256=dependency_aggregate,
            code_dependency_sha256_by_path=dependencies,
            config_identity_sha256="d" * 64,
        )
        self.assertEqual(bound.encoder_asset["checkpoint_sha256"], p2["checkpoint_sha256"])
        self.assertEqual(
            bound.encoder_asset["asset_manifest_identity_sha256"],
            manifest["manifest_identity_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
