from __future__ import annotations

import json
import copy
import hashlib
import shutil
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.multidataset_pipeline.beats_temporal import HFRawInterval
from baseline.multidataset_pipeline.contracts import ObservationState, PREDICTION_UNITS, WaveformSample
from baseline.multidataset_pipeline.hf_data import HFSampleRecord
from baseline.multidataset_pipeline.joint_native import JointNativeProjector
from baseline.multidataset_pipeline.hf_thresholds import (
    HF_THRESHOLD_SELECTION_POLICY,
    threshold_receipt_payload,
    write_hf_threshold_receipt,
)
from baseline.multidataset_pipeline.l40_preflight import (
    _complete_state_snapshot,
    validate_pipeline_adapter_identity,
)
from baseline.multidataset_pipeline.preflight import (
    CandidateDimensionAdapter,
    FOUR_DATASET_SUBTRAIN_UNITS,
    P1_P5_BATCH_SIZE,
    P1_P5_UPDATE_BUDGET,
    P1_P5_VALIDATION_INTERVAL_UPDATES,
    SharedWindowEncoderOutput,
    source_proportional_validation_selection_loss,
)
from baseline.multidataset_pipeline.real_subtrain_provider import (
    FrozenNativeUnit,
    FrozenProviderIndex,
    NativeWindowBatch,
    PROVIDER_SCHEMA_VERSION,
    build_frozen_provider_index,
    canonical_json_sha256,
)
from baseline.multidataset_pipeline.runner_embedding_cache import (
    RunnerEmbeddingCacheSet,
    build_or_load_runner_embedding_caches,
)
from baseline.multidataset_pipeline.sliding_window import collate_sliding_windows
from baseline.multidataset_pipeline.terminal_scoring import (
    HFTemporalTerminalBatch,
    MulticlassTerminalBatch,
    ProductionTerminalScorer,
    TerminalScoringInput,
    terminal_provider_identity_sha256,
)
from baseline.multidataset_pipeline.train_shared_window import (
    LEARNING_RATE,
    RUNNER_SCHEMA_VERSION,
    VALIDATION_SELECTION_SCHEMA_VERSION,
    WEIGHT_DECAY,
    SourceProportionalBatchPlanner,
    TrainingRunnerConfig,
    assemble_trainable_modules,
    build_optimizer,
    cached_native_batch_loss,
    derive_phase_execution_identity,
    initialize_or_validate_execution_contract,
    load_and_validate_approval,
    load_training_checkpoint,
    native_batch_loss,
    native_loss_from_shared_output,
    prepare_phase_execution_root,
    save_training_checkpoint,
    sha256_path,
    structured_state_sha256,
    terminal_score_gate,
    trainable_scope_receipt,
    write_validation_selection_receipt,
)


class StructuredStateHashTest(unittest.TestCase):
    def test_zero_dim_tensor_is_hashed_deterministically(self):
        scalar = torch.tensor(1.25, dtype=torch.float32)
        self.assertEqual(
            structured_state_sha256(scalar),
            structured_state_sha256(scalar.clone()),
        )
        self.assertNotEqual(
            structured_state_sha256(scalar),
            structured_state_sha256(torch.tensor(2.25, dtype=torch.float32)),
        )


def _sample(
    sample_id: str,
    dataset: str,
    partition: str,
    group_id: str,
    audio_path: Path,
    targets: dict[str, object],
) -> Sample:
    return Sample(
        sample_id=sample_id,
        dataset=dataset,
        partition=partition,
        group_id=group_id,
        audio_path=str(audio_path),
        crop_start_s=None,
        crop_end_s=None,
        targets=targets,
        metadata={"patient_id": group_id, "recording_id": sample_id},
    )


def _canonical_identity_receipt() -> dict[str, object]:
    return {
        "status": "four_dataset_sample_contract_passed",
        "rows": 5,
        "unique_ids": 5,
        "ordered_id_sha256": "1" * 64,
        "dataset_rows": {"icbhi": 2, "sprsound": 1, "hf_lung": 1, "kauh": 1},
        "datasets": {
            "icbhi": {
                "manifest_sha256": "2" * 64,
                "partition": {"subtrain": 1, "validation": 0, "test": 1},
                "validation": "fixture_grouped_fold_0",
            },
            "sprsound": {
                "source_commit": "3" * 40,
                "subtrain_events": 1,
                "validation_events": 0,
                "validation": "fixture_patient_fold_0",
                "test_manifest_label_free": True,
            },
            "hf_lung": {
                "assignment_sha256": "4" * 64,
                "partition": {"subtrain": 1, "validation": 0, "test": 0},
                "date_proxy_counts": {"subtrain": 1, "validation": 0, "test": 0},
            },
            "kauh": {
                "outer_fold": 0,
                "partition": {"subtrain": 1, "validation": 0, "test": 0},
                "partition_patients": {"subtrain": 1, "validation": 0, "test": 0},
                "outer_test_ordered_id_sha256": "5" * 64,
            },
        },
    }


def _hf_annotation_identity_receipt() -> dict[str, object]:
    return {
        "status": "hf_manifest_annotation_audit_passed",
        "recordings": 1,
        "assignment_sha256": "4" * 64,
        "ordered_record_sha256": "6" * 64,
        "own_label_tree_sha256": "7" * 64,
        "accepted_label_tree_sha256_reference": "8" * 64,
        "label_tree_identity_status": "reference_not_reproduced_serialization_unknown",
        "partition_proxy_counts": {"subtrain": 1, "validation": 0, "test": 0},
        "recording_states": {"observed": 1, "empty": 0},
        "interval_semantics": "raw positive intervals only",
        "gap_semantics": "not_annotated_never_negative",
        "explicit_negative_intervals": 0,
        "independent_verifier_status": "HOLD",
    }


def _provider_rows(root: Path) -> tuple[list[Sample], HFSampleRecord]:
    hf_root = root / "hf_lung_v1" / "source_original"
    rows = [
        _sample("icbhi:one", "icbhi", "subtrain", "p1", root / "i.wav", {"icbhi_flat4": 1}),
        _sample("spr:one", "sprsound", "subtrain", "p2", root / "s.wav", {"spr_binary": 1, "spr_seven": 2}),
        _sample("hf:train:observed", "hf_lung", "subtrain", "date_proxy:2020-01-01", hf_root / "train" / "observed.wav", {}),
        _sample("kauh:P1:B", "kauh", "subtrain", "P1", root / "k.wav", {"kauh_raw9": 3}),
        _sample("icbhi:test", "icbhi", "test", "outer", root / "outer.wav", {}),
    ]
    record = HFSampleRecord(
        sample_id="hf:train:observed",
        source_split="train",
        partition="subtrain",
        date_proxy="2020-01-01",
        group_id="date_proxy:2020-01-01",
        wav_relative_path="train/observed.wav",
        label_relative_path="train/observed_label.txt",
        raw_intervals=(HFRawInterval("I", 0.0, 1.0),),
        recording_state=ObservationState.OBSERVED,
    )
    return rows, record


class RealProviderIndexTest(unittest.TestCase):
    def test_frozen_split_four_lane_and_outer_isolation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hf_root = root / "hf_lung_v1" / "source_original"
            rows, hf = _provider_rows(root)

            def canonical_loader(_root: Path, fold: int):
                self.assertEqual(fold, 0)
                return rows, _canonical_identity_receipt()

            def hf_loader(_root: Path):
                self.assertEqual(_root, hf_root)
                return (hf,), _hf_annotation_identity_receipt()

            index = build_frozen_provider_index(
                root,
                canonical_loader=canonical_loader,
                hf_loader=hf_loader,
                enforce_real_counts=False,
            )
            self.assertEqual(set(index.lanes), set(PREDICTION_UNITS))
            self.assertEqual({lane: len(value) for lane, value in index.lanes.items()}, {
                "ICBHI": 1,
                "SPRSound": 1,
                "HF": 1,
                "KAUH": 1,
            })
            self.assertFalse(index.receipt["outer_test_accessed"])
            self.assertEqual(index.receipt["outer_test_samples_emitted"], 0)
            self.assertNotIn("icbhi:test", {
                unit.sample.sample_id for values in index.lanes.values() for unit in values
            })
            self.assertIs(index.unit("HF").hf_record, hf)
            self.assertEqual(
                index.receipt["data_identity"]["provider_schema_version"],
                PROVIDER_SCHEMA_VERSION,
            )
            self.assertEqual(
                index.receipt["identity_binding_status"],
                "identity_bound_to_canonical_split_and_hf_annotation_v2",
            )
            self.assertEqual(index.receipt["independent_verifier_status"], "HOLD")
            self.assertEqual(
                index.receipt["hf_label_tree_equivalence_status"],
                "not_verified_equivalent_reference_differs",
            )

    def test_authority_and_annotation_mutations_change_identity_with_same_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows, hf = _provider_rows(root)
            canonical_base = _canonical_identity_receipt()
            annotation_base = _hf_annotation_identity_receipt()

            def build(canonical, annotation):
                return build_frozen_provider_index(
                    root,
                    canonical_loader=lambda _root, _fold: (rows, canonical),
                    hf_loader=lambda _root: ((hf,), annotation),
                    enforce_real_counts=False,
                )

            baseline = build(canonical_base, annotation_base)
            baseline_ids = baseline.receipt["sample_ids"]
            baseline_sha = baseline.receipt["data_identity_sha256"]
            baseline_identity = baseline.receipt["data_identity"]
            mutations = (
                ("canonical", ("datasets", "icbhi", "manifest_sha256")),
                ("canonical", ("datasets", "sprsound", "source_commit")),
                ("canonical", ("datasets", "sprsound", "subtrain_events")),
                ("canonical", ("datasets", "sprsound", "validation")),
                ("canonical", ("datasets", "sprsound", "test_manifest_label_free")),
                ("canonical", ("datasets", "hf_lung", "assignment_sha256")),
                ("canonical", ("datasets", "hf_lung", "date_proxy_counts")),
                ("canonical", ("datasets", "kauh", "outer_fold")),
                ("canonical", ("datasets", "kauh", "partition")),
                ("canonical", ("datasets", "kauh", "partition_patients")),
                ("canonical", ("datasets", "kauh", "outer_test_ordered_id_sha256")),
                ("annotation", ("assignment_sha256",)),
                ("annotation", ("ordered_record_sha256",)),
                ("annotation", ("own_label_tree_sha256",)),
                ("annotation", ("accepted_label_tree_sha256_reference",)),
                ("annotation", ("label_tree_identity_status",)),
                ("annotation", ("interval_semantics",)),
                ("annotation", ("gap_semantics",)),
                ("annotation", ("explicit_negative_intervals",)),
                ("annotation", ("recording_states",)),
            )
            for source, path in mutations:
                with self.subTest(source=source, path=path):
                    canonical = copy.deepcopy(canonical_base)
                    annotation = copy.deepcopy(annotation_base)
                    target = canonical if source == "canonical" else annotation
                    parent = target
                    for key in path[:-1]:
                        parent = parent[key]
                    old = parent[path[-1]]
                    if isinstance(old, bool):
                        parent[path[-1]] = not old
                    elif isinstance(old, int):
                        parent[path[-1]] = old + 1
                    elif isinstance(old, str):
                        parent[path[-1]] = old + "_changed"
                    else:
                        changed = copy.deepcopy(old)
                        first = next(iter(changed))
                        changed[first] = int(changed[first]) + 1
                        parent[path[-1]] = changed
                    changed_index = build(canonical, annotation)
                    self.assertEqual(changed_index.receipt["sample_ids"], baseline_ids)
                    self.assertNotEqual(
                        changed_index.receipt["data_identity_sha256"], baseline_sha
                    )
                    self.assertEqual(
                        changed_index.receipt["data_identity_sha256"],
                        canonical_json_sha256(changed_index.receipt["data_identity"]),
                    )

            def scalar_paths(value, prefix=()):
                if isinstance(value, dict):
                    for key, child in value.items():
                        yield from scalar_paths(child, (*prefix, key))
                else:
                    yield prefix

            for path in scalar_paths(baseline_identity):
                with self.subTest(all_bound_scalar=path):
                    changed = copy.deepcopy(baseline_identity)
                    parent = changed
                    for key in path[:-1]:
                        parent = parent[key]
                    old = parent[path[-1]]
                    if isinstance(old, bool):
                        parent[path[-1]] = not old
                    elif isinstance(old, int):
                        parent[path[-1]] = old + 1
                    else:
                        parent[path[-1]] = str(old) + "_changed"
                    self.assertNotEqual(canonical_json_sha256(changed), baseline_sha)

    def test_test_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "terminal-only"):
            build_frozen_provider_index(Path("missing"), partition="test")


def _waveform(lane: str, sample_id: str, samples: int = 32_000) -> WaveformSample:
    return WaveformSample(
        waveform=torch.linspace(-0.5, 0.5, samples, dtype=torch.float32),
        sample_id=sample_id,
        dataset_id=lane,
        prediction_unit=PREDICTION_UNITS[lane],
        source_start_s=0.0,
        source_end_s=samples / 16_000,
        lineage={
            "partition": "subtrain",
            "outer_test_accessed": "false",
            "group_id": f"group:{sample_id}",
            "source_id": f"source:{sample_id}",
        },
    )


class _FakeBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)


class _FakeAdapter(nn.Module):
    def __init__(self, identity: str = "AST"):
        super().__init__()
        self.encoder_identity = identity
        self.backend = _FakeBackend()
        self.dimension_adapter = CandidateDimensionAdapter(identity)
        self.register_buffer("nonpersistent_probe", torch.tensor([7.0]), persistent=False)
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        values = batch.waveform_windows.mean(dim=-1, keepdim=True).expand(-1, -1, 768)
        values = self.dimension_adapter(values)
        values = torch.where(batch.window_mask.unsqueeze(-1), values, torch.zeros_like(values))
        return SharedWindowEncoderOutput(
            embeddings=values,
            window_mask=batch.window_mask,
            time_map=batch.time_map,
            encoder_identity=self.encoder_identity,
            sample_ids=batch.sample_ids,
            dataset_ids=batch.dataset_ids,
            prediction_units=batch.prediction_units,
        )

    def receipt(self):
        return {
            "provenance": {
                "encoder_identity": self.encoder_identity,
                "source_url": "fixture",
                "source_revision": "fixture",
                "source_license": "fixture",
                "checkpoint_name": "fixture",
                "checkpoint_source": "fixture",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_size_bytes": 1,
                "asset_status": "fixture",
                "license_boundary": "fixture",
            },
            "dimension_adapter": self.dimension_adapter.receipt(),
        }


FIXTURE_PROVIDER_SPECIFICATION = "fixture.module:provider"
FIXTURE_PROVIDER_SOURCE = "def provider(path):\n    raise RuntimeError('not executed')\n"
FIXTURE_PROVIDER_IMPLEMENTATION_SHA256 = hashlib.sha256(
    FIXTURE_PROVIDER_SOURCE.encode("utf-8")
).hexdigest()
FIXTURE_PROVIDER_IDENTITY_SHA256 = terminal_provider_identity_sha256(
    FIXTURE_PROVIDER_SPECIFICATION, FIXTURE_PROVIDER_IMPLEMENTATION_SHA256
)


def _terminal_scorer(
    data_identity_sha256: str, threshold_receipt_sha256: str = "7" * 64
) -> ProductionTerminalScorer:
    ids = {
        "ICBHI_flat4": ("icbhi-0", "icbhi-1"),
        "SPRSound_binary": ("spr-0", "spr-1"),
        "SPRSound_raw7": ("spr-0", "spr-1"),
        "HF_temporal4": ("hf-0", "hf-1"),
        "KAUH_raw9": ("kauh-0", "kauh-1"),
    }
    multiclass = tuple(
        MulticlassTerminalBatch(
            task=task,
            prediction_ids=ids[task],
            targets=torch.tensor([0, 1]),
            predicted_classes=torch.tensor([0, 1]),
        )
        for task in (
            "ICBHI_flat4",
            "SPRSound_binary",
            "SPRSound_raw7",
            "KAUH_raw9",
        )
    )
    hf_targets = torch.tensor(
        [
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        ]
    )
    hf = HFTemporalTerminalBatch(
        prediction_ids=ids["HF_temporal4"],
        probabilities=hf_targets * 0.8 + (1 - hf_targets) * 0.2,
        targets=hf_targets,
        window_mask=torch.ones(2, 2, dtype=torch.bool),
        annotation_mask=torch.ones(2, 2, 4, dtype=torch.bool),
        valid_mask=torch.ones(2, 2, 4, dtype=torch.bool),
        time_map=torch.tensor(
            [[[0.0, 2.0], [1.0, 3.0]], [[0.0, 2.0], [1.0, 3.0]]],
            dtype=torch.float64,
        ),
        thresholds=torch.full((4,), 0.5),
        threshold_receipt_sha256=threshold_receipt_sha256,
    )

    def provider(_checkpoint: Path) -> TerminalScoringInput:
        artifacts = {}
        for name in ("sprsound_label_free_predictions", "terminal_joined_predictions"):
            path = _checkpoint.parent / f"{name}.fixture"
            path.write_bytes(name.encode("utf-8"))
            artifacts[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        return TerminalScoringInput(
            batches=(*multiclass, hf),
            expected_prediction_ids_by_task=ids,
            data_identity_sha256=data_identity_sha256,
            provider_identity_sha256=FIXTURE_PROVIDER_IDENTITY_SHA256,
            prediction_artifacts=artifacts,
        )

    return ProductionTerminalScorer(
        provider,
        expected_provider_identity_sha256=FIXTURE_PROVIDER_IDENTITY_SHA256,
        provider_specification=FIXTURE_PROVIDER_SPECIFICATION,
    )


def _register_terminal_provider(root: Path) -> None:
    implementation = root / "fixture_provider.py"
    implementation.write_text(FIXTURE_PROVIDER_SOURCE)
    manifest = root / "baseline/multidataset_pipeline/terminal_provider_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "terminal_provider_registration_v1",
                "status": "registered",
                "provider_specification": FIXTURE_PROVIDER_SPECIFICATION,
                "provider_identity_sha256": FIXTURE_PROVIDER_IDENTITY_SHA256,
                "implementation_path": "fixture_provider.py",
                "implementation_sha256": FIXTURE_PROVIDER_IMPLEMENTATION_SHA256,
                "scorer_schema_version": "shared_window_terminal_scorer_v2",
                "native_tasks": [
                    "ICBHI_flat4",
                    "SPRSound_binary",
                    "SPRSound_raw7",
                    "HF_temporal4",
                    "KAUH_raw9",
                ],
                "outer_test_access_policy": "terminal_score_only_after_exact_selection_checkpoint_and_approval",
            }
        ),
        encoding="utf-8",
    )


def _native_batch(lane: str) -> NativeWindowBatch:
    sample_count = 240_000 if lane == "HF" else 32_000
    windows = collate_sliding_windows(
        [_waveform(lane, f"{lane}:one", samples=sample_count)]
    )
    targets = {
        "ICBHI": {"icbhi_flat4": torch.tensor([1])},
        "SPRSound": {"spr_binary": torch.tensor([1]), "spr_seven": torch.tensor([2])},
        "HF": {},
        "KAUH": {"kauh_raw9": torch.tensor([3])},
    }[lane]
    return NativeWindowBatch(
        lane=lane,
        windows=windows,
        targets=targets,
        hf_intervals=((HFRawInterval("I", 0.0, 1.1),),) if lane == "HF" else (),
        hf_recording_states=(ObservationState.OBSERVED,) if lane == "HF" else (),
    )


def _cache_fixture_index(partition: str) -> FrozenProviderIndex:
    datasets = {
        "ICBHI": "icbhi",
        "SPRSound": "sprsound",
        "HF": "hf_lung",
        "KAUH": "kauh",
    }
    target_values = {
        "ICBHI": {"icbhi_flat4": 1},
        "SPRSound": {"spr_binary": 1, "spr_seven": 2},
        "HF": {},
        "KAUH": {"kauh_raw9": 3},
    }
    lanes = {}
    for lane, dataset in datasets.items():
        sample_id = f"{lane.lower()}:{partition}:one"
        group_id = (
            "date_proxy:2020-01-01"
            if lane == "HF"
            else f"group:{lane}:{partition}"
        )
        sample = _sample(
            sample_id,
            dataset,
            partition,
            group_id,
            Path("fixture") / f"{sample_id}.wav",
            target_values[lane],
        )
        record = None
        if lane == "HF":
            record = HFSampleRecord(
                sample_id=sample_id,
                source_split="train",
                partition=partition,
                date_proxy="2020-01-01",
                group_id=group_id,
                wav_relative_path=f"train/{sample_id}.wav",
                label_relative_path=f"train/{sample_id}_label.txt",
                raw_intervals=(HFRawInterval("I", 0.0, 1.1),),
                recording_state=ObservationState.OBSERVED,
            )
        lanes[lane] = (FrozenNativeUnit(lane, sample, record),)
    manifest_ids = {lane: canonical_json_sha256({"lane": lane}) for lane in lanes}
    data_identity = {
        "manifest_ordered_id_sha256_by_dataset": manifest_ids,
        "fixture_partition": partition,
    }
    return FrozenProviderIndex(
        partition=partition,
        lanes=lanes,
        receipt={
            "partition": partition,
            "canonical_receipt": {"fixture": True, "partition": partition},
            "data_identity": data_identity,
            "data_identity_sha256": canonical_json_sha256(data_identity),
            "outer_test_accessed": False,
        },
    )


def _cache_fixture_batch_loader(units) -> NativeWindowBatch:
    lane = units[0].lane
    samples = []
    for unit in units:
        sample_count = 240_000 if lane == "HF" else 48_000
        samples.append(
            WaveformSample(
                waveform=torch.linspace(-0.5, 0.5, sample_count),
                sample_id=unit.sample.sample_id,
                dataset_id=lane,
                prediction_unit=PREDICTION_UNITS[lane],
                source_start_s=0.0,
                source_end_s=sample_count / 16_000,
                lineage={
                    "partition": unit.sample.partition,
                    "outer_test_accessed": "false",
                    "group_id": unit.sample.group_id,
                    "source_id": unit.sample.sample_id,
                },
            )
        )
    targets = {
        key: torch.tensor([int(unit.sample.targets[key]) for unit in units])
        for key in {
            "ICBHI": ("icbhi_flat4",),
            "SPRSound": ("spr_binary", "spr_seven"),
            "HF": (),
            "KAUH": ("kauh_raw9",),
        }[lane]
    }
    return NativeWindowBatch(
        lane=lane,
        windows=collate_sliding_windows(samples),
        targets=targets,
        hf_intervals=tuple(
            unit.hf_record.raw_intervals for unit in units if unit.hf_record
        ),
        hf_recording_states=tuple(
            unit.hf_record.recording_state for unit in units if unit.hf_record
        ),
    )
class TrainingAssemblyTest(unittest.TestCase):
    def test_runner_cache_complete_hit_and_cached_uncached_numeric_equivalence(self):
        repo_root = Path(__file__).resolve().parents[1]
        indexes = {
            partition: _cache_fixture_index(partition)
            for partition in ("subtrain", "validation")
        }
        adapter = _FakeAdapter("AST").eval()
        with TemporaryDirectory() as directory:
            cache_root = Path(directory) / "runner-cache"
            cache_set = build_or_load_runner_embedding_caches(
                repo_root=repo_root,
                cache_root=cache_root,
                pipeline_id="P1",
                config_identity_sha256="b" * 64,
                adapter=adapter,
                indexes=indexes,
                device=torch.device("cpu"),
                batch_size=2,
                batch_loader=_cache_fixture_batch_loader,
            )
            cache_set.validate_complete()
            self.assertEqual(len(cache_set.entries), 8)
            self.assertEqual(adapter.forward_calls, 8)
            first_calls = adapter.forward_calls
            hit_set = build_or_load_runner_embedding_caches(
                repo_root=repo_root,
                cache_root=cache_root,
                pipeline_id="P1",
                config_identity_sha256="b" * 64,
                adapter=adapter,
                indexes=indexes,
                device=torch.device("cpu"),
                batch_size=2,
                batch_loader=_cache_fixture_batch_loader,
            )
            self.assertEqual(adapter.forward_calls, first_calls)
            self.assertTrue(
                all(
                    entry.receipt["cache_status"] == "hit_verified_existing"
                    for entry in hit_set.entries.values()
                )
            )
            model = JointNativeProjector()
            uncached_losses = {}
            cached_losses = {}
            for lane in ("ICBHI", "SPRSound", "HF", "KAUH"):
                native = _cache_fixture_batch_loader(indexes["validation"].lanes[lane])
                uncached_output = adapter(native.windows)
                uncached_loss, uncached_receipt, uncached_logits = native_loss_from_shared_output(
                    model,
                    lane=lane,
                    output=uncached_output,
                    targets=native.targets,
                    hf_intervals=native.hf_intervals,
                    hf_recording_states=native.hf_recording_states,
                    device=torch.device("cpu"),
                )
                cached = hit_set.batch("validation", lane, (0,), device=torch.device("cpu"))
                runner_cached_loss, runner_cached_receipt = cached_native_batch_loss(
                    model, cached, device=torch.device("cpu")
                )
                cached_loss, cached_receipt, cached_logits = native_loss_from_shared_output(
                    model,
                    lane=lane,
                    output=cached.output,
                    targets=cached.targets,
                    hf_intervals=cached.hf_intervals,
                    hf_recording_states=cached.hf_recording_states,
                    device=torch.device("cpu"),
                )
                torch.testing.assert_close(cached_loss, uncached_loss, rtol=0, atol=0)
                torch.testing.assert_close(
                    runner_cached_loss, uncached_loss, rtol=0, atol=0
                )
                self.assertEqual(
                    runner_cached_receipt["encoder_execution"],
                    "cache_hit_no_encoder_call",
                )
                self.assertEqual(set(cached_logits), set(uncached_logits))
                for task in cached_logits:
                    torch.testing.assert_close(
                        cached_logits[task], uncached_logits[task], rtol=0, atol=0
                    )
                uncached_losses.update(uncached_receipt["native_task_losses"])
                cached_losses.update(cached_receipt["native_task_losses"])
            uncached_selection = source_proportional_validation_selection_loss(
                uncached_losses
            )
            cached_selection = source_proportional_validation_selection_loss(
                cached_losses
            )
            self.assertEqual(cached_selection, uncached_selection)
            incomplete = dict(cache_set.entries)
            incomplete.pop(("validation", "HF"))
            with self.assertRaisesRegex(RuntimeError, "all eight"):
                RunnerEmbeddingCacheSet(
                    "P1", incomplete, cache_set.receipt
                ).validate_complete()

    def test_l40_snapshot_includes_nonpersistent_buffers(self):
        adapter = _FakeAdapter("AST")
        model = JointNativeProjector()
        snapshot = _complete_state_snapshot(adapter, model)
        self.assertIn("adapter.buffer.nonpersistent_probe", snapshot)
        self.assertNotIn("adapter.state.nonpersistent_probe", snapshot)

    def test_l40_pipeline_adapter_identity_gate(self):
        adapter = _FakeAdapter("AST")
        validate_pipeline_adapter_identity("P1", adapter)
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            validate_pipeline_adapter_identity("P2", adapter)

    def test_frozen_budget_scope_optimizer_and_planner_resume(self):
        with TemporaryDirectory() as directory:
            config = TrainingRunnerConfig.frozen("P1", Path(directory), phase="full")
            config.validate()
            self.assertEqual(
                config.sha256(),
                TrainingRunnerConfig.frozen("P1", Path(directory), phase="preflight").sha256(),
            )
            self.assertEqual(config.batch_size, P1_P5_BATCH_SIZE)
            self.assertEqual(config.update_budget, P1_P5_UPDATE_BUDGET)
            self.assertEqual(config.validation_interval_updates, P1_P5_VALIDATION_INTERVAL_UPDATES)
            adapter = _FakeAdapter()
            model = assemble_trainable_modules(adapter, device=torch.device("cpu"))
            scope = trainable_scope_receipt(adapter, model)
            self.assertEqual(scope["candidate_encoder"]["scope"], "frozen")
            self.assertTrue(scope["hf_uses_same_shared_projector"])
            optimizer, receipt = build_optimizer(adapter, model)
            self.assertFalse(receipt["frozen_encoder_in_optimizer"])
            self.assertEqual(receipt["learning_rate"], LEARNING_RATE)
            self.assertEqual(receipt["weight_decay"], WEIGHT_DECAY)
            optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            self.assertNotIn(id(adapter.backend.weight), optimized)
            self.assertIn(id(model.projector.weight), optimized)

            planner = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
            planner.next()
            state = planner.state_dict()
            expected = planner.next()
            restored = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
            restored.load_state_dict(state)
            self.assertEqual(restored.next(), expected)

    def test_phase_execution_roots_are_isolated_fresh_and_resume_bound(self):
        with TemporaryDirectory() as directory:
            repo_root = Path(directory)
            data_identity = "d" * 64
            scope = {"scope": "fixture"}
            optimizer_receipt = {"optimizer": "fixture"}
            smoke_cache = {
                "policy": "uncached_engineering_smoke",
                "outer_test_cached": False,
            }
            full_cache = {
                "cache_status": "miss_computed_and_written",
                "receipt_sha256": "1" * 64,
                "identity": "cache-fixture",
                "outer_test_cached": False,
            }

            def approval(config: TrainingRunnerConfig, digest: str) -> dict[str, object]:
                return {
                    "status": "approved",
                    "pipeline_id": config.pipeline_id,
                    "phase": config.phase,
                    "config_sha256": config.sha256(),
                    "data_identity_sha256": data_identity,
                    "authorized_by": "management-fixture",
                    "outer_test_authorized": False,
                    "approval_receipt_sha256": digest,
                }

            smoke = TrainingRunnerConfig.frozen("P1", repo_root, phase="smoke")
            full = TrainingRunnerConfig.frozen("P1", repo_root, phase="full")
            self.assertEqual(smoke.sha256(), full.sha256())
            smoke_approval = approval(smoke, "a" * 64)
            full_approval = approval(full, "b" * 64)
            smoke_root, smoke_identity = prepare_phase_execution_root(
                smoke,
                smoke_approval,
                data_identity,
                resume=None,
                resume_sha256=None,
            )
            full_root, full_identity = prepare_phase_execution_root(
                full,
                full_approval,
                data_identity,
                resume=None,
                resume_sha256=None,
            )
            self.assertNotEqual(smoke_root, full_root)
            self.assertEqual(smoke_root.parent.name, "smoke")
            self.assertEqual(full_root.parent.name, "full")
            initialize_or_validate_execution_contract(
                smoke_root,
                identity=smoke_identity,
                config=smoke,
                approval=smoke_approval,
                scope=scope,
                optimizer_receipt=optimizer_receipt,
                cache_receipt=smoke_cache,
                resume=False,
            )
            initialize_or_validate_execution_contract(
                full_root,
                identity=full_identity,
                config=full,
                approval=full_approval,
                scope=scope,
                optimizer_receipt=optimizer_receipt,
                cache_receipt=full_cache,
                resume=False,
            )
            (smoke_root / "train_log.jsonl").write_text(
                json.dumps({"update": 1, "phase": "smoke"}) + "\n",
                encoding="utf-8",
            )
            full_log = json.dumps({"update": 1_725, "phase": "full"}) + "\n"
            (full_root / "train_log.jsonl").write_text(full_log, encoding="utf-8")
            (full_root / "validation_log.jsonl").write_text(
                full_log, encoding="utf-8"
            )
            self.assertNotIn("smoke", (full_root / "train_log.jsonl").read_text())
            self.assertNotIn("full", (smoke_root / "train_log.jsonl").read_text())
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                prepare_phase_execution_root(
                    full,
                    full_approval,
                    data_identity,
                    resume=None,
                    resume_sha256=None,
                )

            checkpoint = full_root / "checkpoints" / "update_001725.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"synthetic-checkpoint-artifact")
            checkpoint_sha = sha256_path(checkpoint)
            checkpoint_receipt = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": checkpoint_sha,
                "update": 1_725,
                "selection_scalar": 0.5,
                "config_sha256": full.sha256(),
                "data_identity_sha256": data_identity,
                "approval_receipt_sha256": full_approval["approval_receipt_sha256"],
                "component_state_sha256": {},
                "outer_test_accessed": False,
                "native_metrics_only": True,
            }
            checkpoint.with_suffix(".receipt.json").write_text(
                json.dumps(checkpoint_receipt), encoding="utf-8"
            )
            resumed_root, resumed_identity = prepare_phase_execution_root(
                full,
                full_approval,
                data_identity,
                resume=checkpoint,
                resume_sha256=checkpoint_sha,
            )
            self.assertEqual(resumed_root, full_root)
            self.assertEqual(resumed_identity, full_identity)
            initialize_or_validate_execution_contract(
                resumed_root,
                identity=resumed_identity,
                config=full,
                approval=full_approval,
                scope=scope,
                optimizer_receipt=optimizer_receipt,
                cache_receipt={
                    **full_cache,
                    "cache_status": "hit_verified_existing",
                    "receipt_sha256": "2" * 64,
                },
                resume=True,
            )
            wrong_approval = approval(full, "c" * 64)
            with self.assertRaisesRegex(RuntimeError, "execution root is missing"):
                prepare_phase_execution_root(
                    full,
                    wrong_approval,
                    data_identity,
                    resume=checkpoint,
                    resume_sha256=checkpoint_sha,
                )
            self.assertEqual(
                derive_phase_execution_identity(full, full_approval, data_identity),
                full_identity,
            )

    def test_phase_execution_claim_is_atomic_and_partial_root_cannot_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainingRunnerConfig.frozen("P1", root, phase="full")
            data_identity = "d" * 64
            approval = {
                "status": "approved",
                "pipeline_id": "P1",
                "phase": "full",
                "config_sha256": config.sha256(),
                "data_identity_sha256": data_identity,
                "authorized_by": "management-fixture",
                "outer_test_authorized": False,
                "approval_receipt_sha256": "a" * 64,
            }
            barrier = threading.Barrier(2)

            def claim() -> str:
                barrier.wait()
                try:
                    execution_root, _ = prepare_phase_execution_root(
                        config,
                        approval,
                        data_identity,
                        resume=None,
                        resume_sha256=None,
                    )
                    return f"winner:{execution_root}"
                except FileExistsError:
                    return "rejected"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(lambda _: claim(), range(2)))
            self.assertEqual(sum(value.startswith("winner:") for value in outcomes), 1)
            self.assertEqual(outcomes.count("rejected"), 1)
            execution_root = Path(
                next(value.split(":", 1)[1] for value in outcomes if value.startswith("winner:"))
            )
            checkpoint = execution_root / "checkpoints" / "update_001725.pt"
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                prepare_phase_execution_root(
                    config,
                    approval,
                    data_identity,
                    resume=checkpoint,
                    resume_sha256="b" * 64,
                )

    def test_engineering_one_step_native_and_hf_masks(self):
        torch.manual_seed(20260728)
        adapter = _FakeAdapter()
        model = JointNativeProjector()
        before = model.projector.weight.detach().clone()
        optimizer, _ = build_optimizer(adapter, model)
        for lane in ("ICBHI", "HF"):
            optimizer.zero_grad(set_to_none=True)
            loss, receipt = native_batch_loss(
                adapter, model, _native_batch(lane), device=torch.device("cpu")
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNone(adapter.backend.weight.grad)
            self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
            if lane == "HF":
                self.assertEqual(
                    receipt["target_receipt"]["negative_semantics"],
                    "source_task_constructed_not_raw_normal",
                )
                self.assertFalse(receipt["target_receipt"]["shared_label_eligible"])
            optimizer.step()
        self.assertFalse(torch.equal(before, model.projector.weight))

    def test_approval_and_terminal_selection_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _register_terminal_provider(root)
            smoke = TrainingRunnerConfig.frozen("P2", root, phase="smoke")
            approval = root / "approval.json"
            approval.write_text(json.dumps({
                "status": "approved",
                "pipeline_id": "P2",
                "phase": "smoke",
                "config_sha256": smoke.sha256(),
                "data_identity_sha256": "a" * 64,
                "authorized_by": "management-fixture",
                "outer_test_authorized": False,
            }), encoding="utf-8")
            validated = load_and_validate_approval(approval, smoke)
            self.assertEqual(len(validated["approval_receipt_sha256"]), 64)
            with self.assertRaisesRegex(PermissionError, "authority/annotation identity"):
                load_and_validate_approval(
                    approval,
                    smoke,
                    expected_data_identity_sha256="b" * 64,
                )
            bad = json.loads(approval.read_text())
            bad["outer_test_authorized"] = True
            approval.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "outer_test_authorized"):
                load_and_validate_approval(approval, smoke)

            full = TrainingRunnerConfig.frozen("P2", root, phase="full")
            adapter = _FakeAdapter("BEATs")
            model = assemble_trainable_modules(adapter, device=torch.device("cpu"))
            optimizer, _ = build_optimizer(adapter, model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=full.update_budget
            )
            planner = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
            full_approval_sha = "f" * 64
            first = save_training_checkpoint(
                root / "update_001725.pt",
                config=full,
                update=1_725,
                adapter=adapter,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                planner=planner,
                validation_history=[],
                data_identity_sha256="c" * 64,
                approval_receipt_sha256=full_approval_sha,
                selection_scalar=0.4,
            )
            second = save_training_checkpoint(
                root / "update_003450.pt",
                config=full,
                update=3_450,
                adapter=adapter,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                planner=planner,
                validation_history=[],
                data_identity_sha256="c" * 64,
                approval_receipt_sha256=full_approval_sha,
                selection_scalar=0.3,
            )
            selection_path = root / "validation_selection_receipt.json"
            selection = write_validation_selection_receipt(
                selection_path,
                config=full,
                data_identity_sha256="c" * 64,
                full_approval_receipt_sha256=full_approval_sha,
                hf_validation_identity={
                    "validation_data_identity_sha256": "3" * 64,
                    "hf_validation_manifest_identity_sha256": "1" * 64,
                    "hf_validation_ordered_prediction_ids_sha256": "2" * 64,
                },
                candidates=[first, second],
            )
            self.assertEqual(
                selection["schema_version"], VALIDATION_SELECTION_SCHEMA_VERSION
            )
            self.assertEqual(selection["selected_update"], 3_450)

            terminal = TrainingRunnerConfig.frozen("P2", root, phase="terminal-score")
            threshold_path = root / "hf_threshold_receipt.json"
            threshold_artifact = write_hf_threshold_receipt(
                threshold_path,
                threshold_receipt_payload(
                    thresholds=(0.5, 0.5, 0.5, 0.5),
                    validation_data_identity_sha256="3" * 64,
                    hf_validation_manifest_identity_sha256="1" * 64,
                    hf_validation_ordered_prediction_ids_sha256="2" * 64,
                    full_approval_receipt_sha256=full_approval_sha,
                    validation_selection_receipt_sha256=selection[
                        "selection_receipt_artifact"
                    ]["sha256"],
                    selected_checkpoint_sha256=second["sha256"],
                    validation_prediction_identity_sha256="4" * 64,
                    per_channel_selection=[
                        {
                            "channel": channel,
                            "threshold": 0.5,
                            "max_f1": 1.0,
                            "candidate_count": 1,
                            "valid_count": 2,
                            "positive_support": 1,
                            "negative_support": 1,
                            "tp": 1,
                            "fp": 0,
                            "fn": 0,
                        }
                        for channel in ("I", "E", "CAS", "DAS")
                    ],
                    scorer_schema_version="shared_window_terminal_scorer_v2",
                ),
            )
            approval.write_text(json.dumps({
                "status": "approved",
                "pipeline_id": "P2",
                "phase": "terminal-score",
                "config_sha256": terminal.sha256(),
                "data_identity_sha256": "c" * 64,
                "selection_receipt_sha256": selection["selection_receipt_artifact"]["sha256"],
                "selected_checkpoint_path": second["path"],
                "selected_checkpoint_sha256": second["sha256"],
                "selected_checkpoint_size_bytes": second["size_bytes"],
                "selected_checkpoint_update": second["update"],
                "terminal_scorer_schema_version": "shared_window_terminal_scorer_v2",
                "terminal_provider_identity_sha256": FIXTURE_PROVIDER_IDENTITY_SHA256,
                "hf_threshold_receipt_sha256": threshold_artifact["sha256"],
                "hf_validation_data_identity_sha256": "3" * 64,
                "hf_validation_manifest_identity_sha256": "1" * 64,
                "hf_validation_ordered_prediction_ids_sha256": "2" * 64,
                "hf_threshold_selection_policy": HF_THRESHOLD_SELECTION_POLICY,
                "authorized_by": "management-fixture",
                "outer_test_authorized": True,
            }), encoding="utf-8")
            result = terminal_score_gate(
                terminal,
                approval,
                selection_path,
                selection["selection_receipt_artifact"]["sha256"],
                Path(second["path"]),
                threshold_path,
                threshold_artifact["sha256"],
                scorer=_terminal_scorer(
                    "c" * 64, threshold_artifact["sha256"]
                ),
            )
            self.assertFalse(result["cross_dataset_pooled_performance"])
            approved_terminal = json.loads(approval.read_text())
            wrong_checkpoint_approval = dict(approved_terminal)
            wrong_checkpoint_approval["selected_checkpoint_sha256"] = "a" * 64
            approval.write_text(json.dumps(wrong_checkpoint_approval), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "exact selected checkpoint"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=_terminal_scorer(
                        "c" * 64, threshold_artifact["sha256"]
                    ),
                )
            approval.write_text(json.dumps(approved_terminal), encoding="utf-8")
            wrong_threshold_approval = dict(approved_terminal)
            wrong_threshold_approval["hf_threshold_receipt_sha256"] = "a" * 64
            approval.write_text(json.dumps(wrong_threshold_approval), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "threshold receipt/policy"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=_terminal_scorer(
                        "c" * 64, threshold_artifact["sha256"]
                    ),
                )
            approval.write_text(json.dumps(approved_terminal), encoding="utf-8")
            wrong_provider = dict(approved_terminal)
            wrong_provider["terminal_provider_identity_sha256"] = "9" * 64
            approval.write_text(json.dumps(wrong_provider), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "scorer/provider identity"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=_terminal_scorer(
                        "c" * 64, threshold_artifact["sha256"]
                    ),
                )
            approval.write_text(json.dumps(approved_terminal), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity/task/isolation"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=_terminal_scorer(
                        "d" * 64, threshold_artifact["sha256"]
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "production native-task scorer"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                )
            with self.assertRaisesRegex(RuntimeError, "production native-task scorer"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=lambda path: {
                        **_terminal_scorer("c" * 64)(path),
                        "pooled_score": 0.5,
                    },
                )
            with self.assertRaisesRegex(RuntimeError, "not the exact"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(first["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=lambda _: {},
                )

            tampered_selection = root / "tampered_selection.json"
            tampered = json.loads(selection_path.read_text())
            tampered["selection_rule"] = "tampered"
            tampered_selection.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                terminal_score_gate(
                    terminal,
                    approval,
                    tampered_selection,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=lambda _: {},
                )
            with self.assertRaisesRegex(RuntimeError, "identity/rule"):
                terminal_score_gate(
                    terminal,
                    approval,
                    tampered_selection,
                    sha256_path(tampered_selection),
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=lambda _: {},
                )

            tampered_threshold_path = root / "tampered_hf_threshold_receipt.json"
            shutil.copy2(threshold_path, tampered_threshold_path)
            with tampered_threshold_path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "threshold receipt byte SHA256"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    tampered_threshold_path,
                    threshold_artifact["sha256"],
                    scorer=_terminal_scorer(
                        "c" * 64, threshold_artifact["sha256"]
                    ),
                )

            wrong_data = json.loads(approval.read_text())
            wrong_data["data_identity_sha256"] = "d" * 64
            approval.write_text(json.dumps(wrong_data), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "authority/annotation"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    threshold_path,
                    threshold_artifact["sha256"],
                    scorer=lambda _: {},
                )

    def test_checkpoint_resume_schema(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainingRunnerConfig.frozen("P1", root, phase="full")
            adapter = _FakeAdapter()
            model = assemble_trainable_modules(adapter, device=torch.device("cpu"))
            optimizer, _ = build_optimizer(adapter, model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.update_budget
            )
            planner = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
            planner.next()
            path = root / "checkpoint.pt"
            receipt = save_training_checkpoint(
                path,
                config=config,
                update=1_725,
                adapter=adapter,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                planner=planner,
                validation_history=[],
                data_identity_sha256="c" * 64,
                approval_receipt_sha256="a" * 64,
                selection_scalar=0.25,
            )
            self.assertEqual(receipt["schema_version"], RUNNER_SCHEMA_VERSION)
            self.assertFalse(receipt["outer_test_accessed"])
            self.assertEqual(receipt["size_bytes"], path.stat().st_size)
            self.assertEqual(receipt["sha256"], sha256_path(path))
            self.assertEqual(receipt["approval_receipt_sha256"], "a" * 64)
            self.assertEqual(
                set(receipt["component_state_sha256"]),
                {"dimension_adapter", "joint_native_model", "optimizer", "scheduler", "planner"},
            )
            self.assertTrue(path.with_suffix(".receipt.json").is_file())
            restored_adapter = _FakeAdapter()
            restored_model = assemble_trainable_modules(
                restored_adapter, device=torch.device("cpu")
            )
            restored_optimizer, _ = build_optimizer(restored_adapter, restored_model)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=config.update_budget
            )
            restored_planner = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
            update, history, loaded_receipt = load_training_checkpoint(
                path,
                config=config,
                adapter=restored_adapter,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                planner=restored_planner,
                expected_data_identity_sha256="c" * 64,
                expected_checkpoint_sha256=receipt["sha256"],
                expected_approval_receipt_sha256="a" * 64,
            )
            self.assertEqual(update, 1_725)
            self.assertEqual(history, [])
            self.assertEqual(loaded_receipt["sha256"], receipt["sha256"])
            torch.testing.assert_close(model.projector.weight, restored_model.projector.weight)
            with self.assertRaisesRegex(RuntimeError, "resume identity"):
                load_training_checkpoint(
                    path,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="d" * 64,
                    expected_checkpoint_sha256=receipt["sha256"],
                    expected_approval_receipt_sha256="a" * 64,
                )

            with self.assertRaisesRegex(RuntimeError, "resume identity"):
                load_training_checkpoint(
                    path,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256=receipt["sha256"],
                    expected_approval_receipt_sha256="b" * 64,
                )

            with self.assertRaisesRegex(RuntimeError, "byte SHA256 mismatch"):
                load_training_checkpoint(
                    path,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256="e" * 64,
                    expected_approval_receipt_sha256="a" * 64,
                )

            payload = torch.load(path, weights_only=False)
            payload["update"] = 1_725
            payload["schema_version"] = "shared_window_training_v2"
            old_schema = root / "old_schema.pt"
            torch.save(payload, old_schema)
            with self.assertRaisesRegex(RuntimeError, "resume identity"):
                load_training_checkpoint(
                    old_schema,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256=sha256_path(old_schema),
                    expected_approval_receipt_sha256="a" * 64,
                )

            replacement = root / "replacement_same_payload.pt"
            shutil.copy2(path, replacement)
            with replacement.open("ab") as handle:
                handle.write(b"replacement")
            self.assertEqual(torch.load(replacement, weights_only=False)["update"], 1_725)
            with self.assertRaisesRegex(RuntimeError, "byte SHA256 mismatch"):
                load_training_checkpoint(
                    replacement,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256=receipt["sha256"],
                    expected_approval_receipt_sha256="a" * 64,
                )

            payload = torch.load(path, weights_only=False)
            payload["update"] = config.update_budget + 1
            out_of_range = root / "out_of_range.pt"
            torch.save(payload, out_of_range)
            with self.assertRaisesRegex(RuntimeError, "resume identity"):
                load_training_checkpoint(
                    out_of_range,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256=sha256_path(out_of_range),
                    expected_approval_receipt_sha256="a" * 64,
                )

            payload = torch.load(path, weights_only=False)
            payload["update"] = 1
            non_validation_update = root / "non_validation_update.pt"
            torch.save(payload, non_validation_update)
            with self.assertRaisesRegex(RuntimeError, "resume identity"):
                load_training_checkpoint(
                    non_validation_update,
                    config=config,
                    adapter=restored_adapter,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    planner=restored_planner,
                    expected_data_identity_sha256="c" * 64,
                    expected_checkpoint_sha256=sha256_path(non_validation_update),
                    expected_approval_receipt_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
