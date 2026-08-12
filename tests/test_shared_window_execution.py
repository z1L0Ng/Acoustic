from __future__ import annotations

import json
import copy
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.multidataset_pipeline.beats_temporal import HFRawInterval
from baseline.multidataset_pipeline.contracts import ObservationState, PREDICTION_UNITS, WaveformSample
from baseline.multidataset_pipeline.hf_data import HFSampleRecord
from baseline.multidataset_pipeline.joint_native import JointNativeProjector
from baseline.multidataset_pipeline.l40_preflight import (
    validate_pipeline_adapter_identity,
)
from baseline.multidataset_pipeline.preflight import (
    CandidateDimensionAdapter,
    FOUR_DATASET_SUBTRAIN_UNITS,
    P1_P5_BATCH_SIZE,
    P1_P5_UPDATE_BUDGET,
    P1_P5_VALIDATION_INTERVAL_UPDATES,
    SharedWindowEncoderOutput,
)
from baseline.multidataset_pipeline.real_subtrain_provider import (
    FrozenNativeUnit,
    NativeWindowBatch,
    PROVIDER_SCHEMA_VERSION,
    build_frozen_provider_index,
    canonical_json_sha256,
)
from baseline.multidataset_pipeline.sliding_window import collate_sliding_windows
from baseline.multidataset_pipeline.train_shared_window import (
    LEARNING_RATE,
    RUNNER_SCHEMA_VERSION,
    VALIDATION_SELECTION_SCHEMA_VERSION,
    WEIGHT_DECAY,
    SourceProportionalBatchPlanner,
    TrainingRunnerConfig,
    assemble_trainable_modules,
    build_optimizer,
    load_and_validate_approval,
    load_training_checkpoint,
    native_batch_loss,
    save_training_checkpoint,
    sha256_path,
    terminal_score_gate,
    trainable_scope_receipt,
    write_validation_selection_receipt,
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

    def forward(self, batch):
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


class TrainingAssemblyTest(unittest.TestCase):
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
                candidates=[first, second],
            )
            self.assertEqual(
                selection["schema_version"], VALIDATION_SELECTION_SCHEMA_VERSION
            )
            self.assertEqual(selection["selected_update"], 3_450)

            terminal = TrainingRunnerConfig.frozen("P2", root, phase="terminal-score")
            approval.write_text(json.dumps({
                "status": "approved",
                "pipeline_id": "P2",
                "phase": "terminal-score",
                "config_sha256": terminal.sha256(),
                "data_identity_sha256": "c" * 64,
                "selection_receipt_sha256": selection["selection_receipt_artifact"]["sha256"],
                "authorized_by": "management-fixture",
                "outer_test_authorized": True,
            }), encoding="utf-8")
            result = terminal_score_gate(
                terminal,
                approval,
                selection_path,
                selection["selection_receipt_artifact"]["sha256"],
                Path(second["path"]),
                scorer=lambda _: {"HF": {"I": {"accuracy": 0.5}}},
            )
            self.assertFalse(result["cross_dataset_pooled_performance"])
            with self.assertRaisesRegex(RuntimeError, "terminal scorer HOLD"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                )
            with self.assertRaisesRegex(RuntimeError, "forbidden pooled"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(second["path"]),
                    scorer=lambda _: {"pooled_score": 0.5},
                )
            with self.assertRaisesRegex(RuntimeError, "not the exact"):
                terminal_score_gate(
                    terminal,
                    approval,
                    selection_path,
                    selection["selection_receipt_artifact"]["sha256"],
                    Path(first["path"]),
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
                    scorer=lambda _: {},
                )
            with self.assertRaisesRegex(RuntimeError, "identity/rule"):
                terminal_score_gate(
                    terminal,
                    approval,
                    tampered_selection,
                    sha256_path(tampered_selection),
                    Path(second["path"]),
                    scorer=lambda _: {},
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
