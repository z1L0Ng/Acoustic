"""Registered terminal-only inference provider for the shared-window runner.

The checkpoint loader is also reused by the validation-only HF exporter without
reading data.  Terminal rows are built only by ``provide_terminal_inputs`` after
the exact checkpoint, approval, provider registration, and HF threshold gates.
This module never participates in training or checkpoint selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    build_samples,
    load_terminal_spr_test_targets,
)

from .adapter_factory import AdapterFactoryConfig, build_production_adapter
from .beats_temporal import HFTargetPolicy, TokenAlignmentPolicy, raw_intervals_to_token_supervision
from .contracts import PREDICTION_UNITS, SAMPLE_RATE, WaveformSample
from .hf_data import HFSampleRecord, build_hf_manifest, load_hf_waveform
from .hf_thresholds import VerifiedHFThresholdReceipt
from .joint_native import JointNativeProjector
from .sliding_window import collate_sliding_windows, masked_mean_window_embeddings
from .terminal_scoring import (
    HFTemporalTerminalBatch,
    MulticlassTerminalBatch,
    NATIVE_TASKS,
    TERMINAL_SCORER_SCHEMA_VERSION,
    TerminalScoringInput,
    terminal_provider_identity_sha256,
)
from .train_shared_window import (
    RUNNER_SCHEMA_VERSION,
    TrainingRunnerConfig,
    assemble_trainable_modules,
    structured_state_sha256,
)


PROVIDER_SPECIFICATION = (
    "baseline.multidataset_pipeline.production_terminal_provider:provide_terminal_inputs"
)
TERMINAL_BATCH_SIZE = 16
EXPECTED_TERMINAL_UNITS = {
    "ICBHI": 2756,
    "SPRSound": 1429,
    "HF": 1956,
    "KAUH": 69,
}
LANE_BY_DATASET = {
    "icbhi": "ICBHI",
    "sprsound": "SPRSound",
    "hf_lung": "HF",
    "kauh": "KAUH",
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def production_provider_identity_sha256() -> str:
    return terminal_provider_identity_sha256(
        PROVIDER_SPECIFICATION, _sha256_path(_implementation_path())
    )


@dataclass(frozen=True)
class TerminalUnit:
    lane: str
    sample: Sample
    hf_record: HFSampleRecord | None = None

    def __post_init__(self) -> None:
        if self.lane not in PREDICTION_UNITS:
            raise ValueError(f"unknown terminal lane: {self.lane}")
        if self.sample.partition != "test":
            raise ValueError("terminal provider accepts only frozen test rows")
        if LANE_BY_DATASET.get(self.sample.dataset) != self.lane:
            raise ValueError("terminal canonical dataset/lane mismatch")
        if (self.lane == "HF") != (self.hf_record is not None):
            raise ValueError("exactly the HF terminal lane requires a raw record")
        if self.hf_record is not None and (
            self.hf_record.sample_id != self.sample.sample_id
            or self.hf_record.partition != "test"
            or self.hf_record.group_id != self.sample.group_id
        ):
            raise RuntimeError("HF terminal canonical/raw-record join mismatch")


@dataclass(frozen=True)
class TerminalIndex:
    lanes: Mapping[str, tuple[TerminalUnit, ...]]
    canonical_receipt: Mapping[str, object]
    terminal_identity_sha256: str


def build_terminal_index(
    dataset_root: Path,
    *,
    kauh_outer_fold: int,
    enforce_real_counts: bool = True,
) -> TerminalIndex:
    """Build the exact accepted terminal rows without loading SPR terminal labels."""

    if kauh_outer_fold != 0:
        raise RuntimeError("current selected shared-window checkpoint is bound to KAUH fold 0")
    samples, canonical_receipt = build_samples(dataset_root, kauh_outer_fold)
    hf_root = dataset_root / "hf_lung_v1" / "source_original"
    hf_records, hf_receipt = build_hf_manifest(hf_root)
    hf_by_id = {record.sample_id: record for record in hf_records}
    if len(hf_by_id) != len(hf_records):
        raise RuntimeError("duplicate HF terminal record ID")

    lanes: dict[str, list[TerminalUnit]] = {lane: [] for lane in PREDICTION_UNITS}
    for sample in samples:
        if sample.partition != "test":
            continue
        lane = LANE_BY_DATASET.get(sample.dataset)
        if lane is None:
            raise RuntimeError(f"unknown canonical dataset: {sample.dataset}")
        record = hf_by_id.get(sample.sample_id) if lane == "HF" else None
        lanes[lane].append(TerminalUnit(lane, sample, record))
    frozen = {
        lane: tuple(sorted(rows, key=lambda unit: unit.sample.sample_id))
        for lane, rows in lanes.items()
    }
    counts = {lane: len(rows) for lane, rows in frozen.items()}
    if any(not rows for rows in frozen.values()):
        raise RuntimeError(f"terminal partition lacks a native lane: {counts}")
    if enforce_real_counts and counts != EXPECTED_TERMINAL_UNITS:
        raise RuntimeError(
            f"terminal native-unit count gate failed: {counts} != {EXPECTED_TERMINAL_UNITS}"
        )
    spr_rows = frozen["SPRSound"]
    if any(unit.sample.targets or "raw_label" in unit.sample.metadata for unit in spr_rows):
        raise RuntimeError("SPRSound terminal labels leaked into the inference manifest")
    ordered_ids = {
        lane: [unit.sample.sample_id for unit in rows] for lane, rows in frozen.items()
    }
    identity = {
        "schema_version": "shared_window_terminal_data_v1",
        "canonical_loader": "baseline.four_dataset_frozen_encoder.data.build_samples",
        "canonical_receipt_sha256": _canonical_sha256(canonical_receipt),
        "hf_manifest_receipt_sha256": _canonical_sha256(hf_receipt),
        "kauh_outer_fold": kauh_outer_fold,
        "terminal_unit_counts": counts,
        "ordered_prediction_ids": ordered_ids,
        "sprsound_labels_loaded": False,
    }
    return TerminalIndex(frozen, canonical_receipt, _canonical_sha256(identity))


def _load_non_hf_waveform(unit: TerminalUnit) -> WaveformSample:
    try:
        import torchaudio
    except (ImportError, OSError) as error:
        raise RuntimeError("torchaudio is required for terminal waveform decoding") from error
    sample = unit.sample
    waveform, source_rate = torchaudio.load(sample.audio_path)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise RuntimeError(f"terminal source audio must be mono: {sample.audio_path}")
    waveform = waveform.squeeze(0).to(torch.float32)
    source_start_s = float(sample.crop_start_s or 0.0)
    if sample.crop_start_s is not None or sample.crop_end_s is not None:
        if sample.crop_start_s is None or sample.crop_end_s is None:
            raise RuntimeError("terminal crop start/end must both be present")
        start = round(sample.crop_start_s * source_rate)
        end = round(sample.crop_end_s * source_rate)
        if not 0 <= start < end <= waveform.numel():
            raise RuntimeError(f"invalid terminal crop for {sample.sample_id}")
        waveform = waveform[start:end]
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    waveform = waveform.contiguous().to(torch.float32)
    return WaveformSample(
        waveform=waveform,
        sample_id=sample.sample_id,
        dataset_id=unit.lane,
        prediction_unit=PREDICTION_UNITS[unit.lane],
        source_start_s=source_start_s,
        source_end_s=source_start_s + waveform.numel() / SAMPLE_RATE,
        lineage={
            "partition": "test",
            "outer_test_accessed": "true_terminal_score_only",
            "group_id": sample.group_id,
            "source_audio_path": str(Path(sample.audio_path).resolve()),
            "source_sample_rate": str(source_rate),
        },
    )


def _load_terminal_waveform(unit: TerminalUnit, dataset_root: Path) -> WaveformSample:
    if unit.lane != "HF":
        return _load_non_hf_waveform(unit)
    if unit.hf_record is None:
        raise RuntimeError("HF terminal record join missing")
    sample, _ = load_hf_waveform(
        dataset_root / "hf_lung_v1" / "source_original", unit.hf_record
    )
    return sample


def load_exact_selected_model(
    selected_checkpoint: Path,
) -> tuple[TrainingRunnerConfig, torch.nn.Module, JointNativeProjector, Mapping[str, object]]:
    """Load and verify one exact selected full checkpoint without reading data."""
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise RuntimeError("selected checkpoint is missing normalized runner config")
    normalized = dict(raw_config)
    normalized["dataset_root"] = Path(str(normalized["dataset_root"]))
    normalized["run_root"] = Path(str(normalized["run_root"]))
    config = TrainingRunnerConfig(**normalized)
    config.validate()
    if config.phase != "full" or config.pipeline_id not in {"P1", "P2", "P3"}:
        raise RuntimeError(
            "registered terminal provider currently supports selected P1/P2/P3 full checkpoints"
        )
    try:
        selected_checkpoint.resolve().relative_to(config.run_root.resolve())
    except ValueError as error:
        raise RuntimeError("selected checkpoint is outside its frozen run root") from error
    expected_components = payload.get("component_state_sha256")
    states = {
        "dimension_adapter": payload.get("dimension_adapter_state"),
        "joint_native_model": payload.get("joint_native_state"),
        "optimizer": payload.get("optimizer_state"),
        "scheduler": payload.get("scheduler_state"),
        "planner": payload.get("planner_state"),
    }
    actual_components = {
        name: structured_state_sha256(state) for name, state in states.items()
    }
    if (
        payload.get("schema_version") != RUNNER_SCHEMA_VERSION
        or payload.get("config_sha256") != config.sha256()
        or payload.get("outer_test_accessed") is not False
        or payload.get("native_metrics_only") is not True
        or not isinstance(expected_components, Mapping)
        or dict(expected_components) != actual_components
    ):
        raise RuntimeError("selected checkpoint provider identity/isolation gate failed")

    repo_root = config.dataset_root.resolve().parents[1]
    adapter = build_production_adapter(
        AdapterFactoryConfig(config.pipeline_id, repo_root, device="cpu")
    )
    torch.manual_seed(config.seed)
    model = assemble_trainable_modules(adapter, device=torch.device("cpu"))
    adapter.dimension_adapter.load_state_dict(payload["dimension_adapter_state"])
    model.load_state_dict(payload["joint_native_state"])
    adapter.eval()
    adapter.backend.eval()
    model.eval()
    return config, adapter, model, payload


def _write_spr_label_free_predictions(
    selected_checkpoint: Path,
    ids: Sequence[str],
    binary_predictions: torch.Tensor,
    raw7_predictions: torch.Tensor,
) -> Path:
    rows = [
        {
            "sample_id": sample_id,
            "spr_binary_prediction": int(binary_predictions[index]),
            "spr_raw7_prediction": int(raw7_predictions[index]),
        }
        for index, sample_id in enumerate(ids)
    ]
    raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode("utf-8")
    root = selected_checkpoint.parent.parent / "terminal_provider"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{selected_checkpoint.stem}_sprsound_predictions_label_free.jsonl"
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("existing SPRSound label-free prediction artifact differs")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return path


def _artifact_receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _write_terminal_joined_predictions(
    selected_checkpoint: Path,
    batches: Sequence[MulticlassTerminalBatch | HFTemporalTerminalBatch],
) -> Path:
    """Persist post-join scorer inputs without overwriting an existing artifact."""

    payload: dict[str, object] = {
        "schema_version": "shared_window_terminal_joined_predictions_v1",
        "selected_checkpoint_sha256": _sha256_path(selected_checkpoint),
        "native_tasks": list(NATIVE_TASKS),
        "prediction_before_spr_label_join": True,
        "outer_test_accessed": True,
        "tasks": {},
    }
    tasks: dict[str, object] = {}
    for batch in batches:
        batch.validate()
        if isinstance(batch, MulticlassTerminalBatch):
            tasks[batch.task] = {
                "prediction_ids": list(batch.prediction_ids),
                "targets": batch.targets.detach().cpu(),
                "predicted_classes": batch.predicted_classes.detach().cpu(),
            }
        else:
            entries = tasks.setdefault(
                batch.task,
                {
                    "prediction_ids": [],
                    "probabilities": [],
                    "targets": [],
                    "window_mask": [],
                    "annotation_mask": [],
                    "valid_mask": [],
                    "time_map": [],
                    "thresholds": batch.thresholds.detach().cpu(),
                    "threshold_receipt_sha256": batch.threshold_receipt_sha256,
                },
            )
            entries["prediction_ids"].extend(batch.prediction_ids)
            for key in (
                "probabilities",
                "targets",
                "window_mask",
                "annotation_mask",
                "valid_mask",
                "time_map",
            ):
                entries[key].append(getattr(batch, key).detach().cpu())
    if set(tasks) != set(NATIVE_TASKS):
        raise RuntimeError("terminal joined artifact lacks an exact native task")
    payload["tasks"] = tasks
    root = selected_checkpoint.parent.parent / "terminal_provider"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{selected_checkpoint.stem}_terminal_joined_predictions.pt"
    temporary = root / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        torch.save(payload, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"terminal joined prediction artifact already exists: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def provide_terminal_inputs(
    selected_checkpoint: Path,
    *,
    verified_hf_threshold_receipt: VerifiedHFThresholdReceipt,
) -> TerminalScoringInput:
    """Infer exact terminal rows and join SPR labels only after durable predictions."""

    if not isinstance(verified_hf_threshold_receipt, VerifiedHFThresholdReceipt):
        raise TypeError("production provider requires the gate-verified HF threshold receipt")
    config, adapter, model, payload = load_exact_selected_model(selected_checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter.to(device).eval()
    adapter.backend.eval()
    model.to(device).eval()
    index = build_terminal_index(
        config.dataset_root, kauh_outer_fold=config.kauh_outer_fold
    )
    thresholds = torch.tensor(
        verified_hf_threshold_receipt.thresholds, dtype=torch.float64
    )
    predicted: dict[str, list[torch.Tensor]] = {task: [] for task in NATIVE_TASKS}
    targets: dict[str, list[torch.Tensor]] = {task: [] for task in NATIVE_TASKS}
    ids: dict[str, list[str]] = {task: [] for task in NATIVE_TASKS}
    hf_batches: list[HFTemporalTerminalBatch] = []

    with torch.no_grad():
        for lane in ("ICBHI", "SPRSound", "HF", "KAUH"):
            units = index.lanes[lane]
            for start in range(0, len(units), TERMINAL_BATCH_SIZE):
                current = units[start : start + TERMINAL_BATCH_SIZE]
                windows = collate_sliding_windows(
                    [_load_terminal_waveform(unit, config.dataset_root) for unit in current]
                ).to(device)
                output = adapter(windows)
                if lane == "HF":
                    logits = model(output.embeddings, "HF")["temporal4"]
                    supervision = raw_intervals_to_token_supervision(
                        output.time_map,
                        output.window_mask,
                        [unit.hf_record.raw_intervals for unit in current],
                        [unit.hf_record.recording_state for unit in current],
                        policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
                        alignment=TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL,
                    )
                    hf_batches.append(
                        HFTemporalTerminalBatch(
                            prediction_ids=tuple(unit.sample.sample_id for unit in current),
                            probabilities=torch.sigmoid(logits).cpu(),
                            targets=supervision.targets.cpu(),
                            window_mask=output.window_mask.cpu(),
                            annotation_mask=supervision.observation_mask.cpu(),
                            valid_mask=supervision.valid_mask.cpu(),
                            time_map=output.time_map.cpu().to(torch.float64),
                            thresholds=thresholds,
                            threshold_receipt_sha256=verified_hf_threshold_receipt.artifact_sha256,
                        )
                    )
                    ids["HF_temporal4"].extend(unit.sample.sample_id for unit in current)
                    continue
                pooled = masked_mean_window_embeddings(output.embeddings, output.window_mask)
                logits = model(pooled, lane)
                if lane == "ICBHI":
                    task_pairs = (("ICBHI_flat4", "flat4", "icbhi_flat4"),)
                elif lane == "SPRSound":
                    task_pairs = (
                        ("SPRSound_binary", "binary", "spr_binary"),
                        ("SPRSound_raw7", "raw7", "spr_seven"),
                    )
                else:
                    task_pairs = (("KAUH_raw9", "raw9", "kauh_raw9"),)
                for task, head, target_key in task_pairs:
                    ids[task].extend(unit.sample.sample_id for unit in current)
                    predicted[task].append(logits[head].argmax(dim=-1).cpu().to(torch.long))
                    if lane != "SPRSound":
                        targets[task].append(
                            torch.tensor(
                                [int(unit.sample.targets[target_key]) for unit in current],
                                dtype=torch.long,
                            )
                        )

    spr_binary = torch.cat(predicted["SPRSound_binary"])
    spr_raw7 = torch.cat(predicted["SPRSound_raw7"])
    spr_label_free_path = _write_spr_label_free_predictions(
        selected_checkpoint, ids["SPRSound_binary"], spr_binary, spr_raw7
    )
    all_samples, _ = build_samples(config.dataset_root, config.kauh_outer_fold)
    spr_targets = load_terminal_spr_test_targets(all_samples)
    ordered_spr = ids["SPRSound_binary"]
    targets["SPRSound_binary"] = [
        torch.tensor([spr_targets[value]["spr_binary"] for value in ordered_spr], dtype=torch.long)
    ]
    targets["SPRSound_raw7"] = [
        torch.tensor([spr_targets[value]["spr_seven"] for value in ordered_spr], dtype=torch.long)
    ]

    batches = []
    for task in ("ICBHI_flat4", "SPRSound_binary", "SPRSound_raw7", "KAUH_raw9"):
        batches.append(
            MulticlassTerminalBatch(
                task=task,
                prediction_ids=tuple(ids[task]),
                targets=torch.cat(targets[task]),
                predicted_classes=torch.cat(predicted[task]),
            )
        )
    batches.extend(hf_batches)
    joined_path = _write_terminal_joined_predictions(selected_checkpoint, batches)
    expected = {task: tuple(ids[task]) for task in NATIVE_TASKS}
    if set(expected) != set(NATIVE_TASKS) or any(not values for values in expected.values()):
        raise RuntimeError("production provider did not cover exactly all native tasks")
    if expected["SPRSound_binary"] != expected["SPRSound_raw7"]:
        raise RuntimeError("SPRSound binary/raw7 terminal ID order differs")
    if tuple(ids["HF_temporal4"]) != tuple(
        unit.sample.sample_id for unit in index.lanes["HF"]
    ):
        raise RuntimeError("HF terminal prediction order differs from frozen index")
    return TerminalScoringInput(
        batches=tuple(batches),
        expected_prediction_ids_by_task=expected,
        data_identity_sha256=str(payload["data_identity_sha256"]),
        provider_identity_sha256=production_provider_identity_sha256(),
        prediction_artifacts={
            "sprsound_label_free_predictions": _artifact_receipt(
                spr_label_free_path
            ),
            "terminal_joined_predictions": _artifact_receipt(joined_path),
        },
        outer_test_accessed=True,
        terminal_targets_loaded=True,
    )
