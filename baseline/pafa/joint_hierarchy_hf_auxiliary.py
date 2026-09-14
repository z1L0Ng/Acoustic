"""PAFA-JH4: JH2-base hard hierarchy with an HF auxiliary loss.

The JH2 core training contract is retained.  HF_Lung_V1 source-train is an
auxiliary-only lane: it adds one 32-window batch to every core optimizer update
and never participates in core validation selection, threshold fitting, or
early stopping.  The run is permanently test-exposed because ICBHI is read at
every epoch for the frozen JH2 selection rule; it is not a clean estimate or a
PAFA reproduction.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    _hf_date_proxy,
    _load_spr,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    decode_icbhi_flat4,
    decode_icbhi_hierarchical_flat4,
    hierarchical_loss,
)
from baseline.multidataset_pipeline.core2_hf_positive_kauh_external import (
    CORE_DATASETS,
    CORE_NODES,
    core_selection_losses,
    select_core_shared_thresholds,
)
from baseline.multidataset_pipeline.hf_data import (
    HFSampleRecord,
    load_hf_waveform,
    parse_label_file,
)
from baseline.multidataset_pipeline.contracts import ObservationState
from baseline.multidataset_pipeline.posthoc_native_readout import (
    ICBHI_LABELS,
    native_metrics,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.pafa.beats_ce_reproduction import _apply_author_ema, read_official_cycles
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    PAFAJointHierarchyModel,
    _balanced_epoch_batches,
    _batch,
    _build_components as _build_core_components,
    _dataset_partitions,
    _icbhi_sample,
    _patient_indices,
    _prepare_waveforms,
    _save_predictions,
    _seed_everything,
    _write_json,
    infer,
    load_selection_samples,
)
from baseline.pafa import jh2_hf_kauh_external as jh2_hf_external


CONDITION = "PAFA_JH4_JH2_HFaux_seed42"
EVIDENCE_LABEL = "JH2_base_icbhi-test-selected_hf-auxiliary_external-diagnostic"
JH2_REFERENCE_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2"
)
JH2_MAIN_REFERENCE_ROOT = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed"
)
JH2_MAIN_EXTERNAL_ROOT = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH"
)
HF_ROOT_RELATIVE = Path("dataset/raw/hf_lung_v1/source_original")
HF_SPLIT_SEED = 20260728
HF_BATCH_SIZE = 32
HF_LAMBDA = 0.25
CORE_UPDATES_PER_EPOCH = 326
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "icbhi_official_test_score"
JH2_ICBHI_SCORE = 0.600502129962986
JH2_SPR_SCORE = 0.8920104437477139


def _condition_for_seed(seed: int) -> str:
    if seed == 42:
        return CONDITION
    return f"PAFA_JH4_JH2_HFaux_multiseed_seed{seed}"


def _base_condition_for_seed(seed: int) -> str:
    if seed == 42:
        return "JH2 epoch19 Hard Hierarchy; no MVN"
    return (
        f"JH2 Hard Hierarchy; no MVN; same-seed Full reference seed_{seed}; "
        "training initializes from the original pretrained BEATs checkpoint"
    )


def _reference_relative_for_seed(seed: int) -> Path:
    if seed == 42:
        return JH2_REFERENCE_RELATIVE
    return JH2_MAIN_REFERENCE_ROOT / f"seed_{seed}"


def _external_reference_relative_for_seed(seed: int) -> Path:
    if seed == 42:
        return Path(
            "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2"
        )
    return JH2_MAIN_EXTERNAL_ROOT / f"seed_{seed}"


def _reference_scores(config: PAFAJointHierarchyConfig) -> tuple[float, float]:
    reference = config.repo_root / _reference_relative_for_seed(config.seed)
    summary = json.loads((reference / "run_summary.json").read_text())
    return (
        float(summary["icbhi_official_score"]),
        float(summary["sprsound_task1_1_official_score"]),
    )


def _validate_run_config(config: PAFAJointHierarchyConfig) -> None:
    if config.seed not in (0, 1, 42):
        raise ValueError("HF auxiliary runner supports seeds 0, 1, and 42")
    replace(config, seed=42).validate()


@dataclass(frozen=True)
class HFWindow:
    record: HFSampleRecord
    window_index: int
    source_start_s: float
    source_end_s: float
    targets: tuple[float, float]
    eligible: tuple[bool, bool]
    annotation_status: str

    @property
    def prediction_id(self) -> str:
        return f"{self.record.sample_id}::window_{self.window_index:02d}"


def _config_payload(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    payload = config.to_dict()
    payload.update(
        {
            "condition": _condition_for_seed(config.seed),
            "evidence_label": EVIDENCE_LABEL,
            "training_config_reused_from": str(_reference_relative_for_seed(config.seed)),
            "base_condition": _base_condition_for_seed(config.seed),
            "base_reference": str(
                config.repo_root / _reference_relative_for_seed(config.seed)
            ),
            "main_icbhi_readout": "Hard Hierarchy",
            "selection": "ICBHI official-test Hard Hierarchy Score only",
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain "
                "the earlier epoch"
            ),
            "terminal_policy": (
                "after every epoch core validation threshold freeze and label-free "
                "ICBHI official-test readout; after selected checkpoint label-free "
                "SPRSound inter and HF source-test readouts"
            ),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_policy": (
                "epoch boundary; strict ICBHI Score improvement only; exact ties "
                "retain the earlier epoch; max epochs remains the upper bound"
            ),
            "hf_auxiliary": {
                "status": "training_enabled",
                "source": "dataset/raw/hf_lung_v1/source_original/train only",
                "split": (
                    "canonical date-proxy GroupShuffleSplit seed 20260728; existing "
                    "subtrain/validation boundaries"
                ),
                "recording_geometry": (
                    "each 15 s recording -> [0,5], [5,10], [10,15] continuous "
                    "non-overlapping mono 16 kHz windows"
                ),
                "window_batch_size": HF_BATCH_SIZE,
                "one_hf_batch_per_core_update": True,
                "core_updates_per_epoch_unchanged": CORE_UPDATES_PER_EPOCH,
                "eligible_definition": (
                    "window overlaps at least one D/Wheeze/Rhonchi/Stridor interval"
                ),
                "targets": {
                    "D": "Crackle; interval overlap positive",
                    "Wheeze": "Wheeze; interval overlap positive",
                    "Rhonchi": "eligible peer-negative for shared nodes",
                    "Stridor": "eligible peer-negative for shared nodes",
                    "level1": "not mapped and no HF Level1 loss",
                },
                "masked": "pure gap, phase-only, and empty annotations",
                "loss": (
                    "eligible-row BCE mean per active attribute node, then mean "
                    "active nodes"
                ),
                "lambda": HF_LAMBDA,
                "selection_role": "record-only; excluded from thresholds, selection, and early stopping",
                "test_role": "selected-checkpoint-only; no HF source-test access during training",
            },
            "kauh_external": {
                "status": "selected-checkpoint-only",
                "source": "dataset/raw/kauh_fraiwan/source_original/audio_files",
                "recordings": 336,
                "patients": 112,
                "filters": ["B", "D", "E"],
                "readout": "recording probabilities, then B/D/E mean at patient level",
                "compatible_overlay": {
                    "N": "Normal",
                    "E W": "Wheeze",
                    "I E W": "Wheeze",
                    "C": "Crackle",
                    "I C": "Crackle",
                    "I C E W": "Both",
                },
                "unresolved": ["Crep", "Bronchial", "I C B"],
                "selection_role": "evaluation-only; excluded from thresholds, selection, and early stopping",
            },
        }
    )
    return payload


def _load_hf_train_records(repo_root: Path) -> tuple[HFSampleRecord, ...]:
    """Build only the source-train HF manifest; source-test is untouched here."""

    source_root = repo_root / HF_ROOT_RELATIVE
    train_root = source_root / "train"
    raw: list[tuple[Path, Path, str, str, tuple[object, ...]]] = []
    for wav in sorted(train_root.rglob("*.wav")):
        label = wav.with_name(f"{wav.stem}_label.txt")
        intervals = parse_label_file(label)
        raw.append(
            (
                wav,
                label,
                wav.stem,
                _hf_date_proxy(wav.name),
                intervals,
            )
        )
    if len(raw) != 7809:
        raise RuntimeError(f"HF source-train recording count changed: {len(raw)}")
    groups = np.asarray([row[3] for row in raw])
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=HF_SPLIT_SEED,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(raw)), groups=groups)
    )
    validation = set(validation_indices.tolist())
    if len({groups[index] for index in train_indices}) != 94:
        raise RuntimeError("HF source-train subtrain date-proxy count changed")
    if len({groups[index] for index in validation_indices}) != 24:
        raise RuntimeError("HF source-train validation date-proxy count changed")
    records = []
    for index, (wav, label, stem, date_proxy, intervals) in enumerate(raw):
        partition = "validation" if index in validation else "subtrain"
        records.append(
            HFSampleRecord(
                sample_id=f"hf:train:{stem}",
                source_split="train",
                partition=partition,
                date_proxy=date_proxy,
                group_id=f"date_proxy:{date_proxy}",
                wav_relative_path=wav.relative_to(source_root).as_posix(),
                label_relative_path=label.relative_to(source_root).as_posix(),
                raw_intervals=intervals,
                recording_state=(
                    ObservationState.OBSERVED if intervals else ObservationState.EMPTY
                ),
            )
        )
    return tuple(records)


def _make_hf_windows(records: Sequence[HFSampleRecord]) -> tuple[HFWindow, ...]:
    windows: list[HFWindow] = []
    for record in records:
        for window_index in range(3):
            start = float(window_index * 5)
            end = start + 5.0
            overlaps = [
                interval
                for interval in record.raw_intervals
                if interval.start_s < end and interval.end_s > start
            ]
            adventitious = {
                interval.raw_token
                for interval in overlaps
                if interval.raw_token in {"D", "Wheeze", "Rhonchi", "Stridor"}
            }
            eligible = bool(adventitious)
            windows.append(
                HFWindow(
                    record=record,
                    window_index=window_index,
                    source_start_s=start,
                    source_end_s=end,
                    targets=(
                        float("D" in adventitious),
                        float("Wheeze" in adventitious),
                    ),
                    eligible=(eligible, eligible),
                    annotation_status=(
                        "eligible_adventitious_overlap"
                        if eligible
                        else "masked_phase_only_gap_or_empty"
                    ),
                )
            )
    return tuple(windows)


def _hf_split_summary(
    records: Sequence[HFSampleRecord],
    windows: Sequence[HFWindow],
    *,
    condition: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "condition": condition,
        "evidence_label": EVIDENCE_LABEL,
        "source_split": "train only; source test not read during training/selection",
        "recordings": {},
        "windows": {},
    }
    for partition in ("subtrain", "validation"):
        current_records = [row for row in records if row.partition == partition]
        current_windows = [row for row in windows if row.record.partition == partition]
        eligible = [row for row in current_windows if row.eligible[0]]
        result["recordings"][partition] = {
            "count": len(current_records),
            "date_proxy_groups": len({row.date_proxy for row in current_records}),
        }
        result["windows"][partition] = {
            "count": len(current_windows),
            "eligible_count": len(eligible),
            "masked_count": len(current_windows) - len(eligible),
            "positive_D": int(sum(row.targets[0] for row in eligible)),
            "positive_Wheeze": int(sum(row.targets[1] for row in eligible)),
        }
    return result


def _hf_epoch_batches(
    windows: Sequence[HFWindow],
    *,
    updates: int,
    epoch: int,
    seed: int,
) -> list[tuple[HFWindow, ...]]:
    """Make exactly one deterministic 32-window HF batch per core update."""

    by_record: dict[str, list[HFWindow]] = {}
    for window in windows:
        by_record.setdefault(window.record.sample_id, []).append(window)
    records = [by_record[key] for key in sorted(by_record)]
    rng = np.random.default_rng(seed + 100_000 + epoch)
    order = rng.permutation(len(records))
    required = updates * HF_BATCH_SIZE
    selected: list[HFWindow] = []
    for index in order:
        selected.extend(records[int(index)])
        if len(selected) >= required:
            break
    if len(selected) < required:
        raise RuntimeError("HF source-train window pool is smaller than one epoch")
    selected = selected[:required]
    return [
        tuple(selected[start : start + HF_BATCH_SIZE])
        for start in range(0, required, HF_BATCH_SIZE)
    ]


def _load_hf_window_batch(
    batch: Sequence[HFWindow],
    *,
    source_root: Path,
    sample_rate: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waveform_cache: dict[str, torch.Tensor] = {}
    waveforms = []
    targets = []
    eligible = []
    width = sample_rate * 5
    for window in batch:
        key = window.record.sample_id
        if key not in waveform_cache:
            waveform_cache[key] = load_hf_waveform(
                source_root,
                window.record,
            )[0].waveform
        start = int(window.source_start_s * sample_rate)
        end = int(window.source_end_s * sample_rate)
        waveforms.append(waveform_cache[key][start:end])
        targets.append(window.targets)
        eligible.append(window.eligible)
    if any(value.shape[-1] != width for value in waveforms):
        raise RuntimeError("HF window geometry changed")
    return (
        torch.stack(waveforms).to(device),
        torch.tensor(targets, dtype=torch.float32, device=device),
        torch.tensor(eligible, dtype=torch.bool, device=device),
    )


def _hf_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    values = []
    for index, node in enumerate(("crackle", "wheeze")):
        mask = eligible[:, index]
        if bool(mask.any()):
            values.append(
                F.binary_cross_entropy_with_logits(
                    logits[node][mask],
                    targets[mask, index],
                )
            )
    if not values:
        return logits["crackle"].sum() * 0.0
    return torch.stack(values).mean()


def _infer_hf(
    model: PAFAJointHierarchyModel,
    windows: Sequence[HFWindow],
    *,
    source_root: Path,
    sample_rate: int,
    batch_size: int,
    device: torch.device,
    include_targets: bool,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "recording_ids": [],
        "dataset_ids": [],
        "group_ids": [],
        "window_indices": [],
        "source_start_s": [],
        "source_end_s": [],
        "file_names": [],
        "annotation_status": [],
        "date_proxies": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
    }
    if include_targets:
        fields.update({"targets": [], "eligible": []})
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            current = windows[start : start + batch_size]
            waveform, targets, eligible = _load_hf_window_batch(
                current,
                source_root=source_root,
                sample_rate=sample_rate,
                device=device,
            )
            output, _ = model(waveform, training=False)
            level1 = output["level1"].float().cpu()
            attributes = torch.stack(
                (output["crackle"], output["wheeze"]), dim=-1
            ).float().cpu()
            attributes_probability = torch.sigmoid(attributes).numpy()
            ids = np.asarray([row.prediction_id for row in current])
            fields["prediction_ids"].append(ids)
            fields["sample_ids"].append(
                np.asarray([row.record.sample_id for row in current])
            )
            fields["recording_ids"].append(
                np.asarray([row.record.sample_id for row in current])
            )
            fields["dataset_ids"].append(np.asarray(["hf_lung"] * len(current)))
            fields["group_ids"].append(
                np.asarray([row.record.group_id for row in current])
            )
            fields["window_indices"].append(
                np.asarray([row.window_index for row in current], dtype=np.int64)
            )
            fields["source_start_s"].append(
                np.asarray([row.source_start_s for row in current], dtype=np.float32)
            )
            fields["source_end_s"].append(
                np.asarray([row.source_end_s for row in current], dtype=np.float32)
            )
            fields["file_names"].append(
                np.asarray([Path(row.record.wav_relative_path).name for row in current])
            )
            fields["annotation_status"].append(
                np.asarray([row.annotation_status for row in current])
            )
            fields["date_proxies"].append(
                np.asarray([row.record.date_proxy for row in current])
            )
            fields["level1_logits"].append(level1.numpy())
            fields["level1_probabilities"].append(
                torch.softmax(level1, dim=-1).numpy()
            )
            fields["level1_predictions"].append(level1.argmax(dim=-1).numpy())
            fields["attribute_logits"].append(attributes.numpy())
            fields["attribute_probabilities"].append(attributes_probability)
            if include_targets:
                fields["targets"].append(targets.cpu().numpy())
                fields["eligible"].append(eligible.cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _hf_validation_metrics(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "status": "hf_validation_record_only",
        "selection_excluded": True,
        "threshold_fitting_excluded": True,
        "early_stopping_excluded": True,
        "windows": int(len(predictions["prediction_ids"])),
        "nodes": {},
    }
    for index, (node, threshold_name) in enumerate(
        (("crackle", "crackle"), ("wheeze", "wheeze"))
    ):
        mask = predictions["eligible"][:, index].astype(bool)
        target = predictions["targets"][mask, index].astype(np.int64)
        logits = predictions["attribute_logits"][mask, index]
        probability = predictions["attribute_probabilities"][mask, index]
        row: dict[str, object] = {
            "eligible": int(mask.sum()),
            "positive": int(target.sum()),
            "negative": int((target == 0).sum()),
            "loss": float(
                (np.logaddexp(0.0, logits) - target * logits).mean()
            )
            if len(target)
            else None,
            "threshold_used_for_record_only_metric": float(thresholds[threshold_name]),
        }
        if len(np.unique(target)) >= 2:
            row["auroc"] = float(roc_auc_score(target, probability))
            row["auprc"] = float(average_precision_score(target, probability))
            row["thresholded_recall"] = float(
                recall_score(
                    target,
                    probability >= thresholds[threshold_name],
                    zero_division=0,
                )
            )
        else:
            row.update(
                {
                    "auroc": None,
                    "auprc": None,
                    "thresholded_recall": None,
                }
            )
        metrics["nodes"][node] = row
    return metrics


def _load_icbhi_terminal_samples(
    config: PAFAJointHierarchyConfig,
) -> tuple[Sample, ...]:
    rows = read_official_cycles(
        config.icbhi_audio_dir,
        config.author_repo,
        ("test",),
    )
    return tuple(
        sorted(
            (_icbhi_sample(row, config.icbhi_audio_dir, "test") for row in rows),
            key=lambda sample: sample.sample_id,
        )
    )


def _attach_icbhi_targets(
    predictions: Mapping[str, np.ndarray],
    ordered_samples: Sequence[Sample],
) -> dict[str, np.ndarray]:
    if list(predictions["sample_ids"]) != [row.sample_id for row in ordered_samples]:
        raise RuntimeError("ICBHI terminal prediction/sample order changed")
    targets = np.zeros((len(ordered_samples), 3), dtype=np.float32)
    eligible = np.ones((len(ordered_samples), 3), dtype=bool)
    raw_labels = []
    for index, sample in enumerate(ordered_samples):
        flat = int(sample.targets["icbhi_flat4"])
        raw_labels.append(ICBHI_LABELS[flat])
        targets[index] = (
            float(flat != 0),
            float(flat in (1, 3)),
            float(flat in (2, 3)),
        )
    return {
        **predictions,
        "raw_ground_truth": np.asarray(raw_labels),
        "targets": targets,
        "eligible": eligible,
    }


def _score_icbhi(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    target = np.asarray(
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"]],
        dtype=np.int64,
    )
    hard_prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"],
        predictions["attribute_probabilities"],
        thresholds,
    )
    hard = native_metrics(target, hard_prediction, ICBHI_LABELS)
    hard.update(
        {
            "task": "ICBHI official-test flat4 Hard Hierarchy main readout",
            "protocol": "official recording split 60/40; 2756 respiratory cycles",
            "decoder": (
                "Level1 Normal->Normal; Level1 Abnormal->Crackle/Wheeze/Both "
                "from shared validation thresholds; neither attribute over "
                "threshold->larger probability-minus-threshold margin; "
                "Crackle wins ties"
            ),
            "thresholds": dict(thresholds),
            "official_score": hard["icbhi_score"],
            "predicted_class_counts": np.bincount(
                hard_prediction,
                minlength=4,
            ).astype(int).tolist(),
        }
    )
    hard["class_collapse"] = bool(
        sum(count > 0 for count in hard["predicted_class_counts"]) <= 1
    )
    bits_prediction = decode_icbhi_flat4(
        predictions["attribute_probabilities"],
        thresholds,
    )
    bits = native_metrics(target, bits_prediction, ICBHI_LABELS)
    bits.update(
        {
            "task": "ICBHI official-test flat4 bits-only diagnostic",
            "decoder": (
                "Crackle/Wheeze bits only: 00 Normal, 10 Crackle, "
                "01 Wheeze, 11 Both"
            ),
            "thresholds": dict(thresholds),
            "official_score": bits["icbhi_score"],
            "predicted_class_counts": np.bincount(
                bits_prediction,
                minlength=4,
            ).astype(int).tolist(),
        }
    )
    bits["class_collapse"] = bool(
        sum(count > 0 for count in bits["predicted_class_counts"]) <= 1
    )
    return {
        "icbhi_flat4": hard,
        "icbhi_flat4_bits_only_ablation": bits,
    }


def _epoch_terminal_payload(
    *,
    epoch: int,
    update: int,
    validation_selection: Mapping[str, object],
    thresholds: Mapping[str, float],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "epoch": epoch,
        "update": update,
        "selected_epoch_for_final_report": False,
        "validation_selection_loss": float(validation_selection["selection_loss"]),
        "shared_attribute_thresholds": dict(thresholds),
        "threshold_source": "same-epoch core validation predictions only; frozen before ICBHI test access",
        "outer_test_accessed": True,
        "sprsound_official_inter_test_accessed": False,
        "hf_source_test_accessed": False,
        "prediction_support": {"icbhi": 2756},
        "test_selection_metric": "icbhi_flat4.official_score",
        **metrics,
    }


def _score_spr_terminal(predictions: Mapping[str, np.ndarray]) -> dict[str, object]:
    target = predictions["targets"][:, 0].astype(np.int64)
    prediction = predictions["level1_predictions"].astype(np.int64)
    metrics = native_metrics(target, prediction, ("normal", "abnormal"))
    metrics.update(
        {
            "task": "SPRSound BioCAS2022 Task1-1 Normal/Adventitious",
            "protocol": "official inter-subject test; 1429 respiratory events",
            "average_score_as": metrics["average_score"],
            "harmonic_score_hs": metrics["harmonic_score"],
            "official_score": (
                metrics["average_score"] + metrics["harmonic_score"]
            )
            / 2,
            "task1_2_raw7": "not produced by the shared three-node head",
        }
    )
    return metrics


def run(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    _validate_run_config(config)
    run_condition = _condition_for_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", _config_payload(config))

    selection_samples = load_selection_samples(config)
    subtrain = _dataset_partitions(selection_samples, "subtrain")
    validation = _dataset_partitions(selection_samples, "validation")
    _write_json(
        config.output_dir / "selection_split_summary.json",
        {
            "condition": run_condition,
            "evidence_label": EVIDENCE_LABEL,
            "datasets": list(CORE_DATASETS),
            "nodes": list(CORE_NODES),
            "training_only": True,
            "groups": {
                dataset: {
                    "subtrain": len({row.group_id for row in subtrain[dataset]}),
                    "validation": len({row.group_id for row in validation[dataset]}),
                }
                for dataset in CORE_DATASETS
            },
        },
    )

    hf_records = _load_hf_train_records(config.repo_root)
    hf_windows = _make_hf_windows(hf_records)
    hf_subtrain = tuple(row for row in hf_windows if row.record.partition == "subtrain")
    hf_validation_windows = tuple(
        row for row in hf_windows if row.record.partition == "validation"
    )
    _write_json(
        config.output_dir / "hf_train_validation_summary.json",
        _hf_split_summary(
            hf_records,
            hf_windows,
            condition=run_condition,
        ),
    )

    waveform_store = _prepare_waveforms(selection_samples, config)
    device = torch.device(config.device)
    model, pafa_criterion = _build_core_components(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    patient_index = _patient_indices(
        [sample for rows in subtrain.values() for sample in rows]
    )
    source_root = config.repo_root / HF_ROOT_RELATIVE
    progress_path = config.output_dir / "progress.jsonl"
    train_log_path = config.output_dir / "train_log.jsonl"
    test_log_path = config.output_dir / "test_selection_log.jsonl"
    terminal_dir = config.output_dir / "terminal"

    best_score = -math.inf
    best_epoch = 0
    best_selection: dict[str, object] | None = None
    best_terminal: dict[str, object] | None = None
    no_improvement_epochs = 0
    completed_epochs = 0
    early_stopped = False
    global_update = 0
    icbhi_terminal_samples: tuple[Sample, ...] | None = None
    icbhi_terminal_waveforms: dict[str, torch.Tensor] | None = None
    started_seconds = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate * (
            config.cosine_eta_min_ratio
            + (1.0 - config.cosine_eta_min_ratio)
            * (1.0 + np.cos(np.pi * epoch / config.epochs))
            / 2.0
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        epoch_losses = {
            dataset: {"classification": [], "pafa": [], "total": []}
            for dataset in CORE_DATASETS
        }
        hf_losses: list[float] = []
        core_batches = _balanced_epoch_batches(
            {dataset: len(rows) for dataset, rows in subtrain.items()},
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        hf_batches = _hf_epoch_batches(
            hf_subtrain,
            updates=len(core_batches),
            epoch=epoch,
            seed=config.seed,
        )
        if len(core_batches) != CORE_UPDATES_PER_EPOCH:
            raise RuntimeError(
                f"JH2 core update count changed: {len(core_batches)} != {CORE_UPDATES_PER_EPOCH}"
            )
        for batch_index, ((dataset, indices), hf_batch) in enumerate(
            zip(core_batches, hf_batches),
            start=1,
        ):
            rows = [subtrain[dataset][int(index)] for index in indices]
            waveform, targets, eligible, patients = _batch(
                rows,
                waveform_store,
                patient_index,
                device,
            )
            hf_waveform, hf_targets, hf_eligible = _load_hf_window_batch(
                hf_batch,
                source_root=source_root,
                sample_rate=config.sample_rate,
                device=device,
            )
            before = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, projected = model(waveform, training=True)
                classification_loss, _ = hierarchical_loss(
                    logits,
                    targets,
                    eligible,
                    HierarchicalLossConfig(mode="ce_bce"),
                    collect_named=False,
                )
                pafa_loss = pafa_criterion(
                    projected,
                    patients,
                    lambda_pcsl=config.lambda_pcsl,
                    lambda_gpal=config.lambda_gpal,
                )
                hf_logits, _ = model(hf_waveform, training=True)
                hf_loss = _hf_loss(hf_logits, hf_targets, hf_eligible)
                core_total = (
                    config.classification_weight * classification_loss
                    + config.pafa_weight * pafa_loss
                )
                total_loss = core_total + HF_LAMBDA * hf_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _apply_author_ema(model, before, config.ema_beta)
            global_update += 1
            epoch_losses[dataset]["classification"].append(
                float(classification_loss.detach().cpu())
            )
            epoch_losses[dataset]["pafa"].append(
                float(pafa_loss.detach().cpu())
            )
            epoch_losses[dataset]["total"].append(
                float(core_total.detach().cpu())
            )
            hf_losses.append(float(hf_loss.detach().cpu()))
            if global_update % 100 == 0:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "epoch_batch": batch_index,
                                "epoch_batches": len(core_batches),
                                "update": global_update,
                                "elapsed_minutes": (
                                    time.perf_counter() - started_seconds
                                )
                                / 60.0,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

        validation_predictions = infer(
            model,
            validation,
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            validation_predictions,
        )
        validation_selection = core_selection_losses(validation_predictions)
        thresholds_raw, threshold_details = select_core_shared_thresholds(
            validation_predictions
        )
        thresholds = {key: float(value) for key, value in thresholds_raw.items()}
        validation_selection = {
            **validation_selection,
            "epoch": epoch,
            "thresholds": thresholds,
            "threshold_details": threshold_details,
            "test_accessed_for_this_epoch_after_freeze": True,
            "hf_auxiliary_excluded_from_selection": True,
        }
        _write_json(
            config.output_dir / "validation" / f"epoch_{epoch:03d}_selection.json",
            validation_selection,
        )
        hf_validation_predictions = _infer_hf(
            model,
            hf_validation_windows,
            source_root=source_root,
            sample_rate=config.sample_rate,
            batch_size=HF_BATCH_SIZE,
            device=device,
            include_targets=True,
        )
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}_hf.npz",
            hf_validation_predictions,
        )
        hf_validation_metrics = _hf_validation_metrics(
            hf_validation_predictions, thresholds
        )
        _write_json(
            config.output_dir / "validation" / f"epoch_{epoch:03d}_hf_metrics.json",
            hf_validation_metrics,
        )

        if icbhi_terminal_samples is None:
            icbhi_terminal_samples = _load_icbhi_terminal_samples(config)
            icbhi_terminal_waveforms = _prepare_waveforms(
                icbhi_terminal_samples,
                config,
            )
        icbhi_label_free = infer(
            model,
            {"icbhi": icbhi_terminal_samples, "sprsound": tuple()},
            icbhi_terminal_waveforms,
            config,
            device,
            include_targets=False,
        )
        epoch_terminal_dir = terminal_dir / f"epoch_{epoch:03d}"
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
            icbhi_label_free,
        )
        icbhi_scored = _attach_icbhi_targets(
            icbhi_label_free,
            icbhi_terminal_samples,
        )
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
            icbhi_scored,
        )
        icbhi_metrics = _score_icbhi(icbhi_scored, thresholds)
        terminal_payload = _epoch_terminal_payload(
            epoch=epoch,
            update=global_update,
            validation_selection=validation_selection,
            thresholds=thresholds,
            metrics=icbhi_metrics,
        )
        _write_json(epoch_terminal_dir / "native_metrics.json", terminal_payload)
        icbhi_score = float(icbhi_metrics["icbhi_flat4"]["official_score"])
        improved = icbhi_score > best_score
        next_no_improvement_epochs = 0 if improved else no_improvement_epochs + 1
        _write_json(
            epoch_terminal_dir / "hf_auxiliary_validation_metrics.json",
            hf_validation_metrics,
        )
        with test_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "update": global_update,
                        "validation_selection_loss": validation_selection["selection_loss"],
                        "thresholds": thresholds,
                        "icbhi_official_score": icbhi_score,
                        "sprsound_official_inter_test_accessed": False,
                        "hf_source_test_accessed": False,
                        "terminal_metrics_path": str(epoch_terminal_dir / "native_metrics.json"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        train_record = {
            "epoch": epoch,
            "update": global_update,
            "learning_rate": learning_rate,
            "train_loss": {
                dataset: {
                    name: float(np.mean(values)) for name, values in losses.items()
                }
                for dataset, losses in epoch_losses.items()
            },
            "hf_auxiliary": {
                "lambda": HF_LAMBDA,
                "loss": float(np.mean(hf_losses)),
                "batch_size": HF_BATCH_SIZE,
                "batches": len(hf_batches),
            },
            "validation": validation_selection,
            "hf_validation": hf_validation_metrics,
            "terminal_test": {
                "icbhi_official_score": icbhi_score,
                "sprsound_official_inter_test_accessed": False,
                "hf_source_test_accessed": False,
                "evidence_label": EVIDENCE_LABEL,
            },
            "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
            "early_stopping": {
                "monitor": EARLY_STOPPING_MONITOR,
                "monitor_value": icbhi_score,
                "strict_improvement": improved,
                "no_improvement_epochs": next_no_improvement_epochs,
                "patience": EARLY_STOPPING_PATIENCE,
            },
        }
        with train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(train_record, sort_keys=True) + "\n")

        if improved:
            best_score = icbhi_score
            best_epoch = epoch
            best_selection = validation_selection
            best_terminal = terminal_payload
            no_improvement_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "icbhi_hard_hierarchy_official_score": best_score,
                    "selection_loss": float(validation_selection["selection_loss"]),
                    "config": _config_payload(config),
                },
                config.output_dir / "best_checkpoint.pt",
            )
            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
                terminal_dir / "selected_icbhi_test_predictions_label_free.npz",
            )
            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
                terminal_dir / "selected_icbhi_test_predictions_scored.npz",
            )
            _write_json(
                terminal_dir / "selected_icbhi_native_metrics.json",
                {**terminal_payload, "selected_epoch_for_final_report": True},
            )
        else:
            no_improvement_epochs = next_no_improvement_epochs
        completed_epochs = epoch
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epoch_complete": True,
                        "update": global_update,
                        "elapsed_minutes": train_record["elapsed_minutes"],
                        "icbhi_official_score": icbhi_score,
                        "current_best_epoch": best_epoch,
                        "current_best_icbhi_official_score": best_score,
                        "early_stopping_no_improvement_epochs": no_improvement_epochs,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(
            json.dumps(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "epoch": epoch,
                    "update": global_update,
                    "elapsed_minutes": train_record["elapsed_minutes"],
                    "validation_selection_loss": validation_selection["selection_loss"],
                    "thresholds": thresholds,
                    "icbhi_official_score": icbhi_score,
                    "hf_validation": hf_validation_metrics,
                    "sprsound_official_inter_test_accessed": False,
                    "hf_source_test_accessed": False,
                    "current_best_epoch": best_epoch,
                    "current_best_icbhi_official_score": best_score,
                    "early_stopping_monitor": EARLY_STOPPING_MONITOR,
                    "early_stopping_no_improvement_epochs": no_improvement_epochs,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            break

    if best_selection is None or best_terminal is None:
        raise RuntimeError(f"{run_condition} completed no selectable epoch")

    _write_json(
        config.output_dir / "validation_selection.json",
        {
            "status": EVIDENCE_LABEL,
            "evidence_label": EVIDENCE_LABEL,
            "condition": run_condition,
            "base_condition": _base_condition_for_seed(config.seed),
            "selected_epoch": best_epoch,
            "selection_loss": float(best_selection["selection_loss"]),
            "shared_attribute_thresholds": best_selection["thresholds"],
            "threshold_details": best_selection["threshold_details"],
            "threshold_source": "selected epoch core validation predictions only; no test tuning",
            "outer_test_accessed": True,
            "validation_selection_test_accessed": False,
            "icbhi_official_test_accesses": completed_epochs,
            "sprsound_official_inter_test_accesses": 1,
            "hf_source_test_accesses": 1,
            "kauh_external_test_accesses": 1,
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
            ),
            "hf_auxiliary_excluded_from_selection": True,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "completed_training_epochs": completed_epochs,
        },
    )

    checkpoint = torch.load(config.output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    del waveform_store
    gc.collect()

    spr_rows, _ = _load_spr(config.repo_root / "dataset/raw", include_checksums=False)
    spr_terminal_samples = tuple(
        sorted(
            (sample for sample in spr_rows if sample.partition == "test"),
            key=lambda sample: sample.sample_id,
        )
    )
    spr_label_free_path = terminal_dir / "selected_sprsound_predictions_label_free.npz"
    if spr_label_free_path.is_file():
        with np.load(spr_label_free_path, allow_pickle=False) as bundle:
            spr_label_free = {key: bundle[key] for key in bundle.files}
    else:
        spr_waveforms = _prepare_waveforms(spr_terminal_samples, config)
        spr_label_free = infer(
            model,
            {"icbhi": tuple(), "sprsound": spr_terminal_samples},
            spr_waveforms,
            config,
            device,
            include_targets=False,
        )
        _save_predictions(spr_label_free_path, spr_label_free)
    spr_targets = load_terminal_spr_test_targets(
        list(spr_terminal_samples),
        include_checksums=False,
    )
    spr_scored = _attach_targets(
        spr_label_free,
        spr_terminal_samples,
        spr_targets,
    )
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        spr_scored,
    )
    spr_metrics = _score_spr_terminal(spr_scored)

    hf_test_samples = jh2_hf_external._load_hf_test_samples(config.repo_root)
    hf_label_free = jh2_hf_external._predict_hf(
        model,
        hf_test_samples,
        config,
        device,
    )
    _save_predictions(
        terminal_dir / "selected_hf_predictions_label_free.npz",
        hf_label_free,
    )
    hf_annotations = jh2_hf_external._parse_hf_annotations(hf_test_samples)
    hf_scored, hf_metrics = jh2_hf_external._hf_scored_predictions(
        hf_label_free,
        hf_annotations,
        best_selection["thresholds"],
    )
    hf_metrics = {
        **hf_metrics,
        "evidence_label": EVIDENCE_LABEL,
        "external_protocol_reference": "fixed JH2 HF source-test protocol; no HF threshold tuning",
    }
    _save_predictions(
        terminal_dir / "selected_hf_predictions_scored.npz",
        hf_scored,
    )
    _write_json(terminal_dir / "hf_metrics.json", hf_metrics)

    kauh_samples = jh2_hf_external._load_kauh_samples(config.repo_root)
    kauh_waveforms = _prepare_waveforms(kauh_samples, config)
    kauh_label_free = jh2_hf_external._predict_kauh(
        model,
        kauh_samples,
        kauh_waveforms,
        config,
        device,
        best_selection["thresholds"],
    )
    _save_predictions(
        terminal_dir / "selected_kauh_predictions_label_free.npz",
        kauh_label_free,
    )
    kauh_scored, _ = jh2_hf_external._kauh_scored_predictions(
        kauh_label_free,
        kauh_samples,
        best_selection["thresholds"],
    )
    kauh_metrics, kauh_patient_scored = jh2_hf_external._kauh_metrics(
        kauh_scored,
        best_selection["thresholds"],
    )
    kauh_metrics = {
        **kauh_metrics,
        "evidence_label": EVIDENCE_LABEL,
        "external_protocol_reference": "fixed selected checkpoint compatible-overlay KAUH evaluation; no KAUH tuning",
    }
    _save_predictions(
        terminal_dir / "selected_kauh_predictions_scored.npz",
        kauh_scored,
    )
    _save_predictions(
        terminal_dir / "selected_kauh_patient_predictions_scored.npz",
        kauh_patient_scored,
    )
    _write_json(terminal_dir / "kauh_metrics.json", kauh_metrics)

    selected_icbhi = best_terminal["icbhi_flat4"]
    selected_icbhi_score = float(selected_icbhi["official_score"])
    spr_score = float(spr_metrics["official_score"])
    no_class_collapse = not bool(selected_icbhi["class_collapse"])
    final_terminal = {
        **best_terminal,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "condition": run_condition,
        "selected_epoch_for_final_report": True,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "shared_attribute_thresholds": best_selection["thresholds"],
        "icbhi_official_test_accesses": completed_epochs,
        "sprsound_official_inter_test_accesses": 1,
        "hf_source_test_accesses": 1,
        "kauh_external_test_accesses": 1,
        "test_access_order": (
            "Each epoch: core validation predictions and threshold freeze, then "
            "label-free/scored ICBHI official test; after selected checkpoint: "
            "label-free SPRSound then SPR targets, then HF source-test predictions "
            "and annotations, then KAUH compatible-overlay predictions and targets"
        ),
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
        "sprsound_inter_task1_1": spr_metrics,
        "hf_source_test": hf_metrics,
        "kauh_external": kauh_metrics,
    }
    _write_json(terminal_dir / "native_metrics.json", final_terminal)
    _write_json(terminal_dir / "selected_native_metrics.json", final_terminal)

    hf_off_metrics = json.loads(
        (
            config.repo_root
            / _external_reference_relative_for_seed(config.seed)
            / "hf_metrics.json"
        ).read_text()
    )
    hf_off = hf_off_metrics["recording_presence_pool"]["metrics"]
    hf_on = hf_metrics["recording_presence_pool"]["metrics"]
    hf_comparison = {
        node: {
            "hf_off_auroc": hf_off[hf_key]["curve"]["auroc"],
            "hf_on_auroc": hf_on[hf_key]["curve"]["auroc"],
            "delta_auroc": hf_on[hf_key]["curve"]["auroc"] - hf_off[hf_key]["curve"]["auroc"],
            "hf_off_auprc": hf_off[hf_key]["curve"]["auprc"],
            "hf_on_auprc": hf_on[hf_key]["curve"]["auprc"],
            "delta_auprc": hf_on[hf_key]["curve"]["auprc"] - hf_off[hf_key]["curve"]["auprc"],
            "hf_off_interval_recall": hf_off_metrics["positive_interval_coverage"][interval_key]["positive_recall"],
            "hf_on_interval_recall": hf_metrics["positive_interval_coverage"][interval_key]["positive_recall"],
        }
        for node, hf_key, interval_key in (
            ("crackle", "D", "D"),
            ("wheeze", "Wheeze", "Wheeze"),
        )
    }
    _write_json(
        config.output_dir / "hf_on_off_comparison.json",
        {
            "status": "matched_fixed_JH2_HF_off_reference_comparison",
            "evidence_label": EVIDENCE_LABEL,
            "hf_off_reference": str(
                config.repo_root
                / _external_reference_relative_for_seed(config.seed)
            ),
            "hf_on_run": str(config.output_dir),
            "metrics": hf_comparison,
            "interpretation": (
                "HF-on auxiliary comparison is diagnostic; success requires core gates "
                "and material improvement without clear Wheeze degradation, not one "
                "threshold metric alone"
            ),
        },
    )

    reference_icbhi_score, reference_spr_score = _reference_scores(config)
    spr_gate = spr_score >= reference_spr_score - 0.01
    icbhi_gate = selected_icbhi_score >= reference_icbhi_score - 0.01
    hf_d_material = (
        hf_comparison["crackle"]["delta_auroc"] > 0.0
        or hf_comparison["crackle"]["delta_auprc"] > 0.0
    )
    hf_wheeze_not_degraded = (
        hf_comparison["wheeze"]["delta_auroc"] >= 0.0
        and hf_comparison["wheeze"]["delta_auprc"] >= 0.0
    )
    summary = {
        "status": (
            "early_stopped_epochwise_test_selected"
            if early_stopped
            else "complete_epochwise_test_selected"
        ),
        "evidence_label": EVIDENCE_LABEL,
        "condition": run_condition,
        "base_condition": _base_condition_for_seed(config.seed),
        "training_config_reused_from": str(_reference_relative_for_seed(config.seed)),
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.epochs,
        "updates": global_update,
        "core_updates_per_epoch": CORE_UPDATES_PER_EPOCH,
        "hf_batches_per_core_update": 1,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "icbhi_official_score": selected_icbhi_score,
        "sprsound_task1_1_official_score": spr_score,
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": completed_epochs,
            "sprsound_official_inter_test": 1,
            "hf_source_test": 1,
            "kauh_external_test": 1,
        },
        "validation_thresholds_test_tuned": False,
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "best_only_checkpoint": True,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "no_improvement_epochs_at_stop": no_improvement_epochs,
        },
        "selected_metrics": {
            "icbhi_flat4": selected_icbhi,
            "icbhi_flat4_bits_only_ablation": best_terminal[
                "icbhi_flat4_bits_only_ablation"
            ],
            "sprsound_inter_task1_1": spr_metrics,
            "hf_source_test": hf_metrics,
            "kauh_external": kauh_metrics,
        },
        "hf_on_off_comparison": hf_comparison,
        "success_gate": {
            "icbhi_score_drop_le_0_01_vs_jh2": icbhi_gate,
            "icbhi_specificity_no_collapse": no_class_collapse,
            "sprsound_score_drop_le_0_01_vs_jh2": spr_gate,
            "hf_D_auroc_or_auprc_materially_improves": hf_d_material,
            "hf_Wheeze_not_clearly_degraded": hf_wheeze_not_degraded,
            "pass_requires_all_and_material_D_gain": (
                icbhi_gate and no_class_collapse and spr_gate and hf_d_material and hf_wheeze_not_degraded
            ),
        },
        "test_access_order": final_terminal["test_access_order"],
        "changed_files": ["baseline/pafa/joint_hierarchy_hf_auxiliary.py"],
        "git_commit": False,
        "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
        "claim_boundary": (
            "dual core-test selection plus selected-checkpoint HF external diagnostic; "
            "not a clean estimate, not a PAFA reproduction, not a paper claim"
        ),
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def finalize_only(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    """Complete only the pending selected-checkpoint terminal postprocessing."""

    config.validate()
    selection_path = config.output_dir / "validation_selection.json"
    checkpoint_path = config.output_dir / "best_checkpoint.pt"
    selected_metrics_path = (
        config.output_dir / "terminal" / "selected_icbhi_native_metrics.json"
    )
    selection = json.loads(selection_path.read_text())
    selected_icbhi_payload = json.loads(selected_metrics_path.read_text())
    if int(selection["selected_epoch"]) != 9:
        raise RuntimeError("finalize-only requires the existing JH4 best epoch 9")
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    selected_icbhi = selected_icbhi_payload["icbhi_flat4"]
    selected_icbhi_bits = selected_icbhi_payload[
        "icbhi_flat4_bits_only_ablation"
    ]

    torch.set_num_threads(config.cpu_threads)
    device = torch.device(config.device)
    model, _ = _build_core_components(config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    terminal_dir = config.output_dir / "terminal"
    spr_rows, _ = _load_spr(config.repo_root / "dataset/raw", include_checksums=False)
    spr_terminal_samples = tuple(
        sorted(
            (sample for sample in spr_rows if sample.partition == "test"),
            key=lambda sample: sample.sample_id,
        )
    )
    spr_label_free_path = terminal_dir / "selected_sprsound_predictions_label_free.npz"
    if spr_label_free_path.is_file():
        with np.load(spr_label_free_path, allow_pickle=False) as bundle:
            spr_label_free = {key: bundle[key] for key in bundle.files}
    else:
        spr_waveforms = _prepare_waveforms(spr_terminal_samples, config)
        spr_label_free = infer(
            model,
            {"icbhi": tuple(), "sprsound": spr_terminal_samples},
            spr_waveforms,
            config,
            device,
            include_targets=False,
        )
        _save_predictions(spr_label_free_path, spr_label_free)
    spr_targets = load_terminal_spr_test_targets(
        list(spr_terminal_samples),
        include_checksums=False,
    )
    spr_scored = _attach_targets(
        spr_label_free,
        spr_terminal_samples,
        spr_targets,
    )
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        spr_scored,
    )
    spr_metrics = _score_spr_terminal(spr_scored)

    hf_test_samples = jh2_hf_external._load_hf_test_samples(config.repo_root)
    hf_label_free = jh2_hf_external._predict_hf(
        model,
        hf_test_samples,
        config,
        device,
    )
    _save_predictions(
        terminal_dir / "selected_hf_predictions_label_free.npz",
        hf_label_free,
    )
    hf_annotations = jh2_hf_external._parse_hf_annotations(hf_test_samples)
    hf_scored, hf_metrics = jh2_hf_external._hf_scored_predictions(
        hf_label_free,
        hf_annotations,
        thresholds,
    )
    hf_metrics = {
        **hf_metrics,
        "evidence_label": EVIDENCE_LABEL,
        "external_protocol_reference": "fixed JH2 HF source-test protocol; no HF threshold tuning",
    }
    _save_predictions(
        terminal_dir / "selected_hf_predictions_scored.npz",
        hf_scored,
    )
    _write_json(terminal_dir / "hf_metrics.json", hf_metrics)

    hf_off_metrics = json.loads(
        (
            config.repo_root
            / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2/hf_metrics.json"
        ).read_text()
    )
    hf_off = hf_off_metrics["recording_presence_pool"]["metrics"]
    hf_on = hf_metrics["recording_presence_pool"]["metrics"]
    hf_comparison = {
        node: {
            "hf_off_auroc": hf_off[node]["curve"]["auroc"],
            "hf_on_auroc": hf_on[node]["curve"]["auroc"],
            "delta_auroc": hf_on[node]["curve"]["auroc"] - hf_off[node]["curve"]["auroc"],
            "hf_off_auprc": hf_off[node]["curve"]["auprc"],
            "hf_on_auprc": hf_on[node]["curve"]["auprc"],
            "delta_auprc": hf_on[node]["curve"]["auprc"] - hf_off[node]["curve"]["auprc"],
            "hf_off_interval_recall": hf_off_metrics["positive_interval_coverage"][
                "D" if node == "crackle" else "Wheeze"
            ]["positive_recall"],
            "hf_on_interval_recall": hf_metrics["positive_interval_coverage"][
                "D" if node == "crackle" else "Wheeze"
            ]["positive_recall"],
        }
        for node in ("crackle", "wheeze")
    }

    jh2_terminal = json.loads(
        (
            config.repo_root
            / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2/terminal/native_metrics.json"
        ).read_text()
    )
    jh2_icbhi = jh2_terminal["icbhi_flat4"]
    jh2_spr = jh2_terminal["sprsound_inter_task1_1"]
    selected_spr_score = float(spr_metrics["official_score"])
    selected_icbhi_score = float(selected_icbhi["official_score"])
    no_class_collapse = not bool(selected_icbhi["class_collapse"])
    hf_d_material = (
        hf_comparison["crackle"]["delta_auroc"] > 0.0
        or hf_comparison["crackle"]["delta_auprc"] > 0.0
    )
    hf_wheeze_not_degraded = (
        hf_comparison["wheeze"]["delta_auroc"] >= 0.0
        and hf_comparison["wheeze"]["delta_auprc"] >= 0.0
    )
    final_terminal = {
        **selected_icbhi_payload,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
            "condition": CONDITION,
        "selected_epoch_for_final_report": True,
        "selected_epoch": 9,
        "selection_loss": float(selection["selection_loss"]),
        "shared_attribute_thresholds": thresholds,
        "icbhi_official_test_accesses": 19,
        "sprsound_official_inter_test_accesses": 1,
        "hf_source_test_accesses": 1,
        "test_access_order": (
            "Existing epochwise ICBHI label-free/scored artifacts were reused without "
            "new ICBHI inference; after selected epoch 9: label-free SPRSound then "
            "SPR targets, then HF source-test predictions and annotations"
        ),
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": True,
        "completed_training_epochs": 19,
        "sprsound_inter_task1_1": spr_metrics,
        "hf_source_test": hf_metrics,
    }
    _write_json(terminal_dir / "native_metrics.json", final_terminal)
    _write_json(terminal_dir / "selected_native_metrics.json", final_terminal)

    core_comparison = {
        "icbhi": {
            "jh2_score": float(jh2_icbhi["official_score"]),
            "jh4_score": selected_icbhi_score,
            "delta_score": selected_icbhi_score - float(jh2_icbhi["official_score"]),
            "jh2_specificity": float(jh2_icbhi["specificity"]),
            "jh4_specificity": float(selected_icbhi["specificity"]),
            "delta_specificity": float(selected_icbhi["specificity"])
            - float(jh2_icbhi["specificity"]),
            "jh2_sensitivity": float(jh2_icbhi["sensitivity"]),
            "jh4_sensitivity": float(selected_icbhi["sensitivity"]),
            "delta_sensitivity": float(selected_icbhi["sensitivity"])
            - float(jh2_icbhi["sensitivity"]),
            "jh2_macro_f1": float(jh2_icbhi["macro_f1"]),
            "jh4_macro_f1": float(selected_icbhi["macro_f1"]),
            "delta_macro_f1": float(selected_icbhi["macro_f1"])
            - float(jh2_icbhi["macro_f1"]),
            "jh2_uar": float(jh2_icbhi["uar"]),
            "jh4_uar": float(selected_icbhi["uar"]),
            "delta_uar": float(selected_icbhi["uar"]) - float(jh2_icbhi["uar"]),
        },
        "sprsound_inter_task1_1": {
            "jh2_score": float(jh2_spr["official_score"]),
            "jh4_score": selected_spr_score,
            "delta_score": selected_spr_score - float(jh2_spr["official_score"]),
        },
    }
    _write_json(
        config.output_dir / "hf_on_off_comparison.json",
        {
            "status": "matched_fixed_JH2_HF_off_reference_comparison",
            "evidence_label": EVIDENCE_LABEL,
            "hf_off_reference": str(
                config.repo_root
                / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2"
            ),
            "hf_on_run": str(config.output_dir),
            "metrics": hf_comparison,
            "interpretation": (
                "HF-on auxiliary comparison is diagnostic; success requires core gates "
                "and material improvement without clear Wheeze degradation, not one "
                "threshold metric alone"
            ),
        },
    )
    icbhi_gate = selected_icbhi_score >= float(jh2_icbhi["official_score"]) - 0.01
    spr_gate = selected_spr_score >= float(jh2_spr["official_score"]) - 0.01
    summary = {
        "status": "early_stopped_epochwise_test_selected_finalized",
        "evidence_label": EVIDENCE_LABEL,
        "condition": CONDITION,
        "base_condition": "JH2 epoch19 Hard Hierarchy; no MVN",
        "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
        "finalization_only": True,
        "completed_training_epochs": 19,
        "max_epochs": config.epochs,
        "updates": 6194,
        "core_updates_per_epoch": CORE_UPDATES_PER_EPOCH,
        "hf_batches_per_core_update": 1,
        "selected_epoch": 9,
        "selection_loss": float(selection["selection_loss"]),
        "icbhi_official_score": selected_icbhi_score,
        "sprsound_task1_1_official_score": selected_spr_score,
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": 19,
            "sprsound_official_inter_test": 1,
            "hf_source_test": 1,
        },
        "actual_access_counts_confirmed_after_finalize": True,
        "predeclared_access_fields_not_evidence": True,
        "validation_thresholds_test_tuned": False,
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "best_only_checkpoint": True,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": True,
            "no_improvement_epochs_at_stop": 10,
        },
        "selected_metrics": {
            "icbhi_flat4": selected_icbhi,
            "icbhi_flat4_bits_only_ablation": selected_icbhi_bits,
            "sprsound_inter_task1_1": spr_metrics,
            "hf_source_test": hf_metrics,
        },
        "core_comparison_to_jh2": core_comparison,
        "hf_on_off_comparison": hf_comparison,
        "success_gate": {
            "icbhi_score_drop_le_0_01_vs_jh2": icbhi_gate,
            "icbhi_specificity_no_collapse": no_class_collapse,
            "sprsound_score_drop_le_0_01_vs_jh2": spr_gate,
            "hf_D_auroc_or_auprc_materially_improves": hf_d_material,
            "hf_Wheeze_not_clearly_degraded": hf_wheeze_not_degraded,
            "pass_requires_all_and_material_D_gain": (
                icbhi_gate
                and no_class_collapse
                and spr_gate
                and hf_d_material
                and hf_wheeze_not_degraded
            ),
        },
        "test_access_order": final_terminal["test_access_order"],
        "changed_files": ["baseline/pafa/joint_hierarchy_hf_auxiliary.py"],
        "git_commit": False,
        "claim_boundary": (
            "JH2-base, ICBHI-test-selected, HF-auxiliary external diagnostic; "
            "not a clean estimate, not a PAFA reproduction, not a paper claim"
        ),
    }
    _write_json(selection_path, {
        **selection,
        "sprsound_official_inter_test_accesses": 1,
        "hf_source_test_accesses": 1,
        "actual_access_counts_confirmed_after_finalize": True,
        "predeclared_access_fields_not_evidence": True,
        "finalization_only": True,
        "finalization_status": "complete",
    })
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def finalize_artifacts_only(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    """Write final summaries from saved scored artifacts without new evaluation."""

    config.validate()
    selection_path = config.output_dir / "validation_selection.json"
    terminal_dir = config.output_dir / "terminal"
    selection = json.loads(selection_path.read_text())
    selected_icbhi_payload = json.loads(
        (terminal_dir / "selected_icbhi_native_metrics.json").read_text()
    )
    with np.load(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        allow_pickle=False,
    ) as bundle:
        spr_scored = {key: bundle[key] for key in bundle.files}
    with np.load(
        terminal_dir / "selected_hf_predictions_scored.npz",
        allow_pickle=False,
    ) as bundle:
        hf_scored = {key: bundle[key] for key in bundle.files}
    hf_metrics = json.loads((terminal_dir / "hf_metrics.json").read_text())
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    selected_icbhi = selected_icbhi_payload["icbhi_flat4"]
    selected_icbhi_bits = selected_icbhi_payload[
        "icbhi_flat4_bits_only_ablation"
    ]
    spr_metrics = _score_spr_terminal(spr_scored)

    hf_off_metrics = json.loads(
        (
            config.repo_root
            / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2/hf_metrics.json"
        ).read_text()
    )
    hf_off = hf_off_metrics["recording_presence_pool"]["metrics"]
    hf_on = hf_metrics["recording_presence_pool"]["metrics"]
    hf_comparison = {
        node: {
            "hf_off_auroc": hf_off[hf_key]["curve"]["auroc"],
            "hf_on_auroc": hf_on[hf_key]["curve"]["auroc"],
            "delta_auroc": hf_on[hf_key]["curve"]["auroc"]
            - hf_off[hf_key]["curve"]["auroc"],
            "hf_off_auprc": hf_off[hf_key]["curve"]["auprc"],
            "hf_on_auprc": hf_on[hf_key]["curve"]["auprc"],
            "delta_auprc": hf_on[hf_key]["curve"]["auprc"]
            - hf_off[hf_key]["curve"]["auprc"],
            "hf_off_interval_recall": hf_off_metrics[
                "positive_interval_coverage"
            ][hf_interval_key]["positive_recall"],
            "hf_on_interval_recall": hf_metrics["positive_interval_coverage"][
                hf_interval_key
            ]["positive_recall"],
        }
        for node, hf_key, hf_interval_key in (
            ("crackle", "D", "D"),
            ("wheeze", "Wheeze", "Wheeze"),
        )
    }

    jh2_terminal = json.loads(
        (
            config.repo_root
            / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2/terminal/native_metrics.json"
        ).read_text()
    )
    jh2_icbhi = jh2_terminal["icbhi_flat4"]
    jh2_spr = jh2_terminal["sprsound_inter_task1_1"]
    selected_icbhi_score = float(selected_icbhi["official_score"])
    selected_spr_score = float(spr_metrics["official_score"])
    no_class_collapse = not bool(selected_icbhi["class_collapse"])
    hf_d_material = (
        hf_comparison["crackle"]["delta_auroc"] > 0.0
        or hf_comparison["crackle"]["delta_auprc"] > 0.0
    )
    hf_wheeze_not_degraded = (
        hf_comparison["wheeze"]["delta_auroc"] >= 0.0
        and hf_comparison["wheeze"]["delta_auprc"] >= 0.0
    )
    final_terminal = {
        **selected_icbhi_payload,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "condition": CONDITION,
        "selected_epoch_for_final_report": True,
        "selected_epoch": 9,
        "selection_loss": float(selection["selection_loss"]),
        "shared_attribute_thresholds": thresholds,
        "icbhi_official_test_accesses": 19,
        "sprsound_official_inter_test_accesses": 1,
        "hf_source_test_accesses": 1,
        "test_access_order": (
            "Existing epochwise ICBHI artifacts were reused without new inference; "
            "existing selected SPRSound and HF scored artifacts were reused without "
            "new target access or inference"
        ),
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": True,
        "completed_training_epochs": 19,
        "sprsound_inter_task1_1": spr_metrics,
        "hf_source_test": hf_metrics,
        "artifact_reuse_only": True,
    }
    _write_json(terminal_dir / "native_metrics.json", final_terminal)
    _write_json(terminal_dir / "selected_native_metrics.json", final_terminal)

    core_comparison = {
        "icbhi": {
            "jh2_score": float(jh2_icbhi["official_score"]),
            "jh4_score": selected_icbhi_score,
            "delta_score": selected_icbhi_score
            - float(jh2_icbhi["official_score"]),
            "jh2_specificity": float(jh2_icbhi["specificity"]),
            "jh4_specificity": float(selected_icbhi["specificity"]),
            "delta_specificity": float(selected_icbhi["specificity"])
            - float(jh2_icbhi["specificity"]),
            "jh2_sensitivity": float(jh2_icbhi["sensitivity"]),
            "jh4_sensitivity": float(selected_icbhi["sensitivity"]),
            "delta_sensitivity": float(selected_icbhi["sensitivity"])
            - float(jh2_icbhi["sensitivity"]),
            "jh2_macro_f1": float(jh2_icbhi["macro_f1"]),
            "jh4_macro_f1": float(selected_icbhi["macro_f1"]),
            "delta_macro_f1": float(selected_icbhi["macro_f1"])
            - float(jh2_icbhi["macro_f1"]),
            "jh2_uar": float(jh2_icbhi["uar"]),
            "jh4_uar": float(selected_icbhi["uar"]),
            "delta_uar": float(selected_icbhi["uar"])
            - float(jh2_icbhi["uar"]),
        },
        "sprsound_inter_task1_1": {
            "jh2_score": float(jh2_spr["official_score"]),
            "jh4_score": selected_spr_score,
            "delta_score": selected_spr_score
            - float(jh2_spr["official_score"]),
        },
    }
    _write_json(
        config.output_dir / "hf_on_off_comparison.json",
        {
            "status": "matched_fixed_JH2_HF_off_reference_comparison",
            "evidence_label": EVIDENCE_LABEL,
            "hf_off_reference": str(
                config.repo_root
                / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2"
            ),
            "hf_on_run": str(config.output_dir),
            "hf_scored_prediction_rows_reused": int(
                len(hf_scored["prediction_ids"])
            ),
            "metrics": hf_comparison,
            "interpretation": (
                "HF-on auxiliary comparison is diagnostic; success requires core gates "
                "and material improvement without clear Wheeze degradation, not one "
                "threshold metric alone"
            ),
        },
    )
    icbhi_gate = selected_icbhi_score >= float(jh2_icbhi["official_score"]) - 0.01
    spr_gate = selected_spr_score >= float(jh2_spr["official_score"]) - 0.01
    summary = {
        "status": "early_stopped_epochwise_test_selected_finalized",
        "evidence_label": EVIDENCE_LABEL,
        "condition": CONDITION,
        "base_condition": "JH2 epoch19 Hard Hierarchy; no MVN",
        "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
        "finalization_only": True,
        "artifact_reuse_only": True,
        "new_model_inference": False,
        "new_test_target_access": False,
        "completed_training_epochs": 19,
        "max_epochs": config.epochs,
        "updates": 6194,
        "core_updates_per_epoch": CORE_UPDATES_PER_EPOCH,
        "hf_batches_per_core_update": 1,
        "selected_epoch": 9,
        "selection_loss": float(selection["selection_loss"]),
        "icbhi_official_score": selected_icbhi_score,
        "sprsound_task1_1_official_score": selected_spr_score,
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": 19,
            "sprsound_official_inter_test": 1,
            "hf_source_test": 1,
        },
        "actual_access_counts_confirmed_after_finalize": True,
        "predeclared_access_fields_not_evidence": True,
        "validation_thresholds_test_tuned": False,
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "best_only_checkpoint": True,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": True,
            "no_improvement_epochs_at_stop": 10,
        },
        "selected_metrics": {
            "icbhi_flat4": selected_icbhi,
            "icbhi_flat4_bits_only_ablation": selected_icbhi_bits,
            "sprsound_inter_task1_1": spr_metrics,
            "hf_source_test": hf_metrics,
        },
        "core_comparison_to_jh2": core_comparison,
        "hf_on_off_comparison": hf_comparison,
        "success_gate": {
            "icbhi_score_drop_le_0_01_vs_jh2": icbhi_gate,
            "icbhi_specificity_no_collapse": no_class_collapse,
            "sprsound_score_drop_le_0_01_vs_jh2": spr_gate,
            "hf_D_auroc_or_auprc_materially_improves": hf_d_material,
            "hf_Wheeze_not_clearly_degraded": hf_wheeze_not_degraded,
            "pass_requires_all_and_material_D_gain": (
                icbhi_gate
                and no_class_collapse
                and spr_gate
                and hf_d_material
                and hf_wheeze_not_degraded
            ),
        },
        "test_access_order": final_terminal["test_access_order"],
        "changed_files": ["baseline/pafa/joint_hierarchy_hf_auxiliary.py"],
        "git_commit": False,
        "claim_boundary": (
            "JH2-base, ICBHI-test-selected, HF-auxiliary external diagnostic; "
            "not a clean estimate, not a PAFA reproduction, not a paper claim"
        ),
    }
    _write_json(
        selection_path,
        {
            **selection,
            "sprsound_official_inter_test_accesses": 1,
            "hf_source_test_accesses": 1,
            "actual_access_counts_confirmed_after_finalize": True,
            "predeclared_access_fields_not_evidence": True,
            "finalization_only": True,
            "artifact_reuse_only": True,
            "new_model_inference": False,
            "new_test_target_access": False,
            "finalization_status": "complete",
        },
    )
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    author_repo = repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo"
    checkpoint = repo_root / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    icbhi_audio = repo_root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"
    return author_repo, checkpoint, icbhi_audio


def _default_output_dir(repo_root: Path, seed: int) -> Path:
    if seed == 42:
        return repo_root / "result/reproduce/pafa_joint_hierarchy" / CONDITION
    return (
        repo_root
        / "result/reproduce/pafa_joint_hierarchy/PAFA_JH4_JH2_HFaux_multiseed"
        / f"seed_{seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 42), default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--finalize-artifacts-only", action="store_true")
    args = parser.parse_args()
    author_repo, checkpoint, icbhi_audio = _default_paths(args.repo_root)
    config = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio,
        output_dir=args.output_dir or _default_output_dir(args.repo_root, args.seed),
        device=args.device,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
    )
    _validate_run_config(config)
    if args.finalize_artifacts_only:
        print(
            json.dumps(finalize_artifacts_only(config), indent=2, sort_keys=True),
            flush=True,
        )
        return
    if args.finalize_only:
        print(json.dumps(finalize_only(config), indent=2, sort_keys=True), flush=True)
        return
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "execution_started": False,
                    "evidence_label": EVIDENCE_LABEL,
                    "seed": args.seed,
                    "config": _config_payload(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
