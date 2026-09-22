"""Fixed-checkpoint SPR four-class and HF D-presence diagnostics.

This module reuses selected LSAA and Native+C/W checkpoints without changing
their selection artifacts.  It only adds the explicitly approved SPR linear
head reference and writes all outputs below an independent result root.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except Exception:
    from baseline.pafa.native_attributes_external_posthoc import (
        _install_numpy_metric_compat,
    )

    _install_numpy_metric_compat()
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from torch import nn

from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.frozen_method_baselines.source_transfer_common import (
    load_hf_test_units,
    load_spr_inter_units,
    load_spr_train_units,
)
from baseline.pafa import jh2_hf_kauh_external as external
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _build_components as build_hierarchy_components,
    _prepare_waveforms,
)
from baseline.pafa.table2_benchmark_controls import BenchmarkConfig
from baseline.pafa.table2_clean_controls import build_components as build_native_components


SEEDS = (0, 1, 42)
SPR_CLASSES = ("Normal", "Crackle", "Wheeze", "Both")
RAW_TO_CLASS = {
    "Normal": 0,
    "Fine Crackle": 1,
    "Coarse Crackle": 1,
    "Wheeze": 2,
    "Wheeze+Crackle": 3,
}
EXCLUDED_SPR_LABELS = ("Rhonchi", "Stridor")
OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_TASK_ADAPTATION_20260921"
)
LSAA_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed"
)
NATIVE_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/native_attributes"
)
NATIVE_SEED42_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/native_attributes/seed_42"
)
NATIVE_HF_POSTHOC_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_attributes_external_posthoc"
)
INFERENCE_BATCH_SIZE = 16
HF_RECORDING_BATCH_SIZE = 4
CPU_THREADS = 2


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def map_spr_label(raw_label: str) -> int | None:
    if raw_label in RAW_TO_CLASS:
        return RAW_TO_CLASS[raw_label]
    if raw_label in EXCLUDED_SPR_LABELS:
        return None
    raise ValueError(f"unknown SPR label: {raw_label}")


def binary_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    true_positive = int(np.count_nonzero(target & prediction))
    false_positive = int(np.count_nonzero(~target & prediction))
    false_negative = int(np.count_nonzero(target & ~prediction))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def fit_spr_attribute_thresholds(
    raw_labels: Sequence[str], attribute_probabilities: np.ndarray
) -> tuple[dict[str, float], dict[str, object]]:
    """Fit C/W thresholds on compatible SPR validation events only."""

    labels = np.asarray(
        [
            -1 if (mapped := map_spr_label(str(value))) is None else int(mapped)
            for value in raw_labels
        ],
        dtype=np.int64,
    )
    compatible = labels >= 0
    thresholds: dict[str, float] = {}
    details: dict[str, object] = {}
    for column, name in enumerate(("crackle", "wheeze")):
        target = np.isin(labels[compatible], (1, 3)) if column == 0 else np.isin(labels[compatible], (2, 3))
        probabilities = np.asarray(attribute_probabilities[compatible, column], dtype=np.float64)
        candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
        best_threshold = 0.5
        best_value = -1.0
        for threshold in candidates:
            value = binary_f1(target, probabilities >= threshold)
            if value > best_value or (value == best_value and float(threshold) > best_threshold):
                best_value = value
                best_threshold = float(threshold)
        thresholds[name] = best_threshold
        details[name] = {
            "threshold": best_threshold,
            "validation_f1": best_value,
            "eligible_events": int(compatible.sum()),
            "positive_events": int(target.sum()),
            "negative_events": int((~target).sum()),
            "tie_break": "higher_threshold",
        }
    return thresholds, details


def decode_spr_hierarchy(
    level1_probabilities: np.ndarray,
    attribute_probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """Decode Normal/Abnormal plus C/W with the fixed tie policy."""

    level1 = np.asarray(level1_probabilities, dtype=np.float64)
    attributes = np.asarray(attribute_probabilities, dtype=np.float64)
    normal = level1[:, 0] >= level1[:, 1]
    crackle = attributes[:, 0] >= float(thresholds["crackle"])
    wheeze = attributes[:, 1] >= float(thresholds["wheeze"])
    output = np.zeros(len(level1), dtype=np.int64)
    abnormal = ~normal
    both = abnormal & crackle & wheeze
    only_crackle = abnormal & crackle & ~wheeze
    only_wheeze = abnormal & ~crackle & wheeze
    unresolved = abnormal & ~crackle & ~wheeze
    output[both] = 3
    output[only_crackle] = 1
    output[only_wheeze] = 2
    margin_crackle = attributes[:, 0] - float(thresholds["crackle"])
    margin_wheeze = attributes[:, 1] - float(thresholds["wheeze"])
    output[unresolved & (margin_wheeze > margin_crackle)] = 2
    output[unresolved & (margin_wheeze <= margin_crackle)] = 1
    return output


def _class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    groups: Sequence[str] | None = None,
    label_indices: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.arange(4, dtype=np.int64)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    support = confusion.sum(axis=1)
    per_class: dict[str, object] = {}
    recalls: dict[str, float | None] = {}
    f1_values = []
    selected_indices = tuple(int(value) for value in label_indices)
    for index in selected_indices:
        name = SPR_CLASSES[index]
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / int(support[index]) if support[index] else None
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
        f1_values.append(f1)
        recalls[name] = recall
        per_class[name] = {
            "events": int(support[index]),
            "precision": float(precision),
            "recall": recall,
            "f1": float(f1),
            "patient_support": (
                int(len({str(groups[i]) for i, value in enumerate(targets) if value == index}))
                if groups is not None
                else None
            ),
        }
    normal_recall = float(confusion[0, 0] / support[0]) if support[0] else None
    abnormal_indices = np.asarray([index for index in selected_indices if index != 0], dtype=np.int64)
    abnormal_support = int(support[abnormal_indices].sum())
    abnormal_correct = int(sum(confusion[index, index] for index in abnormal_indices))
    abnormal_recall = abnormal_correct / abnormal_support if abnormal_support else None
    score = (
        (normal_recall + abnormal_recall) / 2.0
        if normal_recall is not None and abnormal_recall is not None
        else None
    )
    return {
        "events": int(len(targets)),
        "confusion": confusion.tolist(),
        "support": {name: int(value) for name, value in zip(SPR_CLASSES, support)},
        "specificity_normal_recall": normal_recall,
        "sensitivity_abnormal_correct_recall": abnormal_recall,
        "score": score,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "label_order": [SPR_CLASSES[index] for index in selected_indices],
        "zero_division": 0,
    }


def non_both_metrics(targets: np.ndarray, predictions: np.ndarray, groups: Sequence[str]) -> dict[str, object]:
    mask = np.asarray(targets) != 3
    report = _class_metrics(
        np.asarray(targets)[mask],
        np.asarray(predictions)[mask],
        groups=np.asarray(groups)[mask],
        label_indices=(0, 1, 2),
    )
    report["scope"] = "true Normal/Crackle/Wheeze events only; predicted Both remains incorrect"
    report["excluded_true_both_events"] = int((~mask).sum())
    return report


def _source_dir(root: Path, condition: str, seed: int) -> Path:
    if condition == "lsaa":
        return root / LSAA_ROOT_RELATIVE / f"seed_{seed}"
    if seed == 42:
        return root / NATIVE_SEED42_RELATIVE
    return root / NATIVE_ROOT_RELATIVE / f"seed_{seed}"


def _selected_source(root: Path, condition: str, seed: int) -> dict[str, object]:
    run_dir = _source_dir(root, condition, seed)
    summary = json.loads((run_dir / "run_summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    if condition == "lsaa":
        selection = json.loads((run_dir / "validation_selection.json").read_text())
        thresholds = selection["shared_attribute_thresholds"]
    else:
        selection = json.loads((run_dir / "selection.json").read_text())
        thresholds = selection.get("thresholds", {})
    return {
        "condition": condition,
        "seed": seed,
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "backbone_checkpoint": Path(str(config["checkpoint"])),
        "selected_epoch": int(summary["selected_epoch"]),
        "original_thresholds": {key: float(value) for key, value in thresholds.items()},
        "status": str(summary.get("status", "")),
    }


def _split_reference(root: Path, seed: int) -> dict[str, object]:
    return json.loads((_source_dir(root, "native", seed) / "split_reference.json").read_text())


def _sample_from_train_unit(unit: object) -> Sample:
    raw = str(unit.metadata["raw_label"])
    return Sample(
        sample_id=str(unit.sample_id),
        dataset="sprsound",
        partition=str(unit.metadata["internal_partition"]),
        group_id=str(unit.group_id),
        audio_path=str(unit.audio_path),
        crop_start_s=float(unit.start_s),
        crop_end_s=float(unit.end_s),
        targets={"spr_binary": int(raw != "Normal"), "spr_raw_label": raw},
        metadata={"raw_label": raw, "recording_id": str(unit.audio_path).split("/")[-1].removesuffix(".wav")},
    )


def _sample_from_inter_unit(unit: object) -> Sample:
    raw = str(unit.metadata["raw_label"])
    return Sample(
        sample_id=str(unit.sample_id),
        dataset="sprsound",
        partition="test",
        group_id=str(unit.group_id),
        audio_path=str(unit.audio_path),
        crop_start_s=float(unit.start_s),
        crop_end_s=float(unit.end_s),
        targets={"spr_binary": int(raw != "Normal"), "spr_raw_label": raw},
        metadata={"raw_label": raw, "recording_id": str(unit.audio_path).split("/")[-1].removesuffix(".wav")},
    )


def _spr_samples(root: Path, seed: int) -> dict[str, list[Sample]]:
    reference = _split_reference(root, seed)
    partition_by_group = {
        str(row["group_id"]): str(row["partition"])
        for row in reference["datasets"]["sprsound"]["records"]
    }
    train_units = load_spr_train_units(root, partition_by_group)
    inter_units = load_spr_inter_units(root, include_targets=True)
    rows = {
        "subtrain": sorted(
            [_sample_from_train_unit(unit) for unit in train_units if unit.metadata["internal_partition"] == "subtrain"],
            key=lambda row: row.sample_id,
        ),
        "validation": sorted(
            [_sample_from_train_unit(unit) for unit in train_units if unit.metadata["internal_partition"] == "validation"],
            key=lambda row: row.sample_id,
        ),
        "test": sorted([_sample_from_inter_unit(unit) for unit in inter_units], key=lambda row: row.sample_id),
    }
    return rows


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validation_archive(root: Path, condition: str, seed: int, epoch: int) -> dict[str, np.ndarray]:
    return _archive(_source_dir(root, condition, seed) / "validation" / f"epoch_{epoch:03d}.npz")


def _test_archive(root: Path, condition: str, seed: int) -> dict[str, np.ndarray]:
    if condition == "lsaa":
        path = _source_dir(root, condition, seed) / "terminal/selected_sprsound_predictions_scored.npz"
    else:
        path = _source_dir(root, condition, seed) / "terminal/other_target_predictions_scored.npz"
    return _archive(path)


def _spr_rows(archive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = archive["dataset_ids"].astype(str) == "sprsound"
    rows = {key: value[mask] for key, value in archive.items() if value.shape[0] == mask.shape[0]}
    return rows


def _reorder(rows: Mapping[str, np.ndarray], sample_ids: Sequence[str]) -> dict[str, np.ndarray]:
    ids = rows["sample_ids"].astype(str)
    positions = {value: index for index, value in enumerate(ids)}
    if len(positions) != len(ids):
        raise RuntimeError("duplicate prediction sample ID")
    requested = [str(value) for value in sample_ids]
    if set(requested) != set(ids.tolist()):
        raise RuntimeError(f"prediction/sample ID mismatch: requested={len(requested)} archive={len(ids)}")
    order = np.asarray([positions[value] for value in requested], dtype=np.int64)
    return {key: value[order] for key, value in rows.items() if value.shape[0] == len(ids)}


def _build_lsaa_model(
    root: Path,
    source: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, PAFAJointHierarchyConfig]:
    config = PAFAJointHierarchyConfig(
        repo_root=root,
        author_repo=root / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        checkpoint=Path(source["backbone_checkpoint"]),
        icbhi_audio_dir=root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
        output_dir=output_dir,
        device=str(device),
        seed=int(source["seed"]),
        cpu_threads=CPU_THREADS,
    )
    model, _ = build_hierarchy_components(config, device)
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def _build_native_model(root: Path, source: Mapping[str, object], device: torch.device) -> tuple[nn.Module, object]:
    config = BenchmarkConfig(
        repo_root=root,
        variant="native_attributes",
        output_dir=Path(source["run_dir"]),
        device=str(device),
        cpu_threads=CPU_THREADS,
        seed=int(source["seed"]),
    )
    model, _ = build_native_components(config.core, device)
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config.core.base_config()


def _infer_native_spr(
    model: nn.Module,
    samples: Sequence[Sample],
    base_config: object,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "group_ids": [],
        "raw_ground_truth": [],
        "level1_probabilities": [],
        "attribute_probabilities": [],
        "icbhi_native_probabilities": [],
    }
    with torch.no_grad():
        for start in range(0, len(samples), INFERENCE_BATCH_SIZE):
            current = list(samples[start : start + INFERENCE_BATCH_SIZE])
            waveforms = _prepare_waveforms(current, base_config)
            batch = torch.stack([waveforms[row.sample_id] for row in current]).to(device)
            output, _ = model(batch, training=False)
            spr_probability = torch.softmax(output["sprsound_native"].float(), dim=-1).cpu().numpy()
            attribute_probability = torch.sigmoid(
                torch.stack((output["crackle"], output["wheeze"]), dim=-1).float()
            ).cpu().numpy()
            icbhi_probability = torch.softmax(output["icbhi_native"].float(), dim=-1).cpu().numpy()
            fields["sample_ids"].append(np.asarray([row.sample_id for row in current]))
            fields["group_ids"].append(np.asarray([row.group_id for row in current]))
            fields["raw_ground_truth"].append(np.asarray([row.metadata["raw_label"] for row in current]))
            fields["level1_probabilities"].append(spr_probability)
            fields["attribute_probabilities"].append(attribute_probability)
            fields["icbhi_native_probabilities"].append(icbhi_probability)
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _infer_native_hf(
    model: nn.Module,
    samples: Sequence[Sample],
    base_config: object,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "recording_ids": [],
        "window_indices": [],
        "source_start_s": [],
        "source_end_s": [],
        "file_names": [],
        "attribute_probabilities": [],
    }
    with torch.no_grad():
        for start in range(0, len(samples), HF_RECORDING_BATCH_SIZE):
            current = list(samples[start : start + HF_RECORDING_BATCH_SIZE])
            windows: list[torch.Tensor] = []
            recording_ids: list[str] = []
            window_indices: list[int] = []
            starts: list[float] = []
            ends: list[float] = []
            file_names: list[str] = []
            for sample in current:
                values, time_map = external._load_hf_windows(Path(sample.audio_path), base_config.sample_rate)
                for index, (window, (source_start, source_end)) in enumerate(zip(values, time_map)):
                    windows.append(window)
                    recording_ids.append(str(sample.metadata["recording_id"]))
                    window_indices.append(index)
                    starts.append(float(source_start))
                    ends.append(float(source_end))
                    file_names.append(Path(sample.audio_path).name)
            batch = torch.stack(windows).to(device)
            output, _ = model(batch, training=False)
            attribute_probability = torch.sigmoid(
                torch.stack((output["crackle"], output["wheeze"]), dim=-1).float()
            ).cpu().numpy()
            ids = np.asarray([
                f"hf:test:{recording}::window_{index:02d}"
                for recording, index in zip(recording_ids, window_indices)
            ])
            fields["prediction_ids"].append(ids)
            fields["sample_ids"].append(np.asarray([f"hf:test:{value}" for value in recording_ids]))
            fields["recording_ids"].append(np.asarray(recording_ids))
            fields["window_indices"].append(np.asarray(window_indices, dtype=np.int64))
            fields["source_start_s"].append(np.asarray(starts, dtype=np.float32))
            fields["source_end_s"].append(np.asarray(ends, dtype=np.float32))
            fields["file_names"].append(np.asarray(file_names))
            fields["attribute_probabilities"].append(attribute_probability)
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _extract_lsaa_h(
    model: nn.Module,
    samples_by_partition: Mapping[str, Sequence[Sample]],
    base_config: object,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "partition": [],
        "group_ids": [],
        "raw_ground_truth": [],
        "features": [],
    }
    with torch.no_grad():
        for partition in ("subtrain", "validation", "test"):
            samples = samples_by_partition[partition]
            for start in range(0, len(samples), INFERENCE_BATCH_SIZE):
                current = list(samples[start : start + INFERENCE_BATCH_SIZE])
                waveforms = _prepare_waveforms(current, base_config)
                batch = torch.stack([waveforms[row.sample_id] for row in current]).to(device)
                features = model.beats(batch, training=False)
                hidden = model.head.projector(features.mean(dim=1)).float().cpu().numpy()
                fields["sample_ids"].append(np.asarray([row.sample_id for row in current]))
                fields["partition"].append(np.asarray([partition] * len(current)))
                fields["group_ids"].append(np.asarray([row.group_id for row in current]))
                fields["raw_ground_truth"].append(np.asarray([row.metadata["raw_label"] for row in current]))
                fields["features"].append(hidden)
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _load_lsaa_hf(root: Path, seed: int) -> dict[str, np.ndarray]:
    path = root / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH" / f"seed_{seed}/hf_predictions_label_free.npz"
    return _archive(path)


def _load_native_hf(root: Path, seed: int) -> dict[str, np.ndarray]:
    path = root / NATIVE_HF_POSTHOC_RELATIVE / f"seed_{seed}/hf_window_predictions_label_free.npz"
    return _archive(path)


def _hf_targets(root: Path, samples: Sequence[object]) -> dict[str, int]:
    output: dict[str, int] = {}
    for sample in samples:
        tokens = []
        for line in Path(str(sample.metadata["label_path"])).read_text().splitlines():
            if line.strip():
                tokens.append(line.split()[0])
        if set(tokens) & {"D", "Wheeze", "Rhonchi", "Stridor"}:
            output[str(sample.metadata["recording_id"])] = int("D" in tokens)
    return output


def _hf_recording_scores(
    recording_ids: np.ndarray,
    scores: np.ndarray,
    targets: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    recording_ids = recording_ids.astype(str)
    rows = []
    for recording_id in sorted(set(recording_ids.tolist())):
        if recording_id not in targets:
            continue
        indices = np.flatnonzero(recording_ids == recording_id)
        rows.append((recording_id, int(targets[recording_id]), float(np.max(scores[indices]))))
    target = np.asarray([row[1] for row in rows], dtype=np.int64)
    value = np.asarray([row[2] for row in rows], dtype=np.float64)
    scored = {
        "recording_ids": np.asarray([row[0] for row in rows]),
        "d_presence_targets": target,
        "recording_max_scores": value.astype(np.float32),
    }
    metrics = {
        "support": int(len(target)),
        "d_positive": int(target.sum()),
        "other": int((target == 0).sum()),
        "auroc": float(roc_auc_score(target, value)),
        "auprc": float(average_precision_score(target, value)),
        "readout": "maximum over the existing three 5-second windows",
        "target": "D label present versus other eligible D/Wheeze/Rhonchi/Stridor recordings",
        "other_semantics": "no D record in this annotation protocol; not clinical normal and not guaranteed Crackle-negative",
    }
    return scored, metrics


def _linear_head_train(
    seed: int,
    feature_rows: Mapping[str, np.ndarray],
    samples: Mapping[str, Sequence[Sample]],
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    set_seed(seed)
    torch.set_num_threads(CPU_THREADS)
    ids = feature_rows["sample_ids"].astype(str)
    features = feature_rows["features"].astype(np.float32)
    feature_index = {value: index for index, value in enumerate(ids)}

    def subset(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = [row for row in samples[partition] if map_spr_label(str(row.metadata["raw_label"])) is not None]
        positions = np.asarray([feature_index[row.sample_id] for row in rows], dtype=np.int64)
        labels = np.asarray([map_spr_label(str(row.metadata["raw_label"])) for row in rows], dtype=np.int64)
        groups = np.asarray([row.group_id for row in rows])
        return features[positions], labels, groups

    train_x, train_y, _ = subset("subtrain")
    val_x, val_y, val_groups = subset("validation")
    test_x, test_y, test_groups = subset("test")
    counts = np.bincount(train_y, minlength=4).astype(np.float64)
    weights = len(train_y) / (4.0 * counts)
    model = nn.Linear(256, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    rng = np.random.default_rng(seed)
    history_path = output_dir / "linear_head_history.jsonl"
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    no_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), 128):
            batch_indices = order[start : start + 128]
            values = torch.from_numpy(train_x[batch_indices])
            labels = torch.from_numpy(train_y[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_predictions = model(torch.from_numpy(val_x)).argmax(dim=1).numpy()
        validation_report = _class_metrics(val_y, validation_predictions, groups=val_groups)
        score = float(validation_report["score"])
        improved = score > best_score
        no_improvement = 0 if improved else no_improvement + 1
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
                "validation_macro_f1": validation_report["macro_f1"],
                "improved": improved,
                "no_improvement_epochs": no_improvement,
            },
        )
        if no_improvement >= 10:
            break
    if best_state is None:
        raise RuntimeError("linear head did not produce a selected validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.from_numpy(test_x)).argmax(dim=1).numpy()
    test_report = _class_metrics(test_y, test_predictions, groups=test_groups)
    test_report["non_both"] = non_both_metrics(test_y, test_predictions, test_groups)
    torch.save(
        {
            "seed": seed,
            "state_dict": best_state,
            "selected_epoch": best_epoch,
            "validation_score": best_score,
            "class_order": list(SPR_CLASSES),
            "input": "LSAA selected-checkpoint h_i, 256-D shared classification projection",
        },
        output_dir / "linear_head.pt",
    )
    predictions = np.asarray(test_predictions, dtype=np.int64)
    summary = {
        "seed": seed,
        "selected_epoch": best_epoch,
        "completed_epochs": epoch,
        "validation_score": best_score,
        "training_seconds": time.perf_counter() - started,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "patience": 10,
        "min_delta": 0.0,
        "class_weights": [float(value) for value in weights],
        "subtrain_support": {name: int(value) for name, value in zip(SPR_CLASSES, counts)},
        "validation_support": {name: int(value) for name, value in zip(SPR_CLASSES, np.bincount(val_y, minlength=4))},
        "test_support": {name: int(value) for name, value in zip(SPR_CLASSES, np.bincount(test_y, minlength=4))},
        "test_metrics": test_report,
        "actions_not_performed": ["No augmentation", "No EMA", "No PAFA", "No hyperparameter search"],
    }
    write_json(output_dir / "linear_head_summary.json", summary)
    return predictions, summary


def _support_payload(samples: Mapping[str, Sequence[Sample]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for partition, rows in samples.items():
        raw = Counter(str(row.metadata["raw_label"]) for row in rows)
        mapped = Counter()
        patients: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            target = map_spr_label(str(row.metadata["raw_label"]))
            if target is None:
                continue
            mapped[SPR_CLASSES[target]] += 1
            patients[SPR_CLASSES[target]].add(row.group_id)
        payload[partition] = {
            "events_total": len(rows),
            "patients_total": len({row.group_id for row in rows}),
            "raw_classes": dict(sorted(raw.items())),
            "compatible_four_class_events": {name: int(mapped[name]) for name in SPR_CLASSES},
            "compatible_four_class_patients": {name: len(patients[name]) for name in SPR_CLASSES},
            "excluded_events": {name: int(raw[name]) for name in EXCLUDED_SPR_LABELS},
            "excluded_labels": list(EXCLUDED_SPR_LABELS),
        }
    return payload


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _stat(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def _run_seed(root: Path, seed: int, output_root: Path, device_name: str) -> dict[str, object]:
    seed_output = output_root / f"seed_{seed}"
    if seed_output.exists() and any(seed_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {seed_output}")
    seed_output.mkdir(parents=True, exist_ok=True)
    lsaa = _selected_source(root, "lsaa", seed)
    native = _selected_source(root, "native", seed)
    samples = _spr_samples(root, seed)
    support = _support_payload(samples)
    write_json(seed_output / "support.json", support)

    lsaa_validation = _reorder(
        _spr_rows(_validation_archive(root, "lsaa", seed, int(lsaa["selected_epoch"]))),
        [row.sample_id for row in samples["validation"]],
    )
    lsaa_test = _reorder(
        _spr_rows(_test_archive(root, "lsaa", seed)),
        [row.sample_id for row in samples["test"]],
    )
    native_validation_existing = _reorder(
        _spr_rows(_validation_archive(root, "native", seed, int(native["selected_epoch"]))),
        [row.sample_id for row in samples["validation"]],
    )
    native_test_existing = _reorder(
        _spr_rows(_test_archive(root, "native", seed)),
        [row.sample_id for row in samples["test"]],
    )
    lsaa_thresholds, lsaa_threshold_details = fit_spr_attribute_thresholds(
        lsaa_validation["raw_ground_truth"], lsaa_validation["attribute_probabilities"]
    )
    native_thresholds, native_threshold_details = fit_spr_attribute_thresholds(
        native_validation_existing["raw_ground_truth"], native_validation_existing["attribute_probabilities"]
    )

    device = torch.device(device_name)
    native_model, native_base = _build_native_model(root, native, device)
    native_test = _infer_native_spr(native_model, samples["test"], native_base, device)
    hf_samples = external._load_hf_test_samples(root)
    native_hf = _infer_native_hf(native_model, hf_samples, native_base, device)
    del native_model
    if device.type == "mps":
        torch.mps.empty_cache()

    lsaa_model, lsaa_base = _build_lsaa_model(root, lsaa, seed_output, device)
    lsaa_features = _extract_lsaa_h(lsaa_model, samples, lsaa_base, device)
    del lsaa_model
    if device.type == "mps":
        torch.mps.empty_cache()

    _save_npz(
        seed_output / "lsaa_h_features.npz",
        sample_ids=lsaa_features["sample_ids"],
        partition=lsaa_features["partition"],
        group_ids=lsaa_features["group_ids"],
        raw_ground_truth=lsaa_features["raw_ground_truth"],
        features=lsaa_features["features"],
    )
    linear_predictions, linear_summary = _linear_head_train(
        seed, lsaa_features, samples, seed_output
    )

    target_test = np.asarray([map_spr_label(str(row.metadata["raw_label"])) for row in samples["test"]], dtype=np.int64)
    group_test = np.asarray([row.group_id for row in samples["test"]])
    compatible_test = target_test >= 0
    target_test = target_test[compatible_test]
    group_test = group_test[compatible_test]
    lsaa_test_prob = lsaa_test["attribute_probabilities"][compatible_test]
    lsaa_test_level1 = lsaa_test["level1_probabilities"][compatible_test]
    native_test_prob = native_test_existing["attribute_probabilities"][compatible_test]
    native_test_level1 = native_test_existing["level1_probabilities"][compatible_test]
    native_icbhi_test_prob = native_test["icbhi_native_probabilities"][compatible_test]
    lsaa_prediction = decode_spr_hierarchy(lsaa_test_level1, lsaa_test_prob, lsaa_thresholds)
    native_prediction = decode_spr_hierarchy(native_test_level1, native_test_prob, native_thresholds)
    native_borrowed_prediction = native_icbhi_test_prob.argmax(axis=1).astype(np.int64)
    linear_prediction = linear_predictions
    spr_metrics = {
        "LSAA_attribute_readout": {
            **_class_metrics(target_test, lsaa_prediction, groups=group_test),
            "non_both": non_both_metrics(target_test, lsaa_prediction, group_test),
            "thresholds": lsaa_thresholds,
            "threshold_details": lsaa_threshold_details,
        },
        "Native+C/W_attribute_readout": {
            **_class_metrics(target_test, native_prediction, groups=group_test),
            "non_both": non_both_metrics(target_test, native_prediction, group_test),
            "thresholds": native_thresholds,
            "threshold_details": native_threshold_details,
        },
        "Native+C/W_borrowed_ICBHI_head": {
            **_class_metrics(target_test, native_borrowed_prediction, groups=group_test),
            "non_both": non_both_metrics(target_test, native_borrowed_prediction, group_test),
            "thresholds": None,
            "readout": "fixed Native+C/W checkpoint ICBHI four-class softmax argmax",
        },
        "LSAA_fixed_h_linear_head": linear_summary["test_metrics"],
    }
    _save_npz(
        seed_output / "spr_test_predictions.npz",
        sample_ids=np.asarray([row.sample_id for row in samples["test"]])[compatible_test],
        group_ids=group_test,
        raw_ground_truth=np.asarray([row.metadata["raw_label"] for row in samples["test"]])[compatible_test],
        true_class=target_test,
        lsaa_attribute_predictions=lsaa_prediction,
        native_attribute_predictions=native_prediction,
        native_borrowed_icbhi_predictions=native_borrowed_prediction,
        linear_head_predictions=linear_prediction,
        lsaa_attribute_probabilities=lsaa_test_prob,
        native_attribute_probabilities=native_test_prob,
        native_icbhi_probabilities=native_icbhi_test_prob,
    )
    _save_npz(
        seed_output / "spr_validation_predictions.npz",
        sample_ids=np.asarray([row.sample_id for row in samples["validation"]]),
        group_ids=np.asarray([row.group_id for row in samples["validation"]]),
        raw_ground_truth=np.asarray([row.metadata["raw_label"] for row in samples["validation"]]),
        lsaa_attribute_probabilities=lsaa_validation["attribute_probabilities"],
        native_level1_probabilities=native_validation_existing["level1_probabilities"],
        native_attribute_probabilities=native_validation_existing["attribute_probabilities"],
    )

    hf_targets = _hf_targets(root, hf_samples)
    lsaa_hf = _load_lsaa_hf(root, seed)
    native_hf_existing = _load_native_hf(root, seed)
    lsaa_hf_scored, lsaa_hf_metrics = _hf_recording_scores(
        lsaa_hf["recording_ids"], lsaa_hf["attribute_probabilities"][:, 0], hf_targets
    )
    native_hf_c_scored, native_hf_c_metrics = _hf_recording_scores(
        native_hf["recording_ids"], native_hf["attribute_probabilities"][:, 0], hf_targets
    )
    native_hf_icbhi_scored, native_hf_icbhi_metrics = _hf_recording_scores(
        native_hf_existing["recording_ids"],
        native_hf_existing["icbhi_native_probabilities"][:, 1]
        + native_hf_existing["icbhi_native_probabilities"][:, 3],
        hf_targets,
    )
    expected_recording_ids = np.asarray(sorted(hf_targets))
    expected_targets = np.asarray([hf_targets[value] for value in expected_recording_ids], dtype=np.int64)
    for scored in (lsaa_hf_scored, native_hf_c_scored, native_hf_icbhi_scored):
        if not np.array_equal(scored["recording_ids"], expected_recording_ids):
            raise RuntimeError("HF DAS readout recording IDs do not share the 957-recording pool")
        if not np.array_equal(scored["d_presence_targets"], expected_targets):
            raise RuntimeError("HF DAS readout target alignment changed across methods")
    _save_npz(
        seed_output / "hf_das_predictions.npz",
        recording_ids=lsaa_hf_scored["recording_ids"],
        d_presence_targets=lsaa_hf_scored["d_presence_targets"],
        lsaa_max_p_crackle=lsaa_hf_scored["recording_max_scores"],
        native_cw_max_p_crackle=native_hf_c_scored["recording_max_scores"],
        native_icbhi_max_p_crackle_or_both=native_hf_icbhi_scored["recording_max_scores"],
    )
    hf_metrics = {
        "LSAA_C": lsaa_hf_metrics,
        "Native+C/W_C": native_hf_c_metrics,
        "Native+C/W_ICBHI_C_or_Both": native_hf_icbhi_metrics,
        "shared_pool": {
            "recordings": len(hf_targets),
            "d_positive": sum(hf_targets.values()),
            "other": len(hf_targets) - sum(hf_targets.values()),
            "recording_ids_source": "same 957 eligible source-test recordings for all three readouts",
        },
    }
    _save_npz(
        seed_output / "hf_window_predictions.npz",
        lsaa_recording_ids=lsaa_hf["recording_ids"],
        lsaa_attribute_probabilities=lsaa_hf["attribute_probabilities"],
        native_recording_ids=native_hf["recording_ids"],
        native_window_indices=native_hf["window_indices"],
        native_attribute_probabilities=native_hf["attribute_probabilities"],
        native_icbhi_recording_ids=native_hf_existing["recording_ids"],
        native_icbhi_window_indices=native_hf_existing["window_indices"],
        native_icbhi_native_probabilities=native_hf_existing["icbhi_native_probabilities"],
    )
    metrics = {
        "status": "complete_fixed_checkpoint_spr4_hf_das_seed",
        "evidence_label": "fixed_selected_checkpoint_posthoc_diagnostic_with_supervised_spr_linear_reference",
        "seed": seed,
        "selected_checkpoints": {
            "LSAA": {key: str(value) if isinstance(value, Path) else value for key, value in lsaa.items()},
            "Native+C/W": {key: str(value) if isinstance(value, Path) else value for key, value in native.items()},
        },
        "spr_support": support,
        "spr_metrics": spr_metrics,
        "hf_das_metrics": hf_metrics,
        "linear_head": linear_summary,
        "threshold_policy": "SPRSound validation compatible Normal/Crackle/Wheeze/Both only; per-attribute max F1, >=, higher-threshold tie",
        "test_policy": "test used only after fixed checkpoint and validation thresholds/head selection",
        "actions_not_performed": [
            "No encoder update",
            "No existing head/projector update",
            "No checkpoint reselection",
            "No HF threshold/head/checkpoint selection",
            "No Native-only optional arm",
            "No training, inference, or feature extraction on official data outside this fixed protocol",
        ],
    }
    write_json(seed_output / "metrics.json", metrics)
    return metrics


def _aggregate(output_root: Path, seed_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    condition_names = (
        "LSAA_attribute_readout",
        "Native+C/W_attribute_readout",
        "Native+C/W_borrowed_ICBHI_head",
        "LSAA_fixed_h_linear_head",
    )
    spr_aggregate = {}
    for condition in condition_names:
        rows = [row["spr_metrics"][condition] for row in seed_metrics]
        spr_aggregate[condition] = {
            "score": _stat([float(row["score"]) for row in rows]),
            "macro_f1": _stat([float(row["macro_f1"]) for row in rows]),
            "non_both_score": _stat([float(row["non_both"]["score"]) for row in rows]),
            "non_both_macro_f1": _stat([float(row["non_both"]["macro_f1"]) for row in rows]),
        }
    hf_names = ("LSAA_C", "Native+C/W_C", "Native+C/W_ICBHI_C_or_Both")
    hf_aggregate = {}
    for name in hf_names:
        rows = [row["hf_das_metrics"][name] for row in seed_metrics]
        hf_aggregate[name] = {
            "auroc": _stat([float(row["auroc"]) for row in rows]),
            "auprc": _stat([float(row["auprc"]) for row in rows]),
        }
    summary = {
        "status": "complete_fixed_checkpoint_spr4_hf_das_three_seed",
        "evidence_label": "fixed_selected_checkpoint_posthoc_diagnostic_with_supervised_spr_linear_reference",
        "seeds": list(SEEDS),
        "spr_metrics": spr_aggregate,
        "hf_das_metrics": hf_aggregate,
        "per_seed": list(seed_metrics),
        "limitations": [
            "SPRSound test contains only one true Both event and one Both patient.",
            "The non-Both sensitivity analysis is supplementary and does not replace the four-class primary result.",
            "HF Other means no D annotation within the eligible D/Wheeze/Rhonchi/Stridor pool; it is not clinical normal and does not imply Crackle absence.",
            "The linear head is a supervised reference on fixed LSAA h_i, not an unseen-class or universal upper bound.",
        ],
        "actions_not_performed": [
            "No old run modification",
            "No paper, figure, Notion, Git, server, or DCASE/PC-MCL operation",
            "No Native-only optional arm",
        ],
    }
    write_json(output_root / "summary.json", summary)
    lines = [
        "# SPR four-class and HF DAS fixed-checkpoint diagnostic",
        "",
        "Evidence: fixed selected checkpoints; SPR thresholds are calibrated only on the same-seed SPR validation compatible four-class events. The new linear head is a supervised fixed-representation reference.",
        "",
        "## SPR compatible four-class event readout",
        "",
        "| Condition | Score mean±SD | Macro-F1 mean±SD | Non-Both Score mean±SD |",
        "|---|---:|---:|---:|",
    ]
    for condition in condition_names:
        row = spr_aggregate[condition]
        lines.append(
            f"| {condition} | {row['score']['mean']:.4f}±{row['score']['sample_sd']:.4f} | "
            f"{row['macro_f1']['mean']:.4f}±{row['macro_f1']['sample_sd']:.4f} | "
            f"{row['non_both_score']['mean']:.4f}±{row['non_both_score']['sample_sd']:.4f} |"
        )
    lines.extend([
        "",
        "Primary SPR score is ICBHI-style on the compatible four-class event pool: (Normal recall + all-abnormal correct-class recall)/2. Rhonchi and Stridor are excluded from this pool.",
        "",
        "## HF D-presence ranking",
        "",
        "| Readout | AUROC mean±SD | AUPRC mean±SD |",
        "|---|---:|---:|",
    ])
    for name in hf_names:
        row = hf_aggregate[name]
        lines.append(
            f"| {name} | {row['auroc']['mean']:.4f}±{row['auroc']['sample_sd']:.4f} | "
            f"{row['auprc']['mean']:.4f}±{row['auprc']['sample_sd']:.4f} |"
        )
    lines.extend([
        "",
        "HF uses the same 957 eligible recordings for every readout: D-positive=368 and Other=589. Other is not clinical normal.",
        "",
        "## Evidence ceiling",
        "",
        "This is a fixed-checkpoint posthoc diagnostic plus a supervised SPR linear-head reference. It does not change the original selected checkpoints, selection metrics, primary results, or manuscript.",
        "",
    ])
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(root: Path, output_root: Path, device_name: str) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "config.json",
        {
            "status": "running_fixed_checkpoint_spr4_hf_das",
            "seeds": list(SEEDS),
            "device": device_name,
            "precision": "FP32",
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "hf_recording_batch_size": HF_RECORDING_BATCH_SIZE,
            "cpu_threads": CPU_THREADS,
            "output_root": str(output_root),
            "spr_class_order": list(SPR_CLASSES),
            "excluded_spr_labels": list(EXCLUDED_SPR_LABELS),
            "linear_head": "LSAA fixed 256-D h_i -> Linear(256,4), CPU, Adam 1e-3, wd 1e-4, batch 128, max 50, patience 10",
            "hf_das": "same 957 eligible source-test recordings; D presence versus Other; no threshold",
            "actions_not_performed": ["No smoke", "No probe", "No profile", "No hash/checksum", "No paper/Notion/Git/server changes"],
        },
    )
    def json_source(source: Mapping[str, object]) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in source.items()
        }

    asset_audit = {}
    for seed in SEEDS:
        asset_audit[str(seed)] = {
            "LSAA": json_source(_selected_source(root, "lsaa", seed)),
            "Native+C/W": json_source(_selected_source(root, "native", seed)),
            "spr_support": _support_payload(_spr_samples(root, seed)),
        }
    write_json(output_root / "asset_audit.json", asset_audit)
    results = []
    for seed in SEEDS:
        print(json.dumps({"event": "seed_start", "seed": seed}, sort_keys=True), flush=True)
        results.append(_run_seed(root, seed, output_root, device_name))
        print(json.dumps({"event": "seed_complete", "seed": seed}, sort_keys=True), flush=True)
    return _aggregate(output_root, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root or args.repo_root / OUTPUT_ROOT_RELATIVE
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "output_root": str(output_root)}, indent=2))
        return
    torch.set_num_threads(CPU_THREADS)
    summary = run(args.repo_root.resolve(), output_root.resolve(), args.device)
    print(json.dumps({"status": summary["status"], "output_root": str(output_root)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
