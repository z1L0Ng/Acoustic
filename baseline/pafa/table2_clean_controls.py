"""Validation-selected Table 2 controls for the ICBHI + SPRSound paper.

The default command only prints the frozen design.  ``--prepare-splits`` reads
official-train metadata and writes the canonical split manifest.  ``--run`` is
the later training entrypoint and must not be used without a separate user
execution instruction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from baseline.four_dataset_frozen_encoder.data import Sample, load_terminal_spr_test_targets
from baseline.multidataset_pipeline.beats_nal_protocol import (
    CORE_NODES,
    decode_icbhi_hierarchical_flat4,
    mapped_targets,
)
from baseline.multidataset_pipeline.posthoc_native_readout import (
    ICBHI_LABELS,
    native_metrics,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    PAFAJointHierarchyModel,
    _apply_author_ema,
    _build_components as _build_hierarchy_components,
    _icbhi_sample,
    _patient_indices,
    _prepare_waveforms,
    load_terminal_samples as _load_terminal_samples,
)
from baseline.pafa.joint_hierarchy_hf_auxiliary import (
    HF_BATCH_SIZE,
    HF_LAMBDA,
    HF_ROOT_RELATIVE,
    _hf_epoch_batches,
    _hf_loss,
    _load_hf_train_records,
    _load_hf_window_batch,
    _make_hf_windows,
)


VARIANTS = (
    "full_hf_off",
    "full_hf_on",
    "icbhi_only",
    "sprsound_only",
    "coarse_spr",
    "native_attributes",
)
MODEL_SEEDS = (0, 1, 42)
CORE_DATASETS = ("icbhi", "sprsound")
ACTIVE_DATASETS = {
    "full_hf_off": CORE_DATASETS,
    "full_hf_on": CORE_DATASETS,
    "icbhi_only": ("icbhi",),
    "sprsound_only": ("sprsound",),
    "coarse_spr": CORE_DATASETS,
    "native_attributes": CORE_DATASETS,
}
ICBHI_OUTER_SEED = 20260712
SPR_OUTER_SEED = 20260722
CALIBRATION_SPLIT_SEED = 42
CORE_UPDATES_PER_POINT = 326
MAX_VALIDATION_POINTS = 50
MAX_CORE_UPDATES = CORE_UPDATES_PER_POINT * MAX_VALIDATION_POINTS
EARLY_STOPPING_PATIENCE = 10
NODE_WEIGHT = 1.0 / 3.0
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-6
COSINE_ETA_MIN_RATIO = 1e-3
EMA_BETA = 0.5
SPLIT_MANIFEST_RELATIVE = Path("baseline/pafa/table2_clean_split_manifest.json")
RESULT_ROOT_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy/Table2_clean_controls")

SPR_LABELS = (
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve_spr_root(repo_root: Path) -> Path:
    base = repo_root / "dataset/raw/sprsound"
    direct = base.resolve()
    if (direct / "train2022_json").is_dir():
        return direct
    matches = list(base.glob("source_original/*/BioCAS2022"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one BioCAS2022 source under {base}")
    return matches[0].resolve()


def _read_icbhi_train_records(repo_root: Path) -> list[dict[str, str]]:
    path = repo_root / "dataset/processed/manifests/icbhi_2017_cycles.csv"
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["official_split"] == "train"]
    return sorted(
        (
            {
                "dataset": "icbhi",
                "sample_id": f"icbhi:{row['cycle_id']}",
                "group_id": str(row["patient_id"]),
                "native_label": str(row["native_four_class_label"]).lower(),
            }
            for row in rows
        ),
        key=lambda row: row["sample_id"],
    )


def _read_spr_train_records(repo_root: Path) -> list[dict[str, str]]:
    root = _resolve_spr_root(repo_root)
    records: list[dict[str, str]] = []
    for annotation_path in sorted((root / "train2022_json").glob("*.json")):
        recording_id = annotation_path.stem
        payload = json.loads(annotation_path.read_text())
        for event_index, event in enumerate(payload.get("event_annotation", [])):
            raw = str(event["type"])
            if raw not in SPR_LABELS:
                raise ValueError(f"unsupported SPR label {raw!r}")
            records.append(
                {
                    "dataset": "sprsound",
                    "sample_id": f"spr:train:{recording_id}:event_{event_index:03d}",
                    "group_id": recording_id.split("_", 1)[0],
                    "native_label": raw,
                }
            )
    return sorted(records, key=lambda row: row["sample_id"])


def _icbhi_inner_rebalance(
    records: Sequence[Mapping[str, str]], heldout: np.ndarray
) -> tuple[set[int], set[int], dict[str, object]]:
    by_group: dict[str, list[int]] = {}
    for index in heldout.tolist():
        by_group.setdefault(records[index]["group_id"], []).append(index)
    patient_ids = sorted(by_group)
    total_classes = Counter(records[index]["native_label"] for index in heldout)
    total_units = len(heldout)

    def group_sources(indices: Sequence[int], labels: set[str]) -> int:
        return len(
            {
                records[index]["group_id"]
                for index in indices
                if records[index]["native_label"] in labels
            }
        )

    best: tuple[object, ...] | None = None
    best_calibration: tuple[str, ...] | None = None
    all_groups = set(patient_ids)
    for count in range(4, len(patient_ids) - 3):
        for calibration_groups in combinations(patient_ids, count):
            calibration_group_set = set(calibration_groups)
            selection_groups = all_groups - calibration_group_set
            calibration = [
                index
                for group in calibration_groups
                for index in by_group[group]
            ]
            selection = [
                index
                for group in sorted(selection_groups)
                for index in by_group[group]
            ]
            selection_classes = Counter(records[index]["native_label"] for index in selection)
            if (
                selection_classes["normal"] < 100
                or sum(selection_classes[label] for label in ("crackle", "wheeze", "both")) < 100
                or any(selection_classes[label] == 0 for label in ICBHI_LABELS)
            ):
                continue
            valid_group_sources = True
            for side in (calibration, selection):
                if (
                    group_sources(side, {"normal"}) < 2
                    or group_sources(side, {"crackle", "both"}) < 2
                    or group_sources(side, {"wheeze", "both"}) < 2
                ):
                    valid_group_sources = False
                    break
            if not valid_group_sources:
                continue
            calibration_classes = Counter(
                records[index]["native_label"] for index in calibration
            )
            class_deviation = sum(
                Fraction(
                    (2 * calibration_classes[label] - total_classes[label]) ** 2,
                    total_classes[label] ** 2,
                )
                for label in ICBHI_LABELS
            )
            objective = (
                class_deviation,
                abs(2 * len(calibration) - total_units),
                abs(2 * len(calibration_groups) - len(patient_ids)),
                calibration_groups,
            )
            if best is None or objective < best:
                best = objective
                best_calibration = calibration_groups
    if best is None or best_calibration is None:
        raise RuntimeError("no feasible deterministic ICBHI calibration/selection split")
    calibration_groups = set(best_calibration)
    calibration_indices = {
        index for group in calibration_groups for index in by_group[group]
    }
    selection_indices = set(heldout.tolist()) - calibration_indices
    objective_payload = {
        "class_half_deviation": float(best[0]),
        "cycle_half_absolute_difference": int(best[1]),
        "patient_half_absolute_difference": int(best[2]),
        "tie_break": "lexicographically smallest calibration patient ID list",
        "feasibility": {
            "minimum_patients_each_side": 4,
            "minimum_selection_normal": 100,
            "minimum_selection_abnormal": 100,
            "selection_all_native_classes_nonzero": True,
            "each_side_minimum_patient_sources": {
                "normal": 2,
                "crackle_positive": 2,
                "wheeze_positive": 2,
            },
        },
    }
    return calibration_indices, selection_indices, objective_payload


def _assign_three_way_split(
    records: Sequence[Mapping[str, str]], *, outer_seed: int, dataset: str
) -> tuple[list[dict[str, str]], dict[str, object]]:
    labels = np.asarray([row["native_label"] for row in records])
    groups = np.asarray([row["group_id"] for row in records])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    subtrain_indices, validation_indices = next(
        outer.split(np.arange(len(records)), labels, groups)
    )
    heldout = np.asarray(validation_indices, dtype=np.int64)
    if dataset == "icbhi":
        calibration_indices, selection_indices, inner_details = _icbhi_inner_rebalance(
            records, heldout
        )
    else:
        inner = StratifiedGroupKFold(
            n_splits=2,
            shuffle=True,
            random_state=CALIBRATION_SPLIT_SEED,
        )
        calibration_local, selection_local = next(
            inner.split(heldout, labels[heldout], groups[heldout])
        )
        calibration_indices = set(
            heldout[np.asarray(calibration_local, dtype=np.int64)].tolist()
        )
        selection_indices = set(
            heldout[np.asarray(selection_local, dtype=np.int64)].tolist()
        )
        inner_details = {
            "method": "fixed StratifiedGroupKFold two-fold role assignment",
            "seed": CALIBRATION_SPLIT_SEED,
        }
    partition = np.full(len(records), "", dtype="<U11")
    partition[np.asarray(subtrain_indices, dtype=np.int64)] = "subtrain"
    partition[np.asarray(sorted(calibration_indices), dtype=np.int64)] = "calibration"
    partition[np.asarray(sorted(selection_indices), dtype=np.int64)] = "selection"
    return (
        [
            {**dict(row), "partition": str(partition[index])}
            for index, row in enumerate(records)
        ],
        inner_details,
    )


def _node_values(dataset: str, label: str) -> tuple[int | None, int | None, int | None]:
    if dataset == "icbhi":
        return {
            "normal": (0, 0, 0),
            "crackle": (1, 1, 0),
            "wheeze": (1, 0, 1),
            "both": (1, 1, 1),
        }[label]
    return {
        "Normal": (0, 0, 0),
        "Coarse Crackle": (1, 1, 0),
        "Fine Crackle": (1, 1, 0),
        "Wheeze": (1, 0, 1),
        "Wheeze+Crackle": (1, 1, 1),
        "Rhonchi": (1, None, None),
        "Stridor": (1, None, None),
    }[label]


def _partition_support(records: Sequence[Mapping[str, str]]) -> dict[str, object]:
    native = Counter(row["native_label"] for row in records)
    nodes: dict[str, dict[str, int]] = {}
    for index, node in enumerate(("a", "c", "w")):
        values = [
            _node_values(row["dataset"], row["native_label"])[index]
            for row in records
        ]
        eligible = [value for value in values if value is not None]
        nodes[node] = {
            "eligible": len(eligible),
            "positive": sum(value == 1 for value in eligible),
            "negative": sum(value == 0 for value in eligible),
        }
    return {
        "units": len(records),
        "groups": len({row["group_id"] for row in records}),
        "native_classes": dict(sorted(native.items())),
        "nodes": nodes,
    }


def build_split_manifest(repo_root: Path) -> dict[str, object]:
    icbhi, icbhi_inner = _assign_three_way_split(
        _read_icbhi_train_records(repo_root),
        outer_seed=ICBHI_OUTER_SEED,
        dataset="icbhi",
    )
    sprsound, spr_inner = _assign_three_way_split(
        _read_spr_train_records(repo_root),
        outer_seed=SPR_OUTER_SEED,
        dataset="sprsound",
    )
    datasets = {"icbhi": icbhi, "sprsound": sprsound}
    payload: dict[str, object] = {
        "status": "design_frozen_split_support_verified",
        "schema": "table2_clean_split_manifest_v1",
        "source_scope": "official training records only; no official test records",
        "split_policy": {
            "icbhi_outer": (
                "StratifiedGroupKFold(n_splits=5, fold=0, patient_id, "
                f"seed={ICBHI_OUTER_SEED})"
            ),
            "sprsound_outer": (
                "StratifiedGroupKFold(n_splits=5, fold=0, patient_id, "
                f"seed={SPR_OUTER_SEED})"
            ),
            "icbhi_calibration_selection": (
                "deterministic exhaustive assignment of the fixed outer-heldout patient pool; "
                "feasibility constraints then class/cycle/patient balance objectives and "
                "lexicographic calibration-patient tie-break"
            ),
            "sprsound_calibration_selection": (
                "outer validation groups split with StratifiedGroupKFold(n_splits=2, "
                f"fold=0 calibration, remaining fold selection, seed={CALIBRATION_SPLIT_SEED})"
            ),
            "model_seed_changes_split": False,
        },
        "inner_split_details": {
            "icbhi": icbhi_inner,
            "sprsound": spr_inner,
        },
        "datasets": {},
    }
    for dataset, records in datasets.items():
        groups = {
            partition: sorted(
                {row["group_id"] for row in records if row["partition"] == partition}
            )
            for partition in ("subtrain", "calibration", "selection")
        }
        support = {
            partition: _partition_support(
                [row for row in records if row["partition"] == partition]
            )
            for partition in ("subtrain", "calibration", "selection")
        }
        payload["datasets"][dataset] = {
            "records": records,
            "groups": groups,
            "support": support,
        }
    return payload


def prepare_split_manifest(repo_root: Path, output_path: Path) -> dict[str, object]:
    payload = build_split_manifest(repo_root)
    write_json(output_path, payload)
    return payload


def load_split_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def split_reference_payload(
    config: "Table2Config", manifest: Mapping[str, object]
) -> dict[str, object]:
    return {
        "manifest": str(config.split_manifest),
        "active_datasets": list(config.active_datasets),
        "datasets": {
            dataset: {
                "groups": manifest["datasets"][dataset]["groups"],
                "records": manifest["datasets"][dataset]["records"],
                "support": manifest["datasets"][dataset]["support"],
            }
            for dataset in config.active_datasets
        },
    }


@dataclass(frozen=True)
class Table2Config:
    repo_root: Path
    variant: str
    model_seed: int
    output_dir: Path
    split_manifest: Path
    device: str = "mps"
    cpu_threads: int = 4
    batch_size: int = 32
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_validation_points: int = MAX_VALIDATION_POINTS
    updates_per_point: int = CORE_UPDATES_PER_POINT
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    @property
    def active_datasets(self) -> tuple[str, ...]:
        return ACTIVE_DATASETS[self.variant]

    @property
    def max_core_updates(self) -> int:
        return self.max_validation_points * self.updates_per_point

    @property
    def author_repo(self) -> Path:
        return self.repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo"

    @property
    def checkpoint(self) -> Path:
        return self.repo_root / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"

    @property
    def icbhi_audio_dir(self) -> Path:
        return self.repo_root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown Table 2 variant: {self.variant}")
        if self.model_seed not in MODEL_SEEDS:
            raise ValueError(f"model seed must be one of {MODEL_SEEDS}")
        if self.batch_size != 32 or self.updates_per_point != 326:
            raise ValueError("Table 2 batch/cadence contract changed")
        if self.device != "mps" and not self.device.startswith("cuda"):
            raise ValueError("Table 2 controls support only MPS or CUDA FP32")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")

    def base_config(self) -> PAFAJointHierarchyConfig:
        return PAFAJointHierarchyConfig(
            repo_root=self.repo_root,
            author_repo=self.author_repo,
            checkpoint=self.checkpoint,
            icbhi_audio_dir=self.icbhi_audio_dir,
            output_dir=self.output_dir,
            device=self.device,
            seed=self.model_seed,
            cpu_threads=self.cpu_threads,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "repo_root": str(self.repo_root),
            "output_dir": str(self.output_dir),
            "split_manifest": str(self.split_manifest),
            "active_datasets": list(self.active_datasets),
            "max_core_updates": self.max_core_updates,
            "selection": "maximize mean native validation Score over active sources",
            "threshold_source": "calibration groups only",
            "test_policy": "selected checkpoint and thresholds first; each official test once",
            "node_weights": {"a": NODE_WEIGHT, "c": NODE_WEIGHT, "w": NODE_WEIGHT},
            "empty_node_weight_redistributed": False,
            "precision": "FP32",
            "hf_batch_size": HF_BATCH_SIZE if self.variant == "full_hf_on" else None,
            "hf_lambda": HF_LAMBDA if self.variant == "full_hf_on" else None,
        }


class NativeAttributesHead(nn.Module):
    """Native inference heads with shared C/W training auxiliaries."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_projector = nn.Linear(768, 256, bias=True)
        self.icbhi = nn.Linear(256, 4)
        self.sprsound = nn.Linear(256, 2)
        self.crackle = nn.Linear(256, 1)
        self.wheeze = nn.Linear(256, 1)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.shared_projector(embeddings)
        return {
            "icbhi_native": self.icbhi(projected),
            "sprsound_native": self.sprsound(projected),
            "crackle": self.crackle(projected).squeeze(-1),
            "wheeze": self.wheeze(projected).squeeze(-1),
        }


class PAFANativeAttributesModel(nn.Module):
    def __init__(self, beats: nn.Module, pafa_projector: nn.Module) -> None:
        super().__init__()
        self.beats = beats
        self.projector = pafa_projector
        self.head = NativeAttributesHead()

    def forward(
        self, waveform: torch.Tensor, *, training: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = self.beats(waveform, training=training)
        return self.head(features.mean(dim=1)), self.projector(features)


def _zero_loss(logit: torch.Tensor) -> torch.Tensor:
    return logit.sum() * 0.0


def fixed_hierarchy_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    level1_mask = eligible[:, 0]
    level1 = (
        F.cross_entropy(logits["level1"][level1_mask], targets[level1_mask, 0].long())
        if bool(level1_mask.any())
        else _zero_loss(logits["level1"])
    )
    attributes = []
    for index, node in enumerate(("crackle", "wheeze"), start=1):
        mask = eligible[:, index]
        attributes.append(
            F.binary_cross_entropy_with_logits(
                logits[node][mask], targets[mask, index]
            )
            if bool(mask.any())
            else _zero_loss(logits[node])
        )
    return NODE_WEIGHT * (level1 + attributes[0] + attributes[1])


def native_attributes_loss(
    logits: Mapping[str, torch.Tensor],
    dataset: str,
    native_target: torch.Tensor,
    attribute_targets: torch.Tensor,
    attribute_eligible: torch.Tensor,
) -> torch.Tensor:
    native = F.cross_entropy(logits[f"{dataset}_native"], native_target.long())
    attributes = []
    for index, node in enumerate(("crackle", "wheeze"), start=1):
        mask = attribute_eligible[:, index]
        attributes.append(
            F.binary_cross_entropy_with_logits(
                logits[node][mask], attribute_targets[mask, index]
            )
            if bool(mask.any())
            else _zero_loss(logits[node])
        )
    return NODE_WEIGHT * (native + attributes[0] + attributes[1])


def native_only_loss(
    logits: Mapping[str, torch.Tensor],
    dataset: str,
    native_target: torch.Tensor,
) -> torch.Tensor:
    """Keep the frozen one-third native CE coefficient without C/W terms."""

    return NODE_WEIGHT * F.cross_entropy(
        logits[f"{dataset}_native"], native_target.long()
    )


def apply_variant_eligibility(
    eligible: torch.Tensor, dataset: str, variant: str
) -> torch.Tensor:
    result = eligible.clone()
    if variant == "coarse_spr" and dataset == "sprsound":
        result[:, 1:] = False
    return result


def epoch_batches(
    sizes: Mapping[str, int],
    active_datasets: Sequence[str],
    *,
    batch_size: int,
    model_seed: int,
    validation_point: int,
) -> list[tuple[str, np.ndarray]]:
    per_dataset = CORE_UPDATES_PER_POINT // len(active_datasets)
    rng = np.random.default_rng(model_seed + validation_point)
    batches: list[tuple[str, np.ndarray]] = []
    for dataset in active_datasets:
        required = per_dataset * batch_size
        order: list[int] = []
        while len(order) < required:
            order.extend(rng.permutation(sizes[dataset]).tolist())
        values = np.asarray(order[:required], dtype=np.int64)
        batches.extend(
            (dataset, values[start : start + batch_size])
            for start in range(0, required, batch_size)
        )
    rng.shuffle(batches)
    return batches


def fit_attribute_thresholds(
    predictions: Mapping[str, np.ndarray],
    attribute_datasets: Sequence[str],
) -> tuple[dict[str, float], dict[str, object]]:
    from sklearn.metrics import f1_score

    thresholds: dict[str, float] = {}
    details: dict[str, object] = {}
    for column, node in enumerate(("crackle", "wheeze"), start=1):
        dataset_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        candidates = [0.0, 1.0]
        for dataset in attribute_datasets:
            mask = (
                (predictions["dataset_ids"] == dataset)
                & predictions["eligible"][:, column]
            )
            target = predictions["targets"][mask, column].astype(np.int64)
            probability = predictions["attribute_probabilities"][mask, column - 1]
            dataset_values[dataset] = (target, probability)
            candidates.extend(probability.tolist())
        best_value = -1.0
        best_threshold = 0.5
        best_by_dataset: dict[str, float] = {}
        for threshold in np.unique(np.asarray(candidates, dtype=np.float64)):
            by_dataset = {
                dataset: float(
                    f1_score(target, probability >= threshold, zero_division=0)
                )
                for dataset, (target, probability) in dataset_values.items()
            }
            value = float(np.mean(list(by_dataset.values())))
            if value > best_value or (value == best_value and threshold > best_threshold):
                best_value = value
                best_threshold = float(threshold)
                best_by_dataset = by_dataset
        thresholds[node] = best_threshold
        details[node] = {
            "threshold": best_threshold,
            "equal_attribute_source_mean_f1": best_value,
            "f1_by_dataset": best_by_dataset,
            "tie_break": "higher_threshold",
        }
    return thresholds, details


def active_attribute_datasets(config: Table2Config) -> tuple[str, ...]:
    if config.variant == "coarse_spr":
        return ("icbhi",)
    return config.active_datasets


def native_selection_scores(
    predictions: Mapping[str, np.ndarray],
    *,
    variant: str,
    active_datasets: Sequence[str],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    scores: dict[str, float] = {}
    metrics: dict[str, object] = {}
    if "icbhi" in active_datasets:
        mask = predictions["dataset_ids"] == "icbhi"
        target = np.asarray(
            [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"][mask]],
            dtype=np.int64,
        )
        if variant in {"native_attributes", "native_only"}:
            predicted = predictions["native_predictions"][mask].astype(np.int64)
        else:
            predicted = decode_icbhi_hierarchical_flat4(
                predictions["level1_predictions"][mask],
                predictions["attribute_probabilities"][mask],
                thresholds,
            )
        report = native_metrics(target, predicted, ICBHI_LABELS)
        scores["icbhi"] = float(report["average_score"])
        metrics["icbhi"] = report
    if "sprsound" in active_datasets:
        mask = predictions["dataset_ids"] == "sprsound"
        target = np.asarray(
            [int(str(value) != "Normal") for value in predictions["raw_ground_truth"][mask]],
            dtype=np.int64,
        )
        predicted = predictions["level1_predictions"][mask].astype(np.int64)
        report = native_metrics(target, predicted, ("normal", "abnormal"))
        report["official_score"] = (
            float(report["average_score"]) + float(report["harmonic_score"])
        ) / 2.0
        scores["sprsound"] = float(report["official_score"])
        metrics["sprsound"] = report
    return {
        "per_source_scores": scores,
        "selection_utility": float(np.mean([scores[d] for d in active_datasets])),
        "optimization": "maximize",
        "metrics": metrics,
    }


def decode_cw_only(
    attribute_probabilities: np.ndarray, thresholds: Mapping[str, float]
) -> np.ndarray:
    crackle = attribute_probabilities[:, 0] >= float(thresholds["crackle"])
    wheeze = attribute_probabilities[:, 1] >= float(thresholds["wheeze"])
    return crackle.astype(np.int64) + 2 * wheeze.astype(np.int64)


def spr_attribute_auroc(predictions: Mapping[str, np.ndarray]) -> dict[str, object]:
    dataset_mask = predictions["dataset_ids"] == "sprsound"
    values: dict[str, float] = {}
    supports: dict[str, dict[str, int]] = {}
    for column, node in enumerate(("crackle", "wheeze"), start=1):
        mask = dataset_mask & predictions["eligible"][:, column]
        target = predictions["targets"][mask, column].astype(np.int64)
        probability = predictions["attribute_probabilities"][mask, column - 1]
        values[node] = float(roc_auc_score(target, probability))
        supports[node] = {
            "eligible": int(mask.sum()),
            "positive": int(target.sum()),
            "negative": int((target == 0).sum()),
        }
    return {
        "per_attribute": values,
        "mean_cw_auroc": float(np.mean(list(values.values()))),
        "support": supports,
    }


def _split_lookup(manifest: Mapping[str, object]) -> dict[str, str]:
    return {
        str(row["sample_id"]): str(row["partition"])
        for dataset in CORE_DATASETS
        for row in manifest["datasets"][dataset]["records"]
    }


def _load_train_samples(
    config: Table2Config, manifest: Mapping[str, object]
) -> list[Sample]:
    partition = _split_lookup(manifest)
    base = config.base_config()
    samples: list[Sample] = []
    if "icbhi" in config.active_datasets:
        for row in read_official_cycles(base.icbhi_audio_dir, base.author_repo, ("train",)):
            sample_id = f"icbhi:{row.sample_id}"
            samples.append(
                _icbhi_sample(
                    row,
                    base.icbhi_audio_dir,
                    partition[sample_id],
                )
            )
    if "sprsound" in config.active_datasets:
        root = _resolve_spr_root(config.repo_root)
        for annotation_path in sorted((root / "train2022_json").glob("*.json")):
            recording_id = annotation_path.stem
            audio_path = root / "train2022_wav" / f"{recording_id}.wav"
            payload = json.loads(annotation_path.read_text())
            for event_index, event in enumerate(payload.get("event_annotation", [])):
                raw = str(event["type"])
                sample_id = f"spr:train:{recording_id}:event_{event_index:03d}"
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        dataset="sprsound",
                        partition=partition[sample_id],
                        group_id=recording_id.split("_", 1)[0],
                        audio_path=str(audio_path),
                        crop_start_s=float(event["start"]) / 1000.0,
                        crop_end_s=float(event["end"]) / 1000.0,
                        targets={
                            "spr_binary": int(raw != "Normal"),
                            "spr_seven": SPR_LABELS.index(raw),
                        },
                        metadata={
                            "raw_label": raw,
                            "event_id": sample_id.removeprefix("spr:"),
                            "recording_id": recording_id,
                            "patient_id": recording_id.split("_", 1)[0],
                            "event_index": event_index,
                        },
                    )
                )
    return sorted(samples, key=lambda row: row.sample_id)


def _partitioned(
    samples: Sequence[Sample], partition: str, datasets: Sequence[str]
) -> dict[str, tuple[Sample, ...]]:
    return {
        dataset: tuple(
            row for row in samples if row.dataset == dataset and row.partition == partition
        )
        for dataset in datasets
    }


def _native_target(rows: Sequence[Sample], dataset: str) -> torch.Tensor:
    key = "icbhi_flat4" if dataset == "icbhi" else "spr_binary"
    return torch.tensor([int(row.targets[key]) for row in rows], dtype=torch.long)


def _training_batch(
    rows: Sequence[Sample],
    dataset: str,
    waveform_store: Mapping[str, torch.Tensor],
    patient_index: Mapping[tuple[str, str], int],
    device: torch.device,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    waveform = torch.stack([waveform_store[row.sample_id] for row in rows]).to(device)
    targets, eligible, _ = mapped_targets(rows)
    eligible = apply_variant_eligibility(eligible, dataset, variant)
    patients = torch.tensor(
        [patient_index[(row.dataset, row.group_id)] for row in rows],
        dtype=torch.long,
    )
    return (
        waveform,
        targets.to(device),
        eligible.to(device),
        _native_target(rows, dataset).to(device),
        patients.to(device),
    )


def _build_native_components(
    config: Table2Config, device: torch.device
) -> tuple[PAFANativeAttributesModel, nn.Module]:
    sys.path.insert(0, str(config.author_repo))
    try:
        from method.pafa import PAFALoss, ProjectionHead
        from models.beats import BEATsTransferLearningModel
    finally:
        sys.path.pop(0)
    beats = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(config.checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    projector = ProjectionHead(
        input_dim=beats.final_feat_dim,
        hidden_dim=None,
        output_dim=768,
        attention=True,
        proj_type="end2end",
        norm_type="ln",
    )
    return PAFANativeAttributesModel(beats, projector).to(device), PAFALoss().to(device)


def build_components(
    config: Table2Config, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    if config.variant == "native_attributes":
        return _build_native_components(config, device)
    return _build_hierarchy_components(config.base_config(), device)


def _infer(
    model: nn.Module,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    waveform_store: Mapping[str, torch.Tensor],
    config: Table2Config,
    device: torch.device,
    *,
    include_targets: bool,
    variant_override: str | None = None,
) -> dict[str, np.ndarray]:
    variant = variant_override or config.variant
    include_attributes = variant != "native_only"
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "dataset_ids": [],
        "group_ids": [],
        "file_names": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "native_logits": [],
        "native_probabilities": [],
        "native_class_count": [],
        "native_predictions": [],
    }
    if include_attributes:
        fields.update({"attribute_logits": [], "attribute_probabilities": []})
    if include_targets:
        fields.update({"raw_ground_truth": [], "targets": [], "eligible": []})
    model.eval()
    with torch.no_grad():
        for dataset, samples in samples_by_dataset.items():
            for start in range(0, len(samples), config.batch_size):
                current = list(samples[start : start + config.batch_size])
                waveform = torch.stack(
                    [waveform_store[row.sample_id] for row in current]
                ).to(device)
                output, _ = model(waveform, training=False)
                if include_attributes:
                    attributes = torch.stack(
                        (output["crackle"], output["wheeze"]), dim=-1
                    ).float().cpu()
                if variant in {"native_attributes", "native_only"}:
                    native = output[f"{dataset}_native"].float().cpu()
                    native_probability = torch.softmax(native, dim=-1)
                    native_prediction = native.argmax(dim=-1)
                    padded_logits = torch.zeros((len(current), 4), dtype=torch.float32)
                    padded_probability = torch.zeros((len(current), 4), dtype=torch.float32)
                    padded_logits[:, : native.shape[1]] = native
                    padded_probability[:, : native.shape[1]] = native_probability
                    if dataset == "icbhi":
                        level1_probability = torch.stack(
                            (native_probability[:, 0], native_probability[:, 1:].sum(dim=1)),
                            dim=1,
                        )
                    else:
                        level1_probability = native_probability
                    level1_logits = torch.log(level1_probability.clamp_min(1e-12))
                    level1_prediction = level1_probability.argmax(dim=-1)
                    class_count = native.shape[1]
                else:
                    level1_logits = output["level1"].float().cpu()
                    level1_probability = torch.softmax(level1_logits, dim=-1)
                    level1_prediction = level1_logits.argmax(dim=-1)
                    padded_logits = torch.zeros((len(current), 4), dtype=torch.float32)
                    padded_probability = torch.zeros((len(current), 4), dtype=torch.float32)
                    native_prediction = torch.full((len(current),), -1, dtype=torch.long)
                    class_count = 0
                ids = np.asarray([row.sample_id for row in current])
                fields["prediction_ids"].append(ids)
                fields["sample_ids"].append(ids)
                fields["dataset_ids"].append(np.asarray([dataset] * len(current)))
                fields["group_ids"].append(np.asarray([row.group_id for row in current]))
                fields["file_names"].append(
                    np.asarray([Path(row.audio_path).name for row in current])
                )
                fields["level1_logits"].append(level1_logits.numpy())
                fields["level1_probabilities"].append(level1_probability.numpy())
                fields["level1_predictions"].append(level1_prediction.numpy())
                if include_attributes:
                    fields["attribute_logits"].append(attributes.numpy())
                    fields["attribute_probabilities"].append(
                        torch.sigmoid(attributes).numpy()
                    )
                fields["native_logits"].append(padded_logits.numpy())
                fields["native_probabilities"].append(padded_probability.numpy())
                fields["native_class_count"].append(
                    np.full(len(current), class_count, dtype=np.int64)
                )
                fields["native_predictions"].append(native_prediction.numpy())
                if include_targets:
                    targets, eligible, raw = mapped_targets(current)
                    if variant == "coarse_spr" and dataset == "sprsound":
                        eligible[:, 1:] = False
                    fields["raw_ground_truth"].append(np.asarray(raw))
                    fields["targets"].append(targets.numpy())
                    fields["eligible"].append(eligible.numpy())
    return {key: np.concatenate(values, axis=0) for key, values in fields.items()}


def _save_predictions(path: Path, predictions: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate(update: int) -> float:
    progress = update / MAX_CORE_UPDATES
    return LEARNING_RATE * (
        COSINE_ETA_MIN_RATIO
        + (1.0 - COSINE_ETA_MIN_RATIO) * (1.0 + math.cos(math.pi * progress)) / 2.0
    )


def _checkpoint_payload(
    model: nn.Module,
    config: Table2Config,
    *,
    validation_point: int,
    update: int,
    selection: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    return {
        "validation_point": validation_point,
        "update": update,
        "model": copy.deepcopy(model.state_dict()),
        "selection": dict(selection),
        "thresholds": dict(thresholds),
        "config": config.to_dict(),
    }


def run_training(config: Table2Config) -> dict[str, object]:
    """Run one approved training condition; caller owns execution authorization."""

    config.validate()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.model_seed)
    manifest = load_split_manifest(config.split_manifest)
    samples = _load_train_samples(config, manifest)
    partitions = {
        name: _partitioned(samples, name, config.active_datasets)
        for name in ("subtrain", "calibration", "selection")
    }
    write_json(config.output_dir / "config.json", config.to_dict())
    write_json(
        config.output_dir / "split_reference.json",
        split_reference_payload(config, manifest),
    )

    base = config.base_config()
    waveform_store = _prepare_waveforms(samples, base)
    device = torch.device(config.device)
    model, pafa_criterion = build_components(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    patient_index = _patient_indices(
        [row for dataset in partitions["subtrain"].values() for row in dataset]
    )

    hf_windows = None
    hf_source_root = config.repo_root / HF_ROOT_RELATIVE
    if config.variant == "full_hf_on":
        hf_records = _load_hf_train_records(config.repo_root)
        hf_windows = tuple(
            row for row in _make_hf_windows(hf_records) if row.record.partition == "subtrain"
        )

    best_utility = -math.inf
    best_point = 0
    best_thresholds: dict[str, float] = {}
    best_threshold_details: dict[str, object] = {}
    best_selection: dict[str, object] = {}
    no_improvement = 0
    global_update = 0
    train_log = config.output_dir / "train_log.jsonl"
    started = time.perf_counter()

    for validation_point in range(1, config.max_validation_points + 1):
        print(
            json.dumps(
                {
                    "event": "validation_point_start",
                    "variant": config.variant,
                    "model_seed": config.model_seed,
                    "validation_point": validation_point,
                    "next_update": global_update + 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model.train()
        schedule = epoch_batches(
            {dataset: len(rows) for dataset, rows in partitions["subtrain"].items()},
            config.active_datasets,
            batch_size=config.batch_size,
            model_seed=config.model_seed,
            validation_point=validation_point,
        )
        hf_batches = (
            _hf_epoch_batches(
                hf_windows,
                updates=len(schedule),
                epoch=validation_point,
                seed=config.model_seed,
            )
            if hf_windows is not None
            else [None] * len(schedule)
        )
        losses: dict[str, list[float]] = {
            dataset: [] for dataset in config.active_datasets
        }
        hf_losses: list[float] = []
        for (dataset, indices), hf_batch in zip(schedule, hf_batches):
            rows = [partitions["subtrain"][dataset][int(index)] for index in indices]
            waveform, targets, eligible, native_target, patients = _training_batch(
                rows,
                dataset,
                waveform_store,
                patient_index,
                device,
                config.variant,
            )
            before = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits, projected = model(waveform, training=True)
            if config.variant == "native_attributes":
                classification = native_attributes_loss(
                    logits, dataset, native_target, targets, eligible
                )
            else:
                classification = fixed_hierarchy_loss(logits, targets, eligible)
            pafa = pafa_criterion(
                projected,
                patients,
                lambda_pcsl=base.lambda_pcsl,
                lambda_gpal=base.lambda_gpal,
            )
            total = classification + pafa
            if hf_batch is not None:
                hf_waveform, hf_target, hf_eligible = _load_hf_window_batch(
                    hf_batch,
                    source_root=hf_source_root,
                    sample_rate=base.sample_rate,
                    device=device,
                )
                hf_logits, _ = model(hf_waveform, training=True)
                hf_value = _hf_loss(hf_logits, hf_target, hf_eligible)
                total = total + HF_LAMBDA * hf_value
                hf_losses.append(float(hf_value.detach().cpu()))
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(
                    f"non-finite loss at update {global_update + 1} dataset={dataset}"
                )
            total.backward()
            lr = _learning_rate(global_update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            _apply_author_ema(model, before, EMA_BETA)
            global_update += 1
            losses[dataset].append(float(total.detach().cpu()))
            if global_update == 1 or global_update % 32 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "variant": config.variant,
                            "model_seed": config.model_seed,
                            "validation_point": validation_point,
                            "update": global_update,
                            "dataset": dataset,
                            "loss": float(total.detach().cpu()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        calibration_predictions = _infer(
            model,
            partitions["calibration"],
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        selection_predictions = _infer(
            model,
            partitions["selection"],
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        _save_predictions(
            config.output_dir / "calibration" / f"point_{validation_point:03d}.npz",
            calibration_predictions,
        )
        _save_predictions(
            config.output_dir / "selection" / f"point_{validation_point:03d}.npz",
            selection_predictions,
        )
        thresholds, threshold_details = fit_attribute_thresholds(
            calibration_predictions, active_attribute_datasets(config)
        )
        selection = native_selection_scores(
            selection_predictions,
            variant=config.variant,
            active_datasets=config.active_datasets,
            thresholds=thresholds,
        )
        utility = float(selection["selection_utility"])
        improved = utility > best_utility
        no_improvement = 0 if improved else no_improvement + 1
        record = {
            "validation_point": validation_point,
            "update": global_update,
            "learning_rate": lr,
            "train_loss": {
                dataset: float(np.mean(values)) for dataset, values in losses.items()
            },
            "hf_loss": float(np.mean(hf_losses)) if hf_losses else None,
            "thresholds": thresholds,
            "threshold_details": threshold_details,
            "selection": selection,
            "strict_improvement": improved,
            "no_improvement_points": no_improvement,
            "outer_test_accessed": False,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if improved:
            best_utility = utility
            best_point = validation_point
            best_thresholds = thresholds
            best_threshold_details = threshold_details
            best_selection = selection
            torch.save(
                _checkpoint_payload(
                    model,
                    config,
                    validation_point=validation_point,
                    update=global_update,
                    selection=selection,
                    thresholds=thresholds,
                ),
                config.output_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "validation_point": validation_point,
                "update": global_update,
                "model": model.state_dict(),
                "config": config.to_dict(),
            },
            config.output_dir / "last_checkpoint.pt",
        )
        if no_improvement >= config.early_stopping_patience:
            break

    selection_payload = {
        "status": "validation_selected_no_terminal_yet",
        "variant": config.variant,
        "model_seed": config.model_seed,
        "selected_validation_point": best_point,
        "selected_update": best_point * config.updates_per_point,
        "selection": best_selection,
        "thresholds": best_thresholds,
        "threshold_details": best_threshold_details,
        "outer_test_accessed": False,
    }
    write_json(config.output_dir / "validation_selection.json", selection_payload)

    checkpoint = torch.load(config.output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    terminal_samples = _load_terminal_samples(base)
    terminal_by_dataset = _partitioned(terminal_samples, "test", CORE_DATASETS)
    terminal_waveforms = _prepare_waveforms(terminal_samples, base)
    label_free = _infer(
        model,
        terminal_by_dataset,
        terminal_waveforms,
        config,
        device,
        include_targets=False,
    )
    terminal_dir = config.output_dir / "terminal"
    _save_predictions(terminal_dir / "selected_predictions_label_free.npz", label_free)
    spr_targets = load_terminal_spr_test_targets(
        terminal_samples, include_checksums=False
    )
    scored = _attach_targets(label_free, terminal_samples, spr_targets)
    _save_predictions(terminal_dir / "selected_predictions_scored.npz", scored)
    terminal_selection = native_selection_scores(
        scored,
        variant=config.variant,
        active_datasets=CORE_DATASETS,
        thresholds=best_thresholds,
    )
    terminal_payload = {
        "status": "terminal_complete",
        "variant": config.variant,
        "model_seed": config.model_seed,
        "selected_validation_point": best_point,
        "thresholds": best_thresholds,
        "per_native_task": terminal_selection["metrics"],
        "spr_cw": spr_attribute_auroc(scored),
        "outer_test_accessed": True,
        "cross_dataset_pooled_score": None,
        "hf_external_evaluation": (
            "selected_checkpoint evaluator remains a separate implementation step"
            if config.variant == "full_hf_on"
            else None
        ),
    }
    write_json(terminal_dir / "native_metrics.json", terminal_payload)
    summary = {
        "status": "complete",
        "variant": config.variant,
        "model_seed": config.model_seed,
        "active_datasets": list(config.active_datasets),
        "completed_validation_points": validation_point,
        "updates": global_update,
        "selected_validation_point": best_point,
        "selection_utility": best_utility,
        "outer_test_accessed": True,
        "terminal": terminal_payload,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    write_json(config.output_dir / "run_summary.json", summary)
    return summary


def direct_cw_readout(full_run_dir: Path, output_path: Path) -> dict[str, object]:
    selection = json.loads((full_run_dir / "validation_selection.json").read_text())
    thresholds = selection["thresholds"]
    with np.load(
        full_run_dir / "terminal/selected_predictions_scored.npz",
        allow_pickle=False,
    ) as archive:
        predictions = {key: archive[key] for key in archive.files}
    mask = predictions["dataset_ids"] == "icbhi"
    target = np.asarray(
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"][mask]],
        dtype=np.int64,
    )
    full_prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"][mask],
        predictions["attribute_probabilities"][mask],
        thresholds,
    )
    cw_prediction = decode_cw_only(
        predictions["attribute_probabilities"][mask], thresholds
    )
    payload = {
        "status": "zero_training_fixed_scores_fixed_thresholds",
        "source_full_run": str(full_run_dir),
        "thresholds": thresholds,
        "full_hierarchy": native_metrics(target, full_prediction, ICBHI_LABELS),
        "direct_cw": native_metrics(target, cw_prediction, ICBHI_LABELS),
        "prediction_changes": int((full_prediction != cw_prediction).sum()),
        "rows": int(mask.sum()),
        "sprsound_columns": "unchanged; display dashes",
        "interpretation": "final ICBHI conversion rule only; representation unchanged",
    }
    write_json(output_path, payload)
    return payload


def _default_config(repo_root: Path, variant: str, seed: int) -> Table2Config:
    return Table2Config(
        repo_root=repo_root,
        variant=variant,
        model_seed=seed,
        output_dir=repo_root / RESULT_ROOT_RELATIVE / variant / f"seed_{seed}",
        split_manifest=repo_root / SPLIT_MANIFEST_RELATIVE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--prepare-splits", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--direct-cw-full-dir", type=Path)
    parser.add_argument("--direct-cw-output", type=Path)
    args = parser.parse_args()
    if args.prepare_splits:
        payload = prepare_split_manifest(
            args.repo_root, args.repo_root / SPLIT_MANIFEST_RELATIVE
        )
        print(json.dumps({"status": payload["status"], "path": str(args.repo_root / SPLIT_MANIFEST_RELATIVE)}, indent=2))
        return
    if args.direct_cw_full_dir:
        if args.direct_cw_output is None:
            raise ValueError("--direct-cw-output is required")
        print(json.dumps(direct_cw_readout(args.direct_cw_full_dir, args.direct_cw_output), indent=2))
        return
    if args.variant is None or args.seed is None:
        print(
            json.dumps(
                {
                    "status": "IMPLEMENTATION_PREPARED_WAITING_FOR_RUN_AUTHORIZATION",
                    "variants": VARIANTS,
                    "seeds": MODEL_SEEDS,
                    "split_manifest": str(args.repo_root / SPLIT_MANIFEST_RELATIVE),
                },
                indent=2,
            )
        )
        return
    config = _default_config(args.repo_root, args.variant, args.seed)
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "config": config.to_dict()}, indent=2))
        return
    print(json.dumps(run_training(config), indent=2))


if __name__ == "__main__":
    main()
