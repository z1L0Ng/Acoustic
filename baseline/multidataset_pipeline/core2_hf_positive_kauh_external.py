"""Core-2 + HF positive auxiliary + KAUH external evaluation baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    load_terminal_spr_test_targets,
)

from .ast_window_encoder import load_local_ast_window_backend
from .beats_window_encoder import load_local_beats_window_backend
from .panns_window_encoder import load_local_panns_window_backend
from .opera_window_encoder import load_local_opera_ct_window_backend
from .m_unified import (
    PREDICTION_UNITS,
    SEED,
    _prepare_targets,
    _save_predictions,
    binary_metrics,
    build_feature_cache,
    ledger_rows_for_sample,
    load_feature_cache,
    load_canonical_samples,
    map_native_sample,
    set_determinism,
    write_json,
    write_jsonl,
)
from .window_encoder import FrozenWindowBackend


CORE_DATASETS = ("icbhi", "sprsound")
TRAIN_DATASETS = ("icbhi", "sprsound", "hf_lung")
CORE_NODES = ("level1", "crackle", "wheeze")
ROOT_RELATIVE = Path("result/reproduce/core2_hf_positive_kauh_external")
UPDATES_PER_EPOCH = 1_404


class Core2Head(nn.Module):
    """Shared Level1/Crackle/Wheeze head; no Other node exists."""

    def __init__(self, encoder_dim: int = 768) -> None:
        super().__init__()
        self.dimension_adapter = (
            nn.Identity()
            if encoder_dim == 768
            else nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, 768, bias=False),
            )
        )
        self.projector = nn.Linear(768, 256, bias=True)
        self.level1 = nn.Linear(256, 2)
        self.crackle = nn.Linear(256, 1)
        self.wheeze = nn.Linear(256, 1)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.projector(self.dimension_adapter(embeddings))
        return {
            "level1": self.level1(projected),
            "crackle": self.crackle(projected).squeeze(-1),
            "wheeze": self.wheeze(projected).squeeze(-1),
        }


def _core_targets(
    samples: Sequence[Sample],
    stores: Mapping[str, Mapping[str, np.ndarray]],
    *,
    spr_terminal_targets: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base = _prepare_targets(
        samples, stores, spr_terminal_targets=spr_terminal_targets
    )
    return {
        dataset: (targets[..., :3], eligible[..., :3])
        for dataset, (targets, eligible) in base.items()
    }


def core2_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    values = []
    level1_mask = eligible[..., 0]
    if bool(level1_mask.any()):
        values.append(
            F.cross_entropy(
                logits["level1"][level1_mask],
                targets[..., 0][level1_mask].long(),
            )
        )
    for index, node in enumerate(CORE_NODES[1:], start=1):
        mask = eligible[..., index]
        if bool(mask.any()):
            values.append(
                F.binary_cross_entropy_with_logits(
                    logits[node][mask], targets[..., index][mask]
                )
            )
    return torch.stack(values).mean()


def _prediction_id(sample_id: str, dataset: str, window: int) -> str:
    return (
        f"{sample_id}::window_{window:02d}"
        if dataset == "hf_lung"
        else sample_id
    )


def infer_core2(
    model: Core2Head,
    stores: Mapping[str, Mapping[str, np.ndarray]],
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    datasets: Sequence[str],
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "dataset_ids": [],
        "source_start_s": [],
        "source_end_s": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
        "targets": [],
        "eligible": [],
    }
    model.eval()
    with torch.no_grad():
        for dataset in datasets:
            store = stores[dataset]
            target_values, eligible_values = targets[dataset]
            for start in range(0, len(store["sample_ids"]), 8):
                stop = min(start + 8, len(store["sample_ids"]))
                output = model(
                    torch.from_numpy(store["embeddings"][start:stop]).to(device)
                )
                level1_logits = output["level1"].cpu().numpy()
                level1_prob = torch.softmax(output["level1"], dim=-1).cpu().numpy()
                attribute_logits = torch.stack(
                    [output["crackle"], output["wheeze"]], dim=-1
                ).cpu().numpy()
                attribute_prob = torch.sigmoid(
                    torch.from_numpy(attribute_logits)
                ).numpy()
                for local, row in enumerate(range(start, stop)):
                    count = int(store["window_mask"][row].sum())
                    sample_id = str(store["sample_ids"][row])
                    fields["prediction_ids"].append(
                        np.asarray(
                            [
                                _prediction_id(sample_id, dataset, window)
                                for window in range(count)
                            ]
                        )
                    )
                    fields["sample_ids"].append(np.asarray([sample_id] * count))
                    fields["dataset_ids"].append(np.asarray([dataset] * count))
                    fields["source_start_s"].append(store["time_map"][row, :count, 0])
                    fields["source_end_s"].append(store["time_map"][row, :count, 1])
                    fields["level1_logits"].append(level1_logits[local, :count])
                    fields["level1_probabilities"].append(level1_prob[local, :count])
                    fields["level1_predictions"].append(
                        level1_prob[local, :count].argmax(axis=-1)
                    )
                    fields["attribute_logits"].append(attribute_logits[local, :count])
                    fields["attribute_probabilities"].append(attribute_prob[local, :count])
                    fields["targets"].append(target_values[row, :count])
                    fields["eligible"].append(eligible_values[row, :count])
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def core_selection_losses(
    predictions: Mapping[str, np.ndarray]
) -> dict[str, object]:
    dataset_losses = {}
    node_losses = {}
    for dataset in CORE_DATASETS:
        dataset_mask = predictions["dataset_ids"] == dataset
        current = {}
        for index, node in enumerate(CORE_NODES):
            mask = dataset_mask & predictions["eligible"][:, index]
            target = predictions["targets"][mask, index]
            if node == "level1":
                logits = predictions["level1_logits"][mask]
                shifted = logits - logits.max(axis=1, keepdims=True)
                log_probability = shifted - np.log(
                    np.exp(shifted).sum(axis=1, keepdims=True)
                )
                current[node] = float(
                    -log_probability[
                        np.arange(len(target)), target.astype(int)
                    ].mean()
                )
            else:
                logits = predictions["attribute_logits"][mask, index - 1]
                current[node] = float(
                    (np.logaddexp(0.0, logits) - target * logits).mean()
                )
        node_losses[dataset] = current
        dataset_losses[dataset] = float(np.mean(list(current.values())))
    return {
        "dataset_node_losses": node_losses,
        "dataset_losses": dataset_losses,
        "selection_loss": float(
            np.mean([dataset_losses[d] for d in CORE_DATASETS])
        ),
        "selection_definition": "equal mean of ICBHI and SPRSound validation eligible-node loss",
    }


def _safe_threshold_free(
    target: np.ndarray, probability: np.ndarray
) -> tuple[float | None, float | None, bool]:
    if len(np.unique(target)) < 2:
        return None, None, False
    return (
        float(average_precision_score(target, probability)),
        float(roc_auc_score(target, probability)),
        True,
    )


def select_core_shared_thresholds(
    predictions: Mapping[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, object]]:
    thresholds = {}
    details = {}
    for node_index, node in enumerate(CORE_NODES[1:], start=1):
        probabilities = predictions["attribute_probabilities"][:, node_index - 1]
        eligible = predictions["eligible"][:, node_index]
        candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities[eligible])))
        best_objective = -1.0
        best_threshold = 0.5
        best_by_dataset = {}
        for threshold in candidates:
            by_dataset = {}
            for dataset in CORE_DATASETS:
                mask = (
                    (predictions["dataset_ids"] == dataset)
                    & eligible
                )
                by_dataset[dataset] = float(
                    f1_score(
                        predictions["targets"][mask, node_index].astype(int),
                        probabilities[mask] >= threshold,
                        zero_division=0,
                    )
                )
            objective = float(np.mean([by_dataset[d] for d in CORE_DATASETS]))
            if objective > best_objective or (
                objective == best_objective and threshold > best_threshold
            ):
                best_objective = objective
                best_threshold = float(threshold)
                best_by_dataset = by_dataset
        thresholds[node] = best_threshold
        details[node] = {
            "threshold": best_threshold,
            "equal_core_dataset_mean_f1": best_objective,
            "f1_by_dataset": best_by_dataset,
        }
    return thresholds, details


def secondary_dataset_thresholds(
    predictions: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    output = {}
    for dataset in CORE_DATASETS:
        output[dataset] = {}
        for node_index, node in enumerate(CORE_NODES[1:], start=1):
            mask = (
                (predictions["dataset_ids"] == dataset)
                & predictions["eligible"][:, node_index]
            )
            target = predictions["targets"][mask, node_index].astype(int)
            probability = predictions["attribute_probabilities"][mask, node_index - 1]
            best_f1 = -1.0
            best_threshold = 0.5
            for threshold in np.unique(np.concatenate(([0.0, 1.0], probability))):
                value = float(
                    f1_score(target, probability >= threshold, zero_division=0)
                )
                if value > best_f1 or (
                    value == best_f1 and threshold > best_threshold
                ):
                    best_f1 = value
                    best_threshold = float(threshold)
            output[dataset][node] = best_threshold
    return output


def _node_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    auprc, auroc, evaluable = _safe_threshold_free(target, probability)
    metrics = binary_metrics(target, prediction)
    metrics.update(
        {
            "auprc": auprc,
            "auroc": auroc,
            "evaluable_for_macro": evaluable,
            "positive_prediction_rate": float(prediction.mean()),
        }
    )
    return metrics


def score_core_primary(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    per_dataset = {}
    for dataset in CORE_DATASETS:
        dataset_mask = predictions["dataset_ids"] == dataset
        nodes = {}
        level1_mask = dataset_mask & predictions["eligible"][:, 0]
        nodes["level1"] = _node_metrics(
            predictions["targets"][level1_mask, 0].astype(int),
            predictions["level1_probabilities"][level1_mask, 1],
            predictions["level1_predictions"][level1_mask],
        )
        for index, node in enumerate(CORE_NODES[1:], start=1):
            mask = dataset_mask & predictions["eligible"][:, index]
            probability = predictions["attribute_probabilities"][mask, index - 1]
            nodes[node] = _node_metrics(
                predictions["targets"][mask, index].astype(int),
                probability,
                probability >= thresholds[node],
            )
            nodes[node]["core_shared_threshold"] = thresholds[node]
        evaluable_f1 = [
            float(value["f1"])
            for value in nodes.values()
            if value["evaluable_for_macro"]
        ]
        per_dataset[dataset] = {
            "prediction_unit": PREDICTION_UNITS[dataset],
            "nodes": nodes,
            "evaluable_node_macro_f1": float(np.mean(evaluable_f1)),
            "degenerate_nodes_excluded_from_macro": [
                node
                for node, value in nodes.items()
                if not value["evaluable_for_macro"]
            ],
        }
    dataset_scores = {
        dataset: per_dataset[dataset]["evaluable_node_macro_f1"]
        for dataset in CORE_DATASETS
    }
    return {
        "per_dataset": per_dataset,
        "core_dataset_macro_f1": float(np.mean(list(dataset_scores.values()))),
        "worst_core_dataset": min(dataset_scores, key=dataset_scores.get),
        "worst_core_dataset_f1": min(dataset_scores.values()),
        "threshold_policy": "one shared threshold per attribute selected by equal ICBHI/SPRSound validation F1",
    }


def write_dataset_role_contract(repo_root: Path) -> dict[str, object]:
    root = repo_root / ROOT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    samples = load_canonical_samples(repo_root)
    kauh_by_patient: dict[str, str] = {}
    for sample in samples:
        if sample.dataset == "kauh":
            kauh_by_patient[sample.group_id] = str(sample.metadata["raw_sound"])
    eligible_labels = {"N", "E W", "I E W", "C", "I C", "I C E W"}
    excluded = Counter(
        label for label in kauh_by_patient.values() if label not in eligible_labels
    )
    kauh_class = {
        "N": "Normal",
        "E W": "Wheeze",
        "I E W": "Wheeze",
        "C": "Crackle",
        "I C": "Crackle",
        "I C E W": "Both",
    }
    kauh_support = Counter(
        kauh_class[label]
        for label in kauh_by_patient.values()
        if label in kauh_class
    )
    contract = {
        "research_question": (
            "ICBHI+SPRSound full shared-label core with HF explicit-positive auxiliary, "
            "followed by KAUH patient-level external evaluation"
        ),
        "roles": {
            "icbhi": "shared_core; official split; cycle unit",
            "sprsound": "shared_core; BioCAS event; inter primary; intra excluded",
            "hf_lung": (
                "positive auxiliary only; Level1/gap/unannotated/missing/unobserved "
                "attributes masked"
            ),
            "kauh": (
                "external only after selection; B/D/E mean probability at P-number level"
            ),
        },
        "shared_nodes": ["Level1", "Crackle", "Wheeze"],
        "other_node": "deleted; absent from loss, threshold, macro, and decision",
        "sprsound_rhonchi_stridor": (
            "Level1=Abnormal only; Crackle/Wheeze unknown"
        ),
        "hf_rhonchi_stridor": "no eligible shared-node row",
        "selection_datasets": list(CORE_DATASETS),
        "threshold_datasets": list(CORE_DATASETS),
        "training_datasets": list(TRAIN_DATASETS),
        "checkpoint_selection": (
            "equal mean of ICBHI and SPRSound validation eligible-node loss; "
            "HF auxiliary logged only; KAUH absent"
        ),
        "threshold_selection": (
            "one shared threshold per attribute maximizing equal mean of ICBHI and "
            "SPRSound validation F1; no sample-count pooling; HF/KAUH absent"
        ),
        "ast_matched_control": (
            "AST_HF_on versus AST_HF_off; same architecture, seed, windows, "
            "70,200 updates, selection, and threshold policy"
        ),
        "encoder_release_gate": (
            "BEATs/PANNs/OPERA remain blocked until AST HF-on/HF-off management acceptance"
        ),
        "kauh_mapping": {
            "N": "Normal",
            "E W": "Wheeze",
            "I E W": "Wheeze",
            "C": "Crackle",
            "I C": "Crackle",
            "I C E W": "Both",
            "excluded": ["Crep", "Bronchial", "I C B"],
        },
        "kauh_patients": {
            "all": len(kauh_by_patient),
            "eligible": sum(label in eligible_labels for label in kauh_by_patient.values()),
            "excluded": sum(excluded.values()),
            "excluded_by_raw_label": dict(sorted(excluded.items())),
            "evaluable_support_by_unified_class": dict(sorted(kauh_support.items())),
        },
        "kauh_aggregation": {
            "primary": "mean B/D/E probabilities at P-number level",
            "secondary": "max probability sensitivity only",
            "consistency": "filter-view consistency; not device robustness",
        },
        "ground_truth_ledger": str(
            repo_root / "result/reproduce/unified/ground_truth_ledger"
        ),
    }
    write_json(root / "dataset_role_contract.json", contract)
    ledger_support = json.loads(
        (
            repo_root
            / "result/reproduce/unified/ground_truth_ledger/support_summary.json"
        ).read_text()
    )
    def node_support(dataset: str, node: str) -> dict[str, object]:
        return {
            partition: ledger_support["datasets"][dataset]["partitions"][partition][
                "nodes"
            ][node]
            for partition in ("subtrain", "validation", "test")
        }

    delta = {
        "raw_ledger_rebuilt": False,
        "role_change_only": True,
        "shared_nodes": list(CORE_NODES),
        "core_support": {
            dataset: {
                node: node_support(dataset, node) for node in CORE_NODES
            }
            for dataset in CORE_DATASETS
        },
        "hf_positive_auxiliary": {
            node: node_support("hf_lung", node)
            for node in ("crackle", "wheeze")
        },
        "kauh_external": contract["kauh_patients"],
        "prediction_units_unchanged": True,
    }
    write_json(root / "mapping_support_delta.json", delta)
    derived = root / "derived_support_view"
    derived.mkdir(parents=True, exist_ok=True)
    write_json(
        derived / "dataset_role_delta.json",
        {
            "raw_ledger_changed": False,
            "kauh_partition_role": "old 3-way split -> all-external",
            "kauh_evaluation_unit": "recording -> P-number patient",
            "hf_role": "explicit positive auxiliary only",
            "core_role": "ICBHI+SPRSound complete shared labels",
            "superseded_result_path": str(root / "AST/seed_42"),
        },
    )
    write_json(
        derived / "kauh_patient_support.json",
        {
            "all_patients": len(kauh_by_patient),
            "evaluable_patients": sum(kauh_support.values()),
            "support_by_unified_class": dict(sorted(kauh_support.items())),
            "excluded_patients": sum(excluded.values()),
            "excluded_by_raw_label": dict(sorted(excluded.items())),
            "primary_unit": "P-number after mean probability over B/D/E",
            "crackle_low_support_policy": "descriptive only",
        },
    )
    write_json(
        derived / "shared_node_support.json",
        {
            "core": delta["core_support"],
            "hf_positive_auxiliary": delta["hf_positive_auxiliary"],
            "other_node": "removed",
        },
    )
    return contract


def build_ast_gate_summary(repo_root: Path) -> dict[str, object]:
    root = repo_root / ROOT_RELATIVE
    on_path = root / "AST_HF_on/seed_42/run_summary.json"
    off_path = root / "AST_HF_off/seed_42/run_summary.json"
    required_fields = [
        "degenerate_or_not_evaluable_nodes_excluded_from_macro",
        "sprsound_crackle_auprc_precision_positive_prediction_rate",
        "icbhi_per_epoch_validation_curve",
        "hf_on_minus_hf_off_core_auprc_shared_threshold_f1_worst_dataset",
        "hf_positive_prediction_rate_on_vs_off",
    ]
    if not on_path.is_file() or not off_path.is_file():
        summary = {
            "status": "READY_PENDING_USER_START",
            "required_conditions": ["AST_HF_on", "AST_HF_off"],
            "required_gate_fields": required_fields,
            "downstream_encoder_release": "blocked pending AST management acceptance",
            "performance_values": None,
        }
        write_json(root / "ast_hf_on_off_gate_summary.json", summary)
        return summary
    on = json.loads(on_path.read_text())
    off = json.loads(off_path.read_text())
    def validation_curve(condition: str) -> list[dict[str, object]]:
        path = root / condition / "seed_42/train_log.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return [
            {
                "epoch": row["epoch"],
                "icbhi_validation_loss": row["core_validation"]["dataset_losses"]["icbhi"],
                "sprsound_validation_loss": row["core_validation"]["dataset_losses"]["sprsound"],
                "selection_loss": row["core_validation"]["selection_loss"],
            }
            for row in rows
        ]

    comparisons = {}
    for dataset in CORE_DATASETS:
        comparisons[dataset] = {}
        for node in CORE_NODES:
            on_node = on["core_metrics"]["per_dataset"][dataset]["nodes"][node]
            off_node = off["core_metrics"]["per_dataset"][dataset]["nodes"][node]
            comparisons[dataset][node] = {
                "delta_auprc": (
                    None
                    if on_node["auprc"] is None or off_node["auprc"] is None
                    else on_node["auprc"] - off_node["auprc"]
                ),
                "delta_shared_threshold_f1": on_node["f1"] - off_node["f1"],
                "delta_positive_prediction_rate": (
                    on_node["positive_prediction_rate"]
                    - off_node["positive_prediction_rate"]
                ),
            }
    summary = {
        "status": "READY_FOR_MANAGEMENT_ACCEPTANCE",
        "required_gate_fields": required_fields,
        "core_deltas": comparisons,
        "worst_core_dataset_f1_delta": (
            on["core_metrics"]["worst_core_dataset_f1"]
            - off["core_metrics"]["worst_core_dataset_f1"]
        ),
        "sprsound_crackle": {
            condition: value["core_metrics"]["per_dataset"]["sprsound"]["nodes"]["crackle"]
            for condition, value in (("hf_on", on), ("hf_off", off))
        },
        "hf_positive_prediction_rate": {
            condition: {
                node: report["hf_positive_only"]["nodes"][node][
                    "coverage_over_all_hf_windows"
                ]
                for node in CORE_NODES[1:]
            }
            for condition, report in (("hf_on", on), ("hf_off", off))
        },
        "icbhi_validation_curves": {
            "hf_on": validation_curve("AST_HF_on"),
            "hf_off": validation_curve("AST_HF_off"),
        },
        "kauh_filter_view_consistency": {
            condition: report["kauh_external"]["filter_view_consistency"]
            for condition, report in (("hf_on", on), ("hf_off", off))
        },
        "gate_decision_recommendation": (
            "NO_GO_HF_POSITIVE_AUXILIARY; retain AST_HF_off reference"
        ),
        "decision_basis": [
            "HF-on lowers core dataset-macro F1",
            "HF-on lowers SPRSound Crackle AUPRC and precision",
            "HF-on produces near-universal HF positive predictions",
            "small worst-core improvement does not offset the above failures",
        ],
        "downstream_encoder_release": "requires management decision",
    }
    write_json(root / "ast_hf_on_off_gate_summary.json", summary)
    return summary


def require_ast_management_release(repo_root: Path) -> None:
    path = repo_root / ROOT_RELATIVE / "ast_gate_management_decision.json"
    if not path.is_file():
        raise PermissionError(
            "BEATs/PANNs/OPERA blocked until AST HF-on/HF-off management acceptance"
        )
    decision = json.loads(path.read_text())
    if decision.get("status") != "accepted_release_encoders":
        raise PermissionError("AST management gate has not released later encoders")


def _stores(
    samples: Sequence[Sample],
    partition: str,
    datasets: Sequence[str],
    backend: FrozenWindowBackend | None,
    cache_dir: Path,
    *,
    device: torch.device,
    encoder_window_batch_size: int,
    cache_only: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    if cache_only:
        return {
            dataset: load_feature_cache(
                cache_dir,
                partition,
                dataset,
                [
                    sample.sample_id
                    for sample in sorted(
                        (
                            sample
                            for sample in samples
                            if sample.partition == partition
                            and sample.dataset == dataset
                        ),
                        key=lambda sample: sample.sample_id,
                    )
                ],
            )
            for dataset in datasets
        }
    if backend is None:
        raise RuntimeError("feature extraction backend is required")
    return {
        dataset: build_feature_cache(
            samples,
            partition,
            dataset,
            backend,
            cache_dir,
            device=device,
            native_batch_size=8,
            encoder_window_batch_size=encoder_window_batch_size,
        )
        for dataset in datasets
    }


def _epoch_batches(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    epoch: int,
    *,
    hf_auxiliary_enabled: bool,
) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(SEED + epoch)
    batches = []
    datasets = TRAIN_DATASETS if hf_auxiliary_enabled else CORE_DATASETS
    for dataset in datasets:
        active = np.flatnonzero(targets[dataset][1].any(axis=(1, 2)))
        order = rng.permutation(active).tolist()
        batches.extend(
            (dataset, order[start : start + 8])
            for start in range(0, len(order), 8)
        )
    extra = UPDATES_PER_EPOCH - len(batches)
    if extra < 0:
        raise RuntimeError("natural source-proportional batches exceed matched budget")
    if extra:
        available_batches = list(batches)
        sampled = rng.integers(0, len(available_batches), size=extra)
        batches.extend(
            (
                available_batches[index][0],
                list(available_batches[index][1]),
            )
            for index in sampled
        )
    if len(batches) != UPDATES_PER_EPOCH:
        raise RuntimeError("HF-on/HF-off matched update budget changed")
    rng.shuffle(batches)
    return batches


def _filter_predictions(
    predictions: Mapping[str, np.ndarray], datasets: Sequence[str]
) -> dict[str, np.ndarray]:
    mask = np.isin(predictions["dataset_ids"], list(datasets))
    return {key: value[mask] for key, value in predictions.items()}


def _concat_predictions(
    values: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([current[key] for current in values], axis=0)
        for key in values[0]
    }


def train_core2_head(
    stores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
    *,
    device: torch.device,
    hf_auxiliary_enabled: bool,
) -> tuple[Core2Head, dict[str, object]]:
    set_determinism()
    encoder_dims = {
        int(store["embeddings"].shape[-1])
        for partition in stores.values()
        for store in partition.values()
    }
    if len(encoder_dims) != 1:
        raise RuntimeError("core lanes do not share one encoder dimension")
    encoder_dim = encoder_dims.pop()
    model = Core2Head(encoder_dim).to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    active_counts = {
        dataset: int(eligible.any(axis=(1, 2)).sum())
        for dataset, (_, eligible) in targets["subtrain"].items()
    }
    total_updates = 50 * UPDATES_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates
    )
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    best_loss = math.inf
    best_epoch = 0
    best_update = 0
    update = 0
    started = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        core_losses = []
        hf_losses = []
        for dataset, indices in _epoch_batches(
            targets["subtrain"],
            epoch,
            hf_auxiliary_enabled=hf_auxiliary_enabled,
        ):
            embeddings = torch.from_numpy(
                stores["subtrain"][dataset]["embeddings"][indices]
            ).to(device)
            target_values, eligible_values = targets["subtrain"][dataset]
            target = torch.from_numpy(target_values[indices]).to(device)
            eligible = torch.from_numpy(eligible_values[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = core2_loss(model(embeddings), target, eligible)
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            if dataset == "hf_lung":
                hf_losses.append(float(loss.detach()))
            else:
                core_losses.append(float(loss.detach()))
        validation = infer_core2(
            model,
            stores["validation"],
            targets["validation"],
            device=device,
            datasets=tuple(stores["validation"]),
        )
        _save_predictions(
            output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            validation,
        )
        core_validation = _filter_predictions(validation, CORE_DATASETS)
        selection = core_selection_losses(core_validation)
        record = {
            "epoch": epoch,
            "update": update,
            "core_train_loss": float(np.mean(core_losses)),
            "hf_positive_auxiliary_loss": (
                float(np.mean(hf_losses)) if hf_losses else None
            ),
            "hf_auxiliary_enabled": hf_auxiliary_enabled,
            "core_validation": selection,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.perf_counter() - started) / 60,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if float(selection["selection_loss"]) < best_loss:
            best_loss = float(selection["selection_loss"])
            best_epoch = epoch
            best_update = update
            torch.save(
                {
                    "epoch": epoch,
                    "update": update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_loss": best_loss,
                    "encoder_dim": encoder_dim,
                },
                output_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "encoder_dim": encoder_dim,
            },
            output_dir / "last_checkpoint.pt",
        )
        print(
            f"epoch {epoch:02d}/50 update={update} "
            f"core={record['core_train_loss']:.6f} "
            f"hf_aux={record['hf_positive_auxiliary_loss']} "
            f"best_val={best_loss:.6f}",
            flush=True,
        )
    selected = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(selected["model"])
    return model, {
        "selected_epoch": best_epoch,
        "selected_update": best_update,
        "selection_loss": best_loss,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": total_updates,
        "active_units": active_counts,
        "encoder_dim": encoder_dim,
        "hf_auxiliary_enabled": hf_auxiliary_enabled,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    }


def _attach_ground_truth(
    predictions: dict[str, np.ndarray],
    samples: Sequence[Sample],
    spr_targets: Mapping[str, Mapping[str, int]] | None,
) -> None:
    by_prediction = {}
    for sample in samples:
        if sample.dataset not in TRAIN_DATASETS:
            continue
        for row in ledger_rows_for_sample(
            sample, spr_terminal_targets=spr_targets
        ):
            by_prediction[str(row["prediction_id"])] = row
    raw = []
    unified = []
    for prediction_id in predictions["prediction_ids"].tolist():
        row = by_prediction[str(prediction_id)]
        raw.append(json.dumps(row["raw_label"], sort_keys=True))
        unified.append(
            json.dumps(
                {
                    node: {
                        "target": row[f"{node}_target"],
                        "eligible": row[f"{node}_eligible"],
                    }
                    for node in CORE_NODES
                },
                sort_keys=True,
            )
        )
    predictions["raw_ground_truth"] = np.asarray(raw)
    predictions["unified_ground_truth"] = np.asarray(unified)


def hf_positive_report(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    rows = predictions["dataset_ids"] == "hf_lung"
    total_windows = int(rows.sum())
    nodes = {}
    for index, node in enumerate(CORE_NODES[1:], start=1):
        eligible = rows & predictions["eligible"][:, index]
        positive_support = int(eligible.sum())
        probability = predictions["attribute_probabilities"][:, index - 1]
        predicted = probability >= thresholds[node]
        true_positive = int((eligible & predicted).sum())
        nodes[node] = {
            "positive_support": positive_support,
            "positive_recall": (
                true_positive / positive_support if positive_support else None
            ),
            "predicted_positive_windows": int((rows & predicted).sum()),
            "coverage_over_all_hf_windows": (
                float((rows & predicted).sum() / total_windows)
                if total_windows
                else None
            ),
        }
    return {
        "prediction_unit": "2-second source-time window center",
        "total_windows": total_windows,
        "level1": "masked",
        "negative_semantics": "none; gaps and unobserved attributes remain masked",
        "rhonchi_stridor_policy": "no eligible shared-node row",
        "forbidden_metrics": ["detector_f1", "specificity", "auroc"],
        "nodes": nodes,
    }


def kauh_patient_external(
    predictions: Mapping[str, np.ndarray],
    samples: Sequence[Sample],
    thresholds: Mapping[str, float],
    output_dir: Path,
) -> dict[str, object]:
    by_sample = {
        sample.sample_id: sample for sample in samples if sample.dataset == "kauh"
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(predictions["sample_ids"].tolist()):
        grouped[by_sample[str(sample_id)].group_id].append(index)
    patient_rows = []
    metric_nodes = (*CORE_NODES, "both")
    metric_targets: dict[str, list[int]] = {node: [] for node in metric_nodes}
    mean_predictions: dict[str, list[int]] = {node: [] for node in metric_nodes}
    max_predictions: dict[str, list[int]] = {node: [] for node in metric_nodes}
    consistency: dict[str, list[bool]] = {node: [] for node in metric_nodes}
    excluded = Counter()
    for patient_id in sorted(grouped, key=lambda value: int(value.removeprefix("P"))):
        indices = grouped[patient_id]
        source_samples = [by_sample[str(predictions["sample_ids"][i])] for i in indices]
        raw_label = str(source_samples[0].metadata["raw_sound"])
        mapped = map_native_sample(source_samples[0])
        filter_level1 = predictions["level1_probabilities"][indices]
        filter_attribute = predictions["attribute_probabilities"][indices]
        mean_level1_logits = predictions["level1_logits"][indices].mean(axis=0)
        mean_attribute_logits = predictions["attribute_logits"][indices].mean(axis=0)
        level1_probability = filter_level1.mean(axis=0)
        attribute_probability = filter_attribute.mean(axis=0)
        mean_level1_prediction = int(level1_probability.argmax())
        mean_attribute_prediction = {
            node: int(attribute_probability[index] >= thresholds[node])
            for index, node in enumerate(CORE_NODES[1:])
        }
        max_abnormal_probability = float(filter_level1[:, 1].max())
        max_level1_prediction = int(max_abnormal_probability >= 0.5)
        max_attribute_probability = filter_attribute.max(axis=0)
        max_attribute_prediction = {
            node: int(max_attribute_probability[index] >= thresholds[node])
            for index, node in enumerate(CORE_NODES[1:])
        }
        filter_predictions = {
            "level1": filter_level1.argmax(axis=1),
            "crackle": filter_attribute[:, 0] >= thresholds["crackle"],
            "wheeze": filter_attribute[:, 1] >= thresholds["wheeze"],
        }
        filter_predictions["both"] = (
            filter_predictions["crackle"] & filter_predictions["wheeze"]
        )
        for node in metric_nodes:
            consistency[node].append(
                len(set(int(value) for value in filter_predictions[node])) == 1
            )
        row = {
            "patient_id": patient_id,
            "filter_sample_ids": [sample.sample_id for sample in source_samples],
            "raw_label": raw_label,
            "level1_probability": level1_probability.tolist(),
            "mean_level1_logits": mean_level1_logits.tolist(),
            "mean_attribute_logits": {
                node: float(mean_attribute_logits[index])
                for index, node in enumerate(CORE_NODES[1:])
            },
            "attribute_probability": {
                node: float(attribute_probability[index])
                for index, node in enumerate(CORE_NODES[1:])
            },
            "mean_level1_prediction": mean_level1_prediction,
            "mean_attribute_prediction": mean_attribute_prediction,
            "mean_both_prediction": int(
                mean_attribute_prediction["crackle"]
                and mean_attribute_prediction["wheeze"]
            ),
            "max_level1_prediction_secondary": max_level1_prediction,
            "max_attribute_prediction_secondary": max_attribute_prediction,
            "filter_view_consistency": {
                node: consistency[node][-1] for node in metric_nodes
            },
            "excluded": not bool(mapped["level1_eligible"]),
        }
        if row["excluded"]:
            excluded[raw_label] += 1
            row["unified_ground_truth"] = None
        else:
            row["unified_ground_truth"] = {
                node: int(mapped[f"{node}_target"]) for node in CORE_NODES
            }
            row["unified_ground_truth"]["both"] = int(
                row["unified_ground_truth"]["crackle"]
                and row["unified_ground_truth"]["wheeze"]
            )
            mean_by_node = {
                "level1": mean_level1_prediction,
                **mean_attribute_prediction,
            }
            mean_by_node["both"] = int(
                mean_attribute_prediction["crackle"]
                and mean_attribute_prediction["wheeze"]
            )
            max_by_node = {
                "level1": max_level1_prediction,
                **max_attribute_prediction,
            }
            max_by_node["both"] = int(
                max_attribute_prediction["crackle"]
                and max_attribute_prediction["wheeze"]
            )
            for node in metric_nodes:
                metric_targets[node].append(row["unified_ground_truth"][node])
                mean_predictions[node].append(mean_by_node[node])
                max_predictions[node].append(max_by_node[node])
        patient_rows.append(row)
    write_jsonl(output_dir / "kauh_external_patient_predictions.jsonl", patient_rows)
    nodes = {
        node: binary_metrics(
            np.asarray(metric_targets[node]), np.asarray(mean_predictions[node])
        )
        for node in metric_nodes
    }
    for node in metric_nodes:
        nodes[node]["descriptive_only"] = (
            node == "crackle" and nodes[node]["support"]["positive"] < 20
        )
    max_sensitivity = {
        node: binary_metrics(
            np.asarray(metric_targets[node]), np.asarray(max_predictions[node])
        )
        for node in metric_nodes
    }
    f1_values = [
        float(nodes[node]["f1"])
        for node in metric_nodes
        if len(set(metric_targets[node])) == 2
    ]
    report = {
        "unit": "P-number patient after mean probability over B/D/E filters",
        "all_patients": len(patient_rows),
        "eligible_patients": len(metric_targets["level1"]),
        "excluded_patients": sum(excluded.values()),
        "excluded_by_raw_label": dict(sorted(excluded.items())),
        "filter_files_used_for_eligible_patients": 3 * len(metric_targets["level1"]),
        "nodes": nodes,
        "patient_node_macro_f1": float(np.mean(f1_values)),
        "max_probability_aggregation_secondary": max_sensitivity,
        "filter_view_consistency": {
            node: float(np.mean(consistency[node])) for node in metric_nodes
        },
        "filter_view_consistency_name_boundary": (
            "same-patient filter-view consistency; not device robustness"
        ),
        "threshold_source": "ICBHI+SPRSound validation only",
        "selection_participation": False,
        "legacy_partition_cache_role": (
            "subtrain/validation/test filenames are storage-only; all shards are "
            "merged as external and never enter training or selection"
        ),
    }
    write_json(output_dir / "kauh_external_metrics.json", report)
    return report


def run_core2_package(
    repo_root: Path,
    result_dir: Path,
    *,
    backend: FrozenWindowBackend | None,
    cache_dir: Path,
    config: Mapping[str, object],
    package_limitations: Sequence[str],
    hf_auxiliary_enabled: bool,
    device: torch.device,
    encoder_window_batch_size: int,
    cache_only: bool = False,
) -> dict[str, object]:
    set_determinism()
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "config.json", dict(config))
    samples = load_canonical_samples(repo_root)
    training_store_datasets = (
        TRAIN_DATASETS if hf_auxiliary_enabled else CORE_DATASETS
    )
    stores = {
        partition: _stores(
            samples,
            partition,
            training_store_datasets,
            backend,
            cache_dir,
            device=device,
            encoder_window_batch_size=encoder_window_batch_size,
            cache_only=cache_only,
        )
        for partition in ("subtrain", "validation")
    }
    targets = {
        partition: _core_targets(samples, stores[partition])
        for partition in ("subtrain", "validation")
    }
    model, selection = train_core2_head(
        stores,
        targets,
        result_dir,
        device=device,
        hf_auxiliary_enabled=hf_auxiliary_enabled,
    )
    selected_validation = np.load(
        result_dir / "validation" / f"epoch_{selection['selected_epoch']:03d}.npz",
        allow_pickle=False,
    )
    selected_validation_predictions = {
        key: selected_validation[key] for key in selected_validation.files
    }
    core_validation = _filter_predictions(
        selected_validation_predictions, CORE_DATASETS
    )
    thresholds, threshold_details = select_core_shared_thresholds(core_validation)
    secondary_thresholds = secondary_dataset_thresholds(core_validation)
    write_json(
        result_dir / "validation_thresholds.json",
        {
            "thresholds": thresholds,
            "primary_selection_details": threshold_details,
            "source_datasets": list(CORE_DATASETS),
            "policy": (
                "one core-shared threshold per attribute maximizing equal mean of "
                "ICBHI and SPRSound validation F1; tie higher threshold"
            ),
            "secondary_per_dataset_calibration_sensitivity": secondary_thresholds,
            "secondary_not_headline": True,
            "hf_used": False,
            "kauh_used": False,
        },
    )
    _save_predictions(
        result_dir / "selected_validation_predictions.npz",
        selected_validation_predictions,
    )
    test_stores = _stores(
        samples,
        "test",
        TRAIN_DATASETS,
        backend,
        cache_dir,
        device=device,
        encoder_window_batch_size=encoder_window_batch_size,
        cache_only=cache_only,
    )
    label_free_targets = _core_targets(samples, test_stores)
    test_predictions = infer_core2(
        model,
        test_stores,
        label_free_targets,
        device=device,
        datasets=TRAIN_DATASETS,
    )
    label_free = {
        key: value
        for key, value in test_predictions.items()
        if key not in {"targets", "eligible"}
    }
    _attach_ground_truth(label_free, samples, None)
    _save_predictions(
        result_dir / "selected_test_predictions_label_free.npz", label_free
    )
    spr_targets = load_terminal_spr_test_targets(
        samples, include_checksums=False
    )
    scored_targets = _core_targets(
        samples, test_stores, spr_terminal_targets=spr_targets
    )
    target_rows = []
    eligible_rows = []
    for dataset in TRAIN_DATASETS:
        target_values, eligible_values = scored_targets[dataset]
        mask = test_stores[dataset]["window_mask"]
        target_rows.append(target_values[mask])
        eligible_rows.append(eligible_values[mask])
    test_predictions["targets"] = np.concatenate(target_rows)
    test_predictions["eligible"] = np.concatenate(eligible_rows)
    _attach_ground_truth(test_predictions, samples, spr_targets)
    _save_predictions(
        result_dir / "selected_test_predictions.npz", test_predictions
    )
    core_test = _filter_predictions(test_predictions, CORE_DATASETS)
    core_metrics = score_core_primary(core_test, thresholds)
    write_json(result_dir / "core_metrics.json", core_metrics)
    hf_report = hf_positive_report(test_predictions, thresholds)
    write_json(result_dir / "hf_positive_only_metrics.json", hf_report)

    kauh_stores = {
        partition: _stores(
            samples,
            partition,
            ("kauh",),
            backend,
            cache_dir,
            device=device,
            encoder_window_batch_size=encoder_window_batch_size,
            cache_only=cache_only,
        )["kauh"]
        for partition in ("subtrain", "validation", "test")
    }
    kauh_predictions = []
    for partition in ("subtrain", "validation", "test"):
        current_store = {"kauh": kauh_stores[partition]}
        current_targets = _core_targets(samples, current_store)
        kauh_predictions.append(
            infer_core2(
                model,
                current_store,
                current_targets,
                device=device,
                datasets=("kauh",),
            )
        )
    kauh_external = kauh_patient_external(
        _concat_predictions(kauh_predictions), samples, thresholds, result_dir
    )
    summary = {
        "status": "core2_hf_positive_kauh_external_complete",
        "device": str(device),
        "selection": selection,
        "hf_auxiliary_enabled": hf_auxiliary_enabled,
        "thresholds": thresholds,
        "core_metrics": core_metrics,
        "hf_positive_only": hf_report,
        "kauh_external": kauh_external,
        "test_runs": {
            "core_and_hf": 1,
            "kauh_external": 1,
        },
        "issue": list(package_limitations),
        "acceptance_decision": "ACCEPT_WITH_ROLE_AND_PACKAGE_LIMITATIONS",
    }
    write_json(result_dir / "run_summary.json", summary)
    return summary


def _common_config(
    repo_root: Path,
    package: str,
    encoder_dim: int,
    dimension_adapter: str,
    frontend: str,
    source_repo: Path,
    checkpoint: Path,
    device: torch.device,
    microbatch: int,
    hf_auxiliary_enabled: bool,
) -> dict[str, object]:
    return {
        "experiment": "core2_hf_positive_kauh_external",
        "package": package,
        "seed": SEED,
        "sample_rate": 16_000,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "effective_native_batch": 8,
        "encoder_window_microbatch": microbatch,
        "encoder_frozen": True,
        "encoder_embedding_dim": encoder_dim,
        "dimension_adapter": dimension_adapter,
        "dimension_adapter_trainable_parameters": (
            0 if encoder_dim == 768 else 2 * encoder_dim + encoder_dim * 768
        ),
        "comparison_scope": "encoder plus frontend plus required dimension adapter package",
        "shared_projector": "biased Linear 768->256",
        "shared_nodes": ["Level1", "Crackle", "Wheeze"],
        "other_node": "absent from architecture, loss, threshold, macro, and decision",
        "frontend": frontend,
        "source_repo": str(source_repo),
        "checkpoint": str(checkpoint),
        "epochs": 50,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": 50 * UPDATES_PER_EPOCH,
        "optimizer": "Adam",
        "learning_rate": 5e-5,
        "weight_decay": 1e-6,
        "schedule": "cosine per update; no warmup",
        "precision": "FP32",
        "augmentation": False,
        "sampler": "source-proportional homogeneous dataset batches",
        "training_roles": {
            "icbhi": "core",
            "sprsound": "core",
            "hf_lung": (
                "positive auxiliary weight 1.0"
                if hf_auxiliary_enabled
                else "HF-off; evaluation only"
            ),
            "kauh": "external only after selection",
        },
        "kauh_cache_policy": (
            "legacy subtrain/validation/test shard names are all-external storage only; "
            "merged after selection to 112 P-number patients"
        ),
        "hf_auxiliary_enabled": hf_auxiliary_enabled,
        "selection": "equal ICBHI/SPRSound validation eligible-node loss; tie earliest",
        "thresholds": (
            "one core-shared threshold per attribute maximizing equal mean of "
            "ICBHI/SPRSound validation F1; HF/KAUH excluded"
        ),
        "primary_reporting": (
            "per-core-dataset AUPRC/AUROC plus P/R/F1 at the same core-shared threshold"
        ),
        "macro_policy": "degenerate or non-evaluable nodes excluded",
        "dataset_role_contract": str(
            repo_root / ROOT_RELATIVE / "dataset_role_contract.json"
        ),
        "device": str(device),
    }


def run_ast(
    repo_root: Path,
    *,
    microbatch: int = 8,
    hf_auxiliary_enabled: bool = True,
) -> dict[str, object]:
    device = torch.device("cpu")
    source = repo_root / ".cache/icbhi_sprsound_shared_encoder_native_heads/source/repo"
    checkpoint = (
        repo_root
        / ".cache/icbhi_sprsound_shared_encoder_native_heads/checkpoints/hf_ast_legacy_compat.pth"
    )
    cache_dir = repo_root / ".cache/multidataset_pipeline/m_unified_ast_seed42"
    required = {
        cache_dir / f"{partition}_{dataset}.npz"
        for partition in ("subtrain", "validation", "test")
        for dataset in ("icbhi", "sprsound", "hf_lung", "kauh")
    }
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(f"AST cached-only run missing shards: {missing}")
    config = _common_config(
        repo_root,
        "AST_2s_native_grid_v0",
        768,
        "identity",
        "2-second 198-frame AST fbank grid",
        source,
        checkpoint,
        device,
        microbatch,
        hf_auxiliary_enabled,
    )
    condition = "AST_HF_on" if hf_auxiliary_enabled else "AST_HF_off"
    return run_core2_package(
        repo_root,
        repo_root / ROOT_RELATIVE / condition / "seed_42",
        backend=None,
        cache_dir=cache_dir,
        config=config,
        package_limitations=[
            "AST_2s_native_grid_v0 differs from historical 798-frame padded P1",
            "HF is positive-only and cannot support detector F1/specificity",
        ],
        hf_auxiliary_enabled=hf_auxiliary_enabled,
        device=device,
        encoder_window_batch_size=microbatch,
        cache_only=True,
    )


def run_ast_hf_on(repo_root: Path, *, microbatch: int = 8) -> dict[str, object]:
    result = run_ast(
        repo_root, microbatch=microbatch, hf_auxiliary_enabled=True
    )
    build_ast_gate_summary(repo_root)
    return result


def run_ast_hf_off(repo_root: Path, *, microbatch: int = 8) -> dict[str, object]:
    result = run_ast(
        repo_root, microbatch=microbatch, hf_auxiliary_enabled=False
    )
    build_ast_gate_summary(repo_root)
    return result


def run_beats(repo_root: Path, *, microbatch: int = 8) -> dict[str, object]:
    require_ast_management_release(repo_root)
    device = torch.device("cpu")
    source = repo_root / ".cache/multidataset_pipeline/assets/P2/source/repo"
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    )
    cache_dir = repo_root / ".cache/multidataset_pipeline/m_unified_beats_seed42"
    required_core = {
        cache_dir / f"{partition}_{dataset}.npz"
        for partition in ("subtrain", "validation")
        for dataset in CORE_DATASETS
    }
    missing_core = sorted(
        str(path) for path in required_core if not path.is_file()
    )
    if missing_core:
        raise FileNotFoundError(
            f"BEATs HF-off requires existing core caches; rebuilding forbidden: {missing_core}"
        )
    backend = load_local_beats_window_backend(source, checkpoint, device=device)
    config = _common_config(
        repo_root,
        "BEATs_exact_valid_patch_pool_v0",
        768,
        "identity",
        "exact valid-patch mask; frequency mean; valid temporal mean",
        source,
        checkpoint,
        device,
        microbatch,
        False,
    )
    return run_core2_package(
        repo_root,
        repo_root / ROOT_RELATIVE / "BEATs_HF_off/seed_42",
        backend=backend,
        cache_dir=cache_dir,
        config=config,
        package_limitations=[
            "BEATs result is an encoder+frontend+masking+pooling package comparison",
            "HF is evaluation-only and cannot support detector F1/specificity",
        ],
        hf_auxiliary_enabled=False,
        device=device,
        encoder_window_batch_size=microbatch,
        cache_only=False,
    )


def _asset_hold(
    repo_root: Path,
    encoder_dir: str,
    package: str,
    reason: str,
    proposed_frontend: str,
) -> dict[str, object]:
    result_dir = repo_root / ROOT_RELATIVE / encoder_dir / "seed_42"
    result_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "core2_hf_positive_kauh_external",
        "package": package,
        "seed": SEED,
        "scientific_contract": "same Core-2 + HF positive auxiliary + KAUH external contract",
        "proposed_frontend": proposed_frontend,
        "shared_nodes": ["Level1", "Crackle", "Wheeze"],
        "other_node": "absent",
        "comparison_scope": "package-level",
        "panns_dimension_adapter_trainable_parameters": (
            1_576_960 if encoder_dir == "PANNs_Cnn14" else None
        ),
        "opera_pretraining_overlap_caveat": (
            "ICBHI/HF overlap; not clean generalization"
            if encoder_dir == "OPERA_CT"
            else None
        ),
        "status": "HOLD_asset_runtime_unavailable",
    }
    write_json(result_dir / "config.json", config)
    summary = {
        "status": "HOLD_asset_runtime_unavailable",
        "encoder": encoder_dir,
        "single_blocker": reason,
        "training_started": False,
        "test_accessed": False,
        "acceptance_decision": "HOLD",
    }
    write_json(result_dir / "run_summary.json", summary)
    return summary


def run_panns(repo_root: Path, *, microbatch: int = 8) -> dict[str, object]:
    require_ast_management_release(repo_root)
    source = repo_root / ".cache/multidataset_pipeline/assets/P3/source/repo"
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P3/checkpoints/Cnn14_16k_mAP=0.438.pth"
    )
    if not source.is_dir() or not checkpoint.is_file():
        return _asset_hold(
            repo_root,
            "PANNs_Cnn14",
            "PANNs_Cnn14_16k_with_trainable_2048_to_768",
            "official PANNs Cnn14_16k source/checkpoint package is not locally provisioned",
            "official 16 kHz waveform frontend; pooled 2048-d embedding; trainable LayerNorm+Linear 2048->768",
        )
    device = torch.device("cpu")
    backend = load_local_panns_window_backend(
        source, checkpoint, device=device
    )
    config = _common_config(
        repo_root,
        "PANNs_Cnn14_16k_with_trainable_2048_to_768",
        2_048,
        "trainable LayerNorm(2048)+bias-free Linear(2048->768)",
        "official Cnn14_16k waveform frontend and pooled 2048-d embedding",
        source,
        checkpoint,
        device,
        microbatch,
        False,
    )
    return run_core2_package(
        repo_root,
        repo_root / ROOT_RELATIVE / "PANNs_Cnn14/seed_42",
        backend=backend,
        cache_dir=(
            repo_root / ".cache/multidataset_pipeline/core2_panns_seed42"
        ),
        config=config,
        package_limitations=[
            "PANNs comparison includes 1,576,960 trainable dimension-adapter parameters",
            "encoder plus frontend plus dimension adapter is a package-level comparator",
            "HF is evaluation-only and cannot support detector F1/specificity",
        ],
        hf_auxiliary_enabled=False,
        device=device,
        encoder_window_batch_size=microbatch,
        cache_only=False,
    )


def run_opera_ct(repo_root: Path, *, microbatch: int = 8) -> dict[str, object]:
    require_ast_management_release(repo_root)
    source = repo_root / ".cache/multidataset_pipeline/assets/P5/source/repo"
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P5/checkpoints/encoder-operaCT.ckpt"
    )
    if not source.is_dir() or not checkpoint.is_file():
        return _asset_hold(
            repo_root,
            "OPERA_CT",
            "OPERA_CT_2s_to_8s_zero_pad_overlap_aware",
            "official OPERA-CT source/checkpoint package is not locally provisioned",
            "2-second source window zero-padded inside package to official 8-second input; ICBHI/HF pretraining overlap caveat",
        )
    device = torch.device("cpu")
    backend = load_local_opera_ct_window_backend(
        source, checkpoint, device=device
    )
    config = _common_config(
        repo_root,
        "OPERA_CT_2s_to_8s_zero_pad_overlap_aware",
        768,
        "identity",
        "2-second source waveform zero-padded to official 8-second 64-mel HTSAT input",
        source,
        checkpoint,
        device,
        microbatch,
        False,
    )
    config["pretraining_overlap_caveat"] = (
        "standard OPERA pretraining includes ICBHI/HF sources; not clean generalization"
    )
    return run_core2_package(
        repo_root,
        repo_root / ROOT_RELATIVE / "OPERA_CT/seed_42",
        backend=backend,
        cache_dir=(
            repo_root / ".cache/multidataset_pipeline/core2_opera_ct_seed42"
        ),
        config=config,
        package_limitations=[
            "standard OPERA pretraining includes ICBHI/HF sources",
            "2-second source windows are zero-padded to the official 8-second package input",
            "OPERA-CT is an overlap-aware package reference, not clean-generalization evidence",
            "HF is evaluation-only and cannot support detector F1/specificity",
        ],
        hf_auxiliary_enabled=False,
        device=device,
        encoder_window_batch_size=microbatch,
        cache_only=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder",
        choices=("ast_hf_on", "ast_hf_off", "beats", "panns", "opera_ct"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--encoder-window-batch-size", type=int, default=8)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    write_dataset_role_contract(root)
    runners = {
        "ast_hf_on": run_ast_hf_on,
        "ast_hf_off": run_ast_hf_off,
        "beats": run_beats,
        "panns": run_panns,
        "opera_ct": run_opera_ct,
    }
    result = runners[args.encoder](
        root, microbatch=args.encoder_window_batch_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
