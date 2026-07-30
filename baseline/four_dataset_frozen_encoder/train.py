"""Cached-embedding D0-D3 training and evaluation."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from baseline.four_dataset_frozen_encoder.data import (
    HF_ADVENTITIOUS_LABELS,
    HF_PHASE_LABELS,
    KAUH_LABELS,
    Sample,
    load_terminal_spr_test_targets,
)
from baseline.shared_encoder_native_heads.protocol import ICBHI_LABELS, SPR_LABELS


SEED = 20260728
EPOCHS = 5
BATCH_SIZE = 128
TASK_SPECS = {
    "icbhi_flat4": {"kind": "multiclass", "labels": ICBHI_LABELS, "dataset": "icbhi"},
    "spr_binary": {"kind": "multiclass", "labels": ["normal", "adventitious"], "dataset": "sprsound"},
    "spr_seven": {"kind": "multiclass", "labels": SPR_LABELS, "dataset": "sprsound"},
    "hf_phase_presence": {"kind": "multilabel", "labels": HF_PHASE_LABELS, "dataset": "hf_lung"},
    "hf_adventitious_presence": {
        "kind": "multilabel",
        "labels": HF_ADVENTITIOUS_LABELS,
        "dataset": "hf_lung",
    },
    "kauh_raw9": {"kind": "multiclass", "labels": KAUH_LABELS, "dataset": "kauh"},
}
CONDITIONS = {
    "d0_independent_heads": {"shared": False, "sampling": "task", "prior": False},
    "d1_shared_adapter_source_proportional": {
        "shared": True,
        "sampling": "source_proportional",
        "prior": False,
    },
    "d2_shared_adapter_dataset_balanced": {
        "shared": True,
        "sampling": "dataset_balanced",
        "prior": False,
    },
    "d3_shared_adapter_dataset_balanced_prior": {
        "shared": True,
        "sampling": "dataset_balanced",
        "prior": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NativeHead(torch.nn.Module):
    def __init__(self, input_dim: int, classes: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SharedNativeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.heads = torch.nn.ModuleDict(
            {
                task: torch.nn.Linear(256, len(spec["labels"]))
                for task, spec in TASK_SPECS.items()
            }
        )

    def forward(self, values: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.adapter(values))


def _indices(samples: list[Sample], dataset: str, partition: str) -> list[int]:
    return [
        index
        for index, sample in enumerate(samples)
        if sample.dataset == dataset and sample.partition == partition and sample.targets
    ]


def _task_indices(samples: list[Sample], task: str, partition: str) -> list[int]:
    return [
        index
        for index, sample in enumerate(samples)
        if sample.partition == partition and task in sample.targets
    ]


def _prediction_indices(
    samples: list[Sample], task: str, partition: str
) -> list[int]:
    dataset = str(TASK_SPECS[task]["dataset"])
    return [
        index
        for index, sample in enumerate(samples)
        if sample.partition == partition
        and sample.dataset == dataset
        and (
            task in sample.targets
            or (
                partition == "test"
                and dataset == "sprsound"
                and not sample.targets
            )
        )
    ]


def _targets(samples: list[Sample], indices: list[int], task: str, device: torch.device) -> torch.Tensor:
    values = [samples[index].targets[task] for index in indices]
    if TASK_SPECS[task]["kind"] == "multiclass":
        return torch.tensor(values, dtype=torch.long, device=device)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _prior_receipt(samples: list[Sample]) -> dict[str, object]:
    output: dict[str, object] = {}
    for task, spec in TASK_SPECS.items():
        indices = _task_indices(samples, task, "subtrain")
        if spec["kind"] == "multiclass":
            counts = np.bincount(
                [int(samples[index].targets[task]) for index in indices],
                minlength=len(spec["labels"]),
            )
            priors = (counts / counts.sum()).tolist()
            output[task] = {"counts": counts.tolist(), "priors": priors}
        else:
            matrix = np.asarray([samples[index].targets[task] for index in indices], dtype=np.float64)
            positives = matrix.sum(axis=0)
            negatives = len(matrix) - positives
            output[task] = {
                "positives": positives.astype(int).tolist(),
                "negatives": negatives.astype(int).tolist(),
                "pos_weight": (negatives / np.maximum(positives, 1)).tolist(),
            }
    return output


def _loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    task: str,
    prior: dict[str, object],
    use_prior: bool,
) -> torch.Tensor:
    spec = TASK_SPECS[task]
    if spec["kind"] == "multiclass":
        if use_prior:
            probabilities = torch.tensor(prior[task]["priors"], dtype=logits.dtype, device=logits.device)
            logits = logits + torch.log(probabilities.clamp_min(1e-12))
        return torch.nn.functional.cross_entropy(logits, target)
    pos_weight = None
    if use_prior:
        pos_weight = torch.tensor(prior[task]["pos_weight"], dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], task: str) -> dict[str, object]:
    indices = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    metrics: dict[str, object] = {
        "rows": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=indices, average="weighted", zero_division=0)),
        "uar": float(np.mean(recall)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion": matrix.astype(int).tolist(),
    }
    if task in {"icbhi_flat4", "spr_binary", "spr_seven"}:
        specificity = float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
        abnormal_total = matrix[1:].sum()
        sensitivity = float(np.trace(matrix[1:, 1:]) / abnormal_total) if abnormal_total else 0.0
        average = (specificity + sensitivity) / 2
        harmonic = 2 * specificity * sensitivity / (specificity + sensitivity) if specificity + sensitivity else 0.0
        metrics.update(
            {
                "specificity": specificity,
                "sensitivity": sensitivity,
                "average_score": average,
                "harmonic_score": harmonic,
            }
        )
        metrics["native_score"] = (average + harmonic) / 2 if task.startswith("spr_") else average
    return metrics


def _multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, average=None, zero_division=0
    )
    per_class = {}
    for index, label in enumerate(labels):
        has_both = len(np.unique(y_true[:, index])) == 2
        has_positive = bool(y_true[:, index].sum())
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "average_precision": (
                float(
                    average_precision_score(
                        y_true[:, index], probabilities[:, index]
                    )
                )
                if has_positive
                else None
            ),
            "roc_auc": float(roc_auc_score(y_true[:, index], probabilities[:, index])) if has_both else None,
        }
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == predicted, axis=1))),
        "per_class": per_class,
    }


def predict_task(
    model,
    samples: list[Sample],
    embeddings: np.ndarray,
    task: str,
    partition: str,
    device: torch.device,
) -> list[dict[str, object]]:
    indices = _prediction_indices(samples, task, partition)
    if not indices:
        raise RuntimeError(f"no {partition} rows for {task}")
    values = torch.from_numpy(embeddings[indices]).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(values, task) if hasattr(model, "heads") else model(values)
        probabilities = (
            torch.softmax(logits, dim=1)
            if TASK_SPECS[task]["kind"] == "multiclass"
            else torch.sigmoid(logits)
        ).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"non-finite probabilities for {task}")
    spec = TASK_SPECS[task]
    rows = []
    if spec["kind"] == "multiclass":
        predicted = probabilities.argmax(axis=1)
        for local_index, sample_index in enumerate(indices):
            rows.append(
                {
                    "sample_id": samples[sample_index].sample_id,
                    "dataset": samples[sample_index].dataset,
                    "task": task,
                    "partition": partition,
                    "pred_json": json.dumps(int(predicted[local_index])),
                    "probabilities_json": json.dumps(probabilities[local_index].tolist()),
                }
            )
    else:
        predicted = (probabilities >= 0.5).astype(int)
        for local_index, sample_index in enumerate(indices):
            rows.append(
                {
                    "sample_id": samples[sample_index].sample_id,
                    "dataset": samples[sample_index].dataset,
                    "task": task,
                    "partition": partition,
                    "pred_json": json.dumps(predicted[local_index].tolist()),
                    "probabilities_json": json.dumps(probabilities[local_index].tolist()),
                }
            )
    return rows


def score_task_predictions(
    samples: list[Sample],
    task: str,
    rows: list[dict[str, object]],
    terminal_spr_targets: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    targets = []
    scored = []
    for row in rows:
        sample = sample_by_id[str(row["sample_id"])]
        if task in sample.targets:
            target = sample.targets[task]
        elif (
            terminal_spr_targets is not None
            and sample.sample_id in terminal_spr_targets
            and task in terminal_spr_targets[sample.sample_id]
        ):
            target = terminal_spr_targets[sample.sample_id][task]
        else:
            raise RuntimeError(f"missing terminal target for {sample.sample_id} {task}")
        targets.append(target)
        scored.append(
            {
                **row,
                "true_json": json.dumps(target),
            }
        )
    spec = TASK_SPECS[task]
    probabilities = np.asarray(
        [json.loads(str(row["probabilities_json"])) for row in rows],
        dtype=float,
    )
    target_array = np.asarray(targets)
    if spec["kind"] == "multiclass":
        predicted = np.asarray(
            [json.loads(str(row["pred_json"])) for row in rows], dtype=int
        )
        metrics = _multiclass_metrics(
            target_array.astype(int), predicted, spec["labels"], task
        )
    else:
        metrics = _multilabel_metrics(
            target_array.astype(int), probabilities, spec["labels"]
        )
    return metrics, scored


def evaluate_task(
    model,
    samples: list[Sample],
    embeddings: np.ndarray,
    task: str,
    partition: str,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = predict_task(model, samples, embeddings, task, partition, device)
    return score_task_predictions(samples, task, rows)


def _validation_score(model, samples, embeddings, device) -> tuple[float, dict[str, object]]:
    receipts = {}
    values = []
    for task in TASK_SPECS:
        metrics, _ = evaluate_task(model, samples, embeddings, task, "validation", device)
        receipts[task] = metrics
        values.append(float(metrics["macro_f1"]))
    return float(np.mean(values)), receipts


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_outputs(
    output_dir: Path,
    condition: str,
    model,
    samples,
    embeddings,
    history,
    selection,
    priors,
    device,
    dataset_root: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_free_predictions = []
    for task in TASK_SPECS:
        label_free_predictions.extend(
            predict_task(model, samples, embeddings, task, "test", device)
        )
    label_free_path = output_dir / "predictions_label_free.csv.gz"
    _write_predictions(label_free_path, label_free_predictions)
    if not label_free_path.is_file():
        raise RuntimeError("label-free prediction write did not complete")
    terminal_spr_targets = load_terminal_spr_test_targets(samples)
    metrics = {}
    scored_predictions = []
    for task in TASK_SPECS:
        task_rows = [
            row for row in label_free_predictions if row["task"] == task
        ]
        task_metrics, scored_rows = score_task_predictions(
            samples, task, task_rows, terminal_spr_targets
        )
        metrics[task] = task_metrics
        scored_predictions.extend(scored_rows)
    torch.save(
        {
            "condition": condition,
            "model": model.state_dict(),
            "selection": selection,
            "priors": priors,
        },
        output_dir / "best.pth",
    )
    _write_predictions(output_dir / "predictions.csv.gz", scored_predictions)
    payload = {
        "condition": condition,
        "history": history,
        "selection": selection,
        "test_metrics": metrics,
        "prediction_rows": len(scored_predictions),
        "terminal_label_join": {
            "label_free_prediction_path": str(label_free_path),
            "label_free_prediction_sha256": _sha256_file(label_free_path),
            "label_free_rows_written_before_label_load": len(
                label_free_predictions
            ),
            "spr_terminal_labels": len(terminal_spr_targets),
            "spr_test_labels_loaded_after_label_free_write": True,
            "dataset_root": str(dataset_root.resolve()),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def train_d0(
    samples: list[Sample],
    embeddings: np.ndarray,
    output_dir: Path,
    device: torch.device,
    dataset_root: Path,
    guard: Callable[[], None] | None = None,
) -> dict[str, object]:
    set_seed(SEED)
    priors = _prior_receipt(samples)
    heads = torch.nn.ModuleDict()
    history = {}
    selection = {}
    for task, spec in TASK_SPECS.items():
        head = NativeHead(768, len(spec["labels"])).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        train_indices = _task_indices(samples, task, "subtrain")
        best_score = -math.inf
        best_state = None
        task_history = []
        for epoch in range(1, EPOCHS + 1):
            rng = np.random.default_rng(SEED + epoch)
            order = rng.permutation(train_indices).tolist()
            losses = []
            head.train()
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start : start + BATCH_SIZE]
                values = torch.from_numpy(embeddings[indices]).to(device)
                target = _targets(samples, indices, task, device)
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(head(values), target, task, priors, False)
                loss.backward()
                optimizer.step()
                if guard is not None:
                    guard()
                losses.append(float(loss.detach()))
            metrics, _ = evaluate_task(head, samples, embeddings, task, "validation", device)
            score = float(metrics["macro_f1"])
            task_history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_f1": score})
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(head.state_dict())
                selection[task] = {"epoch": epoch, "validation_macro_f1": score}
        if best_state is None:
            raise RuntimeError(f"no D0 state for {task}")
        head.load_state_dict(best_state)
        heads[task] = head
        history[task] = task_history

    class IndependentWrapper(torch.nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.heads = modules

        def forward(self, values, task):
            return self.heads[task](values)

    model = IndependentWrapper(heads).to(device)
    return _save_outputs(
        output_dir,
        "d0_independent_heads",
        model,
        samples,
        embeddings,
        history,
        selection,
        priors,
        device,
        dataset_root,
    )


def _source_batches(samples: list[Sample], policy: str, epoch: int) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(SEED + epoch)
    by_dataset = {
        dataset: _indices(samples, dataset, "subtrain")
        for dataset in ("icbhi", "sprsound", "hf_lung", "kauh")
    }
    chunks = {}
    for dataset, indices in by_dataset.items():
        shuffled = rng.permutation(indices).tolist()
        chunks[dataset] = [
            shuffled[start : start + BATCH_SIZE]
            for start in range(0, len(shuffled), BATCH_SIZE)
        ]
    if policy == "source_proportional":
        schedule = [(dataset, chunk) for dataset, values in chunks.items() for chunk in values]
    elif policy == "dataset_balanced":
        count = max(len(values) for values in chunks.values())
        schedule = [
            (dataset, chunks[dataset][index % len(chunks[dataset])])
            for dataset in chunks
            for index in range(count)
        ]
    else:
        raise ValueError(policy)
    rng.shuffle(schedule)
    return schedule


def train_joint(
    condition: str,
    samples: list[Sample],
    embeddings: np.ndarray,
    output_dir: Path,
    device: torch.device,
    dataset_root: Path,
    guard: Callable[[], None] | None = None,
) -> dict[str, object]:
    config = CONDITIONS[condition]
    set_seed(SEED)
    model = SharedNativeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    priors = _prior_receipt(samples)
    best_score = -math.inf
    best_state = None
    history = []
    selection = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        task_losses = Counter()
        task_steps = Counter()
        for dataset, batch_indices in _source_batches(samples, str(config["sampling"]), epoch):
            values = torch.from_numpy(embeddings[batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            active = []
            for task, spec in TASK_SPECS.items():
                if spec["dataset"] != dataset:
                    continue
                local = [
                    (position, sample_index)
                    for position, sample_index in enumerate(batch_indices)
                    if task in samples[sample_index].targets
                ]
                if not local:
                    continue
                positions = [position for position, _ in local]
                sample_indices = [sample_index for _, sample_index in local]
                logits = model(values[positions], task)
                target = _targets(samples, sample_indices, task, device)
                task_loss = _loss(logits, target, task, priors, bool(config["prior"]))
                active.append(task_loss)
                task_losses[task] += float(task_loss.detach())
                task_steps[task] += 1
            if not active:
                continue
            loss = torch.stack(active).mean()
            loss.backward()
            optimizer.step()
            if guard is not None:
                guard()
            losses.append(float(loss.detach()))
        score, validation = _validation_score(model, samples, embeddings, device)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_mean_macro_f1": score,
                "validation": validation,
                "task_mean_loss": {
                    task: task_losses[task] / task_steps[task] for task in task_steps
                },
                "updates": len(losses),
            }
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            selection = {"epoch": epoch, "validation_mean_macro_f1": score}
    if best_state is None:
        raise RuntimeError(f"no state selected for {condition}")
    model.load_state_dict(best_state)
    return _save_outputs(
        output_dir,
        condition,
        model,
        samples,
        embeddings,
        history,
        selection,
        priors,
        device,
        dataset_root,
    )


def train_conditions(
    samples: list[Sample],
    embeddings: np.ndarray,
    result_dir: Path,
    conditions: list[str],
    device: torch.device,
    dataset_root: Path,
    guard: Callable[[], None] | None = None,
) -> dict[str, object]:
    outputs = {}
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        condition_dir = result_dir / condition
        outputs[condition] = (
            train_d0(
                samples,
                embeddings,
                condition_dir,
                device,
                dataset_root,
                guard,
            )
            if condition == "d0_independent_heads"
            else train_joint(
                condition,
                samples,
                embeddings,
                condition_dir,
                device,
                dataset_root,
                guard,
            )
        )
    return outputs
