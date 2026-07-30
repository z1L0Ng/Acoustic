"""Run cached-feature compatible-head harmonization."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from baseline.four_dataset_frozen_encoder.data import (
    KAUH_LABELS,
    Sample,
    build_samples,
    load_terminal_spr_test_targets,
    sample_to_row,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import (
    BATCH_SIZE,
    EPOCHS,
    TASK_SPECS,
    SharedNativeModel,
    _loss,
    _multiclass_metrics,
    _multilabel_metrics,
    _prior_receipt,
    _source_batches,
    _targets,
    evaluate_task,
    predict_task,
    score_task_predictions,
)
from baseline.shared_encoder_native_heads.protocol import SPR_LABELS


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "four_dataset_shared_compatible_head_harmonization"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
SELECTED_CACHE_SHA256 = (
    "3b3798cc9d01dbdfa8168a1cd641d658eb2fd4553799e59b84b7aae7ad0f5a69"
)
SELECTION_RECEIPT = (
    ROOT
    / "result/four_dataset_representation_attribution/comparison/encoder_selection.json"
)
CONDITIONS = (
    "h0_native_plus_independent_compatible",
    "h1_eligibility_masked_shared",
    "h2_parameter_matched_independent",
)
NARROW_LABELS = ["normal", "crackle", "wheeze", "both"]
SPR_NARROW_MAP = {0: 0, 4: 1, 5: 1, 2: 2, 6: 3}
COMPATIBLE_TASKS = (
    "compat_binary_icbhi",
    "compat_binary_spr",
    "compat_narrow4_icbhi",
    "compat_narrow4_spr",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    fieldnames.extend(
        key
        for row in rows
        for key in row
        if key not in fieldnames
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _ordered_sha(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def set_seed(seed: int = 20260728) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fixed_projection(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(256, 128, generator=generator)
    q, _ = torch.linalg.qr(values, mode="reduced")
    return q.T.contiguous()


class HarmonizationModel(SharedNativeModel):
    def __init__(self, condition: str) -> None:
        super().__init__()
        self.condition = condition
        if condition == CONDITIONS[0]:
            self.compatible = torch.nn.ModuleDict(
                {
                    "binary_icbhi": torch.nn.Linear(256, 2),
                    "binary_spr": torch.nn.Linear(256, 2),
                    "narrow4_icbhi": torch.nn.Linear(256, 4),
                    "narrow4_spr": torch.nn.Linear(256, 4),
                }
            )
        elif condition == CONDITIONS[1]:
            self.compatible = torch.nn.ModuleDict(
                {
                    "binary": torch.nn.Linear(256, 2),
                    "narrow4": torch.nn.Linear(256, 4),
                }
            )
            self.compatible_scales = torch.nn.ParameterDict(
                {
                    "binary": torch.nn.Parameter(torch.ones(2)),
                    "narrow4": torch.nn.Parameter(torch.ones(4)),
                }
            )
        elif condition == CONDITIONS[2]:
            self.register_buffer("binary_projection", _fixed_projection(20260728))
            self.register_buffer("narrow4_projection", _fixed_projection(20260729))
            self.compatible = torch.nn.ModuleDict(
                {
                    "binary_icbhi": torch.nn.Linear(128, 2),
                    "binary_spr": torch.nn.Linear(128, 2),
                    "narrow4_icbhi": torch.nn.Linear(128, 4),
                    "narrow4_spr": torch.nn.Linear(128, 4),
                }
            )
        else:
            raise ValueError(condition)

    def representation(self, values: torch.Tensor) -> torch.Tensor:
        return self.adapter(values)

    def forward(self, values: torch.Tensor, task: str) -> torch.Tensor:
        hidden = self.representation(values)
        return self.heads[task](hidden)

    def forward_compatible(
        self, values: torch.Tensor, ontology: str, dataset: str
    ) -> torch.Tensor:
        hidden = self.representation(values)
        if self.condition == CONDITIONS[0]:
            return self.compatible[f"{ontology}_{dataset}"](hidden)
        if self.condition == CONDITIONS[1]:
            return self.compatible[ontology](hidden) * self.compatible_scales[ontology]
        projection = getattr(self, f"{ontology}_projection")
        return self.compatible[f"{ontology}_{dataset}"](
            torch.nn.functional.linear(hidden, projection)
        )

    def compatible_parameter_count(self) -> int:
        modules = list(self.compatible.parameters())
        if hasattr(self, "compatible_scales"):
            modules.extend(self.compatible_scales.parameters())
        return sum(parameter.numel() for parameter in modules)


def _compatible_target(sample: Sample, ontology: str) -> int | None:
    if sample.dataset == "icbhi":
        native = sample.targets.get("icbhi_flat4")
        if native is None:
            return None
        return int(native != 0) if ontology == "binary" else int(native)
    if sample.dataset != "sprsound":
        return None
    if ontology == "binary":
        target = sample.targets.get("spr_binary")
        return None if target is None else int(target)
    target = sample.targets.get("spr_seven")
    return None if target is None else SPR_NARROW_MAP.get(int(target))


def _compatible_indices(
    samples: list[Sample],
    dataset: str,
    ontology: str,
    partition: str,
    label_free_test: bool = False,
) -> list[int]:
    manifest_dataset = "sprsound" if dataset == "spr" else dataset
    return [
        index
        for index, sample in enumerate(samples)
        if sample.dataset == manifest_dataset
        and sample.partition == partition
        and (
            label_free_test
            or _compatible_target(sample, ontology) is not None
        )
    ]


def _compatible_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, object]:
    indices = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
    abnormal_total = matrix[1:].sum()
    sensitivity = (
        float(np.trace(matrix[1:, 1:]) / abnormal_total)
        if abnormal_total
        else 0.0
    )
    average = (specificity + sensitivity) / 2
    harmonic = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=indices, average="weighted", zero_division=0
            )
        ),
        "uar": float(np.mean(recall)),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "average_score": average,
        "harmonic_score": harmonic,
        "sprsound_score": (average + harmonic) / 2,
        "predicted_abnormal_prevalence": float(np.mean(y_pred != 0)),
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


def _predict_compatible(
    model: HarmonizationModel,
    samples: list[Sample],
    embeddings: np.ndarray,
    ontology: str,
    dataset: str,
    partition: str,
    device: torch.device,
) -> list[dict[str, object]]:
    indices = _compatible_indices(
        samples,
        dataset,
        ontology,
        partition,
        label_free_test=partition == "test" and dataset in {"spr", "sprsound"},
    )
    values = torch.from_numpy(embeddings[indices]).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model.forward_compatible(values, ontology, dataset)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite compatible probabilities")
    predicted = probabilities.argmax(axis=1)
    task = f"compat_{ontology}_{dataset}"
    return [
        {
            "sample_id": samples[index].sample_id,
            "dataset": samples[index].dataset,
            "task": task,
            "partition": partition,
            "pred_json": json.dumps(int(predicted[position])),
            "probabilities_json": json.dumps(probabilities[position].tolist()),
        }
        for position, index in enumerate(indices)
    ]


def _terminal_compatible_target(
    sample: Sample,
    ontology: str,
    dataset: str,
    terminal_spr: dict[str, dict[str, int]],
) -> int | None:
    local = _compatible_target(sample, ontology)
    if local is not None:
        return local
    if dataset == "icbhi":
        return None
    source = terminal_spr[sample.sample_id]
    if ontology == "binary":
        return int(source["spr_binary"])
    return SPR_NARROW_MAP.get(int(source["spr_seven"]))


def _score_compatible(
    rows: list[dict[str, object]],
    samples: list[Sample],
    ontology: str,
    dataset: str,
    terminal_spr: dict[str, dict[str, int]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    scored = []
    excluded = []
    for row in rows:
        target = _terminal_compatible_target(
            sample_by_id[str(row["sample_id"])],
            ontology,
            dataset,
            terminal_spr,
        )
        if target is None:
            excluded.append(str(row["sample_id"]))
            continue
        scored.append({**row, "true_json": json.dumps(target)})
    labels = ["normal", "abnormal"] if ontology == "binary" else NARROW_LABELS
    if not scored:
        raise RuntimeError(
            f"empty compatible scoring: ontology={ontology} dataset={dataset} "
            f"label_free_rows={len(rows)} terminal_spr={len(terminal_spr)}"
        )
    y_true = np.asarray([json.loads(str(row["true_json"])) for row in scored])
    y_pred = np.asarray([json.loads(str(row["pred_json"])) for row in scored])
    metrics = _compatible_metrics(y_true, y_pred, labels)
    return metrics, scored, {
        "label_free_prediction_rows": len(rows),
        "scored_rows": len(scored),
        "excluded_rows": len(excluded),
        "excluded_id_sha256": _ordered_sha(excluded),
    }


def _validation_receipt(
    model: HarmonizationModel,
    samples: list[Sample],
    embeddings: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    receipt: dict[str, object] = {}
    scores = []
    for task in TASK_SPECS:
        metrics, _ = evaluate_task(
            model, samples, embeddings, task, "validation", device
        )
        receipt[task] = metrics
        scores.append(float(metrics["macro_f1"]))
    empty_terminal: dict[str, dict[str, int]] = {}
    for dataset in ("icbhi", "spr"):
        dataset_name = "icbhi" if dataset == "icbhi" else "sprsound"
        for ontology in ("binary", "narrow4"):
            rows = _predict_compatible(
                model,
                samples,
                embeddings,
                ontology,
                dataset,
                "validation",
                device,
            )
            metrics, _, _ = _score_compatible(
                rows, samples, ontology, dataset, empty_terminal
            )
            key = f"compat_{ontology}_{dataset}"
            receipt[key] = metrics
            scores.append(float(metrics["macro_f1"]))
    return float(np.mean(scores)), receipt


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    write_gzip_csv(path, rows)


def train_condition(
    condition: str,
    samples: list[Sample],
    embeddings: np.ndarray,
    output_dir: Path,
    device: torch.device,
    dataset_root: Path,
) -> dict[str, object]:
    set_seed()
    model = HarmonizationModel(condition).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    priors = _prior_receipt(samples)
    history = []
    best_score = -math.inf
    best_state = None
    selection: dict[str, object] = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        route_steps = Counter()
        for dataset, batch_indices in _source_batches(
            samples, "dataset_balanced", epoch
        ):
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
                native = _loss(
                    model(values[positions], task),
                    _targets(samples, sample_indices, task, device),
                    task,
                    priors,
                    False,
                )
                active.append(native)
                route_steps[task] += 1
            if dataset in {"icbhi", "sprsound"}:
                alias = "icbhi" if dataset == "icbhi" else "spr"
                for ontology in ("binary", "narrow4"):
                    local = [
                        (position, sample_index, _compatible_target(samples[sample_index], ontology))
                        for position, sample_index in enumerate(batch_indices)
                    ]
                    local = [row for row in local if row[2] is not None]
                    if not local:
                        continue
                    positions = [row[0] for row in local]
                    targets = torch.tensor(
                        [int(row[2]) for row in local],
                        dtype=torch.long,
                        device=device,
                    )
                    active.append(
                        torch.nn.functional.cross_entropy(
                            model.forward_compatible(
                                values[positions], ontology, alias
                            ),
                            targets,
                        )
                    )
                    route_steps[f"compat_{ontology}_{alias}"] += 1
            if not active:
                continue
            loss = torch.stack(active).mean()
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        score, validation = _validation_receipt(
            model, samples, embeddings, device
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_mean_macro_f1": score,
                "validation": validation,
                "route_steps": dict(sorted(route_steps.items())),
            }
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            selection = {"epoch": epoch, "validation_mean_macro_f1": score}
    if best_state is None:
        raise RuntimeError("no validation-selected state")
    model.load_state_dict(best_state)
    label_free = []
    for task in TASK_SPECS:
        label_free.extend(
            predict_task(model, samples, embeddings, task, "test", device)
        )
    for dataset in ("icbhi", "spr"):
        for ontology in ("binary", "narrow4"):
            label_free.extend(
                _predict_compatible(
                    model,
                    samples,
                    embeddings,
                    ontology,
                    dataset,
                    "test",
                    device,
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    label_free_path = output_dir / "predictions_label_free.csv.gz"
    _write_predictions(label_free_path, label_free)
    terminal_spr = load_terminal_spr_test_targets(samples)
    native_metrics = {}
    compatible_metrics = {}
    coverage = {}
    scored = []
    for task in TASK_SPECS:
        rows = [row for row in label_free if row["task"] == task]
        metrics, scored_rows = score_task_predictions(
            samples, task, rows, terminal_spr
        )
        native_metrics[task] = metrics
        scored.extend(scored_rows)
    for dataset in ("icbhi", "spr"):
        for ontology in ("binary", "narrow4"):
            key = f"compat_{ontology}_{dataset}"
            rows = [row for row in label_free if row["task"] == key]
            metrics, scored_rows, task_coverage = _score_compatible(
                rows, samples, ontology, dataset, terminal_spr
            )
            compatible_metrics[key] = metrics
            coverage[key] = task_coverage
            scored.extend(scored_rows)
    _write_predictions(output_dir / "predictions.csv.gz", scored)
    torch.save(
        {
            "condition": condition,
            "model": model.state_dict(),
            "selection": selection,
        },
        output_dir / "best.pth",
    )
    compatible_parameters = model.compatible_parameter_count()
    expected = json.loads(PROTOCOL_PATH.read_text())["conditions"][condition][
        "compatible_trainable_parameters"
    ]
    if compatible_parameters != expected:
        raise RuntimeError("compatible parameter receipt mismatch")
    payload = {
        "condition": condition,
        "history": history,
        "selection": selection,
        "native_test_metrics": native_metrics,
        "compatible_test_metrics": compatible_metrics,
        "compatible_coverage": coverage,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "compatible": compatible_parameters,
        },
        "prediction_rows": len(scored),
        "terminal_label_join": {
            "label_free_path": str(label_free_path),
            "label_free_rows_written_before_label_load": len(label_free),
            "spr_terminal_labels": len(terminal_spr),
            "spr_test_labels_loaded_after_label_free_write": True,
            "dataset_root": str(dataset_root.resolve()),
        },
    }
    write_json(output_dir / "metrics.json", payload)
    return payload


def data_audit(
    dataset_root: Path, selected_cache: Path, result_root: Path
) -> tuple[list[Sample], np.ndarray]:
    selection = json.loads(SELECTION_RECEIPT.read_text())
    if (
        selection["status"] != "encoder_selection_complete"
        or selection["selected_representation"] != "r1_beats_as2m_audioset_only"
        or selection["selected_cache_sha256"] != SELECTED_CACHE_SHA256
        or sha256_file(selected_cache) != SELECTED_CACHE_SHA256
    ):
        raise RuntimeError("Step 1 encoder-selection dependency failed")
    folds = {}
    canonical = None
    for fold in range(5):
        samples, receipt = build_samples(dataset_root, fold)
        folds[str(fold)] = receipt
        if canonical is None:
            canonical = samples
            write_gzip_csv(
                result_root / "samples_fold_0.csv.gz",
                [sample_to_row(sample) for sample in samples],
            )
        elif [sample.sample_id for sample in samples] != [
            sample.sample_id for sample in canonical
        ]:
            raise RuntimeError("fold changed selected-cache order")
    if canonical is None:
        raise RuntimeError("missing canonical samples")
    embeddings, cache_receipt = load_cache(selected_cache, canonical)
    receipt = {
        "status": "harmonization_data_audit_passed",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selection_receipt_sha256": sha256_file(SELECTION_RECEIPT),
        "selected_cache_sha256": sha256_file(selected_cache),
        "rows": len(canonical),
        "ordered_id_sha256": _ordered_sha(
            [sample.sample_id for sample in canonical]
        ),
        "folds": folds,
        "cache_receipt": cache_receipt,
    }
    write_json(result_root / "data_receipt.json", receipt)
    return canonical, embeddings


def _smoke_samples(samples: list[Sample]) -> list[Sample]:
    selected: set[str] = set()
    for task, spec in TASK_SPECS.items():
        for partition in ("subtrain", "validation", "test"):
            candidates = [
                sample
                for sample in samples
                if sample.dataset == spec["dataset"]
                and sample.partition == partition
                and (
                    task in sample.targets
                    or (
                        sample.dataset == "sprsound"
                        and partition == "test"
                        and not sample.targets
                    )
                )
            ]
            selected.update(sample.sample_id for sample in candidates[:8])
    for partition in ("subtrain", "validation"):
        for raw_label in (1, 3, 4, 5, 6):
            candidate = next(
                (
                    sample
                    for sample in samples
                    if sample.dataset == "sprsound"
                    and sample.partition == partition
                    and sample.targets.get("spr_seven") == raw_label
                ),
                None,
            )
            if candidate:
                selected.add(candidate.sample_id)
    return [sample for sample in samples if sample.sample_id in selected]


def _gradient_route_checks(device: torch.device) -> dict[str, object]:
    values = torch.randn(12, 768, device=device)
    model = HarmonizationModel(CONDITIONS[1]).to(device)
    hf_logits = model(values, "hf_phase_presence")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        hf_logits, torch.zeros_like(hf_logits)
    )
    loss.backward()
    compatible_zero = all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in list(model.compatible.parameters())
        + list(model.compatible_scales.parameters())
    )
    if not compatible_zero:
        raise RuntimeError("HF native loss leaked into compatible head")
    h1 = HarmonizationModel(CONDITIONS[1])
    h2 = HarmonizationModel(CONDITIONS[2])
    if (
        h1.compatible_parameter_count() != 1548
        or h2.compatible_parameter_count() != 1548
    ):
        raise RuntimeError("parameter match failed")
    return {
        "hf_kauh_missing_compatible_label_zero_gradient": True,
        "h1_compatible_parameters": h1.compatible_parameter_count(),
        "h2_compatible_parameters": h2.compatible_parameter_count(),
        "fixed_projection_trainable": False,
    }


def smoke(
    samples: list[Sample],
    embeddings: np.ndarray,
    dataset_root: Path,
    result_root: Path,
    device: torch.device,
) -> dict[str, object]:
    selected = _smoke_samples(samples)
    by_id = {
        sample.sample_id: value for sample, value in zip(samples, embeddings)
    }
    values = np.stack([by_id[sample.sample_id] for sample in selected])
    outputs = {}
    for condition in CONDITIONS:
        outputs[condition] = train_condition(
            condition,
            selected,
            values,
            result_root / "smoke" / condition,
            device,
            dataset_root,
        )
    route = _gradient_route_checks(device)
    receipt = {
        "status": "harmonization_real_data_smoke_passed",
        "samples": len(selected),
        "datasets": dict(
            sorted(Counter(sample.dataset for sample in selected).items())
        ),
        "conditions": list(CONDITIONS),
        "native_tasks": list(TASK_SPECS),
        "compatible_tasks": list(COMPATIBLE_TASKS),
        "gradient_and_parameter_checks": route,
        "finite": all(
            all(np.isfinite(float(row["loss"])) for row in output["history"])
            for output in outputs.values()
        ),
        "spr_label_free_then_terminal_join": all(
            output["terminal_label_join"][
                "spr_test_labels_loaded_after_label_free_write"
            ]
            for output in outputs.values()
        ),
    }
    write_json(result_root / "smoke_receipt.json", receipt)
    return receipt


def train_full(
    dataset_root: Path,
    result_root: Path,
    samples: list[Sample],
    embeddings: np.ndarray,
    device: torch.device,
) -> None:
    for fold in range(5):
        fold_samples, receipt = build_samples(dataset_root, fold)
        if [sample.sample_id for sample in fold_samples] != [
            sample.sample_id for sample in samples
        ]:
            raise RuntimeError("fold changed cache order")
        write_json(result_root / f"fold_{fold}" / "data_receipt.json", receipt)
        for condition in CONDITIONS:
            directory = result_root / f"fold_{fold}" / condition
            artifacts = [
                directory / "best.pth",
                directory / "metrics.json",
                directory / "predictions.csv.gz",
                directory / "predictions_label_free.csv.gz",
            ]
            present = [path.is_file() for path in artifacts]
            if any(present) and not all(present):
                raise RuntimeError(f"partial artifact set: {directory}")
            if all(present):
                print(f"TRAIN_RESUMED fold={fold} condition={condition}", flush=True)
                continue
            train_condition(
                condition,
                fold_samples,
                embeddings,
                directory,
                device,
                dataset_root,
            )
            print(f"TRAIN_COMPLETE fold={fold} condition={condition}", flush=True)


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(result_root: Path) -> dict[str, object]:
    summary = []
    per_class = []
    for condition in CONDITIONS:
        fold_metrics = [
            json.loads(
                (
                    result_root / f"fold_{fold}" / condition / "metrics.json"
                ).read_text()
            )
            for fold in range(5)
        ]
        for task, spec in TASK_SPECS.items():
            if task == "kauh_raw9":
                continue
            runs = [row["native_test_metrics"][task] for row in fold_metrics]
            metric_names = (
                ["macro_f1", "micro_f1"]
                if spec["kind"] == "multilabel"
                else ["macro_f1", "weighted_f1", "uar", "native_score"]
            )
            row = {
                "condition": condition,
                "task": task,
                "dataset": spec["dataset"],
                "evaluation": "five KAUH-fold-conditioned native-test runs",
                "rows": runs[0]["rows"],
                "runs": 5,
            }
            for metric in metric_names:
                values = [float(run[metric]) for run in runs]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sample_std"] = float(np.std(values, ddof=1))
            summary.append(row)
            for label in spec["labels"]:
                class_runs = [run["per_class"][label] for run in runs]
                per_class.append(
                    {
                        "condition": condition,
                        "task": task,
                        "dataset": spec["dataset"],
                        "label": label,
                        "support": class_runs[0]["support"],
                        "recall_mean": float(
                            np.mean([float(run["recall"]) for run in class_runs])
                        ),
                        "recall_sample_std": float(
                            np.std(
                                [float(run["recall"]) for run in class_runs], ddof=1
                            )
                        ),
                    }
                )
        oof = []
        for fold in range(5):
            oof.extend(
                row
                for row in _read_predictions(
                    result_root
                    / f"fold_{fold}"
                    / condition
                    / "predictions.csv.gz"
                )
                if row["task"] == "kauh_raw9"
            )
        ids = [row["sample_id"] for row in oof]
        if len(ids) != 336 or len(ids) != len(set(ids)):
            raise RuntimeError("KAUH OOF coverage failed")
        y_true = np.asarray([json.loads(row["true_json"]) for row in oof])
        y_pred = np.asarray([json.loads(row["pred_json"]) for row in oof])
        metrics = _multiclass_metrics(y_true, y_pred, KAUH_LABELS, "kauh_raw9")
        summary.append(
            {
                "condition": condition,
                "task": "kauh_raw9",
                "dataset": "kauh",
                "evaluation": "five-fold patient-grouped aggregate OOF",
                "rows": 336,
                "runs": 5,
                "macro_f1_mean": metrics["macro_f1"],
                "macro_f1_sample_std": None,
                "weighted_f1_mean": metrics["weighted_f1"],
                "weighted_f1_sample_std": None,
                "uar_mean": metrics["uar"],
                "uar_sample_std": None,
                "native_score_mean": None,
                "native_score_sample_std": None,
            }
        )
        for label in KAUH_LABELS:
            per_class.append(
                {
                    "condition": condition,
                    "task": "kauh_raw9",
                    "dataset": "kauh",
                    "label": label,
                    "support": metrics["per_class"][label]["support"],
                    "recall_mean": metrics["per_class"][label]["recall"],
                    "recall_sample_std": None,
                }
            )
        for compatible_task in COMPATIBLE_TASKS:
            runs = [
                row["compatible_test_metrics"][compatible_task]
                for row in fold_metrics
            ]
            summary.append(
                {
                    "condition": condition,
                    "task": compatible_task,
                    "dataset": (
                        "icbhi"
                        if compatible_task.endswith("icbhi")
                        else "sprsound"
                    ),
                    "evaluation": "five KAUH-fold-conditioned compatible-test runs",
                    "rows": runs[0]["rows"],
                    "runs": 5,
                    **{
                        f"{metric}_mean": float(
                            np.mean([float(run[metric]) for run in runs])
                        )
                        for metric in (
                            "macro_f1",
                            "weighted_f1",
                            "uar",
                            "specificity",
                            "sensitivity",
                            "average_score",
                            "sprsound_score",
                            "predicted_abnormal_prevalence",
                        )
                    },
                    **{
                        f"{metric}_sample_std": float(
                            np.std([float(run[metric]) for run in runs], ddof=1)
                        )
                        for metric in (
                            "macro_f1",
                            "weighted_f1",
                            "uar",
                            "specificity",
                            "sensitivity",
                            "average_score",
                            "sprsound_score",
                            "predicted_abnormal_prevalence",
                        )
                    },
                }
            )
            for label in runs[0]["per_class"]:
                per_class.append(
                    {
                        "condition": condition,
                        "task": compatible_task,
                        "dataset": (
                            "icbhi"
                            if compatible_task.endswith("icbhi")
                            else "sprsound"
                        ),
                        "label": label,
                        "support": runs[0]["per_class"][label]["support"],
                        "recall_mean": float(
                            np.mean(
                                [
                                    float(run["per_class"][label]["recall"])
                                    for run in runs
                                ]
                            )
                        ),
                        "recall_sample_std": float(
                            np.std(
                                [
                                    float(run["per_class"][label]["recall"])
                                    for run in runs
                                ],
                                ddof=1,
                            )
                        ),
                    }
                )
    write_csv(result_root / "summary.csv", summary)
    write_csv(result_root / "per_class_summary.csv", per_class)
    return {"summary": summary, "per_class": per_class}


def analyze(result_root: Path) -> dict[str, object]:
    aggregate_payload = aggregate(result_root)
    summary = aggregate_payload["summary"]
    by_key = {(row["condition"], row["task"]): row for row in summary}
    shared = CONDITIONS[1]
    control = CONDITIONS[2]
    binary_gaps = {}
    primary_pass = True
    specificity_pass = True
    material_improvements = 0
    for dataset in ("icbhi", "spr"):
        task = f"compat_binary_{dataset}"
        h1 = by_key[(shared, task)]
        h2 = by_key[(control, task)]
        gaps = {
            metric: float(h1[f"{metric}_mean"]) - float(h2[f"{metric}_mean"])
            for metric in (
                "macro_f1",
                "uar",
                "specificity",
                "predicted_abnormal_prevalence",
            )
        }
        binary_gaps[dataset] = gaps
        primary_pass &= gaps["macro_f1"] >= -0.02 and gaps["uar"] >= -0.02
        specificity_pass &= gaps["specificity"] >= -0.05
        prevalence = float(h1["predicted_abnormal_prevalence_mean"])
        specificity_pass &= 0.02 <= prevalence <= 0.98
        material_improvements += int(
            gaps["macro_f1"] >= 0.02 or gaps["uar"] >= 0.02
        )
    worst_shared_macro = min(
        float(by_key[(shared, f"compat_binary_{dataset}")]["macro_f1_mean"])
        for dataset in ("icbhi", "spr")
    )
    worst_control_macro = min(
        float(by_key[(control, f"compat_binary_{dataset}")]["macro_f1_mean"])
        for dataset in ("icbhi", "spr")
    )
    worst_shared_uar = min(
        float(by_key[(shared, f"compat_binary_{dataset}")]["uar_mean"])
        for dataset in ("icbhi", "spr")
    )
    worst_control_uar = min(
        float(by_key[(control, f"compat_binary_{dataset}")]["uar_mean"])
        for dataset in ("icbhi", "spr")
    )
    worst_pass = (
        worst_shared_macro - worst_control_macro >= -0.02
        and worst_shared_uar - worst_control_uar >= -0.02
    )
    native_gaps = {}
    material_native_losses = 0
    severe_native_losses = 0
    for task in TASK_SPECS:
        gap = float(by_key[(shared, task)]["macro_f1_mean"]) - float(
            by_key[(control, task)]["macro_f1_mean"]
        )
        native_gaps[task] = gap
        material_native_losses += int(gap < -0.02)
        severe_native_losses += int(gap < -0.05)
    native_pass = material_native_losses <= 1 and severe_native_losses == 0
    go = (
        primary_pass
        and worst_pass
        and specificity_pass
        and material_improvements >= 1
        and native_pass
    )
    receipt = {
        "status": "harmonization_analysis_complete",
        "comparison": "h1 shared minus h2 parameter-matched independent",
        "binary_task_gaps": binary_gaps,
        "worst_dataset_gaps": {
            "macro_f1": worst_shared_macro - worst_control_macro,
            "uar": worst_shared_uar - worst_control_uar,
        },
        "native_macro_f1_gaps": native_gaps,
        "gates": {
            "per_dataset_primary": primary_pass,
            "worst_dataset": worst_pass,
            "specificity_and_collapse": specificity_pass,
            "material_improvement_count": material_improvements,
            "native_side_effect": native_pass,
        },
        "decision": (
            "go_representation_harmonization_supported"
            if go
            else "hold_or_negative_representation_harmonization_not_supported"
        ),
        "narrow4_used_for_decision": False,
        "claim_boundary": json.loads(PROTOCOL_PATH.read_text())["claim_boundary"],
    }
    write_json(result_root / "harmonization_decision.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["audit", "smoke", "train", "analyze", "all"],
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--selected-cache",
        type=Path,
        default=Path(
            ".cache/four_dataset_representation_attribution/"
            "r1_beats_as2m_audioset_only/embeddings.npz"
        ),
    )
    parser.add_argument(
        "--result-root", type=Path, default=Path(f"result/{EXPERIMENT_ID}")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(f".cache/{EXPERIMENT_ID}")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    if result_root.name != EXPERIMENT_ID or result_root.parent.name != "result":
        raise ValueError("invalid result root")
    if cache_root.name != EXPERIMENT_ID or cache_root.parent.name != ".cache":
        raise ValueError("invalid cache root")
    result_root.mkdir(parents=True, exist_ok=True)
    for variable, relative in (
        ("NUMBA_CACHE_DIR", "runtime/numba"),
        ("MPLCONFIGDIR", "runtime/matplotlib"),
        ("XDG_CACHE_HOME", "runtime/xdg"),
    ):
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    samples, embeddings = data_audit(
        args.dataset_root, args.selected_cache, result_root
    )
    if args.phase == "audit":
        return
    if args.phase in {"smoke", "all"}:
        smoke(samples, embeddings, args.dataset_root, result_root, device)
        if args.phase == "smoke":
            return
    if args.phase in {"train", "all"}:
        train_full(
            args.dataset_root,
            result_root,
            samples,
            embeddings,
            device,
        )
        if args.phase == "train":
            return
    analyze(result_root)


if __name__ == "__main__":
    main()
