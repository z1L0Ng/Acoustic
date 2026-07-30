"""Run the preregistered four-dataset cached-feature residual experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    build_samples,
    load_terminal_spr_test_targets,
)
from baseline.four_dataset_frozen_encoder.run import aggregate as baseline_aggregate
from baseline.four_dataset_frozen_encoder.train import (
    TASK_SPECS,
    evaluate_task,
    predict_task,
    score_task_predictions,
)
from model.four_dataset_general_specific_v1.models import (
    CONDITIONS,
    EXPECTED_PARAMETERS,
    GeneralSpecificModel,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
DEFAULT_RESULT = ROOT / "result/model/2026-07-28/four_dataset_general_specific_v1"
FROZEN_RESULT = ROOT / "result/four_dataset_pafa_frozen_encoder"
DATASET_ROOT = ROOT / "dataset/raw"
SEED = 20260728
EPOCHS = 5
BATCH_SIZE = 128
ELIGIBLE_PRIOR_TASKS = {"hf_adventitious_presence", "kauh_raw9"}
EXPECTED_CACHE_SHA256 = (
    "f40ae7fe581457bc86d76b93b1ee811e7ea01bc5e098a6daa73db451f96d1b31"
)
EXPECTED_ROWS = 25084
EXPECTED_DIM = 768


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cache(path: Path) -> tuple[list[str], np.ndarray, dict[str, object]]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"embedding cache SHA mismatch: {actual_sha}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "embeddings", "receipt_json"}:
            raise RuntimeError(f"unexpected cache keys: {payload.files}")
        sample_ids = payload["sample_ids"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        receipt = json.loads(str(payload["receipt_json"]))
    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise RuntimeError(f"unexpected embedding shape: {embeddings.shape}")
    if len(sample_ids) != len(set(sample_ids)) or not np.isfinite(embeddings).all():
        raise RuntimeError("embedding ID uniqueness/finite gate failed")
    return sample_ids, embeddings, {
        "cache_path": str(path.resolve()),
        "cache_sha256": actual_sha,
        "shape": list(embeddings.shape),
        "unique_ids": len(set(sample_ids)),
        "embedded_receipt": receipt,
    }


def assert_sample_alignment(
    samples: list[Sample], sample_ids: list[str], fold: int
) -> dict[str, object]:
    actual = [sample.sample_id for sample in samples]
    if actual != sample_ids:
        raise RuntimeError(f"sample/cache order mismatch for fold {fold}")
    frozen = json.loads(
        (FROZEN_RESULT / f"fold_{fold}/data_receipt.json").read_text()
    )
    dataset_rows = dict(sorted(Counter(sample.dataset for sample in samples).items()))
    if dataset_rows != frozen["dataset_rows"] or len(samples) != EXPECTED_ROWS:
        raise RuntimeError(f"dataset row mismatch for fold {fold}")
    return {
        "fold": fold,
        "rows": len(samples),
        "dataset_rows": dataset_rows,
        "ordered_id_sha256": frozen["ordered_id_sha256"],
        "frozen_receipt_path": str(
            (FROZEN_RESULT / f"fold_{fold}/data_receipt.json").resolve()
        ),
    }


def task_indices(
    samples: list[Sample], task: str, partition: str
) -> list[int]:
    return [
        index
        for index, sample in enumerate(samples)
        if sample.partition == partition and task in sample.targets
    ]


def targets(
    samples: list[Sample],
    indices: list[int],
    task: str,
    device: torch.device,
) -> torch.Tensor:
    values = [samples[index].targets[task] for index in indices]
    dtype = torch.long if TASK_SPECS[task]["kind"] == "multiclass" else torch.float32
    return torch.tensor(values, dtype=dtype, device=device)


def prior_receipt(samples: list[Sample]) -> dict[str, object]:
    output: dict[str, object] = {}
    for task, spec in TASK_SPECS.items():
        indices = task_indices(samples, task, "subtrain")
        if spec["kind"] == "multiclass":
            counts = np.bincount(
                [int(samples[index].targets[task]) for index in indices],
                minlength=len(spec["labels"]),
            )
            output[task] = {
                "counts": counts.tolist(),
                "priors": (counts / counts.sum()).tolist(),
            }
        else:
            matrix = np.asarray(
                [samples[index].targets[task] for index in indices],
                dtype=np.float64,
            )
            positives = matrix.sum(axis=0)
            negatives = len(matrix) - positives
            output[task] = {
                "positives": positives.astype(int).tolist(),
                "negatives": negatives.astype(int).tolist(),
                "pos_weight": (
                    negatives / np.maximum(positives, 1)
                ).tolist(),
            }
    return output


def loss_route(condition: str, task: str) -> str:
    if (
        bool(CONDITIONS[condition]["selective_prior"])
        and task in ELIGIBLE_PRIOR_TASKS
    ):
        return "logit_adjustment" if TASK_SPECS[task]["kind"] == "multiclass" else "pos_weight_bce"
    return "cross_entropy" if TASK_SPECS[task]["kind"] == "multiclass" else "bce"


def task_loss(
    condition: str,
    task: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    priors: dict[str, object],
) -> torch.Tensor:
    route = loss_route(condition, task)
    if route in {"cross_entropy", "logit_adjustment"}:
        adjusted = logits
        if route == "logit_adjustment":
            probabilities = torch.tensor(
                priors[task]["priors"],
                dtype=logits.dtype,
                device=logits.device,
            )
            adjusted = adjusted + torch.log(probabilities.clamp_min(1e-12))
        return torch.nn.functional.cross_entropy(adjusted, target)
    pos_weight = None
    if route == "pos_weight_bce":
        pos_weight = torch.tensor(
            priors[task]["pos_weight"],
            dtype=logits.dtype,
            device=logits.device,
        )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )


def dataset_balanced_batches(
    samples: list[Sample], epoch: int
) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(SEED + epoch)
    by_dataset = {
        dataset: [
            index
            for index, sample in enumerate(samples)
            if sample.dataset == dataset
            and sample.partition == "subtrain"
            and sample.targets
        ]
        for dataset in ("icbhi", "sprsound", "hf_lung", "kauh")
    }
    chunks: dict[str, list[list[int]]] = {}
    for dataset, indices in by_dataset.items():
        shuffled = rng.permutation(indices).tolist()
        chunks[dataset] = [
            shuffled[start : start + BATCH_SIZE]
            for start in range(0, len(shuffled), BATCH_SIZE)
        ]
        if not chunks[dataset]:
            raise RuntimeError(f"no batches for {dataset}")
    count = max(len(values) for values in chunks.values())
    schedule = [
        (dataset, chunks[dataset][index % len(chunks[dataset])])
        for dataset in chunks
        for index in range(count)
    ]
    rng.shuffle(schedule)
    return schedule


def validation_score(
    model: torch.nn.Module,
    samples: list[Sample],
    embeddings: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    metrics = {}
    values = []
    for task in TASK_SPECS:
        task_metrics, _ = evaluate_task(
            model, samples, embeddings, task, "validation", device
        )
        metrics[task] = task_metrics
        values.append(float(task_metrics["macro_f1"]))
    return float(np.mean(values)), metrics


def write_predictions(
    path: Path, rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_multilabel_confusions(
    metrics: dict[str, object],
    scored_rows: list[dict[str, object]],
    task: str,
) -> None:
    if TASK_SPECS[task]["kind"] != "multilabel":
        return
    y_true = np.asarray(
        [json.loads(str(row["true_json"])) for row in scored_rows], dtype=int
    )
    y_pred = np.asarray(
        [json.loads(str(row["pred_json"])) for row in scored_rows], dtype=int
    )
    metrics["confusion"] = {
        label: confusion_matrix(
            y_true[:, index], y_pred[:, index], labels=[0, 1]
        ).astype(int).tolist()
        for index, label in enumerate(TASK_SPECS[task]["labels"])
    }


def save_test_outputs(
    output_dir: Path,
    condition: str,
    model: torch.nn.Module,
    samples: list[Sample],
    embeddings: np.ndarray,
    history: list[dict[str, object]],
    selection: dict[str, object],
    priors: dict[str, object],
    device: torch.device,
    warning_messages: list[str],
    cache_receipt: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_free_rows: list[dict[str, object]] = []
    for task in TASK_SPECS:
        label_free_rows.extend(
            predict_task(model, samples, embeddings, task, "test", device)
        )
    label_free_path = output_dir / "predictions_label_free.csv.gz"
    write_predictions(label_free_path, label_free_rows)
    terminal_targets = load_terminal_spr_test_targets(samples)
    test_metrics = {}
    scored_rows: list[dict[str, object]] = []
    for task in TASK_SPECS:
        current = [row for row in label_free_rows if row["task"] == task]
        metrics, scored = score_task_predictions(
            samples, task, current, terminal_targets
        )
        add_multilabel_confusions(metrics, scored, task)
        test_metrics[task] = metrics
        scored_rows.extend(scored)
    write_predictions(output_dir / "predictions.csv.gz", scored_rows)
    checkpoint = {
        "condition": condition,
        "model": model.state_dict(),
        "selection": selection,
        "priors": priors,
        "loss_routes": {
            task: loss_route(condition, task) for task in TASK_SPECS
        },
        "cache_sha256": cache_receipt["cache_sha256"],
    }
    torch.save(checkpoint, output_dir / "best.pth")
    payload = {
        "condition": condition,
        "history": history,
        "selection": selection,
        "test_metrics": test_metrics,
        "parameters": parameter_count(model),
        "priors": priors,
        "loss_routes": checkpoint["loss_routes"],
        "warning_count": len(warning_messages),
        "warnings": warning_messages,
        "terminal_label_join": {
            "label_free_prediction_path": str(label_free_path.resolve()),
            "label_free_prediction_sha256": sha256_file(label_free_path),
            "label_free_rows_written_before_label_load": len(label_free_rows),
            "spr_terminal_labels": len(terminal_targets),
            "spr_test_labels_loaded_after_label_free_write": True,
        },
        "cache": cache_receipt,
    }
    write_json(output_dir / "metrics.json", payload)
    return payload


def train_condition(
    condition: str,
    samples: list[Sample],
    embeddings: np.ndarray,
    output_dir: Path,
    device: torch.device,
    cache_receipt: dict[str, object],
) -> dict[str, object]:
    set_seed()
    model = GeneralSpecificModel(condition).to(device)
    if parameter_count(model) != EXPECTED_PARAMETERS[condition]:
        raise RuntimeError(f"parameter gate failed for {condition}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    priors = prior_receipt(samples)
    best_score = -math.inf
    best_state = None
    selection: dict[str, object] = {}
    history: list[dict[str, object]] = []
    captured_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            task_losses = Counter()
            task_steps = Counter()
            update_counts = Counter()
            for dataset, batch_indices in dataset_balanced_batches(samples, epoch):
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
                    target = targets(samples, sample_indices, task, device)
                    current_loss = task_loss(
                        condition, task, logits, target, priors
                    )
                    active.append(current_loss)
                    task_losses[task] += float(current_loss.detach())
                    task_steps[task] += 1
                if not active:
                    continue
                loss = torch.stack(active).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for {condition}")
                loss.backward()
                if not all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(f"non-finite gradient for {condition}")
                optimizer.step()
                losses.append(float(loss.detach()))
                update_counts[dataset] += 1
            if len(set(update_counts.values())) != 1:
                raise RuntimeError(
                    f"dataset-balanced update count failed: {update_counts}"
                )
            score, validation = validation_score(
                model, samples, embeddings, device
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_mean_macro_f1": score,
                    "validation": validation,
                    "task_mean_loss": {
                        task: task_losses[task] / task_steps[task]
                        for task in task_steps
                    },
                    "updates": len(losses),
                    "dataset_updates": dict(update_counts),
                }
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                selection = {
                    "epoch": epoch,
                    "validation_mean_macro_f1": score,
                }
        captured_messages = [str(item.message) for item in caught]
    if best_state is None:
        raise RuntimeError(f"no selected state for {condition}")
    model.load_state_dict(best_state)
    return save_test_outputs(
        output_dir,
        condition,
        model,
        samples,
        embeddings,
        history,
        selection,
        priors,
        device,
        captured_messages,
        cache_receipt,
    )


def run_smoke(
    cache_path: Path, result_root: Path, device: torch.device
) -> dict[str, object]:
    sample_ids, embeddings, cache_receipt = load_cache(cache_path)
    samples, data_receipt = build_samples(DATASET_ROOT, 0)
    alignment = assert_sample_alignment(samples, sample_ids, 0)
    spr_test = [
        sample
        for sample in samples
        if sample.dataset == "sprsound" and sample.partition == "test"
    ]
    if len(spr_test) != 1429 or any(
        sample.targets or "raw_label" in sample.metadata for sample in spr_test
    ):
        raise RuntimeError("SPRSound label-free smoke gate failed")
    priors = prior_receipt(samples)
    condition_receipts = {}
    warning_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for condition in CONDITIONS:
            set_seed()
            model = GeneralSpecificModel(condition).to(device)
            count = parameter_count(model)
            if count != EXPECTED_PARAMETERS[condition]:
                raise RuntimeError(
                    f"{condition} parameters {count} != {EXPECTED_PARAMETERS[condition]}"
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            shapes = {}
            for task, spec in TASK_SPECS.items():
                indices = task_indices(samples, task, "subtrain")[:32]
                values = torch.from_numpy(embeddings[indices]).to(device)
                target = targets(samples, indices, task, device)
                logits = model(values, task)
                expected = (len(indices), len(spec["labels"]))
                if tuple(logits.shape) != expected:
                    raise RuntimeError(
                        f"{condition} {task} shape {tuple(logits.shape)} != {expected}"
                    )
                current = task_loss(
                    condition, task, logits, target, priors
                )
                if not torch.isfinite(current):
                    raise RuntimeError(f"non-finite smoke loss {condition} {task}")
                losses.append(current)
                shapes[task] = list(logits.shape)
            loss = torch.stack(losses).mean()
            loss.backward()
            finite_gradients = all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                raise RuntimeError(f"non-finite smoke gradient {condition}")
            optimizer.step()
            condition_receipts[condition] = {
                "parameters": count,
                "loss": float(loss.detach()),
                "finite_gradients": finite_gradients,
                "task_shapes": shapes,
                "loss_routes": {
                    task: loss_route(condition, task) for task in TASK_SPECS
                },
            }
        warning_messages = [str(item.message) for item in caught]
    schedule = dataset_balanced_batches(samples, 1)
    schedule_counts = Counter(dataset for dataset, _ in schedule)
    if len(set(schedule_counts.values())) != 1:
        raise RuntimeError("smoke dataset-balanced schedule failed")
    for task in TASK_SPECS:
        if task in ELIGIBLE_PRIOR_TASKS:
            continue
        if loss_route("d2_task_residual", task) != loss_route(
            "d2_task_residual_selective_prior", task
        ):
            raise RuntimeError(f"non-eligible loss route mismatch for {task}")
    payload = {
        "status": "four_dataset_general_specific_smoke_passed",
        "timestamp_unix": time.time(),
        "device": str(device),
        "cache": cache_receipt,
        "alignment": alignment,
        "data_receipt_status": data_receipt["status"],
        "conditions": condition_receipts,
        "dataset_balanced_updates": dict(schedule_counts),
        "spr_label_free_test_rows": len(spr_test),
        "warning_count": len(warning_messages),
        "warnings": warning_messages,
        "outer_test_metrics_evaluated": False,
    }
    write_json(result_root / "smoke_receipt.json", payload)
    return payload


def write_task_fold_results(result_root: Path) -> None:
    rows = []
    for fold in range(5):
        for condition in CONDITIONS:
            payload = json.loads(
                (
                    result_root
                    / f"fold_{fold}/{condition}/metrics.json"
                ).read_text()
            )
            for task, metrics in payload["test_metrics"].items():
                rows.append(
                    {
                        "fold": fold,
                        "condition": condition,
                        "dataset": TASK_SPECS[task]["dataset"],
                        "task": task,
                        "rows": metrics["rows"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_or_micro_f1": metrics.get(
                            "weighted_f1", metrics.get("micro_f1")
                        ),
                        "uar": metrics.get("uar"),
                        "native_score": metrics.get("native_score"),
                        "selection_epoch": payload["selection"]["epoch"],
                        "validation_mean_macro_f1": payload["selection"][
                            "validation_mean_macro_f1"
                        ],
                        "warning_count": payload["warning_count"],
                    }
                )
    with (result_root / "task_fold_results.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(
    result_root: Path, cache_receipt: dict[str, object]
) -> dict[str, object]:
    baseline_payload = baseline_aggregate(result_root, list(CONDITIONS))
    write_task_fold_results(result_root)
    warning_count = 0
    for fold in range(5):
        for condition in CONDITIONS:
            payload = json.loads(
                (
                    result_root
                    / f"fold_{fold}/{condition}/metrics.json"
                ).read_text()
            )
            warning_count += int(payload["warning_count"])
    manifest = {
        "status": "four_dataset_general_specific_full_complete",
        "protocol": (
            "codex/2026-07-28/"
            "model_design_four_dataset_general_specific_prereg_2026-07-28.md"
        ),
        "conditions": list(CONDITIONS),
        "models": 20,
        "folds": 5,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "selection": "validation unweighted mean macro F1 across six native heads",
        "outer_test_used_for_selection": False,
        "cache": cache_receipt,
        "eligible_prior_tasks": sorted(ELIGIBLE_PRIOR_TASKS),
        "parameters": EXPECTED_PARAMETERS,
        "warning_count": warning_count,
        "reporting": {
            "non_kauh": "mean/sample std over five KAUH-fold-conditioned models",
            "kauh": "five-fold patient-grouped aggregate OOF",
            "sprsound": "label-free prediction before terminal annotation join",
            "hf_missing_label": "not negative",
            "pooled_raw_score": False,
        },
        "baseline_aggregate_receipt": baseline_payload,
        "claim_boundary": [
            "frozen PAFA cached-feature target-supervised method development",
            "not clean zero-shot or source-only generalization",
            "not a universal imbalance loss",
            "no pooled cross-dataset raw Score",
            "no learned router or pathology-routing claim",
        ],
    }
    write_json(result_root / "run_manifest.json", manifest)
    return manifest


def run_full(
    cache_path: Path,
    result_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    smoke_path = result_root / "smoke_receipt.json"
    if not smoke_path.is_file():
        raise RuntimeError("smoke receipt missing")
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke["status"] != "four_dataset_general_specific_smoke_passed"
        or smoke["outer_test_metrics_evaluated"]
        or smoke["warning_count"] != 0
    ):
        raise RuntimeError("smoke gate not eligible for full")
    sample_ids, embeddings, cache_receipt = load_cache(cache_path)
    fold_receipts = []
    for fold in range(5):
        samples, _ = build_samples(DATASET_ROOT, fold)
        fold_receipts.append(assert_sample_alignment(samples, sample_ids, fold))
        for condition in CONDITIONS:
            output_dir = result_root / f"fold_{fold}" / condition
            if (
                not force
                and (output_dir / "metrics.json").is_file()
                and (output_dir / "best.pth").is_file()
            ):
                continue
            train_condition(
                condition,
                samples,
                embeddings,
                output_dir,
                device,
                cache_receipt,
            )
    write_json(
        result_root / "fold_alignment_receipt.json",
        {"folds": fold_receipts},
    )
    return aggregate_results(result_root, cache_receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["smoke", "full", "aggregate", "all"],
        default="all",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.phase in {"smoke", "all"}:
        run_smoke(args.cache, args.result_root, device)
        if args.phase == "smoke":
            return
    if args.phase in {"full", "all"}:
        run_full(args.cache, args.result_root, device, args.force)
        return
    _, _, cache_receipt = load_cache(args.cache)
    aggregate_results(args.result_root, cache_receipt)


if __name__ == "__main__":
    main()
