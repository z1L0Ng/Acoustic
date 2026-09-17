"""Train target-supervised heads on an existing frozen method embedding cache."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import PAFA_METHOD_ENCODER, SEEDS, TASKS
from .models import FrozenTaskAdapter, TASK_DIMS, task_loss


MAX_EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target == 1
    negative = target == 0
    comparisons = score[positive, None] - score[negative][None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def _confusion(target: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (target.astype(int), prediction.astype(int)), 1)
    return matrix


def _classification_metrics(
    target: np.ndarray, prediction: np.ndarray, labels: tuple[str, ...]
) -> dict[str, object]:
    matrix = _confusion(target, prediction, len(labels))
    recall = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(len(labels), dtype=np.float64),
        where=matrix.sum(axis=1) > 0,
    )
    precision = np.divide(
        np.diag(matrix),
        matrix.sum(axis=0),
        out=np.zeros(len(labels), dtype=np.float64),
        where=matrix.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    specificity = float(recall[0])
    abnormal_total = int(matrix[1:].sum())
    sensitivity = (
        float(np.trace(matrix[1:, 1:]) / abnormal_total) if abnormal_total else 0.0
    )
    average = (specificity + sensitivity) / 2.0
    harmonic = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    return {
        "rows": int(len(target)),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "average_score": average,
        "harmonic_score": harmonic,
        "macro_f1": float(f1.mean()),
        "uar": float(recall.mean()),
        "confusion": matrix.tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(matrix[index].sum()),
            }
            for index, label in enumerate(labels)
        },
    }


def metric_value(task: str, target: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, object]]:
    if task == "hf_cas":
        value = _binary_auc(target.astype(int), probability[:, 0])
        return value, {"cas_auroc": value, "support": int(len(target)), "positive": int(target.sum())}
    prediction = probability.argmax(axis=1)
    if task == "icbhi_flat4":
        report = _classification_metrics(
            target.astype(int), prediction, ("normal", "crackle", "wheeze", "both")
        )
        return float(report["average_score"]), report
    if task == "spr_binary":
        report = _classification_metrics(
            target.astype(int), prediction, ("normal", "adventitious")
        )
        report["official_score"] = (float(report["average_score"]) + float(report["harmonic_score"])) / 2.0
        return float(report["official_score"]), report
    matrix = _confusion(target, prediction, 2)
    recall = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(2, dtype=np.float64),
        where=matrix.sum(axis=1) > 0,
    )
    value = float(recall.mean())
    return value, {
        "patient_balanced_accuracy": value,
        "support": int(len(target)),
        "confusion": matrix.tolist(),
    }


def predict(model: FrozenTaskAdapter, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            probability = (
                torch.sigmoid(logits).reshape(-1, 1)
                if logits.shape[1] == 1
                else torch.softmax(logits, dim=1)
            )
            output.append(probability.cpu().numpy())
    return np.concatenate(output, axis=0)


def run(
    *,
    repo_root: Path,
    task: str,
    seed: int,
    kauh_fold: int,
    device_name: str,
) -> dict[str, object]:
    if task not in TASKS or seed not in SEEDS:
        raise ValueError("task or seed outside frozen protocol")
    if task != "kauh_binary" and kauh_fold != 0:
        raise ValueError("non-KAUH tasks use fold 0")
    from baseline.four_dataset_frozen_encoder.data import (
        build_samples,
        load_terminal_spr_test_targets,
    )
    from .data import load_aligned_embeddings, task_arrays

    output = (
        repo_root
        / "result/frozen_method_baselines/pafa_trained_encoder"
        / task
        / f"seed_{seed}"
        / (f"fold_{kauh_fold}" if task == "kauh_binary" else "run")
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    set_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    samples, _ = build_samples(
        repo_root / "dataset/raw", kauh_outer_fold=kauh_fold, include_checksums=False
    )
    cache_path = repo_root / str(PAFA_METHOD_ENCODER.embedding_cache)
    embeddings = load_aligned_embeddings(cache_path, samples)
    terminal_spr_targets = (
        load_terminal_spr_test_targets(samples, include_checksums=False)
        if task == "spr_binary"
        else None
    )
    train = task_arrays(samples, embeddings, task=task, partition="subtrain")
    validation = task_arrays(samples, embeddings, task=task, partition="validation")
    test = task_arrays(
        samples,
        embeddings,
        task=task,
        partition="test",
        terminal_spr_targets=terminal_spr_targets,
    )
    model = FrozenTaskAdapter(TASK_DIMS[task]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(seed)
    best_value = -np.inf
    best_epoch = 0
    best_state = None
    last_state = None
    no_improvement = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train.targets))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            values = torch.from_numpy(train.embeddings[indices]).to(device)
            target = torch.from_numpy(train.targets[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(model(values), target, task)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite {task} loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probability = predict(model, validation.embeddings, BATCH_SIZE, device)
        value, val_metrics = metric_value(task, validation.targets, val_probability)
        improved = value > best_value
        no_improvement = 0 if improved else no_improvement + 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_selection_value": value,
                "validation_metrics": val_metrics,
            }
        )
        last_state = copy.deepcopy(model.state_dict())
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if no_improvement >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no target-head checkpoint selected")
    if last_state is None:
        raise RuntimeError("no target-head epoch completed")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    model.load_state_dict(best_state)
    test_probability = predict(model, test.embeddings, BATCH_SIZE, device)
    _, test_metrics = metric_value(task, test.targets, test_probability)
    np.savez_compressed(
        output / "selected_predictions.npz",
        sample_ids=test.sample_ids,
        group_ids=test.group_ids,
        targets=test.targets,
        probabilities=test_probability,
    )
    torch.save(
        {"model": model.state_dict(), "selected_epoch": best_epoch},
        output / "best.pth",
    )
    torch.save(
        {"model": last_state, "completed_epoch": history[-1]["epoch"]},
        output / "last.pth",
    )
    payload = {
        "status": "complete_target_supervised_frozen_representation_run",
        "method": PAFA_METHOD_ENCODER.method_id,
        "task": task,
        "seed": seed,
        "kauh_fold": kauh_fold if task == "kauh_binary" else None,
        "selected_epoch": best_epoch,
        "selection_value": best_value,
        "history": history,
        "test_metrics": test_metrics,
        "encoder_frozen": True,
        "encoder_checkpoint": PAFA_METHOD_ENCODER.encoder_checkpoint,
        "embedding_cache": PAFA_METHOD_ENCODER.embedding_cache,
        "target_supervised": True,
        "training_config": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": PATIENCE,
            "tie_break": "earlier_epoch",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "run_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--kauh-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "execution_started": False,
                    "method": PAFA_METHOD_ENCODER.to_dict(),
                    "task": args.task,
                    "seed": args.seed,
                    "kauh_fold": args.kauh_fold,
                    "device": args.device,
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            run(
                repo_root=args.repo_root,
                task=args.task,
                seed=args.seed,
                kauh_fold=args.kauh_fold,
                device_name=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
