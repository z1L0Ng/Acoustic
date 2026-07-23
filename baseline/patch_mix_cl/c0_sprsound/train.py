"""Patch-Mix C0 smoke/full training with train-only checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .common import (
    TASK_LABELS,
    CachedEventDataset,
    build_model,
    classification_metrics,
    contrastive_loss,
    ema_update,
    load_cache,
    protocol_args,
    read_csv,
    set_seed,
    task_label,
    write_confusion,
    write_csv,
    write_json,
)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def learning_rate(epoch: int, epochs: int, base: float = 5e-5) -> float:
    eta_min = base * (0.1**3)
    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * epoch / epochs)) / 2


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def evaluate(loader, model, classifier, labels: list[str], device: torch.device):
    model.eval()
    classifier.eval()
    rows: list[dict[str, object]] = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in loader:
            images, targets, event_ids = batch
            images = images.to(device)
            targets = targets.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = classifier(model(images))
                loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            losses.append(float(loss.item()))
            for event_id, target, logit, probability in zip(
                event_ids, targets.cpu().numpy(), logits.cpu().numpy(), probabilities.cpu().numpy()
            ):
                pred = int(np.argmax(probability))
                record: dict[str, object] = {
                    "event_id": event_id,
                    "true_index": int(target),
                    "true_label": labels[int(target)],
                    "pred_index": pred,
                    "pred_label": labels[pred],
                }
                for index, label in enumerate(labels):
                    record[f"logit_{label}"] = float(logit[index])
                    record[f"prob_{label}"] = float(probability[index])
                rows.append(record)
    y_true = np.asarray([row["true_index"] for row in rows], dtype=np.int64)
    y_pred = np.asarray([row["pred_index"] for row in rows], dtype=np.int64)
    metrics, matrix = classification_metrics(y_true, y_pred, labels)
    metrics["loss"] = float(np.mean(losses))
    return metrics, matrix, rows


def infer_label_free(loader, model, classifier, labels: list[str], device: torch.device):
    """Return logits without opening target labels; used by smoke and final pre-score inference."""
    model.eval()
    classifier.eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, event_ids in loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = classifier(model(images))
            probabilities = torch.softmax(logits, dim=1)
            for event_id, logit, probability in zip(
                event_ids, logits.cpu().numpy(), probabilities.cpu().numpy()
            ):
                pred = int(np.argmax(probability))
                record: dict[str, object] = {
                    "event_id": event_id,
                    "pred_index": pred,
                    "pred_label": labels[pred],
                }
                for index, label in enumerate(labels):
                    record[f"logit_{label}"] = float(logit[index])
                    record[f"prob_{label}"] = float(probability[index])
                rows.append(record)
    return rows


def load_inter_scoring_labels(inter_manifest: Path, task: str) -> dict[str, dict[str, object]]:
    """Open inter labels only after final label-free logits exist."""
    labels: dict[str, dict[str, object]] = {}
    for row in read_csv(inter_manifest):
        payload = json.loads(Path(row["annotation_path"]).read_text())
        raw = str(payload["event_annotation"][int(row["event_index"])]["type"])
        mapped = task_label(raw, task)
        labels[row["event_id"]] = {
            "raw_label": raw,
            "task_label": mapped or "",
            "mapping_status": "included" if mapped is not None else "excluded_unsupported",
        }
    return labels


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_LABELS), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-name", default="full")
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    author_repo = (args.author_repo or result_root / "source" / "repo").resolve()
    bootstrap = json.loads((result_root / "receipts" / "bootstrap.json").read_text())
    checkpoint = (args.checkpoint_path or Path(bootstrap["checkpoint_path"])).resolve()
    task_dir = result_root / args.mode / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    labels = TASK_LABELS[args.task]
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    cache_dir = cache_root / "c0_sprsound" / args.cache_name
    train_cache, _, train_index = load_cache(cache_dir / "train")
    inter_cache, _, inter_index = load_cache(cache_dir / "inter")
    all_task_rows = read_csv(result_root / "data" / f"train_{args.task}.csv")
    available = set(train_index)
    subtrain = [row for row in all_task_rows if row["inner_split"] == "subtrain" and row["event_id"] in available]
    validation = [row for row in all_task_rows if row["inner_split"] == "validation" and row["event_id"] in available]
    inter_rows = [
        row
        for row in read_csv(result_root / "data" / "inter_events_label_free.csv")
        if row["event_id"] in inter_index
    ]
    if not subtrain or not validation or not inter_rows:
        raise RuntimeError("cache must include subtrain, validation, and inter rows")

    sys.path.insert(0, str(author_repo))
    from util.augmentation import SpecAugment

    augment = SpecAugment(protocol_args())
    train_dataset = CachedEventDataset(subtrain, train_cache, train_index, labels, augment)
    validation_dataset = CachedEventDataset(validation, train_cache, train_index, labels)
    inter_dataset = CachedEventDataset(inter_rows, inter_cache, inter_index, None)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    inter_loader = torch.utils.data.DataLoader(
        inter_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    if len(train_loader) == 0:
        raise RuntimeError("subtrain smoke subset is smaller than one complete batch")

    work_dir = task_dir / "work"
    pretrained_dir = work_dir / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    expected = pretrained_dir / "audioset_10_10_0.4593.pth"
    if expected.exists() or expected.is_symlink():
        expected.unlink()
    expected.symlink_to(checkpoint)
    previous_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        model, classifier, projector = build_model(author_repo, checkpoint, len(labels), device)
    finally:
        os.chdir(previous_cwd)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
        lr=5e-5,
        weight_decay=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 1 if args.mode == "smoke" else 50
    start_epoch = 1
    best_score = -math.inf
    best_epoch = None
    if args.resume:
        state = torch.load(args.resume.resolve(), map_location=device)
        if state["task"] != args.task or state["seed"] != args.seed:
            raise RuntimeError("resume protocol mismatch")
        model.load_state_dict(state["model"])
        classifier.load_state_dict(state["classifier"])
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_validation_score"])
        best_epoch = state["best_epoch"]
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_random_states"])
        generator.set_state(state["data_generator_state"])

    started = time.perf_counter()
    history: list[dict[str, object]] = []
    finite_gradient = True
    for epoch in range(start_epoch, epochs + 1):
        lr = learning_rate(epoch, 50)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        classifier.train()
        projector.train()
        losses = []
        for step, (images, targets, _) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            previous = [snapshot(module) for module in (model, classifier, projector)]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                features = model(images)
                logits = classifier(features)
                ce = criterion(logits, targets)
                mixed, _, _, lam, index = model(images, targets, patch_mix=True)
                projected = projector(mixed)
                con = contrastive_loss(features.detach(), projected, lam, index)
                loss = ce + con
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradient = finite_gradient and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for module in (model, classifier, projector)
                for parameter in module.parameters()
            )
            if not finite_gradient:
                raise RuntimeError("non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            for module, state in zip((model, classifier, projector), previous):
                ema_update(module, state, 0.5)
            losses.append(float(loss.item()))
            if args.mode == "smoke" and step >= args.max_steps:
                break

        validation_metrics, validation_matrix, validation_predictions = evaluate(
            validation_loader, model, classifier, labels, device
        )
        score = float(validation_metrics["icbhi_score"])
        selection_eligible = args.mode == "smoke" or float(validation_metrics["sensitivity"]) > 5
        if score > best_score and selection_eligible:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                task_dir / "checkpoints" / "best_validation.pth",
                {
                    "protocol": "c0_patch_mix_sprsound_target_native_v1",
                    "task": args.task,
                    "seed": args.seed,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "projector": projector.state_dict(),
                    "validation_metrics": validation_metrics,
                },
            )
        state = {
            "protocol": "c0_patch_mix_sprsound_target_native_v1",
            "task": args.task,
            "seed": args.seed,
            "epoch": epoch,
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_score": best_score,
            "best_epoch": best_epoch,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "data_generator_state": generator.get_state(),
        }
        save_checkpoint(task_dir / "checkpoints" / "last.pth", state)
        write_csv(task_dir / f"validation_predictions_epoch_{epoch:03d}.csv", validation_predictions)
        write_confusion(task_dir / f"validation_confusion_epoch_{epoch:03d}.csv", labels, validation_matrix)
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        write_csv(task_dir / "training_history.csv", history)

    best_path = task_dir / "checkpoints" / "best_validation.pth"
    if not best_path.is_file():
        raise RuntimeError("no checkpoint met the author-derived validation sensitivity >5 gate")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    classifier.load_state_dict(best["classifier"])
    label_free_predictions = infer_label_free(inter_loader, model, classifier, labels, device)
    write_csv(task_dir / "inter_predictions_label_free.csv", label_free_predictions)
    runtime = time.perf_counter() - started

    receipt: dict[str, object] = {
        "status": "smoke_wiring_verified" if args.mode == "smoke" else "full_training_and_final_inter_evaluation_complete",
        "claim": "target-native reference; not zero-shot, pooled training, adaptation, or paper reproduction",
        "task": args.task,
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_inner_validation_score": best_score,
        "subtrain_rows": len(subtrain),
        "validation_rows": len(validation),
        "inter_forward_rows": len(label_free_predictions),
        "inter_metrics_used_for_selection": False,
        "finite_loss_and_gradient": finite_gradient,
        "runtime_seconds": runtime,
        "checkpoint_selection": "inner-validation ICBHI Score only",
        "checkpoint_eligibility": "author-derived sensitivity >5 gate; smoke permits first checkpoint for wiring only",
    }
    if args.mode == "smoke":
        receipt["inter_label_access"] = "none"
        receipt["inter_metric_computation"] = "forbidden_and_not_performed"
    else:
        scoring_labels = load_inter_scoring_labels(
            result_root / "data" / "inter_events_label_free.csv", args.task
        )
        scored = []
        for row in label_free_predictions:
            label_row = scoring_labels[str(row["event_id"])]
            merged = {**row, **label_row}
            if label_row["mapping_status"] == "included":
                merged["true_index"] = labels.index(str(label_row["task_label"]))
            else:
                merged["true_index"] = ""
            scored.append(merged)
        write_csv(task_dir / "inter_predictions.csv", scored)
        included = [row for row in scored if row["mapping_status"] == "included"]
        y_true = np.asarray([int(row["true_index"]) for row in included], dtype=np.int64)
        y_pred = np.asarray([int(row["pred_index"]) for row in included], dtype=np.int64)
        metrics, matrix = classification_metrics(y_true, y_pred, labels)
        write_json(task_dir / "inter_metrics.json", metrics)
        write_confusion(task_dir / "inter_confusion.csv", labels, matrix)
        receipt["inter_scored_rows"] = len(included)
        receipt["inter_excluded_rows"] = len(scored) - len(included)
        receipt["inter_metrics"] = metrics
        receipt["inter_label_access"] = "after label-free logits and best checkpoint were fixed"
    write_json(task_dir / "run_receipt.json", receipt)
    print(
        f"c0_{args.mode}_ok task={args.task} train={len(subtrain)} val={len(validation)} "
        f"inter_forward={len(label_free_predictions)} best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    main()
