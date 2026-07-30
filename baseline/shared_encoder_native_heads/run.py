"""Audit, smoke, profile, and train the minimal shared-encoder baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .protocol import (
    EXPERIMENT_ID,
    ICBHI_LABELS,
    SPR_LABELS,
    TASK_LABELS,
    TRAINING_SEED,
    bootstrap_assets,
    build_model,
    classification_metrics,
    labels_for_rows,
    load_icbhi_rows,
    load_spr_rows,
    parameter_receipt,
    peak_rss_gib,
    preprocess_rows,
    protocol_receipt,
    routed_loss,
    set_seed,
    sha256_file,
    validate_roots,
    write_csv,
    write_json,
)


def row_id(row: dict[str, object]) -> str:
    return str(row.get("cycle_id", row.get("event_id")))


def select_by_label(
    rows: list[dict[str, object]],
    partition: str,
    label_key: str,
    labels: list[str],
) -> list[dict[str, object]]:
    selected = []
    for label in labels:
        selected.append(
            next(
                row
                for row in rows
                if row["partition"] == partition and str(row[label_key]) == label
            )
        )
    return selected


def select_available_labels(
    rows: list[dict[str, object]],
    partition: str,
    label_key: str,
    ordered_labels: list[str],
) -> list[dict[str, object]]:
    available = {
        str(row[label_key]) for row in rows if row["partition"] == partition
    }
    return select_by_label(
        rows,
        partition,
        label_key,
        [label for label in ordered_labels if label in available],
    )


def data_audit(
    dataset_root: Path, result_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    icbhi_rows, icbhi_receipt = load_icbhi_rows(dataset_root / "icbhi_2017")
    spr_rows, spr_receipt = load_spr_rows(dataset_root / "sprsound")
    receipt = {
        "status": "data_protocol_audit_passed",
        "protocol": protocol_receipt(),
        "datasets": {"icbhi_2017": icbhi_receipt, "sprsound_biocas2022": spr_receipt},
    }
    write_json(result_root / "protocol_and_data_receipt.json", receipt)
    return icbhi_rows, spr_rows, receipt


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)


def finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def training_step(
    model,
    optimizer,
    images: torch.Tensor,
    dataset: str,
    rows: list[dict[str, object]],
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = {
        key: value.to(device) for key, value in labels_for_rows(rows, dataset).items()
    }
    dataset_ids = [dataset] * len(rows)
    loss, detail = routed_loss(model, images.to(device), dataset_ids, targets)
    loss.backward()
    if not finite_gradients(model):
        raise RuntimeError("non-finite gradient")
    optimizer.step()
    return float(loss.detach()), detail


def predictions(model, images, task, dataset, device):
    model.eval()
    with torch.inference_mode():
        _, routed = model.routed_logits(images.to(device), [dataset] * len(images))
        _, logits = routed[task]
        probabilities = torch.softmax(logits, dim=1)
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(probabilities).all()
        or not torch.allclose(
            probabilities.sum(dim=1), torch.ones(len(probabilities), device=device), atol=1e-6
        )
    ):
        raise RuntimeError(f"invalid probabilities for {task}")
    return logits.cpu(), probabilities.cpu(), probabilities.argmax(dim=1).cpu().numpy()


def confusion_rows(labels: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "true/pred": label,
            **{prediction: int(value) for prediction, value in zip(labels, values)},
        }
        for label, values in zip(labels, matrix)
    ]


def save_resume_checkpoint(path, model, optimizer, step, audit_sha):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": audit_sha,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_smoke(
    dataset_root: Path,
    result_root: Path,
    cache_root: Path,
    assets: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    icbhi_rows, spr_rows, _ = data_audit(dataset_root, result_root)
    author_repo = Path(str(assets["author_repo"]))
    checkpoint = Path(str(assets["checkpoint"]))
    icbhi_train = select_by_label(
        icbhi_rows, "subtrain", "native_four_class_label", ICBHI_LABELS
    )
    icbhi_validation = select_by_label(
        icbhi_rows, "validation", "native_four_class_label", ICBHI_LABELS
    )
    spr_train = select_by_label(spr_rows, "subtrain", "raw_label", SPR_LABELS)
    spr_validation = select_available_labels(
        spr_rows, "validation", "raw_label", SPR_LABELS
    )
    spr_inter = [row for row in spr_rows if row["partition"] == "inter"][:2]
    spr_intra = [row for row in spr_rows if row["partition"] == "intra"][:2]
    started = time.perf_counter()
    batches = {
        "icbhi_train": preprocess_rows(icbhi_train, author_repo),
        "icbhi_validation": preprocess_rows(icbhi_validation, author_repo),
        "spr_train": preprocess_rows(spr_train, author_repo),
        "spr_validation": preprocess_rows(spr_validation, author_repo),
        "spr_inter_label_free": preprocess_rows(spr_inter, author_repo),
        "spr_intra_label_free": preprocess_rows(spr_intra, author_repo),
    }
    preprocessing_seconds = time.perf_counter() - started
    set_seed(TRAINING_SEED)
    model = build_model(author_repo, checkpoint, cache_root / "work/smoke", device)
    optimizer = optimizer_for(model)
    losses = []
    details = []
    for dataset, rows, images in (
        ("icbhi", icbhi_train, batches["icbhi_train"]),
        ("sprsound", spr_train, batches["spr_train"]),
    ):
        value, detail = training_step(model, optimizer, images, dataset, rows, device)
        losses.append(value)
        details.append(detail)

    # Explicit mixed-dataset routing check: one sample from each dataset.
    mixed_images = torch.cat([batches["icbhi_train"][:1], batches["spr_train"][:1]])
    mixed_targets = {
        "icbhi_flat4": labels_for_rows(icbhi_train[:1], "icbhi")["icbhi_flat4"].to(device),
        **{key: value.to(device) for key, value in labels_for_rows(spr_train[:1], "sprsound").items()},
    }
    optimizer.zero_grad(set_to_none=True)
    mixed_loss, mixed_detail = routed_loss(
        model, mixed_images.to(device), ["icbhi", "sprsound"], mixed_targets
    )
    mixed_loss.backward()
    if not finite_gradients(model):
        raise RuntimeError("mixed routing produced non-finite gradients")
    optimizer.step()
    losses.append(float(mixed_loss.detach()))
    details.append(mixed_detail)

    audit_sha = sha256_file(result_root / "protocol_and_data_receipt.json")
    checkpoint_path = cache_root / "smoke_resume/checkpoint.pth"
    save_resume_checkpoint(checkpoint_path, model, optimizer, 3, audit_sha)
    saved_model_state = deepcopy(model.state_dict())
    del model, optimizer
    resumed = torch.load(checkpoint_path, map_location=device)
    model = build_model(author_repo, checkpoint, cache_root / "work/resume", device)
    optimizer = optimizer_for(model)
    model.load_state_dict(resumed["model"], strict=True)
    optimizer.load_state_dict(resumed["optimizer"])
    if not all(torch.equal(value, model.state_dict()[key]) for key, value in saved_model_state.items()):
        raise RuntimeError("checkpoint reload tensor identity failed")
    resumed_loss, resumed_detail = training_step(
        model, optimizer, batches["icbhi_train"], "icbhi", icbhi_train, device
    )

    validation_receipts = {}
    for task, dataset, rows, images, labels in (
        (
            "icbhi_flat4",
            "icbhi",
            icbhi_validation,
            batches["icbhi_validation"],
            ICBHI_LABELS,
        ),
        ("spr_binary", "sprsound", spr_validation, batches["spr_validation"], TASK_LABELS["spr_binary"]),
        ("spr_seven", "sprsound", spr_validation, batches["spr_validation"], SPR_LABELS),
    ):
        _, probabilities, predicted = predictions(model, images, task, dataset, device)
        target = labels_for_rows(rows, dataset)[task].numpy()
        metrics, matrix = classification_metrics(target, predicted, labels)
        write_csv(result_root / "smoke" / f"{task}_confusion.csv", confusion_rows(labels, matrix))
        validation_receipts[task] = {
            "rows": len(rows),
            "confusion_total": int(matrix.sum()),
            "metrics": metrics,
            "probability_sums_valid": True,
            "predicted_indices": predicted.tolist(),
        }
    label_free_receipt = {}
    for partition in ("inter", "intra"):
        images = batches[f"spr_{partition}_label_free"]
        rows = spr_inter if partition == "inter" else spr_intra
        outputs = {}
        for task in ("spr_binary", "spr_seven"):
            logits, probabilities, predicted = predictions(
                model, images, task, "sprsound", device
            )
            outputs[task] = {
                "logits_finite": bool(torch.isfinite(logits).all()),
                "probabilities_finite": bool(torch.isfinite(probabilities).all()),
                "predicted_indices": predicted.tolist(),
            }
        label_free_receipt[partition] = {
            "ids": [row_id(row) for row in rows],
            "labels_loaded": False,
            "outputs": outputs,
        }
    receipt = {
        "status": "smoke_passed",
        "device": str(device),
        "preprocessing_seconds": preprocessing_seconds,
        "fbank_shapes": {key: list(value.shape) for key, value in batches.items()},
        "losses": losses,
        "loss_details": details,
        "losses_finite": all(math.isfinite(value) for value in losses),
        "gradients_finite": True,
        "mixed_head_routing": mixed_detail,
        "checkpoint_resume": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "saved_step": 3,
            "reload_tensor_identity": True,
            "resumed_step_loss": resumed_loss,
            "resumed_step_detail": resumed_detail,
        },
        "validation": validation_receipts,
        "label_free_test_forward": label_free_receipt,
        "parameters": parameter_receipt(model),
        "peak_rss_gib": peak_rss_gib(),
    }
    write_json(result_root / "smoke_receipt.json", receipt)
    return receipt


def profile_rows(
    icbhi_rows: list[dict[str, object]], spr_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    icbhi = [row for row in icbhi_rows if row["partition"] == "subtrain"]
    spr = [row for row in spr_rows if row["partition"] == "subtrain"]
    total = len(icbhi) + len(spr)
    icbhi_count = round(100 * len(icbhi) / total)
    return icbhi[:icbhi_count], spr[: 100 - icbhi_count]


def source_schedule(policy: str, batch_size: int) -> list[str]:
    icbhi_batches = math.ceil(3055 / batch_size)
    spr_batches = math.ceil(5219 / batch_size)
    if policy == "source_proportional":
        return ["icbhi"] * icbhi_batches + ["sprsound"] * spr_batches
    if policy == "dataset_balanced":
        larger = max(icbhi_batches, spr_batches)
        return ["icbhi"] * larger + ["sprsound"] * larger
    raise ValueError(policy)


def run_profile(
    dataset_root: Path,
    result_root: Path,
    cache_root: Path,
    assets: dict[str, object],
    device: torch.device,
    steps: int,
    batch_size: int,
    sampling_policy: str,
) -> dict[str, object]:
    if not (result_root / "smoke_receipt.json").is_file():
        raise RuntimeError("profile requires a verified smoke receipt")
    icbhi_rows, spr_rows, _ = data_audit(dataset_root, result_root)
    author_repo = Path(str(assets["author_repo"]))
    checkpoint = Path(str(assets["checkpoint"]))
    selected_icbhi, selected_spr = profile_rows(icbhi_rows, spr_rows)
    prep_started = time.perf_counter()
    icbhi_images = preprocess_rows(selected_icbhi, author_repo)
    spr_images = preprocess_rows(selected_spr, author_repo)
    prep_seconds = time.perf_counter() - prep_started
    set_seed(TRAINING_SEED)
    model = build_model(author_repo, checkpoint, cache_root / "work/profile", device)
    optimizer = optimizer_for(model)
    sources = source_schedule(sampling_policy, batch_size)
    random.Random(TRAINING_SEED).shuffle(sources)
    step_times = []
    losses = []
    for step in range(steps):
        dataset = sources[step % len(sources)]
        rows = selected_icbhi if dataset == "icbhi" else selected_spr
        images = icbhi_images if dataset == "icbhi" else spr_images
        start = (step * batch_size) % len(rows)
        indices = [(start + offset) % len(rows) for offset in range(batch_size)]
        batch_rows = [rows[index] for index in indices]
        batch_images = images[indices]
        before = time.perf_counter()
        loss, _ = training_step(model, optimizer, batch_images, dataset, batch_rows, device)
        step_times.append(time.perf_counter() - before)
        losses.append(loss)
    mean_step = float(np.mean(step_times))
    updates_per_epoch = len(source_schedule(sampling_policy, batch_size))
    projected_training_seconds = mean_step * updates_per_epoch * 50
    projected_preprocessing_seconds = prep_seconds / 100 * (6898 + 6656 + 1429 + 1004)
    projected_total = projected_training_seconds + projected_preprocessing_seconds
    cache_bytes = (6898 + 6656 + 1429 + 1004) * 1 * 798 * 128 * 4
    receipt = {
        "status": "profile_passed_server_required"
        if projected_total > 90 * 60 or device.type == "cpu"
        else "profile_passed_local_full_allowed",
        "device": str(device),
        "profile_steps": steps,
        "profile_rows": {"icbhi": len(selected_icbhi), "sprsound": len(selected_spr)},
        "batch_size": batch_size,
        "sampling_policy": sampling_policy,
        "preprocessing_100_events_seconds": prep_seconds,
        "step_seconds": {
            "mean": mean_step,
            "median": float(np.median(step_times)),
            "min": float(np.min(step_times)),
            "max": float(np.max(step_times)),
        },
        "losses_finite": all(math.isfinite(value) for value in losses),
        "gradients_finite": True,
        "updates_per_epoch": updates_per_epoch,
        "max_epochs": 50,
        "projected_cpu_preprocessing_seconds": projected_preprocessing_seconds,
        "projected_cpu_training_seconds": projected_training_seconds,
        "projected_cpu_total_seconds": projected_total,
        "local_90_minute_gate_passed": projected_total <= 90 * 60 and device.type == "cpu",
        "peak_rss_gib": peak_rss_gib(),
        "estimated_fbank_cache_gib": cache_bytes / 1024**3,
        "l40_planning_evidence": {
            "prior_patch_mix_two_forward_peak_vram_gib": 18.614,
            "shared_baseline_expectation": (
                "one AST forward/backward per update plus three small heads; expected not to "
                "exceed the prior two-forward Patch-Mix VRAM, but a server 100-step profile is authoritative"
            ),
            "runtime_estimate": (
                "roughly 2.5-4 hours for 50 epochs based on the accepted Patch-Mix L40 run; "
                "must be replaced by this package's L40 profile"
            ),
        },
    }
    write_json(result_root / "profile_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "bootstrap", "smoke", "profile"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--result-root", type=Path, default=Path("result/icbhi_sprsound_shared_encoder_native_heads")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(".cache/icbhi_sprsound_shared_encoder_native_heads")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sampling-policy",
        choices=["source_proportional", "dataset_balanced"],
        default="source_proportional",
    )
    args = parser.parse_args()
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
    result_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "runtime/numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "runtime/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "runtime/xdg"))
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase == "audit":
        _, _, receipt = data_audit(args.dataset_root, result_root)
        print(json.dumps(receipt["datasets"], indent=2, sort_keys=True))
        return
    assets = bootstrap_assets(cache_root)
    write_json(result_root / "asset_receipt.json", assets)
    if args.phase == "bootstrap":
        print(json.dumps(assets, indent=2, sort_keys=True))
        return
    if args.phase == "smoke":
        receipt = run_smoke(args.dataset_root, result_root, cache_root, assets, device)
    else:
        receipt = run_profile(
            args.dataset_root,
            result_root,
            cache_root,
            assets,
            device,
            args.profile_steps,
            args.batch_size,
            args.sampling_policy,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
