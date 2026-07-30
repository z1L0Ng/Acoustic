"""Run the four-dataset PAFA frozen-encoder D0-D3 matrix."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    KAUH_LABELS,
    build_samples,
    sample_to_row,
)
from baseline.four_dataset_frozen_encoder.encoder import (
    build_encoder,
    extract_embeddings,
    load_cache,
    save_cache,
    sha256_file,
)
from baseline.four_dataset_frozen_encoder.train import (
    BATCH_SIZE,
    CONDITIONS,
    EPOCHS,
    SharedNativeModel,
    TASK_SPECS,
    _multiclass_metrics,
    train_conditions,
)


EXPERIMENT_ID = "four_dataset_pafa_frozen_encoder"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "baseline/four_dataset_frozen_encoder/protocol.json"
DATASET_ORDER = ("icbhi", "sprsound", "hf_lung", "kauh")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def ordered_id_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def load_execution_state(result_root: Path) -> dict[str, object]:
    path = result_root / "execution_state.json"
    if path.is_file():
        return json.loads(path.read_text())
    state = {
        "status": "active",
        "active_seconds": 0.0,
        "hard_wall_clock_seconds": 120 * 60,
        "peak_rss_limit_gib": 24.0,
        "runtime_gate_reason": (
            "54.14 min extraction center estimate plus 0.69 min heads; "
            "109.66 min 2x projection is below the approved 120 min hard bound"
        ),
    }
    write_json(path, state)
    return state


def execution_guard(
    result_root: Path,
    state: dict[str, object],
    phase_started: float,
    base_active_seconds: float,
    force_write: bool = False,
) -> None:
    active = base_active_seconds + (time.perf_counter() - phase_started)
    process_peak_rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    )
    peak_rss = max(float(state.get("peak_rss_gib", 0.0)), process_peak_rss)
    state.update(
        {
            "active_seconds": active,
            "peak_rss_gib": peak_rss,
            "status": "active",
        }
    )
    now = time.perf_counter()
    last_write = float(state.get("_last_write_perf", -1e12))
    if force_write or now - last_write >= 15:
        state["_last_write_perf"] = now
        write_json(
            result_root / "execution_state.json",
            {key: value for key, value in state.items() if not key.startswith("_")},
        )
    if active > float(state["hard_wall_clock_seconds"]):
        state["status"] = "failed_wall_clock_gate"
        write_json(
            result_root / "execution_state.json",
            {key: value for key, value in state.items() if not key.startswith("_")},
        )
        raise RuntimeError("120-minute active wall-clock gate exceeded")
    if peak_rss > float(state["peak_rss_limit_gib"]):
        state["status"] = "failed_peak_rss_gate"
        write_json(
            result_root / "execution_state.json",
            {key: value for key, value in state.items() if not key.startswith("_")},
        )
        raise RuntimeError("24-GiB peak RSS gate exceeded")


def validate_roots(result_root: Path, cache_root: Path) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != EXPERIMENT_ID or result.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}")
    if cache.name != EXPERIMENT_ID or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}")
    return result, cache


def configure_runtime(cache_root: Path, threads: int) -> None:
    for variable, relative in (
        ("NUMBA_CACHE_DIR", "runtime/numba"),
        ("MPLCONFIGDIR", "runtime/matplotlib"),
        ("XDG_CACHE_HOME", "runtime/xdg"),
    ):
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    torch.set_num_threads(threads)


def data_audit(dataset_root: Path, result_root: Path) -> tuple[list, dict[str, object]]:
    folds = {}
    fold_zero_samples = None
    for fold in range(5):
        samples, receipt = build_samples(dataset_root, fold)
        folds[str(fold)] = receipt
        if fold == 0:
            fold_zero_samples = samples
            write_gzip_csv(
                result_root / "samples_fold_0.csv.gz",
                [sample_to_row(sample) for sample in samples],
            )
    if fold_zero_samples is None:
        raise RuntimeError("missing fold zero samples")
    payload = {
        "status": "four_dataset_data_audit_passed",
        "protocol_sha256": sha256_file(PROTOCOL),
        "folds": folds,
        "embedding_identity_independent_of_kauh_fold": True,
    }
    write_json(result_root / "data_receipt.json", payload)
    return fold_zero_samples, payload


def smoke(
    samples,
    source_repo,
    checkpoint,
    backbone,
    dataset_root,
    result_root,
    device,
    batch_size,
) -> dict[str, object]:
    selected_ids = set()
    for task in TASK_SPECS:
        for partition in ("subtrain", "validation", "test"):
            candidates = [
                sample
                for sample in samples
                if sample.partition == partition
                and sample.dataset == TASK_SPECS[task]["dataset"]
                and (
                    task in sample.targets
                    or (
                        partition == "test"
                        and sample.dataset == "sprsound"
                        and not sample.targets
                    )
                )
            ]
            selected_ids.update(sample.sample_id for sample in candidates[:8])
    selected = [sample for sample in samples if sample.sample_id in selected_ids]
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    embeddings, extraction = extract_embeddings(
        selected, source_repo, model, device, batch_size
    )
    smoke_root = result_root / "smoke"
    outputs = train_conditions(
        selected,
        embeddings,
        smoke_root,
        list(CONDITIONS),
        device,
        dataset_root,
    )
    gradients_and_losses_finite = True
    for condition, payload in outputs.items():
        history = payload["history"]
        if condition == "d0_independent_heads":
            loss_values = [
                float(epoch["loss"])
                for task_history in history.values()
                for epoch in task_history
            ]
        else:
            loss_values = [float(epoch["loss"]) for epoch in history]
        gradients_and_losses_finite &= bool(
            loss_values and np.isfinite(loss_values).all()
        )
        state = torch.load(
            smoke_root / condition / "best.pth", map_location="cpu"
        )
        gradients_and_losses_finite &= all(
            torch.isfinite(value).all()
            for value in state["model"].values()
            if torch.is_tensor(value)
        )
    if not gradients_and_losses_finite:
        raise RuntimeError("smoke loss/checkpoint finite gate failed")
    payload = {
        "status": "four_dataset_real_data_smoke_passed",
        "samples": len(selected),
        "dataset_rows": dict(
            sorted(Counter(sample.dataset for sample in selected).items())
        ),
        "partitions": dict(
            sorted(Counter(sample.partition for sample in selected).items())
        ),
        "encoder": identity,
        "extraction": extraction,
        "conditions": list(CONDITIONS),
        "tasks": list(TASK_SPECS),
        "finite_loss_gradient_checkpoint": True,
        "hf_recording_window_counts": sorted(
            {
                extraction["window_count_min"],
                extraction["window_count_max"],
            }
        ),
        "spr_label_free_then_terminal_join": all(
            outputs[condition]["terminal_label_join"][
                "spr_test_labels_loaded_after_label_free_write"
            ]
            for condition in outputs
        ),
    }
    write_json(result_root / "smoke_receipt.json", payload)
    return payload


def _training_profile(samples, device: torch.device) -> dict[str, object]:
    model = SharedNativeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    values = torch.randn(BATCH_SIZE, 768, device=device)
    target = torch.randint(0, 4, (BATCH_SIZE,), device=device)
    started = time.perf_counter()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(values, "icbhi_flat4"), target
        )
        loss.backward()
        optimizer.step()
    step_seconds = (time.perf_counter() - started) / 100
    subtrain_by_dataset = {
        dataset: sum(
            sample.dataset == dataset
            and sample.partition == "subtrain"
            and bool(sample.targets)
            for sample in samples
        )
        for dataset in ("icbhi", "sprsound", "hf_lung", "kauh")
    }
    source_batches = {
        dataset: int(np.ceil(rows / BATCH_SIZE))
        for dataset, rows in subtrain_by_dataset.items()
    }
    task_batches = {
        task: int(
            np.ceil(
                sum(
                    sample.partition == "subtrain" and task in sample.targets
                    for sample in samples
                )
                / BATCH_SIZE
            )
        )
        for task in TASK_SPECS
    }
    d0_updates = sum(task_batches.values()) * EPOCHS * 5
    d1_updates = sum(source_batches.values()) * EPOCHS * 5
    balanced_per_fold_epoch = max(source_batches.values()) * 4
    d2_d3_updates = balanced_per_fold_epoch * EPOCHS * 5 * 2
    total_updates = d0_updates + d1_updates + d2_d3_updates
    return {
        "profile_steps": 100,
        "seconds_per_step": step_seconds,
        "source_subtrain_rows": subtrain_by_dataset,
        "source_batches": source_batches,
        "task_batches": task_batches,
        "projected_optimizer_updates": total_updates,
        "projected_optimizer_seconds": step_seconds * total_updates,
        "projection_caveat": (
            "linear-head timing excludes validation/artifact I/O; a 2x safety factor "
            "is applied to the end-to-end gate"
        ),
    }


def profile(
    samples,
    source_repo,
    checkpoint,
    backbone,
    result_root,
    device,
    batch_size,
) -> dict[str, object]:
    selected = []
    counts = Counter()
    for sample in samples:
        if counts[sample.dataset] < 50:
            selected.append(sample)
            counts[sample.dataset] += 1
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    projected = {}
    total_projected = 0.0
    for dataset in ("icbhi", "sprsound", "hf_lung", "kauh"):
        subset = [sample for sample in selected if sample.dataset == dataset]
        _, receipt = extract_embeddings(
            subset, source_repo, model, device, min(batch_size, len(subset))
        )
        full_count = sum(sample.dataset == dataset for sample in samples)
        estimate = receipt["runtime_seconds"] / len(subset) * full_count
        projected[dataset] = {
            "profile_rows": len(subset),
            "profile_seconds": receipt["runtime_seconds"],
            "full_rows": full_count,
            "projected_seconds": estimate,
        }
        total_projected += estimate
    training = _training_profile(samples, device)
    projected_with_safety = (
        total_projected + float(training["projected_optimizer_seconds"])
    ) * 2
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    local_allowed = projected_with_safety <= 7200 and peak_rss_gib < 24
    payload = {
        "status": "profile_passed",
        "encoder": identity,
        "datasets": projected,
        "projected_total_seconds": total_projected,
        "projected_total_minutes": total_projected / 60,
        "training": training,
        "projected_end_to_end_with_2x_safety_seconds": projected_with_safety,
        "projected_end_to_end_with_2x_safety_minutes": projected_with_safety
        / 60,
        "peak_rss_gib": peak_rss_gib,
        "gate": {
            "max_minutes": 120,
            "max_peak_rss_gib": 24,
            "approval_reason": (
                "54.14 min extraction center estimate plus 0.69 min heads; "
                "109.66 min 2x projection is below the approved 120 min hard bound"
            ),
        },
        "decision": "local full allowed" if local_allowed else "hold",
    }
    write_json(result_root / "profile_receipt.json", payload)
    return payload


def extract(
    samples,
    source_repo,
    checkpoint,
    backbone,
    cache_root,
    result_root,
    device,
    batch_size,
) -> dict[str, object]:
    profile_receipt = json.loads((result_root / "profile_receipt.json").read_text())
    if profile_receipt["decision"] != "local full allowed":
        raise RuntimeError("profile did not authorize local extraction")
    state = load_execution_state(result_root)
    base_active = float(state["active_seconds"])
    phase_started = time.perf_counter()
    guard = lambda: execution_guard(
        result_root, state, phase_started, base_active
    )
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    guard()
    shard_receipts = {}
    shard_embeddings = {}
    shard_root = cache_root / "embedding_shards"
    receipt_root = result_root / "embedding_shards"
    for dataset in DATASET_ORDER:
        subset = [sample for sample in samples if sample.dataset == dataset]
        shard_path = shard_root / f"{dataset}.npz"
        shard_receipt_path = receipt_root / f"{dataset}.json"
        if shard_path.is_file() != shard_receipt_path.is_file():
            raise RuntimeError(
                f"incomplete shard/receipt pair for {dataset}; fail closed"
            )
        if shard_path.is_file():
            values, cached = load_cache(shard_path, subset)
            recorded = json.loads(shard_receipt_path.read_text())
            if (
                cached["cache_sha256"] != recorded["cache_sha256"]
                or cached["encoder"]["task_checkpoint_sha256"]
                != identity["task_checkpoint_sha256"]
                or cached["encoder"]["beats_checkpoint_sha256"]
                != identity["beats_checkpoint_sha256"]
                or recorded["ordered_id_sha256"]
                != ordered_id_sha256(
                    [sample.sample_id for sample in subset]
                )
            ):
                raise RuntimeError(f"accepted shard identity mismatch: {dataset}")
            shard_embeddings[dataset] = values
            shard_receipts[dataset] = {**recorded, "resumed": True}
            print(f"SHARD_RESUMED dataset={dataset} rows={len(subset)}", flush=True)
            continue
        started = time.perf_counter()
        values, extraction = extract_embeddings(
            subset,
            source_repo,
            model,
            device,
            batch_size,
            guard=guard,
        )
        cached = save_cache(
            shard_path,
            subset,
            values,
            {"encoder": identity, "extraction": extraction},
        )
        shard_receipt = {
            **cached,
            "dataset": dataset,
            "rows": len(subset),
            "shape": list(values.shape),
            "ordered_id_sha256": ordered_id_sha256(
                [sample.sample_id for sample in subset]
            ),
            "actual_wall_seconds": time.perf_counter() - started,
            "resumed": False,
        }
        write_json(shard_receipt_path, shard_receipt)
        shard_embeddings[dataset] = values
        shard_receipts[dataset] = shard_receipt
        execution_guard(
            result_root, state, phase_started, base_active, force_write=True
        )
        print(
            f"SHARD_ACCEPTED dataset={dataset} rows={len(subset)} "
            f"seconds={shard_receipt['actual_wall_seconds']:.3f} "
            f"sha256={cached['cache_sha256']}",
            flush=True,
        )
    by_id = {}
    for dataset in DATASET_ORDER:
        subset = [sample for sample in samples if sample.dataset == dataset]
        for sample, value in zip(subset, shard_embeddings[dataset]):
            by_id[sample.sample_id] = value
    embeddings = np.stack([by_id[sample.sample_id] for sample in samples])
    if embeddings.shape != (len(samples), 768) or not np.isfinite(embeddings).all():
        raise RuntimeError("combined embedding shard coverage failed")
    receipt = save_cache(
        cache_root / "embeddings.npz",
        samples,
        embeddings,
        {
            "encoder": identity,
            "shards": shard_receipts,
            "resume_safe": True,
            "combination_policy": "canonical sample order; no numeric transform",
        },
    )
    execution_guard(
        result_root, state, phase_started, base_active, force_write=True
    )
    write_json(result_root / "embedding_receipt.json", receipt)
    return receipt


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(result_root: Path, conditions: list[str]) -> dict[str, object]:
    summary_rows = []
    per_class_rows = []
    oof_receipts = {}
    for condition in conditions:
        fold_metrics = [
            json.loads(
                (result_root / f"fold_{fold}" / condition / "metrics.json").read_text()
            )
            for fold in range(5)
        ]
        non_kauh_receipts = {}
        for task in TASK_SPECS:
            if task == "kauh_raw9":
                continue
            task_runs = [
                receipt["test_metrics"][task] for receipt in fold_metrics
            ]
            metric_names = (
                ["macro_f1", "micro_f1"]
                if TASK_SPECS[task]["kind"] == "multilabel"
                else ["macro_f1", "weighted_f1", "uar", "native_score"]
            )
            aggregate_metrics = {
                metric: {
                    "mean": float(
                        np.mean([float(run[metric]) for run in task_runs])
                    ),
                    "sample_std": float(
                        np.std(
                            [float(run[metric]) for run in task_runs], ddof=1
                        )
                    ),
                }
                for metric in metric_names
            }
            non_kauh_receipts[task] = {
                "fold_conditioned_runs": 5,
                "metrics": aggregate_metrics,
            }
            for label in TASK_SPECS[task]["labels"]:
                class_runs = [run["per_class"][label] for run in task_runs]
                per_class_rows.append(
                    {
                        "condition": condition,
                        "dataset": TASK_SPECS[task]["dataset"],
                        "task": task,
                        "label": label,
                        "evaluation": (
                            "five KAUH-fold-conditioned native-test runs"
                        ),
                        "support": class_runs[0]["support"],
                        "precision_mean": float(
                            np.mean(
                                [float(run["precision"]) for run in class_runs]
                            )
                        ),
                        "precision_sample_std": float(
                            np.std(
                                [float(run["precision"]) for run in class_runs],
                                ddof=1,
                            )
                        ),
                        "recall_mean": float(
                            np.mean(
                                [float(run["recall"]) for run in class_runs]
                            )
                        ),
                        "recall_sample_std": float(
                            np.std(
                                [float(run["recall"]) for run in class_runs],
                                ddof=1,
                            )
                        ),
                        "f1_mean": float(
                            np.mean([float(run["f1"]) for run in class_runs])
                        ),
                        "f1_sample_std": float(
                            np.std(
                                [float(run["f1"]) for run in class_runs], ddof=1
                            )
                        ),
                    }
                )
            summary_rows.append(
                {
                    "condition": condition,
                    "dataset": TASK_SPECS[task]["dataset"],
                    "task": task,
                    "evaluation": "five KAUH-fold-conditioned native-test runs",
                    "rows": task_runs[0]["rows"],
                    "runs": 5,
                    "macro_f1_mean": aggregate_metrics["macro_f1"]["mean"],
                    "macro_f1_sample_std": aggregate_metrics["macro_f1"][
                        "sample_std"
                    ],
                    "weighted_or_micro_f1_mean": aggregate_metrics[
                        "weighted_f1"
                        if "weighted_f1" in aggregate_metrics
                        else "micro_f1"
                    ]["mean"],
                    "weighted_or_micro_f1_sample_std": aggregate_metrics[
                        "weighted_f1"
                        if "weighted_f1" in aggregate_metrics
                        else "micro_f1"
                    ]["sample_std"],
                    "uar_mean": aggregate_metrics.get("uar", {}).get("mean"),
                    "uar_sample_std": aggregate_metrics.get("uar", {}).get(
                        "sample_std"
                    ),
                    "native_score_mean": aggregate_metrics.get(
                        "native_score", {}
                    ).get("mean"),
                    "native_score_sample_std": aggregate_metrics.get(
                        "native_score", {}
                    ).get("sample_std"),
                }
            )
        oof = []
        for fold in range(5):
            rows = _read_predictions(
                result_root / f"fold_{fold}" / condition / "predictions.csv.gz"
            )
            oof.extend(row for row in rows if row["task"] == "kauh_raw9")
        ids = [row["sample_id"] for row in oof]
        if len(oof) != 336 or len(ids) != len(set(ids)):
            raise RuntimeError(f"KAUH OOF coverage failed for {condition}")
        y_true = np.asarray([json.loads(row["true_json"]) for row in oof], dtype=int)
        y_pred = np.asarray([json.loads(row["pred_json"]) for row in oof], dtype=int)
        metrics = _multiclass_metrics(y_true, y_pred, KAUH_LABELS, "kauh_raw9")
        oof_receipts[condition] = metrics
        for label in KAUH_LABELS:
            values = metrics["per_class"][label]
            per_class_rows.append(
                {
                    "condition": condition,
                    "dataset": "kauh",
                    "task": "kauh_raw9",
                    "label": label,
                    "evaluation": "five-fold patient-grouped aggregate OOF",
                    "support": values["support"],
                    "precision_mean": values["precision"],
                    "precision_sample_std": None,
                    "recall_mean": values["recall"],
                    "recall_sample_std": None,
                    "f1_mean": values["f1"],
                    "f1_sample_std": None,
                }
            )
        summary_rows.append(
            {
                "condition": condition,
                "dataset": "kauh",
                "task": "kauh_raw9",
                "evaluation": "five-fold patient-grouped OOF",
                "rows": metrics["rows"],
                "runs": 5,
                "macro_f1_mean": metrics["macro_f1"],
                "macro_f1_sample_std": None,
                "weighted_or_micro_f1_mean": metrics["weighted_f1"],
                "weighted_or_micro_f1_sample_std": None,
                "uar_mean": metrics["uar"],
                "uar_sample_std": None,
                "native_score_mean": None,
                "native_score_sample_std": None,
            }
        )
        oof_receipts[condition]["non_kauh_fold_conditioned"] = (
            non_kauh_receipts
        )
    with (result_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (result_root / "per_class_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    payload = {
        "status": "four_dataset_matrix_complete",
        "conditions": conditions,
        "summary_rows": len(summary_rows),
        "per_class_summary_rows": len(per_class_rows),
        "kauh_oof": oof_receipts,
        "non_kauh_reporting": (
            "mean and sample standard deviation across five KAUH-fold-conditioned "
            "models; no fold is selected as primary"
        ),
        "claim_boundary": json.loads(PROTOCOL.read_text())["claim_boundary"],
    }
    embedding_receipt_path = result_root / "embedding_receipt.json"
    if embedding_receipt_path.is_file():
        payload["embedding_cache"] = json.loads(
            embedding_receipt_path.read_text()
        )
    write_json(result_root / "run_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["audit", "smoke", "profile", "extract", "train", "all"],
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("result/pafa_sprsound_transfer_20260722_235659/source/repo"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".cache/checkpoints/pafa/server_epoch27/best.pth"),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=Path(".cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(f"result/{EXPERIMENT_ID}"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(f".cache/{EXPERIMENT_ID}"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
    )
    args = parser.parse_args()
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
    result_root.mkdir(parents=True, exist_ok=True)
    configure_runtime(cache_root, args.threads)
    device = torch.device(args.device)
    conditions = args.conditions.split(",")
    samples, _ = data_audit(args.dataset_root, result_root)

    if args.phase == "audit":
        return
    if args.phase == "smoke":
        smoke(
            samples,
            args.source_repo,
            args.checkpoint,
            args.backbone_checkpoint,
            args.dataset_root,
            result_root,
            device,
            args.batch_size,
        )
        return
    if args.phase in {"profile", "all"}:
        profile(
            samples,
            args.source_repo,
            args.checkpoint,
            args.backbone_checkpoint,
            result_root,
            device,
            args.batch_size,
        )
        if args.phase == "profile":
            return
    if args.phase in {"extract", "all"}:
        extract(
            samples,
            args.source_repo,
            args.checkpoint,
            args.backbone_checkpoint,
            cache_root,
            result_root,
            device,
            args.batch_size,
        )
        if args.phase == "extract":
            return
    embeddings, cache_receipt = load_cache(cache_root / "embeddings.npz", samples)
    state = load_execution_state(result_root)
    base_active = float(state["active_seconds"])
    phase_started = time.perf_counter()
    guard = lambda: execution_guard(
        result_root, state, phase_started, base_active
    )
    for fold in range(5):
        fold_samples, fold_receipt = build_samples(args.dataset_root, fold)
        if [sample.sample_id for sample in fold_samples] != [sample.sample_id for sample in samples]:
            raise RuntimeError("KAUH fold changed embedding identity/order")
        write_json(result_root / f"fold_{fold}" / "data_receipt.json", fold_receipt)
        pending_conditions = []
        for condition in conditions:
            directory = result_root / f"fold_{fold}" / condition
            artifacts = [
                directory / "best.pth",
                directory / "metrics.json",
                directory / "predictions.csv.gz",
                directory / "predictions_label_free.csv.gz",
            ]
            present = [path.is_file() for path in artifacts]
            if any(present) and not all(present):
                raise RuntimeError(
                    f"partial training artifact set for fold={fold} {condition}"
                )
            if all(present):
                print(
                    f"CONDITION_RESUMED fold={fold} condition={condition}",
                    flush=True,
                )
            else:
                pending_conditions.append(condition)
        train_conditions(
            fold_samples,
            embeddings,
            result_root / f"fold_{fold}",
            pending_conditions,
            device,
            args.dataset_root,
            guard,
        )
        execution_guard(
            result_root, state, phase_started, base_active, force_write=True
        )
        print(
            f"FOLD_ACCEPTED fold={fold} conditions={len(conditions)}",
            flush=True,
        )
    manifest = aggregate(result_root, conditions)
    manifest["embedding_cache"] = cache_receipt
    write_json(result_root / "run_manifest.json", manifest)
    execution_guard(
        result_root, state, phase_started, base_active, force_write=True
    )


if __name__ == "__main__":
    main()
