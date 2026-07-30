"""Run the four-dataset event-sensitive pooling diagnostic."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torchaudio

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    KAUH_LABELS,
    Sample,
    build_samples,
    sample_to_row,
)
from baseline.four_dataset_frozen_encoder.encoder import (
    _waveforms_for_sample,
    build_encoder,
    load_cache,
    save_cache,
    sha256_file,
    verify_source_repo,
)
from baseline.four_dataset_frozen_encoder.train import (
    BATCH_SIZE,
    EPOCHS,
    SEED,
    SharedNativeModel,
    TASK_SPECS,
    _loss,
    _multiclass_metrics,
    _multilabel_metrics,
    _prediction_indices,
    _prior_receipt,
    _source_batches,
    _targets,
    score_task_predictions,
)
from baseline.four_dataset_representation_attribution.run import (
    EXPECTED_R0_SHA256,
    R0,
    ordered_id_sha256,
    write_csv,
    write_gzip_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "four_dataset_event_sensitive_pooling"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
P0 = "p0_r0_d2_pooled_reference"
P1 = "p1_event_sensitive_learned_pooling"
P2 = "p2_parameter_matched_pooled_control"
TRAINED_CONDITIONS = (P1, P2)
ALL_CONDITIONS = (P0, P1, P2)
DATASETS = ("icbhi", "sprsound", "hf_lung", "kauh")
P0_ROOT = (
    ROOT
    / "result/four_dataset_representation_attribution/hf_proxy_fixed_v2"
    / R0
)
R0_CACHE = ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
MAX_CACHE_GIB = 20.0


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


def validate_roots(result_root: Path, cache_root: Path) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != EXPERIMENT_ID or result.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}")
    if cache.name != EXPERIMENT_ID or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}")
    return result, cache


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty prediction file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def data_audit(dataset_root: Path, result_root: Path) -> tuple[list[Sample], dict[str, object]]:
    folds: dict[str, object] = {}
    canonical = None
    for fold in range(5):
        samples, receipt = build_samples(dataset_root, fold)
        folds[str(fold)] = receipt
        if receipt["datasets"]["hf_lung"]["assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256:
            raise RuntimeError("corrected HF assignment gate failed")
        if canonical is None:
            canonical = samples
            write_gzip_csv(
                result_root / "samples_fold_0.csv.gz",
                [sample_to_row(sample) for sample in samples],
            )
        elif [sample.sample_id for sample in samples] != [
            sample.sample_id for sample in canonical
        ]:
            raise RuntimeError("KAUH fold changed canonical sample order")
    if canonical is None or len(canonical) != 25_084:
        raise RuntimeError("four-dataset row count gate failed")
    if sha256_file(R0_CACHE) != EXPECTED_R0_SHA256:
        raise RuntimeError("R0 PAFA pooled cache SHA gate failed")
    receipt = {
        "status": "event_sensitive_pooling_data_audit_passed",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "rows": len(canonical),
        "dataset_rows": dict(sorted(Counter(sample.dataset for sample in canonical).items())),
        "ordered_id_sha256": ordered_id_sha256([sample.sample_id for sample in canonical]),
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "r0_cache_sha256": EXPECTED_R0_SHA256,
        "p0_reference_root": str(P0_ROOT),
        "folds": folds,
    }
    write_json(result_root / "data_receipt.json", receipt)
    return canonical, receipt


def _load_cut_pad(source_repo: Path):
    source_repo = verify_source_repo(source_repo)
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        from util.icbhi_util import cut_pad_sample_torchaudio

    return cut_pad_sample_torchaudio


def _window_embeddings_for_samples(
    samples: list[Sample],
    source_repo: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    guard: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cut_pad = _load_cut_pad(source_repo)
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.audio_path, []).append(sample)
    pending_waveforms: list[torch.Tensor] = []
    pending_ids: list[str] = []
    output: dict[str, list[np.ndarray]] = {sample.sample_id: [] for sample in samples}
    window_counts: dict[str, int] = {}
    started = time.perf_counter()

    def flush() -> None:
        if not pending_waveforms:
            return
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.filterwarnings(
                "ignore",
                message="User provided device_type of 'cuda'.*CUDA is not available.*",
                category=UserWarning,
            )
            frames = encoder(torch.stack(pending_waveforms).to(device), training=False)
            values = frames.mean(dim=1)
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid PAFA window embedding batch: {tuple(values.shape)}")
        for sample_id, value in zip(pending_ids, values.cpu().numpy()):
            output[sample_id].append(value.astype(np.float32, copy=False))
        pending_waveforms.clear()
        pending_ids.clear()
        if guard is not None:
            guard()

    for audio_path in sorted(grouped):
        waveform, sample_rate = torchaudio.load(audio_path)
        for sample in grouped[audio_path]:
            windows = _waveforms_for_sample(waveform, sample_rate, sample, cut_pad)
            if sample.dataset == "hf_lung" and len(windows) != 3:
                raise RuntimeError(f"HF must use exactly three 5 s windows: {sample.sample_id}")
            window_counts[sample.sample_id] = len(windows)
            for window in windows:
                pending_waveforms.append(window)
                pending_ids.append(sample.sample_id)
                if len(pending_waveforms) >= batch_size:
                    flush()
    flush()
    max_windows = max(window_counts.values())
    values = np.zeros((len(samples), max_windows, 768), dtype=np.float32)
    mask = np.zeros((len(samples), max_windows), dtype=bool)
    for index, sample in enumerate(samples):
        current = np.stack(output[sample.sample_id]).astype(np.float32)
        values[index, : len(current)] = current
        mask[index, : len(current)] = True
    if not np.isfinite(values).all() or not mask.any(axis=1).all():
        raise RuntimeError("invalid window cache values/masks")
    receipt = {
        "samples": len(samples),
        "unique_audio_files": len(grouped),
        "shape": list(values.shape),
        "mask_shape": list(mask.shape),
        "finite": True,
        "runtime_seconds": time.perf_counter() - started,
        "window_count_min": int(min(window_counts.values())),
        "window_count_max": int(max(window_counts.values())),
        "window_count_by_dataset": {
            dataset: int(sum(window_counts[s.sample_id] for s in samples if s.dataset == dataset))
            for dataset in DATASETS
        },
        "unit_to_window_policy": json.loads(PROTOCOL_PATH.read_text())["unit_to_window_lineage"],
    }
    return values, mask, receipt


def _save_window_cache(
    path: Path,
    samples: list[Sample],
    values: np.ndarray,
    mask: np.ndarray,
    receipt: dict[str, object],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_ids=np.asarray([sample.sample_id for sample in samples]),
        window_embeddings=values.astype(np.float32),
        window_mask=mask.astype(np.bool_),
        receipt_json=np.asarray(json.dumps(receipt, sort_keys=True)),
    )
    temporary.replace(path)
    return {**receipt, "cache_path": str(path), "cache_sha256": sha256_file(path)}


def _load_window_cache(path: Path, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    archive = np.load(path, allow_pickle=False)
    ids = archive["sample_ids"].astype(str).tolist()
    expected = [sample.sample_id for sample in samples]
    if ids != expected:
        raise RuntimeError("window cache sample order mismatch")
    values = archive["window_embeddings"].astype(np.float32)
    mask = archive["window_mask"].astype(bool)
    if values.shape[:2] != mask.shape or values.shape[0] != len(samples) or values.shape[2] != 768:
        raise RuntimeError("invalid window cache shape")
    if not np.isfinite(values).all() or not mask.any(axis=1).all():
        raise RuntimeError("invalid window cache finite/mask gate")
    receipt = json.loads(str(archive["receipt_json"].item()))
    return values, mask, {**receipt, "cache_path": str(path), "cache_sha256": sha256_file(path)}


def mean_pool(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(np.float32)
    pooled = (values * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)
    return pooled.astype(np.float32)


class PoolingNativeModel(torch.nn.Module):
    def __init__(self, condition: str) -> None:
        super().__init__()
        if condition not in TRAINED_CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        self.pooler = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, 1))
        self.base = SharedNativeModel()

    def pooled(self, windows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.condition == P1:
            scores = self.pooler(windows).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            return torch.sum(windows * weights.unsqueeze(-1), dim=1)
        pooled = torch.sum(windows * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        gate = torch.sigmoid(self.pooler(pooled))
        return pooled * (1.0 + gate)

    def forward(self, windows: torch.Tensor, mask: torch.Tensor, task: str) -> torch.Tensor:
        return self.base(self.pooled(windows, mask), task)


def _select_arrays(values: np.ndarray, mask: np.ndarray, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(values[indices]).to(device),
        torch.from_numpy(mask[indices]).to(device),
    )


def predict_task(
    model: PoolingNativeModel,
    samples: list[Sample],
    values: np.ndarray,
    mask: np.ndarray,
    task: str,
    partition: str,
    device: torch.device,
) -> list[dict[str, object]]:
    indices = _prediction_indices(samples, task, partition)
    if not indices:
        raise RuntimeError(f"no {partition} rows for {task}")
    model.eval()
    probabilities_parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), 2048):
            batch = indices[start : start + 2048]
            batch_values, batch_mask = _select_arrays(values, mask, batch, device)
            logits = model(batch_values, batch_mask, task)
            probabilities = (
                torch.softmax(logits, dim=1)
                if TASK_SPECS[task]["kind"] == "multiclass"
                else torch.sigmoid(logits)
            )
            probabilities_parts.append(probabilities.cpu().numpy())
    probabilities = np.concatenate(probabilities_parts)
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"non-finite probabilities for {task}")
    rows = []
    if TASK_SPECS[task]["kind"] == "multiclass":
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


def evaluate_task(model, samples, values, mask, task, partition, device):
    rows = predict_task(model, samples, values, mask, task, partition, device)
    return score_task_predictions(samples, task, rows)


def _validation_score(model, samples, values, mask, device):
    receipts = {}
    scores = []
    for task in TASK_SPECS:
        metrics, _ = evaluate_task(model, samples, values, mask, task, "validation", device)
        receipts[task] = metrics
        scores.append(float(metrics["macro_f1"]))
    return float(np.mean(scores)), receipts


def _save_outputs(
    output_dir: Path,
    condition: str,
    model: PoolingNativeModel,
    samples: list[Sample],
    values: np.ndarray,
    mask: np.ndarray,
    history: list[dict[str, object]],
    selection: dict[str, object],
    priors: dict[str, object],
    device: torch.device,
    dataset_root: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_free = []
    for task in TASK_SPECS:
        label_free.extend(predict_task(model, samples, values, mask, task, "test", device))
    label_free_path = output_dir / "predictions_label_free.csv.gz"
    _write_predictions(label_free_path, label_free)
    from baseline.four_dataset_frozen_encoder.data import load_terminal_spr_test_targets

    terminal_spr_targets = load_terminal_spr_test_targets(samples)
    metrics = {}
    scored = []
    for task in TASK_SPECS:
        rows = [row for row in label_free if row["task"] == task]
        task_metrics, task_scored = score_task_predictions(samples, task, rows, terminal_spr_targets)
        metrics[task] = task_metrics
        scored.extend(task_scored)
    torch.save(
        {
            "condition": condition,
            "model": model.state_dict(),
            "selection": selection,
            "priors": priors,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        output_dir / "best.pth",
    )
    _write_predictions(output_dir / "predictions.csv.gz", scored)
    payload = {
        "condition": condition,
        "history": history,
        "selection": selection,
        "test_metrics": metrics,
        "prediction_rows": len(scored),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pooler_parameters": sum(parameter.numel() for parameter in model.pooler.parameters()),
        "terminal_label_join": {
            "label_free_prediction_path": str(label_free_path),
            "label_free_prediction_sha256": sha256_file(label_free_path),
            "label_free_rows_written_before_label_load": len(label_free),
            "spr_terminal_labels": len(terminal_spr_targets),
            "spr_test_labels_loaded_after_label_free_write": True,
            "dataset_root": str(dataset_root.resolve()),
        },
    }
    write_json(output_dir / "metrics.json", payload)
    return payload


def train_condition(
    condition: str,
    samples: list[Sample],
    values: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    device: torch.device,
    dataset_root: Path,
    guard: Callable[[], None] | None = None,
) -> dict[str, object]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = PoolingNativeModel(condition).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    priors = _prior_receipt(samples)
    best_score = -float("inf")
    best_state = None
    history = []
    selection = {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        task_losses = Counter()
        task_steps = Counter()
        for dataset, batch_indices in _source_batches(samples, "dataset_balanced", epoch):
            batch_values, batch_mask = _select_arrays(values, mask, batch_indices, device)
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
                logits = model(batch_values[positions], batch_mask[positions], task)
                target = _targets(samples, sample_indices, task, device)
                task_loss = _loss(logits, target, task, priors, False)
                active.append(task_loss)
                task_losses[task] += float(task_loss.detach())
                task_steps[task] += 1
            if not active:
                continue
            loss = torch.stack(active).mean()
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradient in {condition}")
            optimizer.step()
            if guard is not None:
                guard()
            losses.append(float(loss.detach()))
        score, validation = _validation_score(model, samples, values, mask, device)
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
        raise RuntimeError(f"no selected state for {condition}")
    model.load_state_dict(best_state)
    return _save_outputs(output_dir, condition, model, samples, values, mask, history, selection, priors, device, dataset_root)


def _selected_smoke_samples(samples: list[Sample]) -> list[Sample]:
    ids: set[str] = set()
    for task, spec in TASK_SPECS.items():
        for partition in ("subtrain", "validation", "test"):
            candidates = [
                sample
                for sample in samples
                if sample.partition == partition
                and sample.dataset == spec["dataset"]
                and (
                    task in sample.targets
                    or (partition == "test" and sample.dataset == "sprsound" and not sample.targets)
                )
            ]
            ids.update(sample.sample_id for sample in candidates[:8])
    return [sample for sample in samples if sample.sample_id in ids]


def smoke(samples, source_repo, checkpoint, backbone, dataset_root, result_root, device, batch_size) -> dict[str, object]:
    selected = _selected_smoke_samples(samples)
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    values, mask, extraction = _window_embeddings_for_samples(selected, source_repo, model, device, batch_size)
    outputs = {}
    for condition in TRAINED_CONDITIONS:
        outputs[condition] = train_condition(
            condition,
            selected,
            values,
            mask,
            result_root / "smoke" / condition,
            device,
            dataset_root,
        )
    parameter_counts = {condition: outputs[condition]["parameters"] for condition in TRAINED_CONDITIONS}
    if parameter_counts[P1] != parameter_counts[P2]:
        raise RuntimeError("P1/P2 parameter matching failed")
    payload = {
        "status": "event_sensitive_pooling_smoke_passed",
        "samples": len(selected),
        "datasets": dict(sorted(Counter(sample.dataset for sample in selected).items())),
        "encoder": identity,
        "extraction": extraction,
        "conditions": list(TRAINED_CONDITIONS),
        "parameter_counts": parameter_counts,
        "p1_p2_parameter_matched": True,
        "finite_loss_gradient_checkpoint": True,
        "spr_label_free_terminal_join": all(
            outputs[c]["terminal_label_join"]["spr_test_labels_loaded_after_label_free_write"]
            for c in TRAINED_CONDITIONS
        ),
    }
    write_json(result_root / "smoke_receipt.json", payload)
    return payload


def profile(samples, source_repo, checkpoint, backbone, result_root, device, batch_size) -> dict[str, object]:
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    selected = []
    counts = Counter()
    for sample in samples:
        if counts[sample.dataset] < 50:
            selected.append(sample)
            counts[sample.dataset] += 1
    projected = {}
    total = 0.0
    for dataset in DATASETS:
        subset = [sample for sample in selected if sample.dataset == dataset]
        _, _, receipt = _window_embeddings_for_samples(subset, source_repo, model, device, min(batch_size, len(subset)))
        full_count = sum(sample.dataset == dataset for sample in samples)
        estimate = receipt["runtime_seconds"] / len(subset) * full_count
        projected[dataset] = {
            "profile_rows": len(subset),
            "profile_seconds": receipt["runtime_seconds"],
            "full_rows": full_count,
            "projected_seconds": estimate,
        }
        total += estimate
    values = torch.randn(BATCH_SIZE, 3, 768, device=device)
    mask = torch.ones(BATCH_SIZE, 3, dtype=torch.bool, device=device)
    target = torch.randint(0, 4, (BATCH_SIZE,), device=device)
    train_seconds = {}
    for condition in TRAINED_CONDITIONS:
        probe = PoolingNativeModel(condition).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        started = time.perf_counter()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(probe(values, mask, "icbhi_flat4"), target)
            loss.backward()
            optimizer.step()
        train_seconds[condition] = (time.perf_counter() - started) / 100
    updates = max(
        int(np.ceil(sum(sample.dataset == dataset and sample.partition == "subtrain" and bool(sample.targets) for sample in samples) / BATCH_SIZE))
        for dataset in DATASETS
    ) * len(DATASETS) * EPOCHS * 5 * len(TRAINED_CONDITIONS)
    projected_train = max(train_seconds.values()) * updates
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    projected_cache_gib = len(samples) * max(1, max(projected[d]["profile_rows"] for d in projected)) * 768 * 4 / 1024**3
    allowed = (total + projected_train) * 2 <= 3 * 3600 and peak_rss < 24 and projected_cache_gib < MAX_CACHE_GIB
    payload = {
        "status": "event_sensitive_pooling_profile_passed",
        "encoder": identity,
        "datasets": projected,
        "projected_extraction_seconds": total,
        "training_step_seconds": train_seconds,
        "projected_optimizer_updates": updates,
        "projected_training_seconds": projected_train,
        "projected_end_to_end_with_2x_safety_seconds": (total + projected_train) * 2,
        "peak_rss_gib": peak_rss,
        "projected_cache_gib_upper_bound": projected_cache_gib,
        "gate": {"max_seconds": 3 * 3600, "max_peak_rss_gib": 24, "max_new_cache_gib": MAX_CACHE_GIB},
        "decision": "local full allowed" if allowed else "hold",
    }
    write_json(result_root / "profile_receipt.json", payload)
    return payload


def extract(samples, source_repo, checkpoint, backbone, cache_root, result_root, device, batch_size) -> dict[str, object]:
    profile_receipt = json.loads((result_root / "profile_receipt.json").read_text())
    if profile_receipt["decision"] != "local full allowed":
        raise RuntimeError("profile did not authorize local full extraction")
    cache_path = cache_root / "window_embeddings.npz"
    if cache_path.is_file():
        values, mask, receipt = _load_window_cache(cache_path, samples)
        write_json(result_root / "window_embedding_receipt.json", {**receipt, "resumed": True})
        return receipt
    model, identity = build_encoder(source_repo, checkpoint, backbone, device)
    values, mask, extraction = _window_embeddings_for_samples(samples, source_repo, model, device, batch_size)
    pooled = mean_pool(values, mask)
    r0_values, _ = load_cache(R0_CACHE, samples)
    pooled_close_to_r0 = bool(np.allclose(pooled, r0_values, rtol=1e-5, atol=1e-5))
    receipt = _save_window_cache(
        cache_path,
        samples,
        values,
        mask,
        {
            "encoder": identity,
            "extraction": extraction,
            "ordered_id_sha256": ordered_id_sha256([sample.sample_id for sample in samples]),
            "resume_safe": True,
            "r0_mean_pool_numeric_close": pooled_close_to_r0,
            "r0_cache_sha256": EXPECTED_R0_SHA256,
        },
    )
    write_json(result_root / "window_embedding_receipt.json", receipt)
    return receipt


def train(samples, values, mask, result_root, device, dataset_root) -> None:
    for fold in range(5):
        fold_samples, fold_receipt = build_samples(dataset_root, fold)
        if [sample.sample_id for sample in fold_samples] != [sample.sample_id for sample in samples]:
            raise RuntimeError("fold sample order changed")
        write_json(result_root / f"fold_{fold}" / "data_receipt.json", fold_receipt)
        for condition in TRAINED_CONDITIONS:
            directory = result_root / f"fold_{fold}" / condition
            artifacts = [directory / "best.pth", directory / "metrics.json", directory / "predictions.csv.gz", directory / "predictions_label_free.csv.gz"]
            present = [path.is_file() for path in artifacts]
            if any(present) and not all(present):
                raise RuntimeError(f"partial artifact set for fold={fold} condition={condition}")
            if all(present):
                print(f"CONDITION_RESUMED fold={fold} condition={condition}", flush=True)
                continue
            train_condition(condition, fold_samples, values, mask, directory, device, dataset_root)
            print(f"CONDITION_ACCEPTED fold={fold} condition={condition}", flush=True)


def _condition_dir(result_root: Path, fold: int, condition: str) -> Path:
    if condition == P0:
        return P0_ROOT / f"fold_{fold}/d2_shared_adapter_dataset_balanced"
    return result_root / f"fold_{fold}" / condition


def aggregate(result_root: Path) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for condition in ALL_CONDITIONS:
        fold_metrics = [
            json.loads((_condition_dir(result_root, fold, condition) / "metrics.json").read_text())
            for fold in range(5)
        ]
        for task in TASK_SPECS:
            if task == "kauh_raw9":
                continue
            task_runs = [receipt["test_metrics"][task] for receipt in fold_metrics]
            metric_names = (
                ["macro_f1", "micro_f1"]
                if TASK_SPECS[task]["kind"] == "multilabel"
                else ["macro_f1", "weighted_f1", "uar", "native_score"]
            )
            aggregate_metrics = {
                metric: {
                    "mean": float(np.mean([float(run[metric]) for run in task_runs])),
                    "sample_std": float(np.std([float(run[metric]) for run in task_runs], ddof=1)),
                }
                for metric in metric_names
            }
            if "specificity" in task_runs[0]:
                aggregate_metrics["specificity"] = {
                    "mean": float(np.mean([float(run["specificity"]) for run in task_runs])),
                    "sample_std": float(np.std([float(run["specificity"]) for run in task_runs], ddof=1)),
                }
            for label in TASK_SPECS[task]["labels"]:
                class_runs = [run["per_class"][label] for run in task_runs]
                per_class_rows.append(
                    {
                        "condition": condition,
                        "dataset": TASK_SPECS[task]["dataset"],
                        "task": task,
                        "label": label,
                        "evaluation": "five KAUH-fold-conditioned native-test runs",
                        "support": class_runs[0]["support"],
                        "precision_mean": float(np.mean([float(run["precision"]) for run in class_runs])),
                        "precision_sample_std": float(np.std([float(run["precision"]) for run in class_runs], ddof=1)),
                        "recall_mean": float(np.mean([float(run["recall"]) for run in class_runs])),
                        "recall_sample_std": float(np.std([float(run["recall"]) for run in class_runs], ddof=1)),
                        "f1_mean": float(np.mean([float(run["f1"]) for run in class_runs])),
                        "f1_sample_std": float(np.std([float(run["f1"]) for run in class_runs], ddof=1)),
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
                    "macro_f1_sample_std": aggregate_metrics["macro_f1"]["sample_std"],
                    "weighted_or_micro_f1_mean": aggregate_metrics.get("weighted_f1", aggregate_metrics.get("micro_f1"))["mean"],
                    "weighted_or_micro_f1_sample_std": aggregate_metrics.get("weighted_f1", aggregate_metrics.get("micro_f1"))["sample_std"],
                    "uar_mean": aggregate_metrics.get("uar", {}).get("mean"),
                    "uar_sample_std": aggregate_metrics.get("uar", {}).get("sample_std"),
                    "native_score_mean": aggregate_metrics.get("native_score", {}).get("mean"),
                    "native_score_sample_std": aggregate_metrics.get("native_score", {}).get("sample_std"),
                    "specificity_mean": aggregate_metrics.get("specificity", {}).get("mean"),
                    "specificity_sample_std": aggregate_metrics.get("specificity", {}).get("sample_std"),
                }
            )
        oof = []
        for fold in range(5):
            rows = _read_predictions(_condition_dir(result_root, fold, condition) / "predictions.csv.gz")
            oof.extend(row for row in rows if row["task"] == "kauh_raw9")
        ids = [row["sample_id"] for row in oof]
        if len(oof) != 336 or len(ids) != len(set(ids)):
            raise RuntimeError(f"KAUH OOF coverage failed for {condition}")
        y_true = np.asarray([json.loads(row["true_json"]) for row in oof], dtype=int)
        y_pred = np.asarray([json.loads(row["pred_json"]) for row in oof], dtype=int)
        metrics = _multiclass_metrics(y_true, y_pred, KAUH_LABELS, "kauh_raw9")
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
                "specificity_mean": None,
                "specificity_sample_std": None,
            }
        )
    write_csv(result_root / "summary.csv", summary_rows)
    write_csv(result_root / "per_class_summary.csv", per_class_rows)
    decision = decision_receipt(summary_rows)
    write_json(result_root / "decision.json", decision)
    manifest = {
        "status": "event_sensitive_pooling_complete",
        "conditions": list(ALL_CONDITIONS),
        "summary_rows": len(summary_rows),
        "per_class_summary_rows": len(per_class_rows),
        "decision": decision,
        "p0_reference_root": str(P0_ROOT),
        "claim_boundary": json.loads(PROTOCOL_PATH.read_text())["claim_boundary"],
    }
    write_json(result_root / "run_manifest.json", manifest)
    return manifest


def decision_receipt(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row["condition"], row["task"]): row for row in summary_rows}
    votes = []
    details = {}
    for task in TASK_SPECS:
        p1 = by_key[(P1, task)]
        p2 = by_key[(P2, task)]
        macro_delta = float(p1["macro_f1_mean"]) - float(p2["macro_f1_mean"])
        uar_delta = None if p1["uar_mean"] in ("", None) else float(p1["uar_mean"]) - float(p2["uar_mean"])
        native_delta = None if p1["native_score_mean"] in ("", None) else float(p1["native_score_mean"]) - float(p2["native_score_mean"])
        weighted_delta = float(p1["weighted_or_micro_f1_mean"]) - float(p2["weighted_or_micro_f1_mean"])
        specificity_delta = None if p1["specificity_mean"] in ("", None) else float(p1["specificity_mean"]) - float(p2["specificity_mean"])
        positive_secondary = (
            (uar_delta is not None and uar_delta >= -0.03)
            or (native_delta is not None and native_delta >= -0.03)
            or (uar_delta is None and native_delta is None)
        )
        guardrail = weighted_delta >= -0.03 and (specificity_delta is None or specificity_delta >= -0.03)
        material = macro_delta >= 0.03 and positive_secondary and guardrail
        if material:
            votes.append(task)
        details[task] = {
            "p1_minus_p2_macro_f1": macro_delta,
            "p1_minus_p2_uar": uar_delta,
            "p1_minus_p2_native_score": native_delta,
            "p1_minus_p2_weighted_or_micro_f1": weighted_delta,
            "p1_minus_p2_specificity": specificity_delta,
            "material_vote": material,
        }
    return {
        "status": "event_sensitive_pooling_decision_complete",
        "comparison": "P1 event-sensitive learned pooling minus P2 parameter-matched pooled control",
        "material_band": 0.03,
        "material_improvement_count": len(votes),
        "material_improvement_tasks": votes,
        "decision": (
            "event_sensitive_pooling_supported"
            if len(votes) >= 2
            else "not_supported_or_inconclusive"
        ),
        "tasks": details,
        "claim_boundary": json.loads(PROTOCOL_PATH.read_text())["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "smoke-profile", "extract", "train", "aggregate", "all"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--source-repo", type=Path, default=Path("result/pafa_sprsound_transfer_20260722_235659/source/repo"))
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/checkpoints/pafa/server_epoch27/best.pth"))
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path(".cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt"))
    parser.add_argument("--result-root", type=Path, default=Path(f"result/{EXPERIMENT_ID}"))
    parser.add_argument("--cache-root", type=Path, default=Path(f".cache/{EXPERIMENT_ID}"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
    result_root.mkdir(parents=True, exist_ok=True)
    configure_runtime(cache_root, args.threads)
    device = torch.device(args.device)
    samples, _ = data_audit(args.dataset_root, result_root)
    if args.phase == "audit":
        return
    if args.phase == "smoke-profile":
        smoke(samples, args.source_repo, args.checkpoint, args.backbone_checkpoint, args.dataset_root, result_root, device, args.batch_size)
        profile(samples, args.source_repo, args.checkpoint, args.backbone_checkpoint, result_root, device, args.batch_size)
        return
    if args.phase in {"extract", "all"}:
        extract(samples, args.source_repo, args.checkpoint, args.backbone_checkpoint, cache_root, result_root, device, args.batch_size)
        if args.phase == "extract":
            return
    values, mask, receipt = _load_window_cache(cache_root / "window_embeddings.npz", samples)
    write_json(result_root / "window_embedding_receipt.json", receipt)
    if args.phase in {"train", "all"}:
        train(samples, values, mask, result_root, device, args.dataset_root)
        if args.phase == "train":
            return
    if args.phase in {"aggregate", "all"}:
        aggregate(result_root)


if __name__ == "__main__":
    main()
