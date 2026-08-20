"""Local single-seed BEATs ICBHI attribution matrix B1/B2/B3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from baseline.four_dataset_frozen_encoder.data import Sample, load_terminal_spr_test_targets

from .beats_window_encoder import load_local_beats_window_backend
from .contracts import PREDICTION_UNITS, WaveformBatch, collate_waveforms
from .core2_hf_positive_kauh_external import (
    CORE_NODES,
    Core2Head,
    _attach_ground_truth,
    _core_targets,
    _filter_predictions,
    _node_metrics,
    core2_loss,
    infer_core2,
    score_core_primary,
    secondary_dataset_thresholds,
    select_core_shared_thresholds,
)
from .m_unified import (
    SEED,
    _save_predictions,
    load_canonical_samples,
    load_feature_cache,
    set_determinism,
    write_json,
)
from .posthoc_native_readout import ICBHI_LABELS, decode_icbhi_flat4, native_metrics
from .real_subtrain_provider import load_sample_waveform


UPDATES_PER_EPOCH = 1_404
TOTAL_UPDATES = 70_200
ROOT_RELATIVE = Path("result/reproduce/icbhi_attribution_beats")


def matched_batches(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    datasets: Sequence[str],
    epoch: int,
) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(SEED + epoch)
    batches = []
    for dataset in datasets:
        active = np.flatnonzero(targets[dataset][1].any(axis=(1, 2)))
        order = rng.permutation(active).tolist()
        batches.extend(
            (dataset, order[start : start + 8])
            for start in range(0, len(order), 8)
        )
    available = list(batches)
    extra = UPDATES_PER_EPOCH - len(batches)
    sampled = rng.integers(0, len(available), size=extra)
    batches.extend(
        (available[index][0], list(available[index][1])) for index in sampled
    )
    rng.shuffle(batches)
    return batches


def selection_losses(
    predictions: Mapping[str, np.ndarray], datasets: Sequence[str]
) -> dict[str, object]:
    dataset_losses = {}
    node_losses = {}
    for dataset in datasets:
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
        "selection_loss": float(np.mean([dataset_losses[d] for d in datasets])),
    }


def train_hierarchical(
    stores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    datasets: Sequence[str],
    output_dir: Path,
) -> tuple[Core2Head, dict[str, object]]:
    set_determinism()
    model = Core2Head(768).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_UPDATES
    )
    log_path = output_dir / "train_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    best_loss = math.inf
    best_epoch = best_update = update = 0
    started = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        losses = []
        for dataset, indices in matched_batches(targets["subtrain"], datasets, epoch):
            values = torch.from_numpy(
                stores["subtrain"][dataset]["embeddings"][indices]
            )
            target_values, eligible_values = targets["subtrain"][dataset]
            optimizer.zero_grad(set_to_none=True)
            loss = core2_loss(
                model(values),
                torch.from_numpy(target_values[indices]),
                torch.from_numpy(eligible_values[indices]),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(float(loss.detach()))
        validation = infer_core2(
            model,
            stores["validation"],
            targets["validation"],
            device=torch.device("cpu"),
            datasets=datasets,
        )
        _save_predictions(output_dir / "validation" / f"epoch_{epoch:03d}.npz", validation)
        selection = selection_losses(validation, datasets)
        record = {
            "epoch": epoch,
            "update": update,
            "train_loss": float(np.mean(losses)),
            "validation": selection,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if selection["selection_loss"] < best_loss:
            best_loss = selection["selection_loss"]
            best_epoch = epoch
            best_update = update
            torch.save(
                {"epoch": epoch, "update": update, "model": copy.deepcopy(model.state_dict())},
                output_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            output_dir / "last_checkpoint.pt",
        )
        print(
            f"epoch {epoch:02d}/50 update={update} "
            f"train={record['train_loss']:.6f} best_val={best_loss:.6f}",
            flush=True,
        )
    selected = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(selected["model"])
    return model, {
        "selected_epoch": best_epoch,
        "selected_update": best_update,
        "selection_loss": best_loss,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": TOTAL_UPDATES,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    }


def icbhi_hierarchical_metrics(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    nodes = {}
    level_mask = predictions["eligible"][:, 0]
    nodes["level1"] = _node_metrics(
        predictions["targets"][level_mask, 0].astype(int),
        predictions["level1_probabilities"][level_mask, 1],
        predictions["level1_predictions"][level_mask],
    )
    for index, node in enumerate(CORE_NODES[1:], start=1):
        mask = predictions["eligible"][:, index]
        probability = predictions["attribute_probabilities"][mask, index - 1]
        nodes[node] = _node_metrics(
            predictions["targets"][mask, index].astype(int),
            probability,
            probability >= thresholds[node],
        )
        nodes[node]["validation_threshold"] = thresholds[node]
    return {
        "nodes": nodes,
        "evaluable_node_macro_f1": float(np.mean([nodes[n]["f1"] for n in CORE_NODES])),
    }


def icbhi_validation_thresholds(
    predictions: Mapping[str, np.ndarray]
) -> dict[str, float]:
    output = {}
    for index, node in enumerate(CORE_NODES[1:], start=1):
        mask = predictions["eligible"][:, index]
        target = predictions["targets"][mask, index].astype(int)
        probability = predictions["attribute_probabilities"][mask, index - 1]
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in np.unique(np.concatenate(([0.0, 1.0], probability))):
            value = float(f1_score(target, probability >= threshold, zero_division=0))
            if value > best_f1 or (value == best_f1 and threshold > best_threshold):
                best_f1 = value
                best_threshold = float(threshold)
        output[node] = best_threshold
    return output


def native_icbhi_from_hierarchical(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    raw = [json.loads(value) for value in predictions["raw_ground_truth"]]
    target = np.asarray([ICBHI_LABELS.index(value) for value in raw])
    predicted = decode_icbhi_flat4(
        predictions["level1_predictions"],
        predictions["attribute_probabilities"],
        thresholds["crackle"],
        thresholds["wheeze"],
    )
    metrics = native_metrics(target, predicted, ICBHI_LABELS)
    metrics["both_recall"] = metrics["per_class"]["both"]["recall"]
    metrics["readout"] = "POSTHOC unified-head native-format readout"
    return metrics


def load_2s_icbhi_stores(samples: Sequence[Sample]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    cache = Path(".cache/multidataset_pipeline/m_unified_beats_seed42")
    stores = {}
    for partition in ("subtrain", "validation", "test"):
        selected = sorted(
            [s for s in samples if s.dataset == "icbhi" and s.partition == partition],
            key=lambda s: s.sample_id,
        )
        stores[partition] = {
            "icbhi": load_feature_cache(
                cache, partition, "icbhi", [s.sample_id for s in selected]
            )
        }
    return stores


def run_b1(samples: Sequence[Sample], root: Path) -> dict[str, object]:
    output = root / "B1_icbhi_2s_hierarchical/seed_42"
    write_json(output / "config.json", {"condition": "B1", "seed": 42, "encoder": "frozen BEATs iter3+AS2M", "data": "ICBHI-only", "features": "2s/1s pooled-window native aggregation", "head": "hierarchical Level1/Crackle/Wheeze", "epochs": 50, "updates": TOTAL_UPDATES, "optimizer": "Adam lr=5e-5 weight_decay=1e-6 cosine", "claim": "Local single-seed diagnostic"})
    stores = load_2s_icbhi_stores(samples)
    targets = {p: _core_targets(samples, stores[p]) for p in stores}
    model, selection = train_hierarchical(stores, targets, ("icbhi",), output)
    selected_validation = np.load(
        output / "validation" / f"epoch_{selection['selected_epoch']:03d}.npz",
        allow_pickle=False,
    )
    validation = {k: selected_validation[k] for k in selected_validation.files}
    thresholds = icbhi_validation_thresholds(validation)
    write_json(output / "validation_thresholds.json", thresholds)
    test = infer_core2(
        model,
        stores["test"],
        targets["test"],
        device=torch.device("cpu"),
        datasets=("icbhi",),
    )
    _attach_ground_truth(test, samples, None)
    _save_predictions(output / "selected_test_predictions.npz", test)
    hierarchical = icbhi_hierarchical_metrics(test, thresholds)
    native = native_icbhi_from_hierarchical(test, thresholds)
    write_json(output / "hierarchical_metrics.json", hierarchical)
    write_json(output / "native_comparable_metrics.json", native)
    summary = {
        "condition": "B1_icbhi_only_2s_hierarchical",
        "status": "complete_local_single_seed_diagnostic",
        "selection": selection,
        "thresholds": thresholds,
        "hierarchical_metrics": hierarchical,
        "icbhi_flat4": native,
    }
    write_json(output / "run_summary.json", summary)
    return summary


class Flat4Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(768, 256, bias=True)
        self.head = nn.Linear(256, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.projector(values))


def flat4_targets(samples: Sequence[Sample], store: Mapping[str, np.ndarray]) -> np.ndarray:
    by_id = {sample.sample_id: sample for sample in samples}
    return np.asarray(
        [int(by_id[str(sample_id)].targets["icbhi_flat4"]) for sample_id in store["sample_ids"]],
        dtype=np.int64,
    )


def flat4_inference(
    model: Flat4Head,
    store: Mapping[str, np.ndarray],
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    logits = []
    with torch.no_grad():
        for start in range(0, len(target), 8):
            values = torch.from_numpy(store["embeddings"][start : start + 8, 0])
            logits.append(model(values).cpu())
    logits_t = torch.cat(logits)
    probabilities = torch.softmax(logits_t, dim=1).numpy()
    return {
        "sample_ids": store["sample_ids"],
        "logits": logits_t.numpy(),
        "probabilities": probabilities,
        "predictions": probabilities.argmax(axis=1),
        "targets": target,
    }


def run_b2(samples: Sequence[Sample], root: Path) -> dict[str, object]:
    output = root / "B2_icbhi_2s_flat4/seed_42"
    write_json(output / "config.json", {"condition": "B2", "seed": 42, "encoder": "frozen BEATs iter3+AS2M", "data": "ICBHI-only", "features": "same B1 2s cache", "head": "direct Linear 256->4 flat4 after shared 768->256 projector", "loss": "unweighted cross entropy", "epochs": 50, "updates": TOTAL_UPDATES, "optimizer": "Adam lr=5e-5 weight_decay=1e-6 cosine", "claim": "Local single-seed diagnostic"})
    stores = load_2s_icbhi_stores(samples)
    targets = {p: flat4_targets(samples, stores[p]["icbhi"]) for p in stores}
    set_determinism()
    model = Flat4Head()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_UPDATES)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    best_loss = math.inf
    best_epoch = best_update = update = 0
    started = time.perf_counter()
    eligible = np.ones((len(targets["subtrain"]), 1, 1), dtype=bool)
    schedule_targets = {"icbhi": (np.zeros_like(eligible, dtype=np.float32), eligible)}
    for epoch in range(1, 51):
        model.train()
        losses = []
        for _, indices in matched_batches(schedule_targets, ("icbhi",), epoch):
            values = torch.from_numpy(stores["subtrain"]["icbhi"]["embeddings"][indices, 0])
            target = torch.from_numpy(targets["subtrain"][indices])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(values), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(float(loss.detach()))
        validation = flat4_inference(
            model, stores["validation"]["icbhi"], targets["validation"]
        )
        _save_predictions(output / "validation" / f"epoch_{epoch:03d}.npz", validation)
        logits = validation["logits"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        validation_loss = float(
            -log_probability[np.arange(len(targets["validation"])), targets["validation"]].mean()
        )
        record = {"epoch": epoch, "update": update, "train_loss": float(np.mean(losses)), "validation_flat4_ce": validation_loss}
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_update = update
            torch.save({"epoch": epoch, "update": update, "model": copy.deepcopy(model.state_dict())}, output / "best_checkpoint.pt")
        torch.save({"epoch": epoch, "update": update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, output / "last_checkpoint.pt")
        print(f"epoch {epoch:02d}/50 update={update} train={record['train_loss']:.6f} best_val={best_loss:.6f}", flush=True)
    selected = torch.load(output / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(selected["model"])
    test = flat4_inference(model, stores["test"]["icbhi"], targets["test"])
    _save_predictions(output / "selected_test_predictions.npz", test)
    native = native_metrics(test["targets"], test["predictions"], ICBHI_LABELS)
    native["both_recall"] = native["per_class"]["both"]["recall"]
    native["readout"] = "direct flat4 head"
    write_json(output / "native_comparable_metrics.json", native)
    summary = {
        "condition": "B2_icbhi_only_2s_direct_flat4",
        "status": "complete_local_single_seed_diagnostic",
        "selection": {"selected_epoch": best_epoch, "selected_update": best_update, "selection_loss": best_loss, "elapsed_minutes": (time.perf_counter() - started) / 60},
        "icbhi_flat4": native,
    }
    write_json(output / "run_summary.json", summary)
    return summary


def icbhi_whole_store(
    samples: Sequence[Sample], partition: str, archive_path: Path
) -> dict[str, np.ndarray]:
    archive = np.load(archive_path, allow_pickle=False)
    by_cycle = {
        str(cycle): archive["beats"][index]
        for index, cycle in enumerate(archive["cycle_id"])
    }
    selected = sorted(
        [s for s in samples if s.dataset == "icbhi" and s.partition == partition],
        key=lambda s: s.sample_id,
    )
    embeddings = np.stack(
        [by_cycle[s.sample_id.removeprefix("icbhi:")] for s in selected]
    )[:, None, :].astype(np.float32)
    return {
        "sample_ids": np.asarray([s.sample_id for s in selected]),
        "embeddings": embeddings,
        "window_mask": np.ones((len(selected), 1), dtype=bool),
        "time_map": np.asarray(
            [[[float(s.crop_start_s), float(s.crop_end_s)]] for s in selected],
            dtype=np.float64,
        ),
    }


def build_spr_whole_cache(
    samples: Sequence[Sample],
    partition: str,
    cache_dir: Path,
    backend,
) -> dict[str, np.ndarray]:
    path = cache_dir / f"{partition}_sprsound.npz"
    selected = sorted(
        [s for s in samples if s.dataset == "sprsound" and s.partition == partition],
        key=lambda s: s.sample_id,
    )
    if path.is_file():
        return load_feature_cache(cache_dir, partition, "sprsound", [s.sample_id for s in selected])
    cache_dir.mkdir(parents=True, exist_ok=True)
    values_by_id = {}
    started = time.perf_counter()
    geometry = backend.temporal.geometry
    minimum = geometry.frame_length_samples + (geometry.patch_kernel_time - 1) * geometry.frame_shift_samples
    processing_order = sorted(
        selected,
        key=lambda sample: float(sample.crop_end_s - sample.crop_start_s),
        reverse=True,
    )
    cursor = 0
    completed = 0
    while cursor < len(processing_order):
        duration = float(
            processing_order[cursor].crop_end_s
            - processing_order[cursor].crop_start_s
        )
        batch_size = 1 if duration > 4 else 2 if duration > 2.5 else 8
        current = processing_order[cursor : cursor + batch_size]
        loaded = [load_sample_waveform(s, "SPRSound", outer_test_accessed=partition == "test")[0] for s in current]
        batch = collate_waveforms(loaded)
        model_valid = batch.valid_samples.clamp_min(minimum)
        width = max(batch.waveform.shape[1], minimum)
        waveform = torch.zeros(len(current), width, dtype=torch.float32)
        waveform[:, : batch.waveform.shape[1]] = batch.waveform
        model_batch = WaveformBatch(
            waveform=waveform,
            waveform_padding_mask=(torch.arange(width).unsqueeze(0) >= model_valid.unsqueeze(1)),
            valid_samples=model_valid,
            sample_rate=16_000,
            source_start_s=torch.zeros(len(current), dtype=torch.float64),
            source_end_s=model_valid.to(torch.float64) / 16_000,
            sample_ids=batch.sample_ids,
            dataset_ids=batch.dataset_ids,
            prediction_units=batch.prediction_units,
            lineage=batch.lineage,
        )
        with torch.inference_mode():
            current_values = backend.temporal(model_batch).pooled.cpu().numpy()
        values_by_id.update(
            {
                sample.sample_id: current_values[index]
                for index, sample in enumerate(current)
            }
        )
        cursor += len(current)
        completed += len(current)
        if completed == len(selected) or completed % 200 == 0:
            print(f"whole SPR {partition}: {completed}/{len(selected)} ({(time.perf_counter()-started)/60:.1f} min)", flush=True)
    embeddings = np.stack(
        [values_by_id[sample.sample_id] for sample in selected]
    ).astype(np.float32)[:, None, :]
    store = {
        "sample_ids": np.asarray([s.sample_id for s in selected]),
        "embeddings": embeddings,
        "window_mask": np.ones((len(selected), 1), dtype=bool),
        "time_map": np.asarray([[[float(s.crop_start_s), float(s.crop_end_s)]] for s in selected], dtype=np.float64),
    }
    np.savez(path, **store)
    return store


def run_b3(samples: Sequence[Sample], root: Path, repo_root: Path) -> dict[str, object]:
    output = root / "B3_joint_whole_unit_hierarchical/seed_42"
    write_json(output / "config.json", {"condition": "B3", "seed": 42, "encoder": "frozen BEATs iter3+AS2M", "data": "ICBHI+SPRSound", "features": "one full variable-length native unit embedding; no 2s slicing or truncation", "head": "hierarchical Level1/Crackle/Wheeze", "epochs": 50, "updates": TOTAL_UPDATES, "optimizer": "Adam lr=5e-5 weight_decay=1e-6 cosine", "claim": "Local single-seed diagnostic"})
    source = repo_root / ".cache/multidataset_pipeline/assets/P2/source/repo"
    checkpoint = repo_root / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    backend = load_local_beats_window_backend(source, checkpoint, device="cpu")
    cache_dir = repo_root / ".cache/multidataset_pipeline/core2_beats_whole_unit_seed42"
    icbhi_archive = repo_root / ".cache/features/icbhi_2017/beats_frozen_features_full.npz"
    stores = {}
    for partition in ("subtrain", "validation", "test"):
        stores[partition] = {
            "icbhi": icbhi_whole_store(samples, partition, icbhi_archive),
            "sprsound": build_spr_whole_cache(samples, partition, cache_dir, backend),
        }
    write_json(
        output / "whole_unit_cache_audit.json",
        {
            "icbhi": "reused compatible BEATs_iter3_plus_AS2M full-cycle D768 cache; no truncation",
            "sprsound": "rebuilt full official event D768 cache; variable length; no truncation",
            "longest_icbhi_seconds": 16.163,
            "longest_sprsound_seconds": 7.152,
        },
    )
    targets = {p: _core_targets(samples, stores[p]) for p in stores}
    model, selection = train_hierarchical(stores, targets, ("icbhi", "sprsound"), output)
    selected_validation = np.load(output / "validation" / f"epoch_{selection['selected_epoch']:03d}.npz", allow_pickle=False)
    validation = {k: selected_validation[k] for k in selected_validation.files}
    thresholds, details = select_core_shared_thresholds(validation)
    write_json(output / "validation_thresholds.json", {"thresholds": thresholds, "details": details})
    test = infer_core2(model, stores["test"], targets["test"], device=torch.device("cpu"), datasets=("icbhi", "sprsound"))
    spr_targets = load_terminal_spr_test_targets(list(samples), include_checksums=False)
    scored_targets = _core_targets(samples, stores["test"], spr_terminal_targets=spr_targets)
    test["targets"] = np.concatenate([scored_targets[d][0][stores["test"][d]["window_mask"]] for d in ("icbhi", "sprsound")])
    test["eligible"] = np.concatenate([scored_targets[d][1][stores["test"][d]["window_mask"]] for d in ("icbhi", "sprsound")])
    _attach_ground_truth(test, samples, spr_targets)
    _save_predictions(output / "selected_test_predictions.npz", test)
    core = score_core_primary(test, thresholds)
    icbhi_predictions = _filter_predictions(test, ("icbhi",))
    native_icbhi = native_icbhi_from_hierarchical(icbhi_predictions, thresholds)
    spr_predictions = _filter_predictions(test, ("sprsound",))
    spr_native = native_metrics(spr_predictions["targets"][:, 0].astype(int), spr_predictions["level1_predictions"].astype(int), ("normal", "abnormal"))
    write_json(output / "core_metrics.json", core)
    write_json(output / "native_comparable_metrics.json", {"icbhi_flat4": native_icbhi, "sprsound_inter_binary": spr_native})
    summary = {"condition": "B3_joint_whole_native_unit_hierarchical", "status": "complete_local_single_seed_diagnostic", "selection": selection, "thresholds": thresholds, "core_metrics": core, "icbhi_flat4": native_icbhi, "sprsound_binary": spr_native}
    write_json(output / "run_summary.json", summary)
    return summary


def build_comparison(repo_root: Path, b1, b2, b3) -> dict[str, object]:
    root = repo_root / ROOT_RELATIVE
    b0_summary = json.loads((repo_root / "result/reproduce/core2_hf_positive_kauh_external/BEATs_HF_off/seed_42/run_summary.json").read_text())
    b0_native = json.loads((repo_root / "result/reproduce/core2_hf_positive_kauh_external/BEATs_HF_off/seed_42/native_comparable_metrics.json").read_text())
    conditions = {
        "B0": {"icbhi_flat4": b0_native["icbhi_flat4"], "core_metrics": b0_summary["core_metrics"], "sprsound_binary": b0_native["sprsound_inter_binary"]},
        "B1": b1,
        "B2": b2,
        "B3": b3,
    }
    score = {
        name: value["icbhi_flat4"]["icbhi_score"]
        for name, value in conditions.items()
        if "icbhi_flat4" in value
    }
    deltas = {
        "joint_effect_B1_minus_B0": score["B1"] - score["B0"],
        "head_effect_B2_minus_B1": score["B2"] - score["B1"],
        "whole_unit_effect_B3_minus_B0": (
            score["B3"] - score["B0"] if "B3" in score else None
        ),
    }
    evaluable_deltas = {
        key: value for key, value in deltas.items() if value is not None
    }
    best_axis = max(evaluable_deltas, key=evaluable_deltas.get)
    payload = {
        "status": (
            "partial_complete_B1_B2_with_B3_hold"
            if "B3" not in score
            else "complete_local_single_seed_diagnostic"
        ),
        "conditions": conditions,
        "icbhi_score_deltas": deltas,
        "largest_positive_axis": best_axis,
        "next_causal_decision": {
            "joint_effect_B1_minus_B0": "native retention / remove joint interference",
            "head_effect_B2_minus_B1": "direct native flat4 head",
            "whole_unit_effect_B3_minus_B0": "whole-native-unit representation",
        }[best_axis],
        "historical_local_3_seed_references": {
            "BEATs whole-cycle flat4 focal": 0.6260,
            "BEATs whole-cycle flat4 balanced CE": 0.6208,
        },
        "claim_boundary": "Local single-seed diagnostic; not a paper claim",
    }
    write_json(root / "comparison.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = repo_root / ROOT_RELATIVE
    samples = load_canonical_samples(repo_root)
    b1_path = root / "B1_icbhi_2s_hierarchical/seed_42/run_summary.json"
    b2_path = root / "B2_icbhi_2s_flat4/seed_42/run_summary.json"
    b3_path = root / "B3_joint_whole_unit_hierarchical/seed_42/run_summary.json"
    b1 = json.loads(b1_path.read_text()) if b1_path.is_file() else run_b1(samples, root)
    b2 = json.loads(b2_path.read_text()) if b2_path.is_file() else run_b2(samples, root)
    b3 = json.loads(b3_path.read_text()) if b3_path.is_file() else run_b3(samples, root, repo_root)
    comparison = build_comparison(repo_root, b1, b2, b3)
    print(json.dumps(comparison["icbhi_score_deltas"], indent=2))


if __name__ == "__main__":
    main()
