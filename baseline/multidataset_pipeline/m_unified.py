"""AST M-Unified hierarchical baseline for four native respiratory datasets.

The module reuses the canonical sample constructors, waveform decoding,
source-time sliding windows, and AST frontend already present in this package.
It intentionally writes no checksum or hash fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from baseline.four_dataset_frozen_encoder.data import (
    KAUH_LABELS,
    Sample,
    build_samples,
    load_terminal_spr_test_targets,
)
from baseline.shared_encoder_native_heads.protocol import ICBHI_LABELS, SPR_LABELS

from .ast_window_encoder import ASTWindowBackend, load_local_ast_window_backend
from .beats_window_encoder import (
    BEATsWindowBackend,
    load_local_beats_window_backend,
)
from .real_subtrain_provider import (
    LANE_BY_CANONICAL_DATASET,
    load_sample_waveform,
)
from .sliding_window import collate_sliding_windows, masked_mean_window_embeddings
from .window_encoder import FrozenWindowBackend


SEED = 42
DATASETS = ("icbhi", "sprsound", "hf_lung", "kauh")
NODES = ("level1", "crackle", "wheeze", "other")
PREDICTION_UNITS = {
    "icbhi": "respiratory_cycle",
    "sprsound": "respiratory_event",
    "hf_lung": "source_time_window",
    "kauh": "recording",
}
KAUH_ELIGIBLE = {
    "N": (0, 0, 0, 0),
    "E W": (1, 0, 1, 0),
    "I E W": (1, 0, 1, 0),
    "C": (1, 1, 0, 0),
    "I C": (1, 1, 0, 0),
    "I C E W": (1, 1, 1, 0),
}
KAUH_HOLD = {"Crep", "Bronchial", "I C B"}


def set_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _target_row(
    level1: int | None,
    crackle: int | None,
    wheeze: int | None,
    other: int | None,
) -> dict[str, object]:
    values = {
        "level1": level1,
        "crackle": crackle,
        "wheeze": wheeze,
        "other": other,
    }
    row: dict[str, object] = {}
    for node, value in values.items():
        row[f"{node}_target"] = value
        row[f"{node}_eligible"] = value is not None
    return row


def map_native_sample(
    sample: Sample,
    *,
    spr_terminal_targets: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    """Map one non-HF native unit to the frozen hierarchy."""

    if sample.dataset == "icbhi":
        raw = ICBHI_LABELS[int(sample.targets["icbhi_flat4"])]
        mapped = {
            "normal": (0, 0, 0, 0),
            "crackle": (1, 1, 0, 0),
            "wheeze": (1, 0, 1, 0),
            "both": (1, 1, 1, 0),
        }[raw]
        status = "exact_flat4_mapping"
    elif sample.dataset == "sprsound":
        if "spr_seven" in sample.targets:
            raw = SPR_LABELS[int(sample.targets["spr_seven"])]
        elif spr_terminal_targets and sample.sample_id in spr_terminal_targets:
            raw = SPR_LABELS[int(spr_terminal_targets[sample.sample_id]["spr_seven"])]
        else:
            return {
                "raw_label": "terminal_label_deferred",
                "mapping_status": "terminal_label_deferred_until_after_selection",
                **_target_row(None, None, None, None),
            }
        if raw == "Normal":
            mapped = (0, 0, 0, 0)
            status = "exact_compatible_mapping"
        elif raw in {"Coarse Crackle", "Fine Crackle"}:
            mapped = (1, 1, 0, 0)
            status = "exact_compatible_mapping"
        elif raw == "Wheeze":
            mapped = (1, 0, 1, 0)
            status = "exact_compatible_mapping"
        elif raw == "Wheeze+Crackle":
            mapped = (1, 1, 1, 0)
            status = "exact_compatible_mapping"
        elif raw in {"Rhonchi", "Stridor"}:
            mapped = (1, None, None, 1)
            status = "proposed_other_positive_attributes_unknown"
        else:
            raise ValueError(f"unsupported SPRSound raw label: {raw}")
    elif sample.dataset == "kauh":
        raw = str(sample.metadata["raw_sound"])
        if raw in KAUH_ELIGIBLE:
            mapped = KAUH_ELIGIBLE[raw]
            status = "source_defined_eligible_subset"
        elif raw in KAUH_HOLD:
            mapped = (None, None, None, None)
            status = "HOLD_unresolved_source_sound"
        else:
            raise ValueError(f"unsupported KAUH raw label: {raw}")
    else:
        raise ValueError(f"map_native_sample does not accept {sample.dataset}")
    return {
        "raw_label": raw,
        "mapping_status": status,
        **_target_row(*mapped),
    }


def map_hf_window(
    sample: Sample,
    window_index: int,
    source_start_s: float,
    source_end_s: float,
) -> dict[str, object]:
    """Map explicit HF positive intervals at a 2-second window center."""

    center = (source_start_s + source_end_s) / 2
    intervals = sample.metadata.get("raw_intervals", [])
    active_tokens = sorted(
        {
            str(token)
            for token, start, end in intervals
            if float(start) <= center < float(end)
        }
    )
    crackle = 1 if "D" in active_tokens else None
    wheeze = 1 if "Wheeze" in active_tokens else None
    other = 1 if {"Rhonchi", "Stridor"} & set(active_tokens) else None
    return {
        "prediction_id": f"{sample.sample_id}::window_{window_index:02d}",
        "raw_label": active_tokens,
        "raw_semantics": "interval_tokens_at_window_center;I/E_are_phase_only;gap_is_masked",
        "mapping_status": (
            "explicit_positive_attribute_supervision"
            if any(value is not None for value in (crackle, wheeze, other))
            else "masked_phase_or_gap"
        ),
        **_target_row(None, crackle, wheeze, other),
    }


def ledger_rows_for_sample(
    sample: Sample,
    *,
    spr_terminal_targets: Mapping[str, Mapping[str, int]] | None = None,
) -> list[dict[str, object]]:
    common = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "partition": sample.partition,
        "group_id": sample.group_id,
        "prediction_unit": PREDICTION_UNITS[sample.dataset],
        "audio_path": sample.audio_path,
    }
    if sample.dataset != "hf_lung":
        mapped = map_native_sample(sample, spr_terminal_targets=spr_terminal_targets)
        return [
            {
                **common,
                "prediction_id": sample.sample_id,
                "source_start_s": float(sample.crop_start_s or 0.0),
                "source_end_s": (
                    float(sample.crop_end_s)
                    if sample.crop_end_s is not None
                    else None
                ),
                **mapped,
            }
        ]
    rows = []
    for index, start in enumerate(range(0, 14)):
        mapped = map_hf_window(sample, index, float(start), float(start + 2))
        rows.append(
            {
                **common,
                "source_start_s": float(start),
                "source_end_s": float(start + 2),
                **mapped,
            }
        )
    return rows


def support_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        partitions: dict[str, object] = {}
        for partition in ("subtrain", "validation", "test"):
            current = [row for row in selected if row["partition"] == partition]
            nodes = {}
            for node in NODES:
                eligible = [row for row in current if row[f"{node}_eligible"]]
                nodes[node] = {
                    "eligible_prediction_rows": len(eligible),
                    "positive": sum(int(row[f"{node}_target"] == 1) for row in eligible),
                    "negative": sum(int(row[f"{node}_target"] == 0) for row in eligible),
                }
            partitions[partition] = {
                "native_units": len({str(row["sample_id"]) for row in current}),
                "prediction_rows": len(current),
                "nodes": nodes,
            }
        output[dataset] = {
            "prediction_unit": PREDICTION_UNITS[dataset],
            "partitions": partitions,
        }
    return {
        "datasets": output,
        "cross_dataset_total": None,
        "cross_dataset_total_reason": "prediction units differ and are not added",
    }


def ontology_mapping() -> dict[str, object]:
    return {
        "nodes": {
            "level1": ["Normal", "Abnormal"],
            "crackle": "sigmoid binary node",
            "wheeze": "sigmoid binary node",
            "other": "optional sigmoid node on explicitly eligible rows",
        },
        "ICBHI": "exact flat4 mapping",
        "SPRSound": {
            "compatible": [
                "Normal",
                "Coarse Crackle",
                "Fine Crackle",
                "Wheeze",
                "Wheeze+Crackle",
            ],
            "proposed_other": ["Rhonchi", "Stridor"],
            "rhonchi_stridor_attribute_policy": "Crackle/Wheeze unknown; Other positive",
        },
        "HF": {
            "raw_tokens": ["I", "E", "D", "Wheeze", "Rhonchi", "Stridor"],
            "mapping": {
                "D": "Crackle positive candidate",
                "Wheeze": "Wheeze positive",
                "Rhonchi": "Other positive",
                "Stridor": "Other positive",
            },
            "level1": "masked",
            "gap": "masked, never negative",
            "negative_support": "none; positive-only supervision",
        },
        "KAUH": {
            "eligible": sorted(KAUH_ELIGIBLE),
            "hold": sorted(KAUH_HOLD),
            "replicas": "B/D/E remain in the same P-number group",
        },
    }


def load_canonical_samples(repo_root: Path) -> list[Sample]:
    samples, _ = build_samples(
        repo_root / "dataset/raw", kauh_outer_fold=0, include_checksums=False
    )
    return samples


def build_ground_truth_ledger(repo_root: Path, output_dir: Path) -> dict[str, object]:
    samples = load_canonical_samples(repo_root)
    rows = [row for sample in samples for row in ledger_rows_for_sample(sample)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for partition in ("subtrain", "validation"):
        write_jsonl(
            output_dir / f"ledger_{partition}.jsonl",
            [row for row in rows if row["partition"] == partition],
        )
    write_jsonl(
        output_dir / "ledger_test_manifest.jsonl",
        [row for row in rows if row["partition"] == "test"],
    )
    hf_intervals = [
        {
            "sample_id": sample.sample_id,
            "partition": sample.partition,
            "recording_state": sample.metadata.get("recording_state"),
            "raw_intervals": sample.metadata.get("raw_intervals", []),
            "gap_semantics": "unlabeled gap is masked, not negative",
        }
        for sample in samples
        if sample.dataset == "hf_lung"
    ]
    write_jsonl(output_dir / "hf_raw_intervals.jsonl", hf_intervals)
    summary = support_summary(rows)
    write_json(output_dir / "support_summary.json", summary)
    write_json(output_dir / "ontology_mapping.json", ontology_mapping())
    manifest = {
        "status": "ground_truth_ledger_built",
        "prediction_units_kept_separate": True,
        "rows_by_partition": dict(Counter(row["partition"] for row in rows)),
        "native_units_by_dataset": dict(Counter(sample.dataset for sample in samples)),
        "spr_test_labels": "deferred until selected predictions are written",
        "files": [
            "ledger_subtrain.jsonl",
            "ledger_validation.jsonl",
            "ledger_test_manifest.jsonl",
            "hf_raw_intervals.jsonl",
            "support_summary.json",
            "ontology_mapping.json",
        ],
    }
    write_json(output_dir / "ledger_manifest.json", manifest)
    return manifest


def _cache_path(cache_dir: Path, partition: str, dataset: str) -> Path:
    return cache_dir / f"{partition}_{dataset}.npz"


def load_feature_cache(
    cache_dir: Path,
    partition: str,
    dataset: str,
    expected_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    archive = np.load(_cache_path(cache_dir, partition, dataset), allow_pickle=False)
    values = {key: archive[key] for key in archive.files}
    if values["sample_ids"].tolist() != list(expected_ids):
        raise RuntimeError("feature cache order differs from canonical samples")
    return values


def build_feature_cache(
    samples: Sequence[Sample],
    partition: str,
    dataset: str,
    backend: FrozenWindowBackend,
    cache_dir: Path,
    *,
    device: torch.device,
    native_batch_size: int = 8,
    encoder_window_batch_size: int = 2,
) -> dict[str, np.ndarray]:
    selected = sorted(
        [sample for sample in samples if sample.partition == partition and sample.dataset == dataset],
        key=lambda sample: sample.sample_id,
    )
    expected_ids = [sample.sample_id for sample in selected]
    path = _cache_path(cache_dir, partition, dataset)
    if path.is_file():
        print(f"reuse feature cache {path}", flush=True)
        return load_feature_cache(cache_dir, partition, dataset, expected_ids)
    cache_dir.mkdir(parents=True, exist_ok=True)
    unit_embeddings: list[np.ndarray] = []
    unit_time_maps: list[np.ndarray] = []
    started = time.perf_counter()
    lane = LANE_BY_CANONICAL_DATASET[dataset]
    for batch_start in range(0, len(selected), native_batch_size):
        batch_samples = selected[batch_start : batch_start + native_batch_size]
        waveforms = [
            load_sample_waveform(
                sample,
                lane,
                outer_test_accessed=partition == "test",
            )[0]
            for sample in batch_samples
        ]
        window_batch = collate_sliding_windows(waveforms)
        flat_valid = window_batch.window_mask.reshape(-1)
        flat_waveforms = window_batch.waveform_windows.reshape(-1, 32_000)[flat_valid]
        flat_lengths = window_batch.valid_samples.reshape(-1)[flat_valid]
        encoded = []
        with torch.inference_mode():
            for start in range(0, flat_waveforms.shape[0], encoder_window_batch_size):
                values = backend.encode_valid_windows(
                    flat_waveforms[start : start + encoder_window_batch_size].to(device),
                    flat_lengths[start : start + encoder_window_batch_size].to(device),
                )
                encoded.append(values.detach().cpu())
        flat_embeddings = torch.cat(encoded)
        embedding_dim = backend.native_dim
        restored = torch.zeros(
            window_batch.window_mask.numel(), embedding_dim, dtype=torch.float32
        )
        restored[flat_valid] = flat_embeddings
        restored = restored.reshape(
            *window_batch.window_mask.shape, embedding_dim
        )
        if dataset != "hf_lung":
            pooled = masked_mean_window_embeddings(
                restored,
                window_batch.window_mask,
                expected_dim=embedding_dim,
            )
            for row in range(len(batch_samples)):
                count = int(window_batch.window_mask[row].sum())
                unit_embeddings.append(pooled[row : row + 1].numpy())
                unit_time_maps.append(
                    np.asarray(
                        [[
                            float(window_batch.time_map[row, 0, 0]),
                            float(window_batch.time_map[row, count - 1, 1]),
                        ]],
                        dtype=np.float64,
                    )
                )
        else:
            for row in range(len(batch_samples)):
                count = int(window_batch.window_mask[row].sum())
                unit_embeddings.append(restored[row, :count].numpy())
                unit_time_maps.append(window_batch.time_map[row, :count].numpy())
        completed = min(batch_start + native_batch_size, len(selected))
        if completed == len(selected) or completed % 200 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"cache {partition}/{dataset}: {completed}/{len(selected)} units "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
    max_windows = max(values.shape[0] for values in unit_embeddings)
    embedding_dim = unit_embeddings[0].shape[-1]
    embeddings = np.zeros(
        (len(selected), max_windows, embedding_dim), dtype=np.float32
    )
    window_mask = np.zeros((len(selected), max_windows), dtype=bool)
    time_map = np.zeros((len(selected), max_windows, 2), dtype=np.float64)
    for row, (values, times) in enumerate(zip(unit_embeddings, unit_time_maps)):
        count = values.shape[0]
        embeddings[row, :count] = values
        window_mask[row, :count] = True
        time_map[row, :count] = times
    np.savez(
        path,
        sample_ids=np.asarray(expected_ids),
        embeddings=embeddings,
        window_mask=window_mask,
        time_map=time_map,
    )
    return {
        "sample_ids": np.asarray(expected_ids),
        "embeddings": embeddings,
        "window_mask": window_mask,
        "time_map": time_map,
    }


class MUnifiedHead(nn.Module):
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
        self.other = nn.Linear(256, 1)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.projector(self.dimension_adapter(embeddings))
        return {
            "level1": self.level1(projected),
            "crackle": self.crackle(projected).squeeze(-1),
            "wheeze": self.wheeze(projected).squeeze(-1),
            "other": self.other(projected).squeeze(-1),
        }


def target_arrays(
    samples: Sequence[Sample],
    feature_store: Mapping[str, np.ndarray],
    *,
    spr_terminal_targets: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    by_id = {sample.sample_id: sample for sample in samples}
    ids = feature_store["sample_ids"].tolist()
    windows = feature_store["window_mask"].shape[1]
    targets = np.zeros((len(ids), windows, len(NODES)), dtype=np.float32)
    eligible = np.zeros((len(ids), windows, len(NODES)), dtype=bool)
    for row, sample_id in enumerate(ids):
        sample = by_id[str(sample_id)]
        mapped_rows = ledger_rows_for_sample(
            sample, spr_terminal_targets=spr_terminal_targets
        )
        valid_windows = int(feature_store["window_mask"][row].sum())
        if len(mapped_rows) != valid_windows:
            raise RuntimeError("ledger rows and cached windows do not align")
        for window, mapped in enumerate(mapped_rows):
            for node_index, node in enumerate(NODES):
                if mapped[f"{node}_eligible"]:
                    eligible[row, window, node_index] = True
                    targets[row, window, node_index] = float(mapped[f"{node}_target"])
    return targets, eligible


def eligible_node_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    named: dict[str, float] = {}
    level1_mask = eligible[..., 0]
    if bool(level1_mask.any()):
        value = F.cross_entropy(
            logits["level1"][level1_mask], targets[..., 0][level1_mask].long()
        )
        losses.append(value)
        named["level1"] = float(value.detach())
    for index, node in enumerate(NODES[1:], start=1):
        mask = eligible[..., index]
        if bool(mask.any()):
            value = F.binary_cross_entropy_with_logits(
                logits[node][mask], targets[..., index][mask]
            )
            losses.append(value)
            named[node] = float(value.detach())
    if not losses:
        raise RuntimeError("batch contains no eligible hierarchical node")
    return torch.stack(losses).mean(), named


def _stores_for_partition(
    samples: Sequence[Sample],
    partition: str,
    backend: FrozenWindowBackend,
    cache_dir: Path,
    *,
    device: torch.device,
    encoder_window_batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
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
        for dataset in DATASETS
    }


def _prepare_targets(
    samples: Sequence[Sample],
    stores: Mapping[str, Mapping[str, np.ndarray]],
    *,
    spr_terminal_targets: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        dataset: target_arrays(
            samples,
            stores[dataset],
            spr_terminal_targets=spr_terminal_targets,
        )
        for dataset in stores
    }


def _epoch_batches(
    target_sets: Mapping[str, tuple[np.ndarray, np.ndarray]], epoch: int
) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(SEED + epoch)
    batches: list[tuple[str, list[int]]] = []
    for dataset in DATASETS:
        eligible = target_sets[dataset][1]
        active = np.flatnonzero(eligible.any(axis=(1, 2)))
        order = rng.permutation(active).tolist()
        batches.extend(
            (dataset, order[start : start + 8])
            for start in range(0, len(order), 8)
        )
    rng.shuffle(batches)
    return batches


def _prediction_id(sample_id: str, dataset: str, window: int) -> str:
    if dataset == "hf_lung":
        return f"{sample_id}::window_{window:02d}"
    return sample_id


def infer_partition(
    model: MUnifiedHead,
    stores: Mapping[str, Mapping[str, np.ndarray]],
    target_sets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    datasets: Sequence[str] = DATASETS,
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
            targets, eligible = target_sets[dataset]
            for start in range(0, len(store["sample_ids"]), 8):
                stop = min(start + 8, len(store["sample_ids"]))
                values = torch.from_numpy(store["embeddings"][start:stop]).to(device)
                logits = model(values)
                level1_logits = logits["level1"].cpu().numpy()
                level1_prob = torch.softmax(logits["level1"], dim=-1).cpu().numpy()
                attribute_logits = torch.stack(
                    [logits["crackle"], logits["wheeze"], logits["other"]], dim=-1
                ).cpu().numpy()
                attribute_prob = torch.sigmoid(
                    torch.from_numpy(attribute_logits)
                ).numpy()
                for local, global_row in enumerate(range(start, stop)):
                    count = int(store["window_mask"][global_row].sum())
                    sample_id = str(store["sample_ids"][global_row])
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
                    fields["source_start_s"].append(
                        store["time_map"][global_row, :count, 0]
                    )
                    fields["source_end_s"].append(
                        store["time_map"][global_row, :count, 1]
                    )
                    fields["level1_logits"].append(level1_logits[local, :count])
                    fields["level1_probabilities"].append(level1_prob[local, :count])
                    fields["level1_predictions"].append(
                        level1_prob[local, :count].argmax(axis=-1)
                    )
                    fields["attribute_logits"].append(attribute_logits[local, :count])
                    fields["attribute_probabilities"].append(attribute_prob[local, :count])
                    fields["targets"].append(targets[global_row, :count])
                    fields["eligible"].append(eligible[global_row, :count])
    return {key: np.concatenate(values, axis=0) for key, values in fields.items()}


def _save_predictions(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


def validation_losses(
    predictions: Mapping[str, np.ndarray],
    *,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, object]:
    dataset_losses: dict[str, float] = {}
    node_losses: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        rows = predictions["dataset_ids"] == dataset
        current: dict[str, float] = {}
        for node_index, node in enumerate(NODES):
            mask = rows & predictions["eligible"][:, node_index]
            if not mask.any():
                continue
            target = predictions["targets"][mask, node_index]
            if node == "level1":
                logits = predictions["level1_logits"][mask]
                shifted = logits - logits.max(axis=1, keepdims=True)
                log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                current[node] = float(-log_prob[np.arange(len(target)), target.astype(int)].mean())
            else:
                column = NODES.index(node) - 1
                logits = predictions["attribute_logits"][mask, column]
                current[node] = float(
                    (np.logaddexp(0.0, logits) - target * logits).mean()
                )
        if current:
            dataset_losses[dataset] = float(np.mean(list(current.values())))
            node_losses[dataset] = current
    return {
        "dataset_node_losses": node_losses,
        "dataset_losses": dataset_losses,
        "selection_loss": float(np.mean(list(dataset_losses.values()))),
        "dataset_weighting": "equal over active datasets",
        "node_weighting": "equal over active eligible nodes within dataset",
    }


def select_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probability)))
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in candidates:
        value = f1_score(target, probability >= threshold, zero_division=0)
        if value > best_f1 or (value == best_f1 and threshold > best_threshold):
            best_f1 = float(value)
            best_threshold = float(threshold)
    return best_threshold


def select_validation_thresholds(
    predictions: Mapping[str, np.ndarray]
) -> dict[str, float]:
    thresholds = {}
    for node_index, node in enumerate(NODES[1:], start=1):
        mask = predictions["eligible"][:, node_index]
        column = node_index - 1
        thresholds[node] = select_threshold(
            predictions["targets"][mask, node_index].astype(int),
            predictions["attribute_probabilities"][mask, column],
        )
    return thresholds


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    target = target.astype(int)
    prediction = prediction.astype(int)
    tn = int(((target == 0) & (prediction == 0)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    tp = int(((target == 1) & (prediction == 1)).sum())
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "support": {
            "eligible": total,
            "positive": tp + fn,
            "negative": tn + fp,
        },
        "confusion": [[tn, fp], [fn, tp]],
    }


def score_predictions(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, object]:
    per_dataset: dict[str, object] = {}
    for dataset in datasets:
        dataset_rows = predictions["dataset_ids"] == dataset
        nodes: dict[str, object] = {}
        f1_values = []
        for node_index, node in enumerate(NODES):
            mask = dataset_rows & predictions["eligible"][:, node_index]
            if not mask.any():
                nodes[node] = {"status": "masked_no_eligible_support"}
                continue
            target = predictions["targets"][mask, node_index]
            if node == "level1":
                predicted = predictions["level1_predictions"][mask]
            else:
                predicted = (
                    predictions["attribute_probabilities"][mask, node_index - 1]
                    >= thresholds[node]
                )
            metric = binary_metrics(target, predicted)
            nodes[node] = metric
            f1_values.append(float(metric["f1"]))
        per_dataset[dataset] = {
            "prediction_unit": PREDICTION_UNITS[dataset],
            "nodes": nodes,
            "unified_node_macro_f1": float(np.mean(f1_values)),
        }
    dataset_scores = {
        dataset: float(value["unified_node_macro_f1"])
        for dataset, value in per_dataset.items()
    }
    return {
        "per_dataset": per_dataset,
        "dataset_macro_f1": float(np.mean(list(dataset_scores.values()))),
        "dataset_macro_interpretation": (
            "mean of each dataset's eligible-node F1 values, including one-class "
            "supports; not a pooled score and not directly comparable across support semantics"
        ),
        "hf_positive_only_caveat": (
            "HF eligible rows contain positives only; specificity is undefined and "
            "high F1 does not establish a complete detector"
        ),
        "worst_dataset": min(dataset_scores, key=dataset_scores.get),
        "worst_dataset_f1": min(dataset_scores.values()),
        "dataset_scores": dataset_scores,
    }


def pooled_auxiliary(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    nodes = {}
    for node_index, node in enumerate(NODES):
        mask = predictions["eligible"][:, node_index]
        target = predictions["targets"][mask, node_index]
        if node == "level1":
            predicted = predictions["level1_predictions"][mask]
        else:
            predicted = (
                predictions["attribute_probabilities"][mask, node_index - 1]
                >= thresholds[node]
            )
        nodes[node] = binary_metrics(target, predicted)
    return {
        "status": "auxiliary_only",
        "warning": "prediction units differ; pooled confusion is not a primary result",
        "nodes": nodes,
    }


def _active_unit_counts(
    target_sets: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, int]:
    return {
        dataset: int(eligible.any(axis=(1, 2)).sum())
        for dataset, (_, eligible) in target_sets.items()
    }


def train_hierarchical_head(
    stores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    target_sets: Mapping[
        str, Mapping[str, tuple[np.ndarray, np.ndarray]]
    ],
    output_dir: Path,
    *,
    device: torch.device,
) -> tuple[MUnifiedHead, dict[str, object]]:
    set_determinism()
    encoder_dims = {
        int(store["embeddings"].shape[-1])
        for partition in stores.values()
        for store in partition.values()
    }
    if len(encoder_dims) != 1:
        raise RuntimeError("all four lanes must share one encoder embedding dimension")
    model = MUnifiedHead(encoder_dims.pop()).to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    updates_per_epoch = sum(
        math.ceil(value / 8)
        for value in _active_unit_counts(target_sets["subtrain"]).values()
    )
    total_updates = 50 * updates_per_epoch
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
    history = []
    started = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        losses = []
        node_loss_values: dict[str, list[float]] = {node: [] for node in NODES}
        for dataset, indices in _epoch_batches(target_sets["subtrain"], epoch):
            store = stores["subtrain"][dataset]
            targets, eligible = target_sets["subtrain"][dataset]
            embeddings = torch.from_numpy(store["embeddings"][indices]).to(device)
            target = torch.from_numpy(targets[indices]).to(device)
            mask = torch.from_numpy(eligible[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, node_losses = eligible_node_loss(model(embeddings), target, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            losses.append(float(loss.detach()))
            for node, value in node_losses.items():
                node_loss_values[node].append(value)
        validation = infer_partition(model, stores["validation"], target_sets["validation"], device=device)
        validation_path = output_dir / "validation" / f"epoch_{epoch:03d}.npz"
        _save_predictions(validation_path, validation)
        val_loss = validation_losses(validation)
        record = {
            "epoch": epoch,
            "update": update,
            "train_loss": float(np.mean(losses)),
            "train_node_losses": {
                node: float(np.mean(values))
                for node, values in node_loss_values.items()
                if values
            },
            "validation": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.perf_counter() - started) / 60,
        }
        append_jsonl(log_path, record)
        history.append(record)
        if float(val_loss["selection_loss"]) < best_loss:
            best_loss = float(val_loss["selection_loss"])
            best_epoch = epoch
            best_update = update
            torch.save(
                {
                    "epoch": epoch,
                    "update": update,
                    "model": copy.deepcopy(model.state_dict()),
                    "validation_selection_loss": best_loss,
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
            },
            output_dir / "last_checkpoint.pt",
        )
        print(
            f"epoch {epoch:02d}/50 update={update} "
            f"train={record['train_loss']:.6f} val={best_loss:.6f}",
            flush=True,
        )
    selected = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(selected["model"])
    return model, {
        "selected_epoch": best_epoch,
        "selected_update": best_update,
        "selection_loss": best_loss,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "active_units": _active_unit_counts(target_sets["subtrain"]),
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "history_epochs": len(history),
    }


def _complete_test_ledger(
    samples: Sequence[Sample],
    spr_targets: Mapping[str, Mapping[str, int]],
    ledger_dir: Path,
) -> None:
    all_rows = [
        row
        for sample in samples
        for row in ledger_rows_for_sample(sample, spr_terminal_targets=spr_targets)
    ]
    write_jsonl(
        ledger_dir / "ledger_test.jsonl",
        [row for row in all_rows if row["partition"] == "test"],
    )
    write_json(ledger_dir / "support_summary.json", support_summary(all_rows))


def run_ast_m_unified(
    repo_root: Path,
    result_dir: Path,
    *,
    device_name: str = "cpu",
    encoder_window_batch_size: int = 8,
) -> dict[str, object]:
    set_determinism()
    device = torch.device(device_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = repo_root / "result/reproduce/unified/ground_truth_ledger"
    if not (ledger_dir / "ledger_manifest.json").is_file():
        build_ground_truth_ledger(repo_root, ledger_dir)
    samples = load_canonical_samples(repo_root)
    source_repo = (
        repo_root / ".cache/icbhi_sprsound_shared_encoder_native_heads/source/repo"
    )
    requested_checkpoint = (
        repo_root
        / ".cache/icbhi_sprsound_shared_encoder_native_heads/checkpoints/audioset_0.4593_runtime_cv4.pth"
    )
    checkpoint = (
        repo_root
        / ".cache/icbhi_sprsound_shared_encoder_native_heads/checkpoints/hf_ast_legacy_compat.pth"
    )
    config = {
        "experiment": "AST_M_Unified",
        "seed": SEED,
        "sample_rate": 16_000,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "native_batch_size": 8,
        "homogeneous_dataset_batches": True,
        "encoder": "frozen pretrained AST AudioSet",
        "ast_frontend": "two-second 198-frame fbank grid; no extra six-second frontend padding",
        "package_definition": "AST_2s_native_grid_v0",
        "package_caveat": (
            "new 2s-native AST package; not matched to the historical P1 "
            "798-frame padded package"
        ),
        "single_dataset_comparator_requirement": (
            "future Hanlin AST S-Native must use AST_2s_native_grid_v0 before "
            "PH-Unified versus M-Unified comparison"
        ),
        "requested_checkpoint_issue": (
            f"{requested_checkpoint} is an HTML document and is not loadable"
        ),
        "checkpoint_used": str(checkpoint),
        "source_repo": str(source_repo),
        "projector": "trainable biased Linear 768->256",
        "heads": {
            "level1": "Linear 256->2 cross_entropy",
            "crackle": "Linear 256->1 BCE",
            "wheeze": "Linear 256->1 BCE",
            "other": "Linear 256->1 BCE on explicit eligible rows",
        },
        "epochs": 50,
        "optimizer": "Adam",
        "learning_rate": 5e-5,
        "weight_decay": 1e-6,
        "schedule": "cosine per update; no warmup",
        "precision": "FP32",
        "augmentation": False,
        "gradient_accumulation": 1,
        "sampler": "source-proportional homogeneous batches",
        "selection": "equal mean of per-dataset eligible-node validation loss; tie earliest epoch",
        "test_policy": "one inference after validation selection",
        "ledger_reference": str(ledger_dir),
        "canonical_split_note": "existing canonical split assignments retained; training randomness uses seed 42",
        "device": str(device),
    }
    write_json(result_dir / "config.json", config)
    backend = load_local_ast_window_backend(
        source_repo, checkpoint, device=device
    )
    cache_dir = repo_root / ".cache/multidataset_pipeline/m_unified_ast_seed42"
    stores = {
        partition: _stores_for_partition(
            samples,
            partition,
            backend,
            cache_dir,
            device=device,
            encoder_window_batch_size=encoder_window_batch_size,
        )
        for partition in ("subtrain", "validation")
    }
    targets = {
        partition: _prepare_targets(samples, stores[partition])
        for partition in ("subtrain", "validation")
    }
    model, selection = train_hierarchical_head(
        stores,
        targets,
        result_dir,
        device=device,
    )
    selected_validation = np.load(
        result_dir / "validation" / f"epoch_{selection['selected_epoch']:03d}.npz",
        allow_pickle=False,
    )
    validation_predictions = {
        key: selected_validation[key] for key in selected_validation.files
    }
    thresholds = select_validation_thresholds(validation_predictions)
    write_json(
        result_dir / "selected_validation_thresholds.json",
        {
            "thresholds": thresholds,
            "policy": "per-node validation max-F1; tie higher threshold",
            "outer_test_used": False,
        },
    )
    test_stores = _stores_for_partition(
        samples,
        "test",
        backend,
        cache_dir,
        device=device,
        encoder_window_batch_size=encoder_window_batch_size,
    )
    test_targets_label_free = _prepare_targets(samples, test_stores)
    test_predictions = infer_partition(
        model, test_stores, test_targets_label_free, device=device
    )
    label_free = {
        key: value
        for key, value in test_predictions.items()
        if key not in {"targets", "eligible"}
    }
    _save_predictions(result_dir / "selected_test_predictions_label_free.npz", label_free)
    spr_targets = load_terminal_spr_test_targets(samples, include_checksums=False)
    scored_targets = _prepare_targets(
        samples, test_stores, spr_terminal_targets=spr_targets
    )
    scored_target_rows = []
    scored_eligible_rows = []
    for dataset in DATASETS:
        target_values, eligible_values = scored_targets[dataset]
        mask = test_stores[dataset]["window_mask"]
        scored_target_rows.append(target_values[mask])
        scored_eligible_rows.append(eligible_values[mask])
    test_predictions["targets"] = np.concatenate(scored_target_rows)
    test_predictions["eligible"] = np.concatenate(scored_eligible_rows)
    _save_predictions(result_dir / "selected_test_predictions.npz", test_predictions)
    metrics = score_predictions(test_predictions, thresholds)
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        result_dir / "combined_pooled_auxiliary.json",
        pooled_auxiliary(test_predictions, thresholds),
    )
    _complete_test_ledger(samples, spr_targets, ledger_dir)
    summary = {
        "status": "AST_M_Unified_50_epochs_complete",
        "device": str(device),
        "selection": selection,
        "thresholds": thresholds,
        "metrics": metrics,
        "metric_interpretation": (
            "dataset/node F1 values include one-class eligible supports where present; "
            "HF is positive-only and its near-one F1 is not evidence of specificity or "
            "a closed detector"
        ),
        "test_inference_runs": 1,
        "encoder_frozen": True,
        "limitations": [
            "AST_2s_native_grid_v0 is a new package and is not directly matched to the historical 798-frame padded P1 result",
            "Hanlin AST S-Native must use the same AST_2s_native_grid_v0 package for PH-Unified versus M-Unified comparison",
            "HF has positive-only attribute supervision; Level1 and gaps are masked",
            "KAUH Crep/Bronchial/I C B remain masked and outside the claim",
            "SPRSound Rhonchi/Stridor use Proposed Other while Crackle/Wheeze remain unknown",
            "different native prediction units are not a pooled primary result",
        ],
    }
    write_json(result_dir / "run_summary.json", summary)
    return summary


def _run_encoder_m_unified(
    repo_root: Path,
    result_dir: Path,
    *,
    backend: FrozenWindowBackend,
    cache_dir: Path,
    config: Mapping[str, object],
    status: str,
    limitations: Sequence[str],
    device: torch.device,
    encoder_window_batch_size: int,
) -> dict[str, object]:
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "config.json", dict(config))
    samples = load_canonical_samples(repo_root)
    stores = {
        partition: _stores_for_partition(
            samples,
            partition,
            backend,
            cache_dir,
            device=device,
            encoder_window_batch_size=encoder_window_batch_size,
        )
        for partition in ("subtrain", "validation")
    }
    targets = {
        partition: _prepare_targets(samples, stores[partition])
        for partition in ("subtrain", "validation")
    }
    model, selection = train_hierarchical_head(
        stores, targets, result_dir, device=device
    )
    selected_validation = np.load(
        result_dir / "validation" / f"epoch_{selection['selected_epoch']:03d}.npz",
        allow_pickle=False,
    )
    validation_predictions = {
        key: selected_validation[key] for key in selected_validation.files
    }
    thresholds = select_validation_thresholds(validation_predictions)
    write_json(
        result_dir / "selected_validation_thresholds.json",
        {
            "thresholds": thresholds,
            "policy": "per-node validation max-F1; tie higher threshold",
            "outer_test_used": False,
        },
    )
    test_stores = _stores_for_partition(
        samples,
        "test",
        backend,
        cache_dir,
        device=device,
        encoder_window_batch_size=encoder_window_batch_size,
    )
    label_free_targets = _prepare_targets(samples, test_stores)
    test_predictions = infer_partition(
        model, test_stores, label_free_targets, device=device
    )
    _save_predictions(
        result_dir / "selected_test_predictions_label_free.npz",
        {
            key: value
            for key, value in test_predictions.items()
            if key not in {"targets", "eligible"}
        },
    )
    spr_targets = load_terminal_spr_test_targets(
        samples, include_checksums=False
    )
    scored_targets = _prepare_targets(
        samples, test_stores, spr_terminal_targets=spr_targets
    )
    target_rows = []
    eligible_rows = []
    for dataset in DATASETS:
        target_values, eligible_values = scored_targets[dataset]
        mask = test_stores[dataset]["window_mask"]
        target_rows.append(target_values[mask])
        eligible_rows.append(eligible_values[mask])
    test_predictions["targets"] = np.concatenate(target_rows)
    test_predictions["eligible"] = np.concatenate(eligible_rows)
    _save_predictions(
        result_dir / "selected_test_predictions.npz", test_predictions
    )
    metrics = score_predictions(test_predictions, thresholds)
    write_json(result_dir / "metrics.json", metrics)
    write_json(
        result_dir / "combined_pooled_auxiliary.json",
        pooled_auxiliary(test_predictions, thresholds),
    )
    summary = {
        "status": status,
        "device": str(device),
        "selection": selection,
        "thresholds": thresholds,
        "metrics": metrics,
        "metric_interpretation": (
            "dataset/node F1 values include one-class eligible supports where present; "
            "HF is positive-only and its near-one F1 is not evidence of specificity or "
            "a closed detector"
        ),
        "test_inference_runs": 1,
        "encoder_frozen": True,
        "limitations": list(limitations),
    }
    write_json(result_dir / "run_summary.json", summary)
    return summary


def run_beats_m_unified(
    repo_root: Path,
    result_dir: Path,
    *,
    device_name: str = "cpu",
    encoder_window_batch_size: int = 8,
) -> dict[str, object]:
    set_determinism()
    device = torch.device(device_name)
    source_repo = repo_root / ".cache/multidataset_pipeline/assets/P2/source/repo"
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    )
    backend = load_local_beats_window_backend(
        source_repo, checkpoint, device=device
    )
    config = {
        "experiment": "BEATs_M_Unified",
        "seed": SEED,
        "sample_rate": 16_000,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "native_batch_size": 8,
        "encoder_window_microbatch": encoder_window_batch_size,
        "homogeneous_dataset_batches": True,
        "encoder": "frozen pretrained BEATs AudioSet",
        "package_definition": "BEATs_exact_valid_patch_pool_v0",
        "frontend": (
            "BEATs fbank and Conv2d patches; exact valid-patch mask; frequency-patch "
            "mean then valid temporal-token mean for each 2-second source window"
        ),
        "package_caveat": (
            "BEATs plus its frontend/masking/pooling package; differences from AST "
            "cannot be attributed to the backbone alone"
        ),
        "checkpoint_used": str(checkpoint),
        "source_repo": str(source_repo),
        "encoder_embedding_dim": 768,
        "dimension_adapter": "identity",
        "projector": "trainable biased Linear 768->256",
        "heads": {
            "level1": "Linear 256->2 cross_entropy",
            "crackle": "Linear 256->1 BCE",
            "wheeze": "Linear 256->1 BCE",
            "other": "Linear 256->1 BCE on explicit eligible rows",
        },
        "epochs": 50,
        "optimizer": "Adam",
        "learning_rate": 5e-5,
        "weight_decay": 1e-6,
        "schedule": "cosine per update; no warmup",
        "precision": "FP32",
        "augmentation": False,
        "gradient_accumulation": 1,
        "sampler": "source-proportional homogeneous batches",
        "selection": (
            "equal mean of per-dataset eligible-node validation loss; tie earliest epoch"
        ),
        "test_policy": "one inference after validation selection",
        "ledger_reference": str(
            repo_root / "result/reproduce/unified/ground_truth_ledger"
        ),
        "canonical_split_note": (
            "existing canonical split assignments retained; training randomness uses seed 42"
        ),
        "device": str(device),
    }
    return _run_encoder_m_unified(
        repo_root,
        result_dir,
        backend=backend,
        cache_dir=(
            repo_root / ".cache/multidataset_pipeline/m_unified_beats_seed42"
        ),
        config=config,
        status="BEATs_M_Unified_50_epochs_complete",
        limitations=[
            "BEATs_exact_valid_patch_pool_v0 is a frontend-plus-encoder package, not a pure backbone comparison",
            "HF has positive-only attribute supervision; Level1 and gaps are masked",
            "KAUH Crep/Bronchial/I C B remain masked and outside the claim",
            "SPRSound Rhonchi/Stridor use Proposed Other while Crackle/Wheeze remain unknown",
            "different native prediction units are not a pooled primary result",
        ],
        device=device,
        encoder_window_batch_size=encoder_window_batch_size,
    )


def regroup_native_probabilities(
    dataset: str,
    probabilities: Mapping[str, float],
) -> dict[str, object]:
    """Regroup future Hanlin native probabilities without inventing labels."""

    if dataset == "icbhi":
        return {
            "level1_normal": probabilities["normal"],
            "level1_abnormal": sum(
                probabilities[label] for label in ("crackle", "wheeze", "both")
            ),
            "crackle": probabilities["crackle"] + probabilities["both"],
            "wheeze": probabilities["wheeze"] + probabilities["both"],
            "other": 0.0,
        }
    if dataset == "sprsound":
        return {
            "level1_normal": probabilities["Normal"],
            "level1_abnormal": 1.0 - probabilities["Normal"],
            "crackle": sum(
                probabilities[label]
                for label in ("Coarse Crackle", "Fine Crackle", "Wheeze+Crackle")
            ),
            "wheeze": probabilities["Wheeze"] + probabilities["Wheeze+Crackle"],
            "other": probabilities["Rhonchi"] + probabilities["Stridor"],
        }
    if dataset == "kauh":
        eligible_mass = sum(probabilities[label] for label in KAUH_ELIGIBLE)
        unresolved_mass = sum(probabilities[label] for label in KAUH_HOLD)
        return {
            "level1_normal": probabilities["N"],
            "level1_abnormal_eligible_mass": eligible_mass - probabilities["N"],
            "crackle": sum(
                probabilities[label] for label in ("C", "I C", "I C E W")
            ),
            "wheeze": sum(
                probabilities[label] for label in ("E W", "I E W", "I C E W")
            ),
            "other": None,
            "unresolved_probability_mass": unresolved_mass,
            "limitation": "Crep/Bronchial/I C B cannot be assigned without a new mapping",
        }
    raise ValueError("native regrouping supports ICBHI, SPRSound, or KAUH")


def regroup_hf_native_probabilities(probabilities: Mapping[str, float]) -> dict[str, object]:
    return {
        "crackle": probabilities.get("DAS"),
        "wheeze": None,
        "other": None,
        "cas_probability": probabilities.get("CAS"),
        "limitation": "a single CAS logit cannot be split into Wheeze versus Other",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ledger", "train", "train-beats"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-window-batch-size", type=int, default=8)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.command == "ledger":
        result = build_ground_truth_ledger(
            root, root / "result/reproduce/unified/ground_truth_ledger"
        )
    elif args.command == "train":
        result = run_ast_m_unified(
            root,
            root / "result/reproduce/unified/AST_M_Unified/seed_42",
            device_name=args.device,
            encoder_window_batch_size=args.encoder_window_batch_size,
        )
    else:
        result = run_beats_m_unified(
            root,
            root / "result/reproduce/unified/BEATs_M_Unified/seed_42",
            device_name=args.device,
            encoder_window_batch_size=args.encoder_window_batch_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
