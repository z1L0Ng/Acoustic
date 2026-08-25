"""BEATs Core-2 normalization, augmentation, and loss ablation protocol.

This module prepares the controlled ICBHI + SPRSound experiment requested in
the 2026-08-20 meeting follow-up.  It never loads HF/KAUH supervision and never
decodes or scores terminal/test audio.  Calling the CLI without ``--run`` only
prints the configuration; waveform decoding and training require an explicit
run command.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseline.four_dataset_frozen_encoder.data import Sample, _load_icbhi, _load_spr

from .beats_temporal import BEATsTemporalAdapter
from .beats_window_encoder import load_local_beats_model
from .contracts import PREDICTION_UNITS, SAMPLE_RATE, WaveformBatch, WaveformSample
from .core2_hf_positive_kauh_external import (
    CORE_DATASETS,
    CORE_NODES,
    Core2Head,
    UPDATES_PER_EPOCH,
    core_selection_losses,
    select_core_shared_thresholds,
)
from .m_unified import map_native_sample, set_determinism
from .real_subtrain_provider import LANE_BY_CANONICAL_DATASET, load_sample_waveform
from .sliding_window import (
    SlidingWindowBatch,
    collate_sliding_windows,
    masked_mean_window_embeddings,
)


NORMALIZATION_MODES = ("none", "peak", "rms")
AUGMENTATION_MODES = ("none", "gain", "noise", "gain_noise")
LOSS_MODES = ("ce_bce", "focal", "class_balanced")
ENCODER_SCOPES = ("frozen", "full")


@dataclass(frozen=True)
class WaveformNormalizationConfig:
    mode: str = "none"
    peak_target: float = 0.95
    rms_target_dbfs: float | None = None

    def validate(self) -> None:
        if self.mode not in NORMALIZATION_MODES:
            raise ValueError(f"unknown waveform normalization: {self.mode}")
        if self.mode == "rms" and self.rms_target_dbfs is None:
            raise ValueError("rms normalization requires an explicit target dBFS")


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    mode: str = "none"
    probability: float = 0.5
    gain_db: float = 6.0
    noise_snr_db_min: float = 20.0
    noise_snr_db_max: float = 40.0

    def validate(self) -> None:
        if self.mode not in AUGMENTATION_MODES:
            raise ValueError(f"unknown waveform augmentation: {self.mode}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("augmentation probability must lie in [0,1]")
        if self.noise_snr_db_min > self.noise_snr_db_max:
            raise ValueError("noise SNR range is reversed")


@dataclass(frozen=True)
class HierarchicalLossConfig:
    mode: str = "ce_bce"
    focal_gamma: float = 2.0
    effective_number_beta: float = 0.9999

    def validate(self) -> None:
        if self.mode not in LOSS_MODES:
            raise ValueError(f"unknown hierarchical loss: {self.mode}")


@dataclass(frozen=True)
class BEATsNALConfig:
    repo_root: Path
    source_repo: Path
    checkpoint: Path
    output_dir: Path
    device: str = "cpu"
    seed: int = 42
    sample_rate: int = SAMPLE_RATE
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    batch_size: int = 8
    epochs: int = 50
    cpu_threads: int = 4
    encoder_scope: str = "full"
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    normalization: WaveformNormalizationConfig = WaveformNormalizationConfig()
    augmentation: WaveformAugmentationConfig = WaveformAugmentationConfig()
    loss: HierarchicalLossConfig = HierarchicalLossConfig()

    def validate(self) -> None:
        if self.seed != 42:
            raise ValueError("the frozen comparison seed is 42")
        if self.sample_rate != 16_000 or self.window_seconds != 4.0 or self.stride_seconds != 2.0:
            raise ValueError("the Core-2 input contract is 16 kHz, 4 s / 2 s")
        if self.batch_size != 8 or self.epochs != 50:
            raise ValueError("the frozen comparison budget is batch 8 for 50 epochs")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        if self.encoder_scope not in ENCODER_SCOPES:
            raise ValueError(f"unknown encoder scope: {self.encoder_scope}")
        self.normalization.validate()
        self.augmentation.validate()
        self.loss.validate()

    @property
    def condition(self) -> str:
        return (
            f"BEATs_{self.encoder_scope}_"
            f"norm-{self.normalization.mode}_"
            f"aug-{self.augmentation.mode}_"
            f"loss-{self.loss.mode}_seed42"
        )

    @property
    def window_samples(self) -> int:
        return int(self.window_seconds * self.sample_rate)

    @property
    def stride_samples(self) -> int:
        return int(self.stride_seconds * self.sample_rate)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ("repo_root", "source_repo", "checkpoint", "output_dir"):
            value[key] = str(value[key])
        value["condition"] = self.condition
        value["datasets"] = list(CORE_DATASETS)
        value["nodes"] = list(CORE_NODES)
        value["selection"] = (
            "equal mean of ICBHI and SPRSound validation eligible-node loss"
        )
        value["test_access"] = "not part of this training entry"
        value["training_batch_execution"] = "true_native_unit_batch"
        value["decoded_waveform_cache"] = (
            "in_process_post_normalization_pre_augmentation"
        )
        return value


def normalize_waveform(
    waveform: torch.Tensor,
    config: WaveformNormalizationConfig,
) -> torch.Tensor:
    """Normalize one native waveform before windowing."""

    config.validate()
    values = waveform.to(torch.float32)
    if config.mode == "none":
        return values
    if config.mode == "peak":
        scale = values.abs().max()
        return values if float(scale) == 0.0 else values * (config.peak_target / scale)
    rms = values.square().mean().sqrt()
    if float(rms) == 0.0:
        return values
    target = 10.0 ** (float(config.rms_target_dbfs) / 20.0)
    return values * (target / rms)


def _uniform(
    low: float,
    high: float,
    generator: torch.Generator,
) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def augment_waveform(
    waveform: torch.Tensor,
    config: WaveformAugmentationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply label-preserving train-only loudness and/or noise augmentation."""

    config.validate()
    values = waveform.clone()
    if config.mode == "none":
        return values
    if config.mode in {"gain", "gain_noise"} and float(
        torch.rand((), generator=generator)
    ) < config.probability:
        gain = _uniform(-config.gain_db, config.gain_db, generator)
        values = values * (10.0 ** (gain / 20.0))
    if config.mode in {"noise", "gain_noise"} and float(
        torch.rand((), generator=generator)
    ) < config.probability:
        signal_rms = values.square().mean().sqrt()
        if float(signal_rms) > 0.0:
            snr_db = _uniform(
                config.noise_snr_db_min,
                config.noise_snr_db_max,
                generator,
            )
            noise = torch.randn(
                values.shape,
                dtype=values.dtype,
                device=values.device,
                generator=generator,
            )
            noise = noise * (signal_rms / (10.0 ** (snr_db / 20.0)))
            values = values + noise
    return values


def transform_sample(
    sample: WaveformSample,
    normalization: WaveformNormalizationConfig,
    augmentation: WaveformAugmentationConfig,
    *,
    training: bool,
    generator: torch.Generator,
) -> WaveformSample:
    waveform = normalize_waveform(sample.waveform, normalization)
    if training:
        waveform = augment_waveform(waveform, augmentation, generator)
    return replace(sample, waveform=waveform.contiguous())


def mapped_targets(
    samples: Sequence[Sample],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    targets = torch.zeros(len(samples), len(CORE_NODES), dtype=torch.float32)
    eligible = torch.zeros(len(samples), len(CORE_NODES), dtype=torch.bool)
    raw_labels: list[str] = []
    for row, sample in enumerate(samples):
        mapped = map_native_sample(sample)
        raw_labels.append(str(mapped["raw_label"]))
        for column, node in enumerate(CORE_NODES):
            if mapped[f"{node}_eligible"]:
                eligible[row, column] = True
                targets[row, column] = float(mapped[f"{node}_target"])
    return targets, eligible, raw_labels


def effective_number_class_weights(
    samples: Sequence[Sample],
    *,
    beta: float = 0.9999,
) -> dict[str, torch.Tensor]:
    targets, eligible, _ = mapped_targets(samples)
    weights: dict[str, torch.Tensor] = {}
    for index, node in enumerate(CORE_NODES):
        active = targets[eligible[:, index], index].long()
        counts = torch.bincount(active, minlength=2).to(torch.float64)
        if bool((counts == 0).any()):
            raise RuntimeError(f"subtrain support is missing a {node} class")
        value = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights[node] = (value / value.mean()).to(torch.float32)
    return weights


def hierarchical_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
    config: HierarchicalLossConfig,
    class_weights: Mapping[str, torch.Tensor] | None = None,
    *,
    active_nodes: Sequence[str] | None = None,
    collect_named: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equal-node masked loss for Level1, Crackle, and Wheeze."""

    config.validate()
    losses: list[torch.Tensor] = []
    named: dict[str, float] = {}
    active = set(active_nodes) if active_nodes is not None else None
    for index, node in enumerate(CORE_NODES):
        mask = eligible[:, index]
        if active is not None:
            if node not in active:
                continue
        elif not bool(mask.any()):
            continue
        target = targets[mask, index]
        value = _node_loss_values(
            node,
            logits[node][mask],
            target,
            config,
            class_weights,
        )
        node_loss = value.mean()
        losses.append(node_loss)
        if collect_named:
            named[node] = float(node_loss.detach())
    return torch.stack(losses).mean(), named


def _node_loss_values(
    node: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    config: HierarchicalLossConfig,
    class_weights: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
    if node == "level1":
        value = F.cross_entropy(logits, target.long(), reduction="none")
    else:
        value = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    if config.mode == "focal":
        value = (1.0 - torch.exp(-value)).pow(config.focal_gamma) * value
    elif config.mode == "class_balanced":
        if class_weights is None:
            raise ValueError("class-balanced loss requires subtrain-only weights")
        value = value * class_weights[node].to(value.device)[target.long()]
    return value


def hierarchical_loss_contribution(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
    config: HierarchicalLossConfig,
    class_weights: Mapping[str, torch.Tensor] | None,
    eligible_denominators: torch.Tensor,
) -> torch.Tensor:
    """Exact one-unit contribution to the effective native batch loss."""

    active_nodes = int((eligible_denominators > 0).sum())
    contribution = logits["level1"].sum() * 0.0
    for index, node in enumerate(CORE_NODES):
        if not bool(eligible[0, index]):
            continue
        target = targets[:, index]
        value = _node_loss_values(
            node,
            logits[node],
            target,
            config,
            class_weights,
        ).sum()
        contribution = contribution + value / eligible_denominators[index]
    return contribution / active_nodes


class BEATsCore2Model(nn.Module):
    def __init__(self, beats: nn.Module, *, encoder_trainable: bool) -> None:
        super().__init__()
        self.encoder_trainable = encoder_trainable
        self.temporal = BEATsTemporalAdapter(
            beats,
            trainable=encoder_trainable,
        )
        self.head = Core2Head(encoder_dim=768)

    def encode_units(self, windows: SlidingWindowBatch) -> torch.Tensor:
        flat_valid = windows.window_mask.reshape(-1)
        window_width = windows.waveform_windows.shape[-1]
        waveforms = windows.waveform_windows.reshape(-1, window_width)[flat_valid]
        valid_samples = windows.valid_samples.reshape(-1)[flat_valid]
        geometry = self.temporal.geometry
        minimum = (
            geometry.frame_length_samples
            + (geometry.patch_kernel_time - 1) * geometry.frame_shift_samples
        )
        model_valid = valid_samples.clamp_min(minimum)
        count = waveforms.shape[0]
        source_time_dtype = (
            torch.float32 if waveforms.device.type == "mps" else torch.float64
        )
        starts = torch.zeros(count, dtype=source_time_dtype, device=waveforms.device)
        if len(set(windows.dataset_ids)) != 1:
            raise RuntimeError("Core-2 batches must contain one dataset lane")
        flat_dataset_ids = (windows.dataset_ids[0],) * count
        batch = WaveformBatch(
            waveform=waveforms,
            waveform_padding_mask=(
                torch.arange(window_width, device=waveforms.device).unsqueeze(0)
                >= model_valid.unsqueeze(1)
            ),
            valid_samples=model_valid,
            sample_rate=SAMPLE_RATE,
            source_start_s=starts,
            source_end_s=model_valid.to(source_time_dtype) / SAMPLE_RATE,
            sample_ids=tuple(f"window-{index}" for index in range(count)),
            dataset_ids=flat_dataset_ids,
            prediction_units=tuple(
                PREDICTION_UNITS[dataset] for dataset in flat_dataset_ids
            ),
            lineage=tuple({"internal": "core2_native_window"} for _ in range(count)),
        )
        pooled = self.temporal(batch).pooled.to(torch.float32)
        restored = torch.zeros(
            windows.window_mask.numel(),
            pooled.shape[-1],
            dtype=pooled.dtype,
            device=pooled.device,
        )
        restored[flat_valid] = pooled
        restored = restored.reshape(*windows.window_mask.shape, pooled.shape[-1])
        return masked_mean_window_embeddings(
            restored,
            windows.window_mask,
            deep=False,
        )

    def forward(self, windows: SlidingWindowBatch) -> dict[str, torch.Tensor]:
        return self.head(self.encode_units(windows))


def _load_transformed_batch(
    samples: Sequence[Sample],
    config: BEATsNALConfig,
    *,
    training: bool,
    generators: Sequence[torch.Generator],
    waveform_cache: dict[str, WaveformSample],
) -> SlidingWindowBatch:
    if len(generators) != len(samples):
        raise ValueError("one waveform generator is required per sample")
    waveforms: list[WaveformSample] = []
    for sample, generator in zip(samples, generators):
        if sample.partition not in {"subtrain", "validation"}:
            raise RuntimeError("this training entry does not decode terminal/test rows")
        prepared = waveform_cache.get(sample.sample_id)
        if prepared is None:
            lane = LANE_BY_CANONICAL_DATASET[sample.dataset]
            decoded = load_sample_waveform(sample, lane, outer_test_accessed=False)[0]
            if config.normalization.mode == "none":
                prepared = decoded
            else:
                prepared = replace(
                    decoded,
                    waveform=normalize_waveform(
                        decoded.waveform,
                        config.normalization,
                    ).contiguous(),
                )
            waveform_cache[sample.sample_id] = prepared
        if training and config.augmentation.mode != "none":
            prepared = replace(
                prepared,
                waveform=augment_waveform(
                    prepared.waveform,
                    config.augmentation,
                    generator,
                ).contiguous(),
            )
        waveforms.append(prepared)
    return collate_sliding_windows(
        waveforms,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
    )


def _epoch_batches(
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    epoch: int,
    *,
    seed: int,
    batch_size: int,
) -> list[tuple[str, list[int]]]:
    rng = np.random.default_rng(seed + epoch)
    batches: list[tuple[str, list[int]]] = []
    for dataset in CORE_DATASETS:
        order = rng.permutation(len(samples_by_dataset[dataset])).tolist()
        batches.extend(
            (dataset, order[start : start + batch_size])
            for start in range(0, len(order), batch_size)
        )
    extra = UPDATES_PER_EPOCH - len(batches)
    if extra < 0:
        raise RuntimeError("natural batches exceed the matched update budget")
    if extra:
        sampled = rng.integers(0, len(batches), size=extra)
        batches.extend((batches[index][0], list(batches[index][1])) for index in sampled)
    rng.shuffle(batches)
    return batches


def infer_validation(
    model: BEATsCore2Model,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    config: BEATsNALConfig,
    waveform_cache: dict[str, WaveformSample],
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "dataset_ids": [],
        "group_ids": [],
        "file_names": [],
        "raw_ground_truth": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
        "targets": [],
        "eligible": [],
    }
    model.eval()
    generator = torch.Generator().manual_seed(config.seed)
    with torch.no_grad():
        for dataset in CORE_DATASETS:
            samples = samples_by_dataset[dataset]
            for start in range(0, len(samples), config.batch_size):
                current = samples[start : start + config.batch_size]
                windows = _load_transformed_batch(
                    current,
                    config,
                    training=False,
                    generators=[generator] * len(current),
                    waveform_cache=waveform_cache,
                ).to(config.device)
                output = model(windows)
                targets, eligible, raw_labels = mapped_targets(current)
                level1_logits = output["level1"].detach().cpu()
                attributes = torch.stack(
                    (output["crackle"], output["wheeze"]), dim=-1
                ).detach().cpu()
                ids = np.asarray([sample.sample_id for sample in current])
                fields["prediction_ids"].append(ids)
                fields["sample_ids"].append(ids)
                fields["dataset_ids"].append(np.asarray([dataset] * len(current)))
                fields["group_ids"].append(np.asarray([sample.group_id for sample in current]))
                fields["file_names"].append(
                    np.asarray([Path(sample.audio_path).name for sample in current])
                )
                fields["raw_ground_truth"].append(np.asarray(raw_labels))
                fields["level1_logits"].append(level1_logits.numpy())
                fields["level1_probabilities"].append(
                    torch.softmax(level1_logits, dim=-1).numpy()
                )
                fields["level1_predictions"].append(level1_logits.argmax(dim=-1).numpy())
                fields["attribute_logits"].append(attributes.numpy())
                fields["attribute_probabilities"].append(torch.sigmoid(attributes).numpy())
                fields["targets"].append(targets.numpy())
                fields["eligible"].append(eligible.numpy())
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def decode_icbhi_flat4(
    attribute_probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """Reconstruct Normal/Crackle/Wheeze/Both from the two atomic bits."""

    crackle = attribute_probabilities[:, 0] >= thresholds["crackle"]
    wheeze = attribute_probabilities[:, 1] >= thresholds["wheeze"]
    return crackle.astype(np.int64) + 2 * wheeze.astype(np.int64)


def _save_predictions(path: Path, predictions: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **predictions)


def load_core_samples(repo_root: Path) -> list[Sample]:
    """Load only canonical ICBHI and SPRSound rows, without checksums."""

    dataset_root = repo_root / "dataset/raw"
    icbhi, _ = _load_icbhi(dataset_root, include_checksums=False)
    sprsound, _ = _load_spr(dataset_root, include_checksums=False)
    samples = [*icbhi, *sprsound]
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("duplicate Core-2 sample ID")
    return samples


def run_training(config: BEATsNALConfig) -> dict[str, object]:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    set_determinism(config.seed)
    all_samples = load_core_samples(config.repo_root)
    partitions = {
        partition: {
            dataset: tuple(
                sorted(
                    (
                        sample
                        for sample in all_samples
                        if sample.partition == partition and sample.dataset == dataset
                    ),
                    key=lambda sample: sample.sample_id,
                )
            )
            for dataset in CORE_DATASETS
        }
        for partition in ("subtrain", "validation")
    }
    subtrain_samples = [
        sample
        for dataset in CORE_DATASETS
        for sample in partitions["subtrain"][dataset]
    ]
    class_weights = effective_number_class_weights(
        subtrain_samples,
        beta=config.loss.effective_number_beta,
    )
    target_device = torch.device(config.device)
    device_class_weights = {
        node: values.to(target_device) for node, values in class_weights.items()
    }
    beats = load_local_beats_model(
        config.source_repo,
        config.checkpoint,
        device=target_device,
        trainable=config.encoder_scope == "full",
    )
    model = BEATsCore2Model(
        beats,
        encoder_trainable=config.encoder_scope == "full",
    ).to(target_device)
    parameter_groups = [
        {"params": model.head.parameters(), "lr": config.head_learning_rate}
    ]
    if config.encoder_scope == "full":
        parameter_groups.insert(
            0,
            {"params": model.temporal.parameters(), "lr": config.backbone_learning_rate},
        )
    optimizer = torch.optim.Adam(
        parameter_groups,
        weight_decay=config.weight_decay,
    )
    total_updates = config.epochs * UPDATES_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if (config.output_dir / "train_log.jsonl").exists():
        raise RuntimeError("output directory already contains a training log")
    (config.output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (config.output_dir / "class_weights.json").write_text(
        json.dumps(
            {node: values.tolist() for node, values in class_weights.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    log_path = config.output_dir / "train_log.jsonl"
    progress_path = config.output_dir / "progress.jsonl"
    best_loss = math.inf
    best_epoch = 0
    update = 0
    started = time.perf_counter()
    waveform_cache: dict[str, WaveformSample] = {}
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: dict[str, list[float]] = {dataset: [] for dataset in CORE_DATASETS}
        epoch_data_seconds = 0.0
        epoch_step_seconds = 0.0
        epoch_units = 0
        epoch_windows = 0
        for dataset, indices in _epoch_batches(
            partitions["subtrain"],
            epoch,
            seed=config.seed,
            batch_size=config.batch_size,
        ):
            samples = [partitions["subtrain"][dataset][index] for index in indices]
            batch_targets, batch_eligible, _ = mapped_targets(samples)
            active_nodes = tuple(
                node
                for index, node in enumerate(CORE_NODES)
                if bool(batch_eligible[:, index].any())
            )
            generators = [
                torch.Generator().manual_seed(
                    config.seed + epoch * 1_000_000 + update * config.batch_size + offset
                )
                for offset in range(len(samples))
            ]
            data_started = time.perf_counter()
            windows = _load_transformed_batch(
                samples,
                config,
                training=True,
                generators=generators,
                waveform_cache=waveform_cache,
            )
            epoch_units += len(samples)
            epoch_windows += int(windows.window_mask.sum())
            windows = windows.to(target_device)
            batch_targets = batch_targets.to(target_device)
            batch_eligible = batch_eligible.to(target_device)
            epoch_data_seconds += time.perf_counter() - data_started
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = hierarchical_loss(
                model(windows),
                batch_targets,
                batch_eligible,
                config.loss,
                device_class_weights,
                active_nodes=active_nodes,
                collect_named=False,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            batch_loss = float(loss.detach())
            epoch_step_seconds += time.perf_counter() - step_started
            train_losses[dataset].append(batch_loss)
            if update % 100 == 0:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "update": update,
                                "epoch_updates": sum(len(values) for values in train_losses.values()),
                                "epoch_units": epoch_units,
                                "epoch_windows": epoch_windows,
                                "elapsed_minutes": (time.perf_counter() - started) / 60.0,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        validation_started = time.perf_counter()
        predictions = infer_validation(
            model,
            partitions["validation"],
            config,
            waveform_cache,
        )
        validation_seconds = time.perf_counter() - validation_started
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            predictions,
        )
        selection = core_selection_losses(predictions)
        record = {
            "epoch": epoch,
            "update": update,
            "train_loss": {
                dataset: float(np.mean(values))
                for dataset, values in train_losses.items()
            },
            "validation": selection,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "runtime": {
                "train_batch_prepare_seconds": epoch_data_seconds,
                "train_step_wall_seconds": epoch_step_seconds,
                "validation_wall_seconds": validation_seconds,
                "native_units": epoch_units,
                "windows": epoch_windows,
                "decoded_waveform_cache_entries": len(waveform_cache),
            },
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if float(selection["selection_loss"]) < best_loss:
            best_loss = float(selection["selection_loss"])
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "update": update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_loss": best_loss,
                    "config": config.to_dict(),
                },
                config.output_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(),
            },
            config.output_dir / "last_checkpoint.pt",
        )
    with np.load(
        config.output_dir / "validation" / f"epoch_{best_epoch:03d}.npz",
        allow_pickle=False,
    ) as archive:
        selected_predictions = {key: archive[key] for key in archive.files}
    thresholds, threshold_details = select_core_shared_thresholds(
        selected_predictions
    )
    validation_selection = {
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "shared_attribute_thresholds": thresholds,
        "threshold_details": threshold_details,
        "icbhi_flat4_decoder": (
            "Crackle/Wheeze bits: 00 Normal, 10 Crackle, 01 Wheeze, 11 Both; "
            "Level1 is reported separately and does not override flat4"
        ),
        "outer_test_accessed": False,
    }
    (config.output_dir / "validation_selection.json").write_text(
        json.dumps(validation_selection, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "status": "training_complete_validation_selected",
        "condition": config.condition,
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "shared_attribute_thresholds": thresholds,
        "updates": update,
        "outer_test_accessed": False,
        "terminal_result": False,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _default_paths(repo_root: Path) -> tuple[Path, Path]:
    source = repo_root / ".cache/multidataset_pipeline/assets/P2/source/repo"
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    )
    return source, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--encoder-scope", choices=ENCODER_SCOPES, default="full")
    parser.add_argument("--normalization", choices=NORMALIZATION_MODES, default="none")
    parser.add_argument("--rms-target-dbfs", type=float)
    parser.add_argument("--augmentation", choices=AUGMENTATION_MODES, default="none")
    parser.add_argument("--loss", choices=LOSS_MODES, default="ce_bce")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    source, checkpoint = _default_paths(args.repo_root)
    condition = (
        f"BEATs_{args.encoder_scope}_norm-{args.normalization}_"
        f"aug-{args.augmentation}_loss-{args.loss}_seed42"
    )
    output_dir = args.output_dir or (
        args.repo_root / "result/reproduce/beats_nal_ablation" / condition
    )
    config = BEATsNALConfig(
        repo_root=args.repo_root,
        source_repo=args.source_repo or source,
        checkpoint=args.checkpoint or checkpoint,
        output_dir=output_dir,
        device=args.device,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        cpu_threads=args.cpu_threads,
        encoder_scope=args.encoder_scope,
        normalization=WaveformNormalizationConfig(
            mode=args.normalization,
            rms_target_dbfs=args.rms_target_dbfs,
        ),
        augmentation=WaveformAugmentationConfig(mode=args.augmentation),
        loss=HierarchicalLossConfig(mode=args.loss),
    )
    config.validate()
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "execution_started": False,
                    "config": config.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run_training(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
