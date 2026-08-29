"""PAFA-style joint ICBHI + SPRSound hierarchical full fine-tuning.

PAFA-JH1 is a proposed method experiment, not a PAFA paper reproduction.  It
keeps the author BEATs/5-second/EMA/PCSL/GPAL recipe while replacing the direct
flat-four classifier with the project's shared Level1/Crackle/Wheeze hierarchy.
Training and checkpoint selection use only patient-grouped subtrain/validation
partitions.  Terminal ICBHI and SPRSound data are loaded only after the best
validation checkpoint and shared attribute thresholds have been fixed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    _load_spr,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    decode_icbhi_hierarchical_flat4,
    hierarchical_loss,
    mapped_targets,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets, _score
from baseline.multidataset_pipeline.core2_hf_positive_kauh_external import (
    CORE_DATASETS,
    CORE_NODES,
    Core2Head,
    core_selection_losses,
    select_core_shared_thresholds,
)
from baseline.pafa.beats_ce_reproduction import (
    CycleRecord,
    _apply_author_ema,
    read_official_cycles,
)


CONDITION = "PAFA_JH1_joint_hierarchy_seed42"
LABEL_ORDER = ("normal", "crackle", "wheeze", "both")


@dataclass(frozen=True)
class PAFAJointHierarchyConfig:
    repo_root: Path
    author_repo: Path
    checkpoint: Path
    icbhi_audio_dir: Path
    output_dir: Path
    device: str = "cuda"
    seed: int = 42
    sample_rate: int = 16_000
    desired_seconds: float = 5.0
    batch_size: int = 32
    epochs: int = 50
    cpu_threads: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    cosine_eta_min_ratio: float = 1e-3
    ema_beta: float = 0.5
    classification_weight: float = 1.0
    pafa_weight: float = 1.0
    lambda_pcsl: float = 50.0
    lambda_gpal: float = 0.0005
    projection_output_dim: int = 768
    projection_norm: str = "ln"
    projection_attention: bool = True

    def validate(self) -> None:
        expected = {
            "seed": 42,
            "sample_rate": 16_000,
            "desired_seconds": 5.0,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 5e-5,
            "weight_decay": 1e-6,
            "cosine_eta_min_ratio": 1e-3,
            "ema_beta": 0.5,
            "classification_weight": 1.0,
            "pafa_weight": 1.0,
            "lambda_pcsl": 50.0,
            "lambda_gpal": 0.0005,
            "projection_output_dim": 768,
            "projection_norm": "ln",
            "projection_attention": True,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"frozen PAFA-JH1 field changed: {field}")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for field in (
            "repo_root",
            "author_repo",
            "checkpoint",
            "icbhi_audio_dir",
            "output_dir",
        ):
            payload[field] = str(payload[field])
        payload.update(
            {
                "condition": CONDITION,
                "datasets": list(CORE_DATASETS),
                "nodes": list(CORE_NODES),
                "encoder": "BEATs iter3+ AS2M full fine-tuning",
                "input": "mono 16 kHz; 5 s repeat-pad/front-truncate; no SpecAugment",
                "batching": (
                    "dataset-homogeneous native-unit batch 32; equal batch count "
                    "per dataset per epoch"
                ),
                "selection": (
                    "equal mean of ICBHI and SPRSound validation eligible-node loss"
                ),
                "terminal_policy": "one access after validation selection and threshold freeze",
                "checkpoint_policy": "best only; no optimizer resume state",
                "evidence_label": "proposed_method_single_seed",
            }
        )
        return payload


class PAFAJointHierarchyModel(nn.Module):
    def __init__(self, beats: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.beats = beats
        self.projector = projector
        self.head = Core2Head(encoder_dim=768)

    def forward(
        self, waveform: torch.Tensor, *, training: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = self.beats(waveform, training=training)
        logits = self.head(features.mean(dim=1))
        projected = self.projector(features)
        return logits, projected


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _icbhi_sample(
    row: CycleRecord,
    audio_dir: Path,
    partition: str,
) -> Sample:
    return Sample(
        sample_id=f"icbhi:{row.sample_id}",
        dataset="icbhi",
        partition=partition,
        group_id=str(row.group_id),
        audio_path=str(audio_dir / f"{row.recording_id}.wav"),
        crop_start_s=float(row.start_s),
        crop_end_s=float(row.end_s),
        targets={"icbhi_flat4": int(row.ground_truth)},
        metadata={
            "cycle_id": row.sample_id,
            "recording_id": row.recording_id,
            "patient_id": row.group_id,
            "official_split": row.official_split,
        },
    )


def _icbhi_selection_samples(config: PAFAJointHierarchyConfig) -> list[Sample]:
    rows = read_official_cycles(
        config.icbhi_audio_dir,
        config.author_repo,
        ("train",),
    )
    labels = np.asarray([row.ground_truth for row in rows])
    groups = np.asarray([row.group_id for row in rows])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.seed,
    )
    subtrain_indices, validation_indices = next(
        splitter.split(np.zeros(len(rows)), labels, groups)
    )
    validation = set(validation_indices.tolist())
    subtrain = set(subtrain_indices.tolist())
    return [
        _icbhi_sample(
            row,
            config.icbhi_audio_dir,
            "validation" if index in validation else "subtrain",
        )
        for index, row in enumerate(rows)
        if index in subtrain or index in validation
    ]


def load_selection_samples(config: PAFAJointHierarchyConfig) -> list[Sample]:
    icbhi = _icbhi_selection_samples(config)
    sprsound, _ = _load_spr(
        config.repo_root / "dataset/raw",
        include_checksums=False,
    )
    sprsound = [sample for sample in sprsound if sample.partition != "test"]
    samples = [*icbhi, *sprsound]
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("duplicate PAFA-JH1 selection sample ID")
    for dataset in CORE_DATASETS:
        subtrain_groups = {
            sample.group_id
            for sample in samples
            if sample.dataset == dataset and sample.partition == "subtrain"
        }
        validation_groups = {
            sample.group_id
            for sample in samples
            if sample.dataset == dataset and sample.partition == "validation"
        }
        if subtrain_groups & validation_groups:
            raise RuntimeError(f"{dataset} patient overlap in selection split")
    return samples


def load_terminal_samples(config: PAFAJointHierarchyConfig) -> list[Sample]:
    icbhi_rows = read_official_cycles(
        config.icbhi_audio_dir,
        config.author_repo,
        ("test",),
    )
    icbhi = [
        _icbhi_sample(row, config.icbhi_audio_dir, "test")
        for row in icbhi_rows
    ]
    sprsound, _ = _load_spr(
        config.repo_root / "dataset/raw",
        include_checksums=False,
    )
    sprsound = [sample for sample in sprsound if sample.partition == "test"]
    samples = [*sorted(icbhi, key=lambda row: row.sample_id), *sorted(sprsound, key=lambda row: row.sample_id)]
    support = {
        dataset: sum(sample.dataset == dataset for sample in samples)
        for dataset in CORE_DATASETS
    }
    if support != {"icbhi": 2756, "sprsound": 1429}:
        raise RuntimeError(f"terminal support changed: {support}")
    return samples


def partition_summary(samples: Sequence[Sample]) -> dict[str, object]:
    output: dict[str, object] = {}
    for dataset in CORE_DATASETS:
        output[dataset] = {}
        for partition in ("subtrain", "validation"):
            rows = [
                sample
                for sample in samples
                if sample.dataset == dataset and sample.partition == partition
            ]
            targets, eligible, _ = mapped_targets(rows)
            output[dataset][partition] = {
                "units": len(rows),
                "groups": len({sample.group_id for sample in rows}),
                "eligible": {
                    node: int(eligible[:, index].sum())
                    for index, node in enumerate(CORE_NODES)
                },
                "positive": {
                    node: int(targets[:, index][eligible[:, index]].sum())
                    for index, node in enumerate(CORE_NODES)
                },
            }
    return output


def _build_components(
    config: PAFAJointHierarchyConfig,
    device: torch.device,
) -> tuple[PAFAJointHierarchyModel, nn.Module]:
    sys.path.insert(0, str(config.author_repo))
    try:
        from method.pafa import PAFALoss, ProjectionHead
        from models.beats import BEATsTransferLearningModel
    finally:
        sys.path.pop(0)
    beats = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(config.checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    projector = ProjectionHead(
        input_dim=beats.final_feat_dim,
        hidden_dim=None,
        output_dim=config.projection_output_dim,
        attention=config.projection_attention,
        proj_type="end2end",
        norm_type=config.projection_norm,
    )
    return PAFAJointHierarchyModel(beats, projector).to(device), PAFALoss().to(device)


def _prepare_waveforms(
    samples: Sequence[Sample],
    config: PAFAJointHierarchyConfig,
) -> dict[str, torch.Tensor]:
    import torchaudio
    from torchaudio import transforms as T

    target_samples = int(config.sample_rate * config.desired_seconds)
    fade_samples = int(config.sample_rate / 16)
    recording_fade = T.Fade(
        fade_in_len=fade_samples,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )
    repeat_fade = T.Fade(
        fade_in_len=0,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )
    by_audio: dict[str, list[Sample]] = {}
    for sample in samples:
        by_audio.setdefault(sample.audio_path, []).append(sample)

    waveforms: dict[str, torch.Tensor] = {}
    for audio_path, rows in by_audio.items():
        waveform, source_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != config.sample_rate:
            waveform = T.Resample(source_rate, config.sample_rate)(waveform)
        waveform = recording_fade(waveform)
        for sample in rows:
            start = int(float(sample.crop_start_s or 0.0) * config.sample_rate)
            end = int(
                float(sample.crop_end_s) * config.sample_rate
                if sample.crop_end_s is not None
                else waveform.shape[-1]
            )
            start = min(start, waveform.shape[-1])
            end = min(end, waveform.shape[-1])
            unit = waveform[:, start:end]
            if unit.shape[-1] == 0:
                raise RuntimeError(f"empty native unit: {sample.sample_id}")
            if unit.shape[-1] > target_samples:
                unit = unit[:, :target_samples]
            else:
                repeats = int(np.ceil(target_samples / unit.shape[-1]))
                unit = unit.repeat(1, repeats)[:, :target_samples]
                unit = repeat_fade(unit)
            waveforms[sample.sample_id] = unit.squeeze(0).to(torch.float32).contiguous()
    return waveforms


def _balanced_epoch_batches(
    sizes: Mapping[str, int],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed + epoch)
    target_batches = max(size // batch_size for size in sizes.values())
    batches: list[tuple[str, np.ndarray]] = []
    for dataset in CORE_DATASETS:
        needed = target_batches * batch_size
        order: list[int] = []
        while len(order) < needed:
            order.extend(rng.permutation(sizes[dataset]).tolist())
        indices = np.asarray(order[:needed], dtype=np.int64)
        batches.extend(
            (dataset, indices[start : start + batch_size])
            for start in range(0, needed, batch_size)
        )
    rng.shuffle(batches)
    return batches


def _patient_indices(samples: Sequence[Sample]) -> dict[tuple[str, str], int]:
    keys = sorted({(sample.dataset, sample.group_id) for sample in samples})
    return {key: index for index, key in enumerate(keys)}


def _batch(
    rows: Sequence[Sample],
    waveform_store: Mapping[str, torch.Tensor],
    patient_index: Mapping[tuple[str, str], int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    waveform = torch.stack([waveform_store[row.sample_id] for row in rows]).to(device)
    targets, eligible, _ = mapped_targets(rows)
    patients = torch.tensor(
        [patient_index[(row.dataset, row.group_id)] for row in rows],
        dtype=torch.long,
    )
    return waveform, targets.to(device), eligible.to(device), patients.to(device)


def infer(
    model: PAFAJointHierarchyModel,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    waveform_store: Mapping[str, torch.Tensor],
    config: PAFAJointHierarchyConfig,
    device: torch.device,
    *,
    include_targets: bool,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "dataset_ids": [],
        "group_ids": [],
        "file_names": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
    }
    if include_targets:
        fields.update({"raw_ground_truth": [], "targets": [], "eligible": []})
    model.eval()
    with torch.no_grad():
        for dataset in CORE_DATASETS:
            samples = samples_by_dataset[dataset]
            for start in range(0, len(samples), config.batch_size):
                current = list(samples[start : start + config.batch_size])
                waveform = torch.stack(
                    [waveform_store[row.sample_id] for row in current]
                ).to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    output, _ = model(waveform, training=False)
                level1 = output["level1"].float().cpu()
                attributes = torch.stack(
                    (output["crackle"], output["wheeze"]), dim=-1
                ).float().cpu()
                ids = np.asarray([row.sample_id for row in current])
                fields["prediction_ids"].append(ids)
                fields["sample_ids"].append(ids)
                fields["dataset_ids"].append(np.asarray([dataset] * len(current)))
                fields["group_ids"].append(
                    np.asarray([row.group_id for row in current])
                )
                fields["file_names"].append(
                    np.asarray([Path(row.audio_path).name for row in current])
                )
                fields["level1_logits"].append(level1.numpy())
                fields["level1_probabilities"].append(
                    torch.softmax(level1, dim=-1).numpy()
                )
                fields["level1_predictions"].append(
                    level1.argmax(dim=-1).numpy()
                )
                fields["attribute_logits"].append(attributes.numpy())
                fields["attribute_probabilities"].append(
                    torch.sigmoid(attributes).numpy()
                )
                if include_targets:
                    targets, eligible, raw = mapped_targets(current)
                    fields["raw_ground_truth"].append(np.asarray(raw))
                    fields["targets"].append(targets.numpy())
                    fields["eligible"].append(eligible.numpy())
    return {key: np.concatenate(values, axis=0) for key, values in fields.items()}


def _save_predictions(path: Path, predictions: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)


def _dataset_partitions(
    samples: Sequence[Sample], partition: str
) -> dict[str, tuple[Sample, ...]]:
    return {
        dataset: tuple(
            sorted(
                (
                    sample
                    for sample in samples
                    if sample.dataset == dataset and sample.partition == partition
                ),
                key=lambda sample: sample.sample_id,
            )
        )
        for dataset in CORE_DATASETS
    }


def run(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    selection_samples = load_selection_samples(config)
    subtrain = _dataset_partitions(selection_samples, "subtrain")
    validation = _dataset_partitions(selection_samples, "validation")
    split_payload = partition_summary(selection_samples)
    _write_json(config.output_dir / "config.json", config.to_dict())
    _write_json(config.output_dir / "selection_split_summary.json", split_payload)

    waveform_store = _prepare_waveforms(selection_samples, config)
    device = torch.device(config.device)
    model, pafa_criterion = _build_components(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    patient_index = _patient_indices(
        [sample for rows in subtrain.values() for sample in rows]
    )
    best_loss = math.inf
    best_epoch = 0
    global_update = 0
    started = time.perf_counter()
    progress_path = config.output_dir / "progress.jsonl"
    train_log_path = config.output_dir / "train_log.jsonl"

    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate * (
            config.cosine_eta_min_ratio
            + (1.0 - config.cosine_eta_min_ratio)
            * (1.0 + np.cos(np.pi * epoch / config.epochs))
            / 2.0
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        epoch_losses = {
            dataset: {"classification": [], "pafa": [], "total": []}
            for dataset in CORE_DATASETS
        }
        batches = _balanced_epoch_batches(
            {dataset: len(rows) for dataset, rows in subtrain.items()},
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
        for batch_index, (dataset, indices) in enumerate(batches, start=1):
            rows = [subtrain[dataset][int(index)] for index in indices]
            waveform, targets, eligible, patients = _batch(
                rows,
                waveform_store,
                patient_index,
                device,
            )
            before = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, projected = model(waveform, training=True)
                classification_loss, _ = hierarchical_loss(
                    logits,
                    targets,
                    eligible,
                    HierarchicalLossConfig(mode="ce_bce"),
                    collect_named=False,
                )
                pafa_loss = pafa_criterion(
                    projected,
                    patients,
                    lambda_pcsl=config.lambda_pcsl,
                    lambda_gpal=config.lambda_gpal,
                )
                total_loss = (
                    config.classification_weight * classification_loss
                    + config.pafa_weight * pafa_loss
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _apply_author_ema(model, before, config.ema_beta)
            global_update += 1
            epoch_losses[dataset]["classification"].append(
                float(classification_loss.detach().cpu())
            )
            epoch_losses[dataset]["pafa"].append(float(pafa_loss.detach().cpu()))
            epoch_losses[dataset]["total"].append(float(total_loss.detach().cpu()))
            if global_update % 100 == 0:
                _append_jsonl(
                    progress_path,
                    {
                        "epoch": epoch,
                        "epoch_batch": batch_index,
                        "epoch_batches": len(batches),
                        "update": global_update,
                        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
                    },
                )

        predictions = infer(
            model,
            validation,
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            predictions,
        )
        selection = core_selection_losses(predictions)
        record = {
            "epoch": epoch,
            "update": global_update,
            "learning_rate": learning_rate,
            "train_loss": {
                dataset: {
                    name: float(np.mean(values))
                    for name, values in losses.items()
                }
                for dataset, losses in epoch_losses.items()
            },
            "validation": selection,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        _append_jsonl(train_log_path, record)
        if float(selection["selection_loss"]) < best_loss:
            best_loss = float(selection["selection_loss"])
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_loss": best_loss,
                    "config": config.to_dict(),
                },
                config.output_dir / "best_checkpoint.pt",
            )

    checkpoint = torch.load(
        config.output_dir / "best_checkpoint.pt",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    with np.load(
        config.output_dir / "validation" / f"epoch_{best_epoch:03d}.npz",
        allow_pickle=False,
    ) as archive:
        selected_predictions = {key: archive[key] for key in archive.files}
    thresholds, threshold_details = select_core_shared_thresholds(selected_predictions)
    selection_payload = {
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "shared_attribute_thresholds": thresholds,
        "threshold_details": threshold_details,
        "outer_test_accessed": False,
        "decoder": (
            "Level1 Normal->Normal; Level1 Abnormal->Crackle/Wheeze/Both; "
            "if neither attribute crosses its shared validation threshold, use "
            "the larger probability-minus-threshold margin with Crackle winning ties"
        ),
    }
    _write_json(config.output_dir / "validation_selection.json", selection_payload)

    del waveform_store
    gc.collect()
    terminal_samples = load_terminal_samples(config)
    terminal_by_dataset = {
        dataset: tuple(
            sample for sample in terminal_samples if sample.dataset == dataset
        )
        for dataset in CORE_DATASETS
    }
    terminal_waveforms = _prepare_waveforms(terminal_samples, config)
    label_free = infer(
        model,
        terminal_by_dataset,
        terminal_waveforms,
        config,
        device,
        include_targets=False,
    )
    terminal_dir = config.output_dir / "terminal"
    label_free_path = terminal_dir / "selected_test_predictions_label_free.npz"
    _save_predictions(label_free_path, label_free)
    spr_targets = load_terminal_spr_test_targets(
        terminal_samples,
        include_checksums=False,
    )
    scored = _attach_targets(label_free, terminal_samples, spr_targets)
    _save_predictions(
        terminal_dir / "selected_test_predictions_scored.npz",
        scored,
    )
    metrics = _score(scored, thresholds)
    terminal_payload = {
        "status": "terminal_native_scores_complete",
        "condition": CONDITION,
        "selected_epoch": best_epoch,
        "threshold_source": "validation only; no test tuning",
        "outer_test_accessed": True,
        "terminal_targets_loaded_after_label_free_prediction_write": True,
        "prediction_support": {"icbhi": 2756, "sprsound": 1429},
        **metrics,
    }
    _write_json(terminal_dir / "native_metrics.json", terminal_payload)
    icbhi_score = float(terminal_payload["icbhi_flat4"]["official_score"])
    spr_score = float(
        terminal_payload["sprsound_inter_task1_1"]["official_score"]
    )
    summary = {
        "status": "complete",
        "condition": CONDITION,
        "selected_epoch": best_epoch,
        "selection_loss": best_loss,
        "updates": global_update,
        "outer_test_accessed": True,
        "icbhi_official_score": icbhi_score,
        "sprsound_task1_1_official_score": spr_score,
        "mainline_gate": {
            "icbhi_score_at_least_0_59": icbhi_score >= 0.59,
            "sprsound_score_at_least_0_90": spr_score >= 0.90,
            "numeric_gate_passed": icbhi_score >= 0.59 and spr_score >= 0.90,
        },
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    author_repo = (
        repo_root
        / "result/pafa_sprsound_transfer_20260722_235659/source/repo"
    )
    checkpoint = (
        repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    )
    icbhi_audio = (
        repo_root
        / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"
    )
    return author_repo, checkpoint, icbhi_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    author_repo, checkpoint, icbhi_audio = _default_paths(args.repo_root)
    config = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio,
        output_dir=args.output_dir
        or args.repo_root / "result/reproduce/pafa_joint_hierarchy" / CONDITION,
        device=args.device,
        cpu_threads=args.cpu_threads,
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
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
