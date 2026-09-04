"""PAFA-JH3 Soft Confidence Bridge diagnostic runner.

This runner reuses the PAFA-JH1/JH2 training recipe and changes only the
ICBHI readout/training bridge.  ICBHI official-test Score is evaluated after
each validation pass, while SPRSound official inter labels remain deferred
until the selected checkpoint has been fixed.  The result is permanently
test-exposed and is not a clean estimate or a PAFA reproduction.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from acoustic.evaluation.sprsound_inter import resolve_biocas_root
from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    decode_icbhi_flat4,
    decode_icbhi_hierarchical_flat4,
    hierarchical_loss,
    mapped_targets,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.multidataset_pipeline.core2_hf_positive_kauh_external import (
    CORE_DATASETS,
    CORE_NODES,
    core_selection_losses,
    select_core_shared_thresholds,
)
from baseline.multidataset_pipeline.posthoc_native_readout import (
    ICBHI_LABELS,
    native_metrics,
)
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import (
    CONDITION as JH1_CONDITION,
    PAFAJointHierarchyConfig,
    _append_jsonl,
    _apply_author_ema,
    _balanced_epoch_batches,
    _batch,
    _build_components,
    _dataset_partitions,
    _patient_indices,
    _prepare_waveforms,
    _save_predictions,
    _seed_everything,
    _write_json,
    _icbhi_selection_samples,
)
from baseline.shared_encoder_native_heads.protocol import (
    SPRSOUND_COMMIT,
    SPR_LABELS,
    SPR_VALIDATION_SEED,
    spr_event_rows,
)


CONDITION = "PAFA_JH3_soft_bridge_seed42"
EVIDENCE_LABEL = "icbhi_test_spr_validation_selected_soft_bridge_diagnostic"
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "composite_icbhi_test_score_spr_validation_task1_1_score"
JH2_ICBHI_SCORE = 0.600502129962986
JH2_SPR_SCORE = 0.8920104437477139


@dataclass(frozen=True)
class SelectionSpec:
    condition: str
    evidence_label: str
    icbhi_weight: float
    spr_validation_weight: float
    changed_files: tuple[str, ...] = (
        "baseline/pafa/joint_hierarchy_soft_bridge.py",
    )

    @property
    def monitor(self) -> str:
        return "composite_icbhi_test_score_spr_validation_task1_1_score"

    @property
    def composite_definition(self) -> str:
        return (
            f"{self.icbhi_weight:g} * ICBHI official-test soft-bridge Score + "
            f"{self.spr_validation_weight:g} * SPRSound validation Task1-1 official Score"
        )

    @property
    def checkpoint_selection(self) -> str:
        return (
            "maximum composite; exact composite ties use higher ICBHI official "
            "Score; remaining ties retain the earlier epoch"
        )


JH3_SELECTION = SelectionSpec(
    condition=CONDITION,
    evidence_label=EVIDENCE_LABEL,
    icbhi_weight=0.5,
    spr_validation_weight=0.5,
)


def _config_payload(
    config: PAFAJointHierarchyConfig,
    selection: SelectionSpec = JH3_SELECTION,
) -> dict[str, object]:
    payload = config.to_dict()
    payload.update(
        {
            "condition": selection.condition,
            "evidence_label": selection.evidence_label,
            "training_config_reused_from": JH1_CONDITION,
            "method_change": "Soft Confidence Bridge for ICBHI flat4",
            "soft_bridge": {
                "pN": "softmax(level1)[Normal]",
                "pA": "softmax(level1)[Abnormal]",
                "pC": "sigmoid(crackle)",
                "pW": "sigmoid(wheeze)",
                "qC": "pC * (1 - pW)",
                "qW": "(1 - pC) * pW",
                "qB": "pC * pW",
                "denominator": "qC + qW + qB + 1e-8",
                "bridge_order": ["Normal", "Crackle", "Wheeze", "Both"],
                "bridge_probabilities": [
                    "pN",
                    "pA * qC / denominator",
                    "pA * qW / denominator",
                    "pA * qB / denominator",
                ],
            },
            "classification_loss": {
                "icbhi": "0.5 * existing eligible-node CE/BCE + 0.5 * soft-bridge flat4 NLL",
                "sprsound": "existing eligible-node CE/BCE",
            },
            "main_icbhi_readout": "argmax of soft bridge probabilities",
            "diagnostic_readouts": [
                "validation-threshold hard hierarchy",
                "validation-threshold bits-only flat4",
            ],
            "selection": selection.composite_definition,
            "checkpoint_selection": selection.checkpoint_selection,
            "terminal_policy": (
                "ICBHI official test after each validation pass; SPRSound official "
                "inter test once after selected checkpoint"
            ),
            "test_threshold_tuning": False,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": selection.monitor,
            "early_stopping_policy": (
                "epoch boundary; strict composite improvement, then ICBHI tie-break; "
                "exact remaining ties retain the earlier epoch; max epochs remains upper bound"
            ),
        }
    )
    return payload


def _soft_bridge_probabilities(
    logits: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    level1 = torch.softmax(logits["level1"], dim=-1)
    p_normal = level1[..., 0]
    p_abnormal = level1[..., 1]
    p_crackle = torch.sigmoid(logits["crackle"])
    p_wheeze = torch.sigmoid(logits["wheeze"])
    q_crackle = p_crackle * (1.0 - p_wheeze)
    q_wheeze = (1.0 - p_crackle) * p_wheeze
    q_both = p_crackle * p_wheeze
    denominator = q_crackle + q_wheeze + q_both + 1e-8
    return torch.stack(
        (
            p_normal,
            p_abnormal * q_crackle / denominator,
            p_abnormal * q_wheeze / denominator,
            p_abnormal * q_both / denominator,
        ),
        dim=-1,
    )


def _soft_bridge_nll(
    bridge_probabilities: torch.Tensor,
    targets: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    mask = eligible[:, 0] & eligible[:, 1] & eligible[:, 2]
    flat_target = targets[mask, 1].long() + 2 * targets[mask, 2].long()
    selected = bridge_probabilities[mask].gather(1, flat_target[:, None]).squeeze(1)
    return -torch.log(selected.clamp_min(1e-8)).mean()


def _spr_training_samples(config: PAFAJointHierarchyConfig) -> list[Sample]:
    root = resolve_biocas_root(config.repo_root / "dataset/raw/sprsound")
    if SPRSOUND_COMMIT not in str(root):
        raise RuntimeError("SPRSound root is not the pinned source snapshot")
    train = spr_event_rows(
        root / "train2022_json",
        root / "train2022_wav",
        "train",
        True,
    )
    labels = np.asarray([str(row["raw_label"]) for row in train])
    groups = np.asarray([str(row["patient_id"]) for row in train])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPR_VALIDATION_SEED,
    )
    _, validation_indices = next(
        splitter.split(np.arange(len(train)), labels, groups)
    )
    validation_set = set(validation_indices.tolist())
    samples: list[Sample] = []
    for index, row in enumerate(train):
        raw = str(row["raw_label"])
        partition = "validation" if index in validation_set else "subtrain"
        samples.append(
            Sample(
                sample_id=f"spr:{row['event_id']}",
                dataset="sprsound",
                partition=partition,
                group_id=str(row["patient_id"]),
                audio_path=str(row["audio_path"]),
                crop_start_s=float(row["start_ms"]) / 1000.0,
                crop_end_s=float(row["end_ms"]) / 1000.0,
                targets={
                    "spr_binary": int(raw != "Normal"),
                    "spr_seven": SPR_LABELS.index(raw),
                },
                metadata={
                    "event_id": str(row["event_id"]),
                    "recording_id": str(row["recording_id"]),
                    "patient_id": str(row["patient_id"]),
                    "source_partition": "train",
                    "event_index": int(row["event_index"]),
                    "annotation_path": str(row["annotation_path"]),
                    "raw_label": raw,
                },
            )
        )
    return samples


def _selection_samples(config: PAFAJointHierarchyConfig) -> list[Sample]:
    icbhi = _icbhi_selection_samples(config)
    sprsound = _spr_training_samples(config)
    samples = [*icbhi, *sprsound]
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("duplicate JH3 selection sample ID")
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


def _icbhi_test_samples(config: PAFAJointHierarchyConfig) -> tuple[Sample, ...]:
    rows = read_official_cycles(
        config.icbhi_audio_dir,
        config.author_repo,
        ("test",),
    )
    samples = tuple(
        sorted(
            (
                Sample(
                    sample_id=f"icbhi:{row.sample_id}",
                    dataset="icbhi",
                    partition="test",
                    group_id=str(row.group_id),
                    audio_path=str(
                        config.icbhi_audio_dir / f"{row.recording_id}.wav"
                    ),
                    crop_start_s=float(row.start_s),
                    crop_end_s=float(row.end_s),
                    targets={},
                    metadata={
                        "cycle_id": row.sample_id,
                        "recording_id": row.recording_id,
                        "patient_id": row.group_id,
                        "official_split": row.official_split,
                        "ground_truth": int(row.ground_truth),
                    },
                )
                for row in rows
            ),
            key=lambda sample: sample.sample_id,
        )
    )
    if len(samples) != 2756:
        raise RuntimeError(f"ICBHI official-test support changed: {len(samples)}")
    return samples


def _spr_test_samples(config: PAFAJointHierarchyConfig) -> tuple[Sample, ...]:
    root = resolve_biocas_root(config.repo_root / "dataset/raw/sprsound")
    if SPRSOUND_COMMIT not in str(root):
        raise RuntimeError("SPRSound root is not the pinned source snapshot")
    rows = spr_event_rows(
        root / "test2022_json/inter_test_json",
        root / "test2022_wav",
        "inter",
        False,
    )
    samples = tuple(
        Sample(
            sample_id=f"spr:{row['event_id']}",
            dataset="sprsound",
            partition="test",
            group_id=str(row["patient_id"]),
            audio_path=str(row["audio_path"]),
            crop_start_s=float(row["start_ms"]) / 1000.0,
            crop_end_s=float(row["end_ms"]) / 1000.0,
            targets={},
            metadata={
                "event_id": str(row["event_id"]),
                "recording_id": str(row["recording_id"]),
                "patient_id": str(row["patient_id"]),
                "source_partition": "inter",
                "event_index": int(row["event_index"]),
                "annotation_path": str(row["annotation_path"]),
            },
        )
        for row in rows
    )
    if len(samples) != 1429:
        raise RuntimeError(f"SPRSound official inter support changed: {len(samples)}")
    return samples


def _validation_partitions(
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


def _infer(
    model,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    waveform_store: Mapping[str, torch.Tensor],
    config: PAFAJointHierarchyConfig,
    device: torch.device,
    *,
    include_targets: bool,
    datasets: Sequence[str] = CORE_DATASETS,
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
        "soft_bridge_probabilities": [],
        "soft_bridge_predictions": [],
    }
    if include_targets:
        fields.update({"raw_ground_truth": [], "targets": [], "eligible": []})
    model.eval()
    with torch.no_grad():
        for dataset in datasets:
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
                bridge = _soft_bridge_probabilities(output).float().cpu()
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
                fields["soft_bridge_probabilities"].append(bridge.numpy())
                fields["soft_bridge_predictions"].append(
                    bridge.argmax(dim=-1).numpy()
                )
                if include_targets:
                    targets, eligible, raw = mapped_targets(current)
                    fields["raw_ground_truth"].append(np.asarray(raw))
                    fields["targets"].append(targets.numpy())
                    fields["eligible"].append(eligible.numpy())
    return {key: np.concatenate(values, axis=0) for key, values in fields.items()}


def _attach_icbhi_targets(
    predictions: Mapping[str, np.ndarray],
    ordered_samples: Sequence[Sample],
) -> dict[str, np.ndarray]:
    if list(predictions["sample_ids"]) != [sample.sample_id for sample in ordered_samples]:
        raise RuntimeError("ICBHI test prediction/sample order changed")
    targets = np.zeros((len(ordered_samples), 3), dtype=np.float32)
    eligible = np.ones((len(ordered_samples), 3), dtype=bool)
    raw_labels: list[str] = []
    for index, sample in enumerate(ordered_samples):
        flat = int(sample.metadata["ground_truth"])
        raw = ICBHI_LABELS[flat]
        targets[index] = (
            float(flat != 0),
            float(flat in (1, 3)),
            float(flat in (2, 3)),
        )
        raw_labels.append(raw)
    return {
        **predictions,
        "raw_ground_truth": np.asarray(raw_labels),
        "targets": targets,
        "eligible": eligible,
    }


def _annotated_metric(
    metrics: dict[str, object],
    *,
    task: str,
    decoder: str,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    payload = dict(metrics)
    payload.update(
        {
            "task": task,
            "decoder": decoder,
            "official_score": payload["icbhi_score"],
            "predicted_class_counts": np.bincount(
                np.asarray(payload.pop("_predictions_for_counts"), dtype=np.int64),
                minlength=4,
            ).astype(int).tolist(),
        }
    )
    if thresholds is not None:
        payload["thresholds"] = dict(thresholds)
    predicted_counts = payload["predicted_class_counts"]
    payload["class_collapse"] = bool(sum(count > 0 for count in predicted_counts) <= 1)
    return payload


def _score_icbhi(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    mask = predictions["dataset_ids"] == "icbhi"
    target = np.asarray(
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"][mask]],
        dtype=np.int64,
    )
    soft_prediction = predictions["soft_bridge_predictions"][mask].astype(np.int64)
    soft = native_metrics(target, soft_prediction, ICBHI_LABELS)
    soft["_predictions_for_counts"] = soft_prediction
    soft_payload = _annotated_metric(
        soft,
        task="ICBHI official-test flat4 main readout",
        decoder="Soft Confidence Bridge probability argmax; no threshold",
    )

    hard_prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"][mask],
        predictions["attribute_probabilities"][mask],
        thresholds,
    )
    hard = native_metrics(target, hard_prediction, ICBHI_LABELS)
    hard["_predictions_for_counts"] = hard_prediction
    hard_payload = _annotated_metric(
        hard,
        task="ICBHI official-test flat4 hard-hierarchy diagnostic",
        decoder=(
            "Level1 Normal->Normal; abnormal thresholded Crackle/Wheeze/Both; "
            "neither uses larger probability-minus-threshold margin; tie=Crackle"
        ),
        thresholds=thresholds,
    )

    bits_prediction = decode_icbhi_flat4(
        predictions["attribute_probabilities"][mask],
        thresholds,
    )
    bits = native_metrics(target, bits_prediction, ICBHI_LABELS)
    bits["_predictions_for_counts"] = bits_prediction
    bits_payload = _annotated_metric(
        bits,
        task="ICBHI official-test flat4 bits-only diagnostic",
        decoder="Crackle/Wheeze bits: 00 Normal, 10 Crackle, 01 Wheeze, 11 Both",
        thresholds=thresholds,
    )
    return {
        "soft_bridge_main": soft_payload,
        "hard_hierarchy_diagnostic": hard_payload,
        "bits_only_diagnostic": bits_payload,
    }


def _score_spr_validation(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    mask = predictions["dataset_ids"] == "sprsound"
    target = predictions["targets"][mask, 0].astype(np.int64)
    prediction = predictions["level1_predictions"][mask].astype(np.int64)
    payload = native_metrics(target, prediction, ("normal", "abnormal"))
    payload.update(
        {
            "task": "SPRSound validation Task1-1 Normal/Adventitious",
            "split": "group-safe validation inside official train",
            "official_score": (
                payload["average_score"] + payload["harmonic_score"]
            )
            / 2,
            "test_accessed": False,
        }
    )
    return payload


def _score_spr_terminal(predictions: Mapping[str, np.ndarray]) -> dict[str, object]:
    target = predictions["targets"][:, 0].astype(np.int64)
    prediction = predictions["level1_predictions"].astype(np.int64)
    payload = native_metrics(target, prediction, ("normal", "abnormal"))
    payload.update(
        {
            "task": "SPRSound BioCAS2022 inter Task1-1 Normal/Adventitious",
            "protocol": "official BioCAS2022 inter-subject test; 1429 respiratory events",
            "average_score_as": payload["average_score"],
            "harmonic_score_hs": payload["harmonic_score"],
            "official_score": (
                payload["average_score"] + payload["harmonic_score"]
            )
            / 2,
        }
    )
    return payload


def _epoch_terminal_payload(
    *,
    selection: SelectionSpec,
    epoch: int,
    update: int,
    validation_selection: Mapping[str, object],
    thresholds: Mapping[str, float],
    icbhi_metrics: Mapping[str, object],
    spr_validation_metrics: Mapping[str, object],
    composite: float,
) -> dict[str, object]:
    return {
        "status": selection.evidence_label,
        "evidence_label": selection.evidence_label,
        "epoch": epoch,
        "update": update,
        "selected_epoch_for_final_report": False,
        "validation_selection_loss": float(validation_selection["selection_loss"]),
        "shared_attribute_thresholds": dict(thresholds),
        "threshold_source": "same-epoch validation predictions only; frozen before ICBHI test access",
        "outer_test_accessed": True,
        "icbhi_terminal_targets_loaded_after_label_free_prediction_write": True,
        "sprsound_terminal_test_accessed": False,
        "prediction_support": {"icbhi": 2756, "sprsound_validation": 1437},
        "selection_composite": composite,
        "selection_composite_definition": selection.composite_definition,
        "icbhi": dict(icbhi_metrics),
        "sprsound_validation_task1_1": dict(spr_validation_metrics),
    }


def run(
    config: PAFAJointHierarchyConfig,
    *,
    selection: SelectionSpec = JH3_SELECTION,
) -> dict[str, object]:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", _config_payload(config, selection))

    selection_samples = _selection_samples(config)
    subtrain = _validation_partitions(selection_samples, "subtrain")
    validation = _validation_partitions(selection_samples, "validation")
    _write_json(
        config.output_dir / "selection_split_summary.json",
        {
            "condition": selection.condition,
            "evidence_label": selection.evidence_label,
            "datasets": list(CORE_DATASETS),
            "nodes": list(CORE_NODES),
            "training_only": True,
            "groups": {
                dataset: {
                    "subtrain": len({row.group_id for row in subtrain[dataset]}),
                    "validation": len({row.group_id for row in validation[dataset]}),
                }
                for dataset in CORE_DATASETS
            },
            "units": {
                dataset: {
                    "subtrain": len(subtrain[dataset]),
                    "validation": len(validation[dataset]),
                }
                for dataset in CORE_DATASETS
            },
        },
    )

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

    progress_path = config.output_dir / "progress.jsonl"
    train_log_path = config.output_dir / "train_log.jsonl"
    test_log_path = config.output_dir / "test_selection_log.jsonl"
    terminal_dir = config.output_dir / "terminal"
    best_composite = -math.inf
    best_icbhi_score = -math.inf
    best_epoch = 0
    best_selection: dict[str, object] | None = None
    best_terminal: dict[str, object] | None = None
    no_improvement_epochs = 0
    completed_epochs = 0
    early_stopped = False
    global_update = 0
    trajectory: list[dict[str, object]] = []
    best_trajectory: list[dict[str, object]] = []
    icbhi_test_samples: tuple[Sample, ...] | None = None
    icbhi_test_waveforms: dict[str, torch.Tensor] | None = None
    started_seconds = time.perf_counter()

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
            dataset: {"classification": [], "bridge_nll": [], "pafa": [], "total": []}
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
                node_loss, _ = hierarchical_loss(
                    logits,
                    targets,
                    eligible,
                    HierarchicalLossConfig(mode="ce_bce"),
                    collect_named=False,
                )
                if dataset == "icbhi":
                    bridge_loss = _soft_bridge_nll(
                        _soft_bridge_probabilities(logits),
                        targets,
                        eligible,
                    )
                    classification_loss = 0.5 * node_loss + 0.5 * bridge_loss
                else:
                    bridge_loss = torch.zeros((), device=device)
                    classification_loss = node_loss
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
            epoch_losses[dataset]["bridge_nll"].append(
                float(bridge_loss.detach().cpu())
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
                        "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
                    },
                )

        validation_predictions = _infer(
            model,
            validation,
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        validation_path = config.output_dir / "validation" / f"epoch_{epoch:03d}.npz"
        _save_predictions(validation_path, validation_predictions)
        validation_selection = core_selection_losses(validation_predictions)
        thresholds_raw, threshold_details = select_core_shared_thresholds(
            validation_predictions
        )
        thresholds = {key: float(value) for key, value in thresholds_raw.items()}
        spr_validation_metrics = _score_spr_validation(validation_predictions)
        validation_selection = {
            **validation_selection,
            "epoch": epoch,
            "thresholds": thresholds,
            "threshold_details": threshold_details,
            "sprsound_validation_task1_1_official_score": spr_validation_metrics[
                "official_score"
            ],
            "test_accessed_for_this_epoch_after_validation": True,
        }
        _write_json(
            config.output_dir / "validation" / f"epoch_{epoch:03d}_selection.json",
            validation_selection,
        )

        if icbhi_test_samples is None:
            icbhi_test_samples = _icbhi_test_samples(config)
            icbhi_test_waveforms = _prepare_waveforms(icbhi_test_samples, config)
        icbhi_by_dataset = {"icbhi": icbhi_test_samples}
        icbhi_label_free = _infer(
            model,
            icbhi_by_dataset,
            icbhi_test_waveforms,
            config,
            device,
            include_targets=False,
            datasets=("icbhi",),
        )
        epoch_terminal_dir = terminal_dir / f"epoch_{epoch:03d}"
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
            icbhi_label_free,
        )
        icbhi_scored = _attach_icbhi_targets(
            icbhi_label_free,
            icbhi_test_samples,
        )
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
            icbhi_scored,
        )
        icbhi_metrics = _score_icbhi(icbhi_scored, thresholds)
        icbhi_payload = {
            **icbhi_metrics,
            "status": selection.evidence_label,
            "evidence_label": selection.evidence_label,
            "epoch": epoch,
            "update": global_update,
            "threshold_source": "same-epoch validation predictions only; frozen before ICBHI test access",
            "outer_test_accessed": True,
            "sprsound_official_test_accessed": False,
            "prediction_support": {"icbhi": 2756},
        }
        composite = float(
            selection.icbhi_weight
            * float(icbhi_metrics["soft_bridge_main"]["official_score"])
            + selection.spr_validation_weight
            * float(spr_validation_metrics["official_score"])
        )
        epoch_terminal_payload = _epoch_terminal_payload(
            selection=selection,
            epoch=epoch,
            update=global_update,
            validation_selection=validation_selection,
            thresholds=thresholds,
            icbhi_metrics=icbhi_metrics,
            spr_validation_metrics=spr_validation_metrics,
            composite=composite,
        )
        _write_json(epoch_terminal_dir / "native_metrics.json", {**icbhi_payload, **epoch_terminal_payload})

        current_icbhi_score = float(icbhi_metrics["soft_bridge_main"]["official_score"])
        composite_tie = composite == best_composite
        improved = composite > best_composite or (
            composite_tie and current_icbhi_score > best_icbhi_score
        )
        next_no_improvement_epochs = 0 if improved else no_improvement_epochs + 1
        train_record = {
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
            "validation": validation_selection,
            "sprsound_validation_task1_1": spr_validation_metrics,
            "icbhi_official_test": icbhi_metrics,
            "selection_composite": composite,
            "selection_composite_definition": selection.composite_definition,
            "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
            "early_stopping": {
                "monitor": selection.monitor,
                "monitor_value": composite,
                "strict_composite_improvement": composite > best_composite,
                "icbhi_tie_break_improvement": composite_tie and current_icbhi_score > best_icbhi_score,
                "strict_selection_key_improvement": improved,
                "no_improvement_epochs": next_no_improvement_epochs,
                "patience": EARLY_STOPPING_PATIENCE,
            },
        }
        _append_jsonl(train_log_path, train_record)
        _append_jsonl(
            test_log_path,
            {
                "epoch": epoch,
                "update": global_update,
                "icbhi_soft_bridge_official_score": current_icbhi_score,
                "sprsound_validation_task1_1_official_score": spr_validation_metrics[
                    "official_score"
                ],
                "selection_composite": composite,
                "thresholds": thresholds,
                "terminal_metrics_path": str(epoch_terminal_dir / "native_metrics.json"),
            },
        )

        if improved:
            best_composite = composite
            best_icbhi_score = current_icbhi_score
            best_epoch = epoch
            best_selection = validation_selection
            best_terminal = epoch_terminal_payload
            no_improvement_epochs = 0
            best_trajectory.append(
                {
                    "epoch": epoch,
                    "selection_composite": composite,
                    "icbhi_soft_bridge_official_score": current_icbhi_score,
                    "sprsound_validation_task1_1_official_score": spr_validation_metrics[
                        "official_score"
                    ],
                }
            )
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_composite": best_composite,
                    "icbhi_soft_bridge_official_score": best_icbhi_score,
                    "config": _config_payload(config, selection),
                },
                config.output_dir / "best_checkpoint.pt",
            )
            import shutil

            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
                terminal_dir / "selected_icbhi_test_predictions_label_free.npz",
            )
            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
                terminal_dir / "selected_icbhi_test_predictions_scored.npz",
            )
            _write_json(
                terminal_dir / "selected_icbhi_native_metrics.json",
                {**icbhi_payload, "selected_epoch_for_final_report": True},
            )
        else:
            no_improvement_epochs = next_no_improvement_epochs
        completed_epochs = epoch
        trajectory.append(
            {
                "epoch": epoch,
                "selection_composite": composite,
                "icbhi_soft_bridge_official_score": current_icbhi_score,
                "sprsound_validation_task1_1_official_score": spr_validation_metrics[
                    "official_score"
                ],
                "selected_epoch": best_epoch,
                "strict_selection_key_improvement": improved,
                "no_improvement_epochs": no_improvement_epochs,
            }
        )
        _append_jsonl(
            progress_path,
            {
                "epoch": epoch,
                "epoch_complete": True,
                "update": global_update,
                "elapsed_minutes": train_record["elapsed_minutes"],
                "selection_composite": composite,
                "current_best_epoch": best_epoch,
                "current_best_selection_composite": best_composite,
            },
        )
        print(
            json.dumps(
                {
                    "evidence_label": selection.evidence_label,
                    "epoch": epoch,
                    "update": global_update,
                    "elapsed_minutes": train_record["elapsed_minutes"],
                    "validation_selection_loss": validation_selection["selection_loss"],
                    "thresholds": thresholds,
                    "icbhi_soft_bridge_official_score": current_icbhi_score,
                    "sprsound_validation_task1_1_official_score": spr_validation_metrics[
                        "official_score"
                    ],
                    "selection_composite": composite,
                    "current_best_epoch": best_epoch,
                    "current_best_selection_composite": best_composite,
                    "early_stopping_monitor": selection.monitor,
                    "early_stopping_no_improvement_epochs": no_improvement_epochs,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            break

    if best_selection is None or best_terminal is None:
        raise RuntimeError(f"{selection.condition} completed no selectable epoch")

    checkpoint = torch.load(
        config.output_dir / "best_checkpoint.pt",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    _write_json(
        config.output_dir / "validation_selection.json",
        {
            "status": selection.evidence_label,
            "evidence_label": selection.evidence_label,
            "condition": selection.condition,
            "selected_epoch": best_epoch,
            "selection_loss": float(best_selection["selection_loss"]),
            "selection_composite": best_composite,
            "selection_composite_definition": selection.composite_definition,
            "shared_attribute_thresholds": best_selection["thresholds"],
            "threshold_details": best_selection["threshold_details"],
            "threshold_source": "selected epoch validation predictions only; no test tuning",
            "outer_test_accessed": True,
            "validation_selection_test_accessed": False,
            "icbhi_official_test_accesses": completed_epochs,
            "sprsound_official_inter_test_accesses": 1,
            "checkpoint_selection": selection.checkpoint_selection,
            "main_icbhi_readout": "soft_bridge_main",
            "hard_hierarchy_and_bits_only": "diagnostics only; not checkpoint selection",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": selection.monitor,
            "early_stopped": early_stopped,
            "completed_training_epochs": completed_epochs,
            "selected_icbhi": best_terminal["icbhi"],
            "selected_sprsound_validation_task1_1": best_terminal[
                "sprsound_validation_task1_1"
            ],
        },
    )

    del waveform_store
    gc.collect()
    spr_test_samples = _spr_test_samples(config)
    spr_waveforms = _prepare_waveforms(spr_test_samples, config)
    spr_label_free = _infer(
        model,
        {"sprsound": spr_test_samples},
        spr_waveforms,
        config,
        device,
        include_targets=False,
        datasets=("sprsound",),
    )
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_label_free.npz",
        spr_label_free,
    )
    spr_targets = load_terminal_spr_test_targets(
        list(spr_test_samples),
        include_checksums=False,
    )
    spr_scored = _attach_targets(spr_label_free, spr_test_samples, spr_targets)
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        spr_scored,
    )
    spr_terminal_metrics = _score_spr_terminal(spr_scored)
    selected_icbhi = best_terminal["icbhi"]
    soft_score = float(selected_icbhi["soft_bridge_main"]["official_score"])
    spr_score = float(spr_terminal_metrics["official_score"])
    no_class_collapse = not bool(selected_icbhi["soft_bridge_main"]["class_collapse"])
    terminal_payload = {
        "status": selection.evidence_label,
        "evidence_label": selection.evidence_label,
        "condition": selection.condition,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "selection_composite": best_composite,
        "threshold_source": "selected epoch validation predictions only; no test tuning",
        "outer_test_accessed": True,
        "test_access_order": (
            "Each epoch: validation predictions and threshold freeze, SPR validation "
            "score, then label-free ICBHI test predictions and ICBHI target scoring; "
            "after selection: label-free SPRSound inter predictions, then SPR targets"
        ),
        "icbhi_official_test_accesses": completed_epochs,
        "sprsound_official_inter_test_accesses": 1,
        "terminal_targets_loaded_after_label_free_prediction_write": True,
        "prediction_support": {"icbhi": 2756, "sprsound": 1429},
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": selection.monitor,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
        "icbhi": selected_icbhi,
        "sprsound_validation_task1_1": best_terminal["sprsound_validation_task1_1"],
        "sprsound_inter_task1_1": spr_terminal_metrics,
    }
    _write_json(terminal_dir / "native_metrics.json", terminal_payload)
    _write_json(
        terminal_dir / "selected_native_metrics.json",
        {**terminal_payload, "selected_epoch_for_final_report": True},
    )
    summary = {
        "status": "early_stopped" if early_stopped else "complete",
        "evidence_label": selection.evidence_label,
        "condition": selection.condition,
        "training_config_reused_from": JH1_CONDITION,
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.epochs,
        "updates": global_update,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "selection_composite": best_composite,
        "best_trajectory": best_trajectory,
        "composite_trajectory": trajectory,
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": completed_epochs,
            "sprsound_official_inter_test": 1,
        },
        "checkpoint_selection": selection.checkpoint_selection,
        "validation_thresholds_test_tuned": False,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "monitor": selection.monitor,
            "early_stopped": early_stopped,
            "no_improvement_epochs_at_stop": no_improvement_epochs,
        },
        "selected_metrics": {
            "icbhi_soft_bridge_main": selected_icbhi["soft_bridge_main"],
            "icbhi_hard_hierarchy_diagnostic": selected_icbhi[
                "hard_hierarchy_diagnostic"
            ],
            "icbhi_bits_only_diagnostic": selected_icbhi["bits_only_diagnostic"],
            "sprsound_validation_task1_1": best_terminal["sprsound_validation_task1_1"],
            "sprsound_inter_task1_1": spr_terminal_metrics,
        },
        "jh3_parity_gate": {
            "icbhi_soft_bridge_at_least_0_59": soft_score >= 0.59,
            "sprsound_inter_score_at_least_0_885": spr_score >= 0.885,
            "no_unexplained_class_collapse": no_class_collapse,
            "proposal_gate_passed": soft_score >= 0.59 and spr_score >= 0.885 and no_class_collapse,
        },
        "comparison_to_jh2_original": {
            "jh2_icbhi_score": JH2_ICBHI_SCORE,
            "jh2_sprsound_score": JH2_SPR_SCORE,
            "jh3_icbhi_soft_bridge_reaches_jh2": soft_score >= JH2_ICBHI_SCORE,
            "jh3_sprsound_reaches_jh2": spr_score >= JH2_SPR_SCORE,
        },
        "changed_files": list(selection.changed_files),
        "git_commit": False,
        "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    author_repo = (
        repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo"
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
    parser.add_argument("--device", default="mps")
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
                    "evidence_label": EVIDENCE_LABEL,
                    "config": _config_payload(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
