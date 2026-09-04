"""PAFA-JH3.4 Hard Hierarchy with unit-level scalar log-fbank MVN.

This isolated runner reuses the JH2 epochwise official-test-selected loop and
changes only the BEATs frontend normalization.  It is permanently
test-exposed and is not a clean estimate or a PAFA reproduction.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from baseline.four_dataset_frozen_encoder.data import (
    _load_spr,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    decode_icbhi_flat4,
    decode_icbhi_hierarchical_flat4,
    hierarchical_loss,
)
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
from baseline.pafa import joint_hierarchy_test_selected as jh2
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    PAFAJointHierarchyModel,
    _apply_author_ema,
    _balanced_epoch_batches,
    _batch,
    _dataset_partitions,
    _icbhi_sample,
    _patient_indices,
    _prepare_waveforms,
    _save_predictions,
    _seed_everything,
    infer,
    load_selection_samples,
)
from baseline.pafa.joint_hierarchy_soft_bridge_mvn import UnitScalarMVNBEATs


CONDITION = "PAFA_JH3_4_hard_hierarchy_mvn_seed42"
EVIDENCE_LABEL = "icbhi_test_selected_hard_hierarchy_unit_scalar_fbank_mvn_diagnostic"
CHANGED_FILES = ("baseline/pafa/joint_hierarchy_test_selected_mvn.py",)
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "icbhi_official_test_score"
JH2_ICBHI_SCORE = 0.600502129962986
JH2_SPR_SCORE = 0.8920104437477139


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

    transfer_model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(config.checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    beats = UnitScalarMVNBEATs(transfer_model)
    projector = ProjectionHead(
        input_dim=beats.final_feat_dim,
        hidden_dim=None,
        output_dim=config.projection_output_dim,
        attention=config.projection_attention,
        proj_type="end2end",
        norm_type=config.projection_norm,
    )
    return PAFAJointHierarchyModel(beats, projector).to(device), PAFALoss().to(device)


_ORIGINAL_CONFIG_PAYLOAD = jh2._config_payload


def _config_payload(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    payload = _ORIGINAL_CONFIG_PAYLOAD(config)
    payload.update(
        {
            "condition": CONDITION,
            "evidence_label": EVIDENCE_LABEL,
            "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
            "method_change": "unit-level scalar log-fbank MVN frontend only",
            "main_icbhi_readout": "Hard Hierarchy",
            "selection": "ICBHI official-test Hard Hierarchy Score only",
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain "
                "the earlier epoch"
            ),
            "terminal_policy": (
                "after every epoch validation threshold freeze and label-free ICBHI "
                "official-test readout; SPRSound official inter readout once after "
                "selected checkpoint"
            ),
            "frontend_normalization": {
                "waveform": "mono 16 kHz; 5 s repeat-pad/front-truncate",
                "fbank": (
                    "canonical BEATs torchaudio Kaldi log-fbank; num_mel_bins=128; "
                    "sample_frequency=16000; frame_length=25 ms; frame_shift=10 ms"
                ),
                "fixed_beats_affine": "not applied",
                "scope": "each unit independently over all T,F values",
                "mean": "mean over all T,F",
                "variance": "mean((F-mu)^2), unbiased=False",
                "transform": "(F-mu)/sqrt(variance+1e-5)",
                "waveform_peak_rms_normalization": "none",
                "spec_augment": "none",
                "dataset_specific_statistics": "none",
                "additional_normalization": "none",
            },
        }
    )
    return payload


def _install_isolated_overrides() -> None:
    jh2._build_components = _build_components
    jh2._config_payload = _config_payload


def _load_icbhi_terminal_samples(
    config: PAFAJointHierarchyConfig,
) -> tuple[object, ...]:
    rows = read_official_cycles(
        config.icbhi_audio_dir,
        config.author_repo,
        ("test",),
    )
    return tuple(
        sorted(
            (
                _icbhi_sample(row, config.icbhi_audio_dir, "test")
                for row in rows
            ),
            key=lambda sample: sample.sample_id,
        )
    )


def _load_spr_terminal_samples(
    config: PAFAJointHierarchyConfig,
) -> tuple[object, ...]:
    rows, _ = _load_spr(
        config.repo_root / "dataset/raw",
        include_checksums=False,
    )
    return tuple(
        sorted(
            (sample for sample in rows if sample.partition == "test"),
            key=lambda sample: sample.sample_id,
        )
    )


def _attach_icbhi_targets(
    predictions: Mapping[str, np.ndarray],
    ordered_samples: Sequence[object],
) -> dict[str, np.ndarray]:
    if list(predictions["sample_ids"]) != [sample.sample_id for sample in ordered_samples]:
        raise RuntimeError("ICBHI terminal prediction/sample order changed")
    targets = np.zeros((len(ordered_samples), 3), dtype=np.float32)
    eligible = np.ones((len(ordered_samples), 3), dtype=bool)
    raw_labels: list[str] = []
    for index, sample in enumerate(ordered_samples):
        flat = int(sample.targets["icbhi_flat4"])
        raw_labels.append(ICBHI_LABELS[flat])
        targets[index] = (
            float(flat != 0),
            float(flat in (1, 3)),
            float(flat in (2, 3)),
        )
    return {
        **predictions,
        "raw_ground_truth": np.asarray(raw_labels),
        "targets": targets,
        "eligible": eligible,
    }


def _score_icbhi(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    target = np.asarray(
        [
            ICBHI_LABELS.index(str(value))
            for value in predictions["raw_ground_truth"]
        ],
        dtype=np.int64,
    )
    hard_prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"],
        predictions["attribute_probabilities"],
        thresholds,
    )
    hard = native_metrics(target, hard_prediction, ICBHI_LABELS)
    hard.update(
        {
            "task": "ICBHI official-test flat4 Hard Hierarchy main readout",
            "protocol": "official recording split 60/40; 2756 respiratory cycles",
            "decoder": (
                "Level1 Normal->Normal; Level1 Abnormal->Crackle/Wheeze/Both "
                "from shared validation thresholds; neither attribute over "
                "threshold->larger probability-minus-threshold margin; "
                "Crackle wins ties"
            ),
            "thresholds": dict(thresholds),
            "official_score": hard["icbhi_score"],
            "predicted_class_counts": np.bincount(
                hard_prediction,
                minlength=4,
            ).astype(int).tolist(),
        }
    )
    hard["class_collapse"] = bool(
        sum(count > 0 for count in hard["predicted_class_counts"]) <= 1
    )

    bits_prediction = decode_icbhi_flat4(
        predictions["attribute_probabilities"],
        thresholds,
    )
    bits = native_metrics(target, bits_prediction, ICBHI_LABELS)
    bits.update(
        {
            "task": "ICBHI official-test flat4 bits-only diagnostic",
            "decoder": (
                "Crackle/Wheeze bits only: 00 Normal, 10 Crackle, "
                "01 Wheeze, 11 Both"
            ),
            "thresholds": dict(thresholds),
            "official_score": bits["icbhi_score"],
            "predicted_class_counts": np.bincount(
                bits_prediction,
                minlength=4,
            ).astype(int).tolist(),
        }
    )
    bits["class_collapse"] = bool(
        sum(count > 0 for count in bits["predicted_class_counts"]) <= 1
    )
    return {
        "icbhi_flat4": hard,
        "icbhi_flat4_bits_only_ablation": bits,
    }


def _score_spr_terminal(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    target = predictions["targets"][:, 0].astype(np.int64)
    prediction = predictions["level1_predictions"].astype(np.int64)
    metrics = native_metrics(target, prediction, ("normal", "abnormal"))
    metrics.update(
        {
            "task": "SPRSound BioCAS2022 Task1-1 Normal/Adventitious",
            "protocol": "official inter-subject test; 1429 respiratory events",
            "average_score_as": metrics["average_score"],
            "harmonic_score_hs": metrics["harmonic_score"],
            "official_score": (
                metrics["average_score"] + metrics["harmonic_score"]
            )
            / 2,
            "task1_2_raw7": "not produced by the shared three-node head",
        }
    )
    return metrics


def _epoch_terminal_payload(
    *,
    epoch: int,
    update: int,
    validation_selection: Mapping[str, object],
    thresholds: Mapping[str, float],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "epoch": epoch,
        "update": update,
        "selected_epoch_for_final_report": False,
        "validation_selection_loss": float(validation_selection["selection_loss"]),
        "shared_attribute_thresholds": dict(thresholds),
        "threshold_source": "same-epoch validation predictions only; frozen before ICBHI test access",
        "outer_test_accessed": True,
        "sprsound_official_inter_test_accessed": False,
        "prediction_support": {"icbhi": 2756},
        "test_selection_metric": "icbhi_flat4.official_score",
        **metrics,
    }


def run_hard_only(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    jh2._write_json(config.output_dir / "config.json", _config_payload(config))

    selection_samples = load_selection_samples(config)
    subtrain = _dataset_partitions(selection_samples, "subtrain")
    validation = _dataset_partitions(selection_samples, "validation")
    jh2._write_json(
        config.output_dir / "selection_split_summary.json",
        {
            "condition": CONDITION,
            "evidence_label": EVIDENCE_LABEL,
            "datasets": list(CORE_DATASETS),
            "nodes": list(CORE_NODES),
            "training_only": True,
            "groups": {
                dataset: {
                    "subtrain": len({row.group_id for row in rows}),
                    "validation": len(
                        {row.group_id for row in validation[dataset]}
                    ),
                }
                for dataset, rows in subtrain.items()
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
    best_score = -math.inf
    best_epoch = 0
    best_selection: dict[str, object] | None = None
    best_terminal: dict[str, object] | None = None
    no_improvement_epochs = 0
    completed_epochs = 0
    early_stopped = False
    global_update = 0
    icbhi_terminal_samples: tuple[object, ...] | None = None
    icbhi_terminal_waveforms: dict[str, torch.Tensor] | None = None
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
                classification_loss, _ = jh2.hierarchical_loss(
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
            epoch_losses[dataset]["pafa"].append(
                float(pafa_loss.detach().cpu())
            )
            epoch_losses[dataset]["total"].append(
                float(total_loss.detach().cpu())
            )
            if global_update % 100 == 0:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "epoch_batch": batch_index,
                                "epoch_batches": len(batches),
                                "update": global_update,
                                "elapsed_minutes": (
                                    time.perf_counter() - started_seconds
                                )
                                / 60.0,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

        validation_predictions = infer(
            model,
            validation,
            waveform_store,
            config,
            device,
            include_targets=True,
        )
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            validation_predictions,
        )
        validation_selection = core_selection_losses(validation_predictions)
        thresholds_raw, threshold_details = select_core_shared_thresholds(
            validation_predictions
        )
        thresholds = {key: float(value) for key, value in thresholds_raw.items()}
        validation_selection = {
            **validation_selection,
            "epoch": epoch,
            "thresholds": thresholds,
            "threshold_details": threshold_details,
            "test_accessed_for_this_epoch_after_freeze": True,
            "sprsound_official_test_accessed_for_this_epoch": False,
        }
        jh2._write_json(
            config.output_dir / "validation" / f"epoch_{epoch:03d}_selection.json",
            validation_selection,
        )

        if icbhi_terminal_samples is None:
            icbhi_terminal_samples = _load_icbhi_terminal_samples(config)
            icbhi_terminal_waveforms = _prepare_waveforms(
                icbhi_terminal_samples,
                config,
            )
        icbhi_label_free = infer(
            model,
            {"icbhi": icbhi_terminal_samples, "sprsound": tuple()},
            icbhi_terminal_waveforms,
            config,
            device,
            include_targets=False,
        )
        epoch_terminal_dir = terminal_dir / f"epoch_{epoch:03d}"
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
            icbhi_label_free,
        )
        icbhi_scored = _attach_icbhi_targets(
            icbhi_label_free,
            icbhi_terminal_samples,
        )
        _save_predictions(
            epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
            icbhi_scored,
        )
        icbhi_metrics = _score_icbhi(icbhi_scored, thresholds)
        terminal_payload = _epoch_terminal_payload(
            epoch=epoch,
            update=global_update,
            validation_selection=validation_selection,
            thresholds=thresholds,
            metrics=icbhi_metrics,
        )
        jh2._write_json(epoch_terminal_dir / "native_metrics.json", terminal_payload)
        icbhi_score = float(icbhi_metrics["icbhi_flat4"]["official_score"])
        improved = icbhi_score > best_score
        next_no_improvement_epochs = 0 if improved else no_improvement_epochs + 1

        jh2._append_jsonl(
            test_log_path,
            {
                "epoch": epoch,
                "update": global_update,
                "validation_selection_loss": validation_selection[
                    "selection_loss"
                ],
                "thresholds": thresholds,
                "icbhi_hard_hierarchy_official_score": icbhi_score,
                "sprsound_official_inter_test_accessed": False,
                "terminal_metrics_path": str(
                    epoch_terminal_dir / "native_metrics.json"
                ),
            },
        )
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
            "terminal_test": {
                "icbhi_hard_hierarchy_official_score": icbhi_score,
                "sprsound_official_inter_test_accessed": False,
                "evidence_label": EVIDENCE_LABEL,
            },
            "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
            "early_stopping": {
                "monitor": EARLY_STOPPING_MONITOR,
                "monitor_value": icbhi_score,
                "strict_improvement": improved,
                "no_improvement_epochs": next_no_improvement_epochs,
                "patience": EARLY_STOPPING_PATIENCE,
            },
        }
        jh2._append_jsonl(train_log_path, train_record)

        if improved:
            best_score = icbhi_score
            best_epoch = epoch
            best_selection = validation_selection
            best_terminal = terminal_payload
            no_improvement_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "icbhi_hard_hierarchy_official_score": best_score,
                    "selection_loss": float(validation_selection["selection_loss"]),
                    "config": _config_payload(config),
                },
                config.output_dir / "best_checkpoint.pt",
            )
            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_label_free.npz",
                terminal_dir / "selected_icbhi_test_predictions_label_free.npz",
            )
            shutil.copyfile(
                epoch_terminal_dir / "icbhi_test_predictions_scored.npz",
                terminal_dir / "selected_icbhi_test_predictions_scored.npz",
            )
            jh2._write_json(
                terminal_dir / "selected_icbhi_native_metrics.json",
                {
                    **terminal_payload,
                    "selected_epoch_for_final_report": True,
                },
            )
        else:
            no_improvement_epochs = next_no_improvement_epochs
        completed_epochs = epoch
        jh2._append_jsonl(
            progress_path,
            {
                "epoch": epoch,
                "epoch_complete": True,
                "update": global_update,
                "elapsed_minutes": train_record["elapsed_minutes"],
                "icbhi_official_score": icbhi_score,
                "current_best_epoch": best_epoch,
                "current_best_icbhi_official_score": best_score,
                "early_stopping_no_improvement_epochs": no_improvement_epochs,
            },
        )
        print(
            json.dumps(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "epoch": epoch,
                    "update": global_update,
                    "elapsed_minutes": train_record["elapsed_minutes"],
                    "validation_selection_loss": validation_selection[
                        "selection_loss"
                    ],
                    "thresholds": thresholds,
                    "icbhi_official_score": icbhi_score,
                    "sprsound_official_inter_test_accessed": False,
                    "current_best_epoch": best_epoch,
                    "current_best_icbhi_official_score": best_score,
                    "early_stopping_monitor": EARLY_STOPPING_MONITOR,
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
        raise RuntimeError(f"{CONDITION} completed no selectable epoch")

    jh2._write_json(
        config.output_dir / "validation_selection.json",
        {
            "status": EVIDENCE_LABEL,
            "evidence_label": EVIDENCE_LABEL,
            "condition": CONDITION,
            "selected_epoch": best_epoch,
            "selection_loss": float(best_selection["selection_loss"]),
            "shared_attribute_thresholds": best_selection["thresholds"],
            "threshold_details": best_selection["threshold_details"],
            "threshold_source": "selected epoch validation predictions only; no test tuning",
            "outer_test_accessed": True,
            "validation_selection_test_accessed": False,
            "icbhi_official_test_accesses": completed_epochs,
            "sprsound_official_inter_test_accesses": 1,
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
            ),
            "main_icbhi_readout": "icbhi_flat4",
            "diagnostic_bits_only": True,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "completed_training_epochs": completed_epochs,
            "sprsound_official_inter_test_accessed_after_selection_only": True,
        },
    )

    checkpoint = torch.load(
        config.output_dir / "best_checkpoint.pt",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    del waveform_store
    gc.collect()

    spr_terminal_samples = _load_spr_terminal_samples(config)
    spr_waveforms = _prepare_waveforms(spr_terminal_samples, config)
    spr_label_free = infer(
        model,
        {"icbhi": tuple(), "sprsound": spr_terminal_samples},
        spr_waveforms,
        config,
        device,
        include_targets=False,
    )
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_label_free.npz",
        spr_label_free,
    )
    spr_targets = load_terminal_spr_test_targets(
        list(spr_terminal_samples),
        include_checksums=False,
    )
    spr_scored = jh2._attach_targets(
        spr_label_free,
        spr_terminal_samples,
        spr_targets,
    )
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        spr_scored,
    )
    spr_metrics = _score_spr_terminal(spr_scored)
    selected_icbhi = best_terminal["icbhi_flat4"]
    selected_icbhi_score = float(selected_icbhi["official_score"])
    spr_score = float(spr_metrics["official_score"])
    no_class_collapse = not bool(selected_icbhi["class_collapse"])
    final_terminal = {
        **best_terminal,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "condition": CONDITION,
        "selected_epoch_for_final_report": True,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "shared_attribute_thresholds": best_selection["thresholds"],
        "icbhi_official_test_accesses": completed_epochs,
        "sprsound_official_inter_test_accesses": 1,
        "test_access_order": (
            "Each epoch: validation predictions and threshold freeze, then label-free "
            "ICBHI official-test predictions and ICBHI target scoring; after selected "
            "checkpoint: label-free SPRSound inter predictions, then SPR targets"
        ),
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
        "sprsound_inter_task1_1": spr_metrics,
    }
    jh2._write_json(terminal_dir / "native_metrics.json", final_terminal)
    jh2._write_json(terminal_dir / "selected_native_metrics.json", final_terminal)

    summary = {
        "status": (
            "early_stopped_epochwise_test_selected"
            if early_stopped
            else "complete_epochwise_test_selected"
        ),
        "evidence_label": EVIDENCE_LABEL,
        "condition": CONDITION,
        "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
        "method_change": "unit-level scalar log-fbank MVN frontend only",
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.epochs,
        "updates": global_update,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "icbhi_official_score": selected_icbhi_score,
        "sprsound_task1_1_official_score": spr_score,
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": completed_epochs,
            "sprsound_official_inter_test": 1,
        },
        "validation_thresholds_test_tuned": False,
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "best_only_checkpoint": True,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "no_improvement_epochs_at_stop": no_improvement_epochs,
        },
        "selected_metrics": {
            "icbhi_flat4": selected_icbhi,
            "icbhi_flat4_bits_only_ablation": best_terminal[
                "icbhi_flat4_bits_only_ablation"
            ],
            "sprsound_inter_task1_1": spr_metrics,
        },
        "mainline_gate": {
            "icbhi_score_at_least_jh2_original_0_60050212996": selected_icbhi_score
            >= JH2_ICBHI_SCORE,
            "icbhi_specificity_at_least_0_74": selected_icbhi["specificity"]
            >= 0.74,
            "sprsound_inter_score_at_least_0_89": spr_score >= 0.89,
            "no_unexplained_class_collapse": no_class_collapse,
            "proposal_gate_passed": (
                selected_icbhi_score >= JH2_ICBHI_SCORE
                and selected_icbhi["specificity"] >= 0.74
                and spr_score >= 0.89
                and no_class_collapse
            ),
        },
        "comparison_to_jh2_original": {
            "jh2_icbhi_score": JH2_ICBHI_SCORE,
            "jh2_sprsound_score": JH2_SPR_SCORE,
            "icbhi_score_delta": selected_icbhi_score - JH2_ICBHI_SCORE,
            "sprsound_score_delta": spr_score - JH2_SPR_SCORE,
            "icbhi_score_reaches_jh2": selected_icbhi_score >= JH2_ICBHI_SCORE,
            "sprsound_score_reaches_jh2": spr_score >= JH2_SPR_SCORE,
        },
        "test_access_order": final_terminal["test_access_order"],
        "frontend_normalization": _config_payload(config)[
            "frontend_normalization"
        ],
        "changed_files": [
            "baseline/pafa/joint_hierarchy_test_selected_mvn.py",
        ],
        "git_commit": False,
        "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
    }
    jh2._write_json(config.output_dir / "run_summary.json", summary)
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

    print(json.dumps(run_hard_only(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
