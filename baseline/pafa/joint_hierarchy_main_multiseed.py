"""JH2 Hard Hierarchy main-method three-seed confirmation.

This runner keeps the JH2 model, loss, input, batching, optimizer, EMA, and
core data contract.  Each seed uses ICBHI official Hard Hierarchy Score as the
only checkpoint-selection and early-stopping monitor.  SPRSound official
inter evaluation is deferred until after that seed's selected checkpoint.

The outputs are test-selected diagnostics and are not clean estimates,
PAFA-paper reproductions, or paper results.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

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
from baseline.pafa.beats_ce_reproduction import _apply_author_ema, read_official_cycles
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _append_jsonl,
    _balanced_epoch_batches,
    _batch,
    _build_components,
    _dataset_partitions,
    _icbhi_sample,
    _patient_indices,
    _prepare_waveforms,
    _save_predictions,
    _seed_everything,
    _write_json,
    infer,
    load_selection_samples,
)


SEEDS = (0, 1, 42)
CONDITION_PREFIX = "PAFA_JH2_main_multiseed"
EVIDENCE_LABEL = "JH2_main_icbhi-test-selected_multiseed_diagnostic"
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "icbhi_official_test_score"
CORE_UPDATES_PER_EPOCH = 326
JH2_REFERENCE_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2"
)
MULTISEED_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed"
)


def _condition(seed: int) -> str:
    return f"{CONDITION_PREFIX}_seed{seed}"


def _validate_config(config: PAFAJointHierarchyConfig) -> None:
    if config.seed not in SEEDS:
        raise ValueError(f"main multiseed runner accepts only seeds {SEEDS}")
    expected = {
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
        if getattr(config, field) != value:
            raise ValueError(f"frozen JH2 field changed: {field}")
    if config.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")


def _config_payload(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    payload = config.to_dict()
    payload.update(
        {
            "condition": _condition(config.seed),
            "evidence_label": EVIDENCE_LABEL,
            "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
            "main_method": "JH2 Hard Hierarchy",
            "method_change": "none; independent confirmation seed",
            "seed_role": f"formal main-method confirmation seed {config.seed}",
            "selection": "ICBHI official-test Hard Hierarchy Score only",
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain "
                "the earlier epoch"
            ),
            "terminal_policy": (
                "after every epoch core validation threshold freeze and label-free "
                "ICBHI official-test readout; after selected checkpoint label-free "
                "SPRSound inter readout"
            ),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_policy": (
                "epoch boundary; strict ICBHI Score improvement only; exact ties "
                "retain the earlier epoch; max epochs remains the upper bound"
            ),
            "hf_auxiliary": False,
            "kauh_evaluation": False,
            "mvn": False,
            "soft_bridge": False,
            "sprsound_official_per_epoch": False,
            "historical_seed42_included_in_multiseed_statistics": False,
        }
    )
    return payload


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
            (_icbhi_sample(row, config.icbhi_audio_dir, "test") for row in rows),
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
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"]],
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


def _score_spr_terminal(predictions: Mapping[str, np.ndarray]) -> dict[str, object]:
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
        "threshold_source": "same-epoch core validation predictions only; frozen before ICBHI test access",
        "outer_test_accessed": True,
        "sprsound_official_inter_test_accessed": False,
        "prediction_support": {"icbhi": 2756},
        "test_selection_metric": "icbhi_flat4.official_score",
        **metrics,
    }


def run_seed(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    _validate_config(config)
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", _config_payload(config))

    selection_samples = load_selection_samples(config)
    subtrain = _dataset_partitions(selection_samples, "subtrain")
    validation = _dataset_partitions(selection_samples, "validation")
    _write_json(
        config.output_dir / "selection_split_summary.json",
        {
            "condition": _condition(config.seed),
            "evidence_label": EVIDENCE_LABEL,
            "seed": config.seed,
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
        if len(batches) != CORE_UPDATES_PER_EPOCH:
            raise RuntimeError(
                f"JH2 core update count changed: {len(batches)} != {CORE_UPDATES_PER_EPOCH}"
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
            epoch_losses[dataset]["pafa"].append(
                float(pafa_loss.detach().cpu())
            )
            epoch_losses[dataset]["total"].append(
                float(total_loss.detach().cpu())
            )
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
            "sprsound_official_inter_test_accessed": False,
        }
        _write_json(
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
        _write_json(epoch_terminal_dir / "native_metrics.json", terminal_payload)
        icbhi_score = float(icbhi_metrics["icbhi_flat4"]["official_score"])
        improved = icbhi_score > best_score
        next_no_improvement_epochs = 0 if improved else no_improvement_epochs + 1
        _append_jsonl(
            test_log_path,
            {
                "epoch": epoch,
                "update": global_update,
                "validation_selection_loss": validation_selection["selection_loss"],
                "sprsound_validation_loss": validation_selection["dataset_losses"]["sprsound"],
                "thresholds": thresholds,
                "icbhi_official_score": icbhi_score,
                "sprsound_official_inter_test_accessed": False,
                "terminal_metrics_path": str(epoch_terminal_dir / "native_metrics.json"),
            },
        )
        train_record = {
            "epoch": epoch,
            "update": global_update,
            "seed": config.seed,
            "learning_rate": learning_rate,
            "train_loss": {
                dataset: {
                    name: float(np.mean(values)) for name, values in losses.items()
                }
                for dataset, losses in epoch_losses.items()
            },
            "validation": validation_selection,
            "terminal_test": {
                "icbhi_official_score": icbhi_score,
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
        _append_jsonl(train_log_path, train_record)
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
                    "seed": config.seed,
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
            _write_json(
                terminal_dir / "selected_icbhi_native_metrics.json",
                {**terminal_payload, "selected_epoch_for_final_report": True},
            )
        else:
            no_improvement_epochs = next_no_improvement_epochs
        completed_epochs = epoch
        _append_jsonl(
            progress_path,
            {
                "epoch": epoch,
                "epoch_complete": True,
                "update": global_update,
                "seed": config.seed,
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
                    "seed": config.seed,
                    "epoch": epoch,
                    "update": global_update,
                    "elapsed_minutes": train_record["elapsed_minutes"],
                    "validation_selection_loss": validation_selection["selection_loss"],
                    "sprsound_validation_loss": validation_selection["dataset_losses"]["sprsound"],
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
        raise RuntimeError(f"{_condition(config.seed)} completed no selectable epoch")
    _write_json(
        config.output_dir / "validation_selection.json",
        {
            "status": EVIDENCE_LABEL,
            "evidence_label": EVIDENCE_LABEL,
            "condition": _condition(config.seed),
            "seed": config.seed,
            "selected_epoch": best_epoch,
            "selection_loss": float(best_selection["selection_loss"]),
            "shared_attribute_thresholds": best_selection["thresholds"],
            "threshold_details": best_selection["threshold_details"],
            "threshold_source": "selected epoch core validation predictions only; no test tuning",
            "outer_test_accessed": True,
            "validation_selection_test_accessed": False,
            "icbhi_official_test_accesses": completed_epochs,
            "sprsound_official_inter_test_accesses": 1,
            "checkpoint_selection": (
                "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
            ),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "completed_training_epochs": completed_epochs,
            "hf_auxiliary": False,
            "kauh_evaluation": False,
        },
    )

    checkpoint = torch.load(config.output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
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
    spr_scored = __import__(
        "baseline.multidataset_pipeline.beats_nal_terminal",
        fromlist=["_attach_targets"],
    )._attach_targets(spr_label_free, spr_terminal_samples, spr_targets)
    _save_predictions(
        terminal_dir / "selected_sprsound_predictions_scored.npz",
        spr_scored,
    )
    spr_metrics = _score_spr_terminal(spr_scored)
    selected_icbhi = best_terminal["icbhi_flat4"]
    final_terminal = {
        **best_terminal,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "condition": _condition(config.seed),
        "seed": config.seed,
        "selected_epoch_for_final_report": True,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "shared_attribute_thresholds": best_selection["thresholds"],
        "icbhi_official_test_accesses": completed_epochs,
        "sprsound_official_inter_test_accesses": 1,
        "test_access_order": (
            "Each epoch: core validation predictions and threshold freeze, then "
            "label-free/scored ICBHI official test; after selected checkpoint: "
            "label-free SPRSound inter predictions, then SPR targets"
        ),
        "checkpoint_selection": (
            "maximum ICBHI official test Hard Hierarchy Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
        "sprsound_inter_task1_1": spr_metrics,
        "hf_auxiliary": False,
        "kauh_evaluation": False,
    }
    _write_json(terminal_dir / "native_metrics.json", final_terminal)
    _write_json(terminal_dir / "selected_native_metrics.json", final_terminal)
    summary = {
        "status": (
            "early_stopped_epochwise_icbhi_test_selected"
            if early_stopped
            else "complete_epochwise_icbhi_test_selected"
        ),
        "evidence_label": EVIDENCE_LABEL,
        "condition": _condition(config.seed),
        "seed": config.seed,
        "training_config_reused_from": "PAFA_JH2_test_selected_seed42",
        "main_method": "JH2 Hard Hierarchy",
        "hf_auxiliary": False,
        "kauh_evaluation": False,
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.epochs,
        "updates": global_update,
        "core_updates_per_epoch": CORE_UPDATES_PER_EPOCH,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "icbhi_official_score": float(selected_icbhi["official_score"]),
        "sprsound_task1_1_official_score": float(spr_metrics["official_score"]),
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
        "test_access_order": final_terminal["test_access_order"],
        "changed_files": ["baseline/pafa/joint_hierarchy_main_multiseed.py"],
        "git_commit": False,
        "claim_boundary": (
            "ICBHI-test-selected/non-clean main-method confirmation; SPRSound is "
            "selected-only terminal evidence; not a paper result"
        ),
        "elapsed_minutes": (time.perf_counter() - started_seconds) / 60.0,
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def _stat(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "ddof": 1,
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def _metric_stats(
    rows: Sequence[Mapping[str, object]],
    metric_path: Sequence[str],
) -> dict[str, object]:
    values = []
    for row in rows:
        value: object = row
        for key in metric_path:
            value = value[key]
        values.append(float(value))
    return _stat(values)


def _per_class_stats(
    rows: Sequence[Mapping[str, object]],
    metric_path: Sequence[str],
) -> dict[str, object]:
    labels = sorted(
        {
            str(label)
            for row in rows
            for label in _nested(row, metric_path).get("per_class", {})
        }
    )
    return {
        label: _stat(
            [
                float(_nested(row, metric_path)["per_class"][label]["recall"])
                for row in rows
            ]
        )
        for label in labels
    }


def _nested(value: Mapping[str, object], path: Sequence[str]) -> Mapping[str, object]:
    current: object = value
    for key in path:
        current = current[key]
    return current


def aggregate_multiseed(repo_root: Path) -> dict[str, object]:
    root = repo_root / MULTISEED_RELATIVE
    run_summaries = []
    for seed in SEEDS:
        run_dir = root / f"seed_{seed}"
        run_summaries.append(json.loads((run_dir / "run_summary.json").read_text()))
    icbhi_rows = [row["selected_metrics"]["icbhi_flat4"] for row in run_summaries]
    spr_rows = [row["selected_metrics"]["sprsound_inter_task1_1"] for row in run_summaries]
    icbhi_metrics = {
        metric: _metric_stats(icbhi_rows, (metric,))
        for metric in ("specificity", "sensitivity", "official_score", "macro_f1", "uar")
    }
    icbhi_metrics["per_class_recall"] = _per_class_stats(icbhi_rows, ())
    spr_metrics = {
        metric: _metric_stats(spr_rows, (metric,))
        for metric in (
            "sensitivity",
            "specificity",
            "average_score",
            "harmonic_score",
            "official_score",
            "macro_f1",
            "uar",
        )
    }
    spr_metrics["per_class_recall"] = _per_class_stats(spr_rows, ())
    historical_dir = repo_root / JH2_REFERENCE_RELATIVE
    historical_terminal = json.loads(
        (historical_dir / "terminal/native_metrics.json").read_text()
    )
    summary = {
        "status": "complete_jh2_main_multiseed_confirmation",
        "evidence_label": EVIDENCE_LABEL,
        "condition": "PAFA_JH2_main_multiseed",
        "main_method": "JH2 Hard Hierarchy; no MVN, Soft Bridge, HF auxiliary, or KAUH",
        "seeds": list(SEEDS),
        "n": 3,
        "sample_std_ddof": 1,
        "excluded_runs": [
            {
                "seed": 2,
                "status": "STOPPED_BY_USER_EXCLUDED_FROM_FORMAL_3SEED",
                "source": str(root / "seed_2"),
                "included_in_mean_std": False,
            }
        ],
        "selected_epoch": _stat([float(row["selected_epoch"]) for row in run_summaries]),
        "completed_training_epochs": {
            f"seed_{seed}": int(row["completed_training_epochs"])
            for seed, row in zip(SEEDS, run_summaries)
        },
        "icbhi": icbhi_metrics,
        "sprsound_inter_task1_1": spr_metrics,
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "selected_epoch": int(row["selected_epoch"]),
                "completed_training_epochs": int(row["completed_training_epochs"]),
                "icbhi": row["selected_metrics"]["icbhi_flat4"],
                "sprsound_inter_task1_1": row["selected_metrics"]["sprsound_inter_task1_1"],
                "test_access_counts": row["test_access_counts"],
            }
            for row in run_summaries
        ],
        "historical_seed42_external_reference": {
            "included_in_multiseed_statistics": False,
            "source": str(historical_dir),
            "selected_epoch": 19,
            "icbhi": historical_terminal["icbhi_flat4"],
            "sprsound_inter_task1_1": historical_terminal["sprsound_inter_task1_1"],
            "evidence_label": "dual_official_test_exposed_epochwise_diagnostic_icbhi_selected",
        },
        "pafa_paper_rows": {
            "included_in_multiseed_statistics": False,
            "status": "not integrated into this artifact",
        },
        "test_access_boundary": {
            "per_seed_icbhi_official_test_accesses": "completed epochs",
            "per_seed_sprsound_official_inter_test_accesses": 1,
            "sprsound_per_epoch_access": False,
            "hf_access": False,
            "kauh_access": False,
        },
        "claim_boundary": (
            "three-seed JH2 ICBHI-test-selected/non-clean diagnostic; historical seed42 "
            "and paper rows are external reference rows and are not part of mean/std"
        ),
        "changed_files": ["baseline/pafa/joint_hierarchy_main_multiseed.py"],
        "git_commit": False,
    }
    _write_json(root / "multiseed_summary.json", summary)
    (root / "multiseed_summary.md").write_text(
        _multiseed_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _format_stat(value: Mapping[str, object]) -> str:
    return f"{float(value['mean']):.6f} ± {float(value['sample_std']):.6f}"


def _multiseed_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# JH2 main-method three-seed confirmation",
        "",
        f"Evidence: `{summary['evidence_label']}`.",
        "This is an ICBHI-test-selected/non-clean diagnostic. Historical seed42 and paper rows are external references, not members of the three-seed statistics.",
        "",
        "## Contract",
        "",
        "JH2 Hard Hierarchy; BEATs iter3+ AS2M full fine-tuning; ICBHI+SPRSound joint training; 5 s repeat-pad/front-truncate; homogeneous batch32; Adam 5e-5/wd1e-6; cosine; EMA0.5; PCSL50/GPAL0.0005; no MVN, Soft Bridge, HF auxiliary, or KAUH.",
        "Epoch selection and patience-10 early stopping use only ICBHI official Hard Hierarchy Score. SPRSound official inter is accessed once after each seed's selected checkpoint.",
        "",
        "## Three-seed mean ± sample standard deviation (n=3)",
        "",
        "| ICBHI metric | Mean ± sample std |\n|---|---:|",
    ]
    for metric, label in (
        ("specificity", "Sp"),
        ("sensitivity", "Se"),
        ("official_score", "Score"),
        ("macro_f1", "Macro-F1"),
        ("uar", "UAR"),
    ):
        lines.append(f"| {label} | {_format_stat(summary['icbhi'][metric])} |")
    lines.extend(["", "ICBHI per-class recall:"])
    for label, value in summary["icbhi"]["per_class_recall"].items():
        lines.append(f"- {label}: {_format_stat(value)}")
    lines.extend(["", "| SPRSound Task1-1 metric | Mean ± sample std |\n|---|---:|"])
    for metric, label in (
        ("sensitivity", "Se"),
        ("specificity", "Sp"),
        ("average_score", "AS"),
        ("harmonic_score", "HS"),
        ("official_score", "Score"),
        ("macro_f1", "Macro-F1"),
        ("uar", "UAR"),
    ):
        lines.append(f"| {label} | {_format_stat(summary['sprsound_inter_task1_1'][metric])} |")
    lines.extend(["", "SPRSound per-class recall:"])
    for label, value in summary["sprsound_inter_task1_1"]["per_class_recall"].items():
        lines.append(f"- {label}: {_format_stat(value)}")
    lines.extend(["", "## Per-seed selected results", "", "| Seed | Selected epoch | Completed epochs | ICBHI Score | SPRSound Score |", "|---:|---:|---:|---:|---:|"])
    for row in summary["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['selected_epoch']} | {row['completed_training_epochs']} | "
            f"{float(row['icbhi']['official_score']):.6f} | {float(row['sprsound_inter_task1_1']['official_score']):.6f} |"
        )
    lines.extend([
        "",
        "## External reference rows",
        "",
        f"Historical seed42 is kept separately at `{summary['historical_seed42_external_reference']['source']}` and is excluded from mean/std. PAFA paper rows were not integrated into this artifact.",
        "Seed2 partial artifacts are retained separately with status `STOPPED_BY_USER_EXCLUDED_FROM_FORMAL_3SEED` and are excluded from mean/std.",
        "",
        "## Boundary",
        "",
        "No HF/KAUH evaluation, server task, other baseline, Git commit/push, or Notion change was performed.",
    ])
    return "\n".join(lines) + "\n"


def _default_paths(repo_root: Path, seed: int) -> tuple[Path, Path, Path, Path]:
    author_repo = repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo"
    checkpoint = repo_root / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt"
    icbhi_audio = repo_root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"
    output_dir = repo_root / MULTISEED_RELATIVE / f"seed_{seed}"
    return author_repo, checkpoint, icbhi_audio, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate_multiseed(args.repo_root), indent=2, sort_keys=True), flush=True)
        return
    if args.seed is None:
        parser.error("--seed is required unless --aggregate is used")
    author_repo, checkpoint, icbhi_audio, output_dir = _default_paths(args.repo_root, args.seed)
    config = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio,
        output_dir=args.output_dir or output_dir,
        device=args.device,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
    )
    _validate_config(config)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "condition": _condition(args.seed),
                    "evidence_label": EVIDENCE_LABEL,
                    "config": _config_payload(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run_seed(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
