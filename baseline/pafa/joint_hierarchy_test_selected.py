"""PAFA-JH2 epochwise official-test-selected diagnostic runner.

This runner reuses the PAFA-JH1 model and training recipe.  Its only scientific
change is that each epoch freezes validation-only thresholds, evaluates the
official ICBHI and SPRSound test sets, and keeps the checkpoint with the
largest ICBHI official Score.  The result is permanently test-exposed and is
not a clean estimate or a PAFA reproduction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import load_terminal_spr_test_targets
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets, _score
from baseline.multidataset_pipeline.beats_nal_protocol import (
    HierarchicalLossConfig,
    hierarchical_loss,
)
from baseline.multidataset_pipeline.core2_hf_positive_kauh_external import (
    CORE_DATASETS,
    CORE_NODES,
    core_selection_losses,
    select_core_shared_thresholds,
)
from baseline.pafa.joint_hierarchy import (
    CONDITION as JH1_CONDITION,
    PAFAJointHierarchyConfig,
    _append_jsonl,
    _apply_author_ema,
    _balanced_epoch_batches,
    _batch,
    _build_components,
    _dataset_partitions,
    _prepare_waveforms,
    _save_predictions,
    _seed_everything,
    _write_json,
    infer,
    load_selection_samples,
    load_terminal_samples,
    _patient_indices,
)


CONDITION = "PAFA_JH2_test_selected_seed42"
EVIDENCE_LABEL = "dual_official_test_exposed_epochwise_diagnostic_icbhi_selected"
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "icbhi_official_test_score"


def _config_payload(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    payload = config.to_dict()
    payload.update(
        {
            "condition": CONDITION,
            "evidence_label": EVIDENCE_LABEL,
            "terminal_policy": (
                "after every epoch validation threshold freeze; official ICBHI and "
                "SPRSound test readout"
            ),
            "checkpoint_selection": (
                "maximum ICBHI official test Score; exact ties retain the earlier epoch"
            ),
            "checkpoint_policy": "best checkpoint only; per-epoch predictions and metrics retained",
            "test_threshold_tuning": False,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_policy": (
                "epoch boundary; strict improvement only; exact ties retain the earlier epoch; "
                "max epochs remains the upper bound"
            ),
        }
    )
    return payload


def _epoch_terminal_payload(
    *,
    epoch: int,
    update: int,
    validation_selection: dict[str, object],
    thresholds: dict[str, float],
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "epoch": epoch,
        "update": update,
        "selected_epoch_for_final_report": False,
        "validation_selection_loss": float(
            validation_selection["selection_loss"]
        ),
        "shared_attribute_thresholds": thresholds,
        "threshold_source": "same-epoch validation predictions only; frozen before test access",
        "outer_test_accessed": True,
        "terminal_targets_loaded_after_label_free_prediction_write": True,
        "prediction_support": {"icbhi": 2756, "sprsound": 1429},
        "test_selection_metric": "icbhi_flat4.official_score",
        **metrics,
    }


def run(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    config.validate()
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
            "condition": CONDITION,
            "evidence_label": EVIDENCE_LABEL,
            "datasets": list(CORE_DATASETS),
            "nodes": list(CORE_NODES),
            "training_only": True,
            "groups": {
                dataset: {
                    "subtrain": len({row.group_id for row in rows}),
                    "validation": len(
                        {
                            row.group_id
                            for row in validation[dataset]
                        }
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
    terminal_samples = None
    terminal_by_dataset = None
    terminal_waveforms = None
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
        }

        if terminal_samples is None:
            terminal_samples = load_terminal_samples(config)
            terminal_by_dataset = {
                dataset: tuple(
                    sample
                    for sample in terminal_samples
                    if sample.dataset == dataset
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
        epoch_terminal_dir = terminal_dir / f"epoch_{epoch:03d}"
        _save_predictions(
            epoch_terminal_dir / "selected_test_predictions_label_free.npz",
            label_free,
        )
        spr_targets = load_terminal_spr_test_targets(
            terminal_samples,
            include_checksums=False,
        )
        scored = _attach_targets(label_free, terminal_samples, spr_targets)
        _save_predictions(
            epoch_terminal_dir / "selected_test_predictions_scored.npz",
            scored,
        )
        metrics = _score(scored, thresholds)
        terminal_payload = _epoch_terminal_payload(
            epoch=epoch,
            update=global_update,
            validation_selection=validation_selection,
            thresholds=thresholds,
            metrics=metrics,
        )
        _write_json(epoch_terminal_dir / "native_metrics.json", terminal_payload)
        icbhi_score = float(metrics["icbhi_flat4"]["official_score"])
        improved = icbhi_score > best_score
        next_no_improvement_epochs = 0 if improved else no_improvement_epochs + 1
        _append_jsonl(
            test_log_path,
            {
                "epoch": epoch,
                "update": global_update,
                "validation_selection_loss": validation_selection[
                    "selection_loss"
                ],
                "thresholds": thresholds,
                "icbhi_official_score": metrics["icbhi_flat4"][
                    "official_score"
                ],
                "sprsound_task1_1_official_score": metrics[
                    "sprsound_inter_task1_1"
                ]["official_score"],
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
                "icbhi_official_score": metrics["icbhi_flat4"][
                    "official_score"
                ],
                "sprsound_task1_1_official_score": metrics[
                    "sprsound_inter_task1_1"
                ]["official_score"],
                "evidence_label": EVIDENCE_LABEL,
            },
            "elapsed_minutes": (
                time.perf_counter() - started_seconds
            )
            / 60.0,
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
                    "model": copy.deepcopy(model.state_dict()),
                    "icbhi_official_score": best_score,
                    "selection_loss": float(
                        validation_selection["selection_loss"]
                    ),
                    "config": _config_payload(config),
                },
                config.output_dir / "best_checkpoint.pt",
            )
            shutil.copyfile(
                epoch_terminal_dir / "selected_test_predictions_label_free.npz",
                terminal_dir / "selected_test_predictions_label_free.npz",
            )
            shutil.copyfile(
                epoch_terminal_dir / "selected_test_predictions_scored.npz",
                terminal_dir / "selected_test_predictions_scored.npz",
            )
            _write_json(
                terminal_dir / "selected_native_metrics.json",
                {
                    **best_terminal,
                    "selected_epoch_for_final_report": True,
                },
            )
        else:
            no_improvement_epochs = next_no_improvement_epochs
        completed_epochs = epoch

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
                    "sprsound_task1_1_official_score": metrics[
                        "sprsound_inter_task1_1"
                    ]["official_score"],
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

    assert best_selection is not None
    assert best_terminal is not None
    final_selection = {
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "shared_attribute_thresholds": best_selection["thresholds"],
        "threshold_details": best_selection["threshold_details"],
        "threshold_source": "selected epoch validation predictions; no test tuning",
        "outer_test_accessed": True,
        "validation_selection_test_accessed": False,
        "epochwise_terminal_test_accessed": True,
        "checkpoint_selection": (
            "maximum ICBHI official test Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
    }
    _write_json(config.output_dir / "validation_selection.json", final_selection)

    final_terminal = {
        **best_terminal,
        "status": EVIDENCE_LABEL,
        "evidence_label": EVIDENCE_LABEL,
        "selected_epoch_for_final_report": True,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "shared_attribute_thresholds": best_selection["thresholds"],
        "terminal_test_accesses": completed_epochs,
        "checkpoint_selection": (
            "maximum ICBHI official test Score; exact ties retain the earlier epoch"
        ),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopped": early_stopped,
        "completed_training_epochs": completed_epochs,
    }
    _write_json(config.output_dir / "terminal" / "native_metrics.json", final_terminal)
    _write_json(
        config.output_dir / "run_summary.json",
        {
            "status": (
                "early_stopped_epochwise_test_selected"
                if early_stopped
                else "complete_epochwise_test_selected"
            ),
            "evidence_label": EVIDENCE_LABEL,
            "condition": CONDITION,
            "training_config_reused_from": JH1_CONDITION,
            "completed_training_epochs": completed_epochs,
            "max_epochs": config.epochs,
            "updates": global_update,
            "selected_epoch": best_epoch,
            "selection_loss": float(best_selection["selection_loss"]),
            "icbhi_official_score": float(
                final_terminal["icbhi_flat4"]["official_score"]
            ),
            "sprsound_task1_1_official_score": float(
                final_terminal["sprsound_inter_task1_1"]["official_score"]
            ),
            "outer_test_accessed": True,
            "terminal_test_accesses": completed_epochs,
            "validation_thresholds_test_tuned": False,
            "checkpoint_selection": (
                "maximum ICBHI official test Score; exact ties retain the earlier epoch"
            ),
            "best_only_checkpoint": True,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopped": early_stopped,
            "no_improvement_epochs_at_stop": no_improvement_epochs,
            "interrupted_prior_run": {
                "condition": JH1_CONDITION,
                "completed_training_epochs": 25,
                "interrupted_epoch": 26,
                "interrupted_batch": "250/326",
                "interrupted_global_update": 8400,
            },
        },
    )
    return {
        "status": (
            "early_stopped_epochwise_test_selected"
            if early_stopped
            else "complete_epochwise_test_selected"
        ),
        "evidence_label": EVIDENCE_LABEL,
        "selected_epoch": best_epoch,
        "selection_loss": float(best_selection["selection_loss"]),
        "icbhi_official_score": float(
            final_terminal["icbhi_flat4"]["official_score"]
        ),
        "sprsound_task1_1_official_score": float(
            final_terminal["sprsound_inter_task1_1"]["official_score"]
        ),
        "updates": global_update,
    }


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
