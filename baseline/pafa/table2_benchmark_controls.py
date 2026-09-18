"""PAFA/JH2 benchmark controls paired with the original Full model seeds.

These controls retain the historical JH2 benchmark split and test-selected
evidence boundary.  The module does nothing unless ``--run`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    _load_spr,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import (
    _apply_author_ema,
    _icbhi_sample,
    _patient_indices,
    _prepare_waveforms,
    load_selection_samples,
)
from baseline.pafa.table2_clean_controls import (
    CORE_DATASETS,
    CORE_UPDATES_PER_POINT,
    COSINE_ETA_MIN_RATIO,
    EARLY_STOPPING_PATIENCE,
    EMA_BETA,
    LEARNING_RATE,
    MAX_CORE_UPDATES,
    MAX_VALIDATION_POINTS,
    MODEL_SEEDS,
    WEIGHT_DECAY,
    Table2Config,
    _infer,
    _save_predictions,
    _seed_everything,
    _training_batch,
    active_attribute_datasets,
    build_components,
    epoch_batches,
    fit_attribute_thresholds,
    fixed_hierarchy_loss,
    native_attributes_loss,
    native_only_loss,
    native_selection_scores,
    spr_attribute_auroc,
    write_json,
)


BENCHMARK_VARIANTS = (
    "icbhi_only",
    "sprsound_only",
    "coarse_spr",
    "native_attributes",
    "native_only",
)
FULL_REFERENCE_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed"
)
OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42"
)
MULTISEED_OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed"
)
ATTRIBUTION_OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918"
)
SEED = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    variant: str
    output_dir: Path
    device: str = "mps"
    cpu_threads: int = 4
    seed: int = SEED

    @property
    def core(self) -> Table2Config:
        return Table2Config(
            repo_root=self.repo_root,
            variant=("native_attributes" if self.variant == "native_only" else self.variant),
            model_seed=self.seed,
            output_dir=self.output_dir,
            split_manifest=self.repo_root / "baseline/pafa/table2_clean_split_manifest.json",
            device=self.device,
            cpu_threads=self.cpu_threads,
        )

    @property
    def active_datasets(self) -> tuple[str, ...]:
        return self.core.active_datasets

    @property
    def selection_dataset(self) -> str:
        return "sprsound" if self.variant == "sprsound_only" else "icbhi"

    @property
    def full_reference(self) -> Path:
        return self.repo_root / FULL_REFERENCE_ROOT_RELATIVE / f"seed_{self.seed}"

    def validate(self) -> None:
        if self.variant not in BENCHMARK_VARIANTS:
            raise ValueError(f"unsupported benchmark variant: {self.variant}")
        if self.seed not in MODEL_SEEDS:
            raise ValueError(f"benchmark seed must be one of {MODEL_SEEDS}")
        self.core.validate()

    def to_dict(self) -> dict[str, object]:
        payload = {
            **self.core.to_dict(),
            "variant": self.variant,
            "protocol": f"JH2_seed{self.seed}_test_selected_benchmark_control",
            "full_reference": str(self.full_reference),
            "checkpoint": str(self.core.checkpoint),
            "split_manifest": None,
            "split": (
                f"original JH2 seed{self.seed} official-train subtrain/validation split; "
                "sample IDs and support are recorded in split_reference.json"
            ),
            "selection_dataset": self.selection_dataset,
            "selection": (
                "maximize ICBHI official-test native Score"
                if self.selection_dataset == "icbhi"
                else "maximize SPRSound official-inter native Score"
            ),
            "checkpoint_selection": (
                "maximum ICBHI official-test native Score"
                if self.selection_dataset == "icbhi"
                else "maximum SPRSound official-inter native Score"
            ),
            "threshold_source": "original JH2-style validation split only",
            "test_policy": (
                "selection dataset official test is read each epoch; the other target "
                "is first read after source-native checkpoint and thresholds freeze"
            ),
            "selection_evidence": "test-selected benchmark control; not clean evidence",
            "max_epochs": MAX_VALIDATION_POINTS,
            "updates_per_epoch": CORE_UPDATES_PER_POINT,
            "max_updates": MAX_CORE_UPDATES,
            "learning_rate_schedule": "JH2 epoch cosine; constant within each epoch",
            "cosine_eta_min_ratio": COSINE_ETA_MIN_RATIO,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "strict_improvement": True,
            "tie_break": "earlier_epoch",
            "mvn": False,
            "precision": "FP32",
            "physical_batch_size": 32,
        }
        if self.variant == "native_attributes":
            payload["node_weights"] = {
                "native": 1.0 / 3.0,
                "c": 1.0 / 3.0,
                "w": 1.0 / 3.0,
            }
        elif self.variant == "native_only":
            base = self.core.base_config()
            payload["node_weights"] = {
                "native": 1.0 / 3.0,
                "c": 0.0,
                "w": 0.0,
            }
            payload["attribute_supervision"] = False
            payload["attribute_heads"] = "retained for matched initialization; not trained or reported"
            payload["threshold_source"] = "not applicable; no supervised C/W readout"
            payload["pafa_enabled"] = True
            payload["lambda_pcsl"] = base.lambda_pcsl
            payload["lambda_gpal"] = base.lambda_gpal
        return payload


def load_benchmark_training_samples(config: BenchmarkConfig) -> list[Sample]:
    samples = load_selection_samples(config.core.base_config())
    return sorted(
        [row for row in samples if row.dataset in config.active_datasets],
        key=lambda row: row.sample_id,
    )


def benchmark_split_reference(
    config: BenchmarkConfig, samples: Sequence[Sample]
) -> dict[str, object]:
    return {
        "protocol": f"original JH2 seed{config.seed} official-train split",
        "active_datasets": list(config.active_datasets),
        "datasets": {
            dataset: {
                "support": {
                    partition: {
                        "units": sum(
                            row.dataset == dataset and row.partition == partition
                            for row in samples
                        ),
                        "groups": len(
                            {
                                row.group_id
                                for row in samples
                                if row.dataset == dataset and row.partition == partition
                            }
                        ),
                    }
                    for partition in ("subtrain", "validation")
                },
                "groups": {
                    partition: sorted(
                        {
                            row.group_id
                            for row in samples
                            if row.dataset == dataset and row.partition == partition
                        }
                    )
                    for partition in ("subtrain", "validation")
                },
                "records": [
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "partition": row.partition,
                    }
                    for row in samples
                    if row.dataset == dataset
                ],
            }
            for dataset in config.active_datasets
        },
    }


def _partitioned(
    samples: Sequence[Sample], partition: str, datasets: Sequence[str]
) -> dict[str, tuple[Sample, ...]]:
    return {
        dataset: tuple(
            row for row in samples if row.dataset == dataset and row.partition == partition
        )
        for dataset in datasets
    }


def _icbhi_test_samples(config: BenchmarkConfig) -> tuple[Sample, ...]:
    base = config.core.base_config()
    return tuple(
        _icbhi_sample(row, base.icbhi_audio_dir, "test")
        for row in read_official_cycles(base.icbhi_audio_dir, base.author_repo, ("test",))
    )


def _spr_test_samples(config: BenchmarkConfig) -> tuple[Sample, ...]:
    samples, _ = _load_spr(config.repo_root / "dataset/raw", include_checksums=False)
    return tuple(row for row in samples if row.dataset == "sprsound" and row.partition == "test")


def _concat_predictions(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {key: np.concatenate((first[key], second[key]), axis=0) for key in first}


def _selection_test_samples(config: BenchmarkConfig) -> tuple[Sample, ...]:
    return (
        _spr_test_samples(config)
        if config.selection_dataset == "sprsound"
        else _icbhi_test_samples(config)
    )


def _score_selection_test(
    model: torch.nn.Module,
    samples: Sequence[Sample],
    waveform_store: Mapping[str, torch.Tensor],
    config: BenchmarkConfig,
    thresholds: Mapping[str, float],
    device: torch.device,
    spr_targets: Mapping[str, Mapping[str, int]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    by_dataset = {config.selection_dataset: tuple(samples)}
    label_free = _infer(
        model,
        by_dataset,
        waveform_store,
        config.core,
        device,
        include_targets=config.selection_dataset == "icbhi",
        variant_override=config.variant,
    )
    predictions = (
        _attach_targets(label_free, samples, spr_targets or {})
        if config.selection_dataset == "sprsound"
        else label_free
    )
    report = native_selection_scores(
        predictions,
        variant=config.variant,
        active_datasets=(config.selection_dataset,),
        thresholds=thresholds,
    )
    return predictions, report


def _learning_rate(epoch: int) -> float:
    return float(
        LEARNING_RATE
        * (
            COSINE_ETA_MIN_RATIO
            + (1.0 - COSINE_ETA_MIN_RATIO)
            * (1.0 + np.cos(np.pi * epoch / MAX_VALIDATION_POINTS))
            / 2.0
        )
    )


def run_benchmark_control(config: BenchmarkConfig) -> dict[str, object]:
    """Run one explicitly approved benchmark control."""

    config.validate()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    _seed_everything(config.seed)

    samples = load_benchmark_training_samples(config)
    subtrain = _partitioned(samples, "subtrain", config.active_datasets)
    validation = _partitioned(samples, "validation", config.active_datasets)
    write_json(config.output_dir / "config.json", config.to_dict())
    write_json(
        config.output_dir / "split_reference.json",
        benchmark_split_reference(config, samples),
    )

    base = config.core.base_config()
    waveform_store = _prepare_waveforms(samples, base)
    device = torch.device(config.device)
    model, pafa_criterion = build_components(config.core, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    patient_index = _patient_indices(
        [row for rows in subtrain.values() for row in rows]
    )

    selection_test_samples = _selection_test_samples(config)
    selection_test_waveforms = _prepare_waveforms(selection_test_samples, base)
    selection_spr_targets = (
        load_terminal_spr_test_targets(
            list(selection_test_samples), include_checksums=False
        )
        if config.selection_dataset == "sprsound"
        else None
    )

    best_score = -math.inf
    best_epoch = 0
    best_thresholds: dict[str, float] = {}
    best_threshold_details: dict[str, object] = {}
    no_improvement = 0
    global_update = 0
    started = time.perf_counter()
    train_log = config.output_dir / "train_log.jsonl"

    for epoch in range(1, MAX_VALIDATION_POINTS + 1):
        print(
            json.dumps(
                {
                    "event": "epoch_start",
                    "variant": config.variant,
                    "seed": config.seed,
                    "epoch": epoch,
                    "next_update": global_update + 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        lr = _learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        schedule = epoch_batches(
            {dataset: len(rows) for dataset, rows in subtrain.items()},
            config.active_datasets,
            batch_size=32,
            model_seed=config.seed,
            validation_point=epoch,
        )
        losses: dict[str, list[float]] = {
            dataset: [] for dataset in config.active_datasets
        }
        for dataset, indices in schedule:
            rows = [subtrain[dataset][int(index)] for index in indices]
            waveform, targets, eligible, native_target, patients = _training_batch(
                rows,
                dataset,
                waveform_store,
                patient_index,
                device,
                config.variant,
            )
            before = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits, projected = model(waveform, training=True)
            classification = (
                native_only_loss(logits, dataset, native_target)
                if config.variant == "native_only"
                else native_attributes_loss(
                    logits, dataset, native_target, targets, eligible
                )
                if config.variant == "native_attributes"
                else fixed_hierarchy_loss(logits, targets, eligible)
            )
            pafa = pafa_criterion(
                projected,
                patients,
                lambda_pcsl=base.lambda_pcsl,
                lambda_gpal=base.lambda_gpal,
            )
            total = classification + pafa
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(
                    f"non-finite loss at update {global_update + 1} dataset={dataset}"
                )
            total.backward()
            optimizer.step()
            _apply_author_ema(model, before, EMA_BETA)
            global_update += 1
            losses[dataset].append(float(total.detach().cpu()))
            if global_update == 1 or global_update % 32 == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "variant": config.variant,
                            "seed": config.seed,
                            "epoch": epoch,
                            "update": global_update,
                            "dataset": dataset,
                            "loss": float(total.detach().cpu()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        validation_predictions = _infer(
            model,
            validation,
            waveform_store,
            config.core,
            device,
            include_targets=True,
            variant_override=config.variant,
        )
        _save_predictions(
            config.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            validation_predictions,
        )
        if config.variant == "native_only":
            thresholds, threshold_details = {}, {
                "status": "not_applicable",
                "reason": "explicit C/W supervision is disabled",
            }
        else:
            thresholds, threshold_details = fit_attribute_thresholds(
                validation_predictions, active_attribute_datasets(config.core)
            )
        selected_predictions, selection = _score_selection_test(
            model,
            selection_test_samples,
            selection_test_waveforms,
            config,
            thresholds,
            device,
            selection_spr_targets,
        )
        _save_predictions(
            config.output_dir / "selection_test" / f"epoch_{epoch:03d}.npz",
            selected_predictions,
        )
        score = float(selection["selection_utility"])
        improved = score > best_score
        no_improvement = 0 if improved else no_improvement + 1
        record = {
            "epoch": epoch,
            "update": global_update,
            "learning_rate": lr,
            "train_loss": {
                dataset: float(np.mean(values)) for dataset, values in losses.items()
            },
            "validation_thresholds": thresholds,
            "threshold_details": threshold_details,
            "selection_dataset": config.selection_dataset,
            "selection_score": score,
            "selection_metrics": selection["metrics"],
            "strict_improvement": improved,
            "no_improvement_epochs": no_improvement,
            "evidence_boundary": "official-test-selected benchmark control",
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if improved:
            best_score = score
            best_epoch = epoch
            best_thresholds = thresholds
            best_threshold_details = threshold_details
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_score": score,
                    "thresholds": thresholds,
                    "config": config.to_dict(),
                },
                config.output_dir / "best_checkpoint.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "update": global_update,
                "model": model.state_dict(),
                "config": config.to_dict(),
            },
            config.output_dir / "last_checkpoint.pt",
        )
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            break

    write_json(
        config.output_dir / "selection.json",
        {
            "status": "test_selected_benchmark_control",
            "variant": config.variant,
            "seed": config.seed,
            "selected_epoch": best_epoch,
            "selected_update": best_epoch * CORE_UPDATES_PER_POINT,
            "selection_dataset": config.selection_dataset,
            "selection_score": best_score,
            "thresholds": best_thresholds,
            "threshold_details": best_threshold_details,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
    )

    checkpoint = torch.load(config.output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    with np.load(
        config.output_dir / "selection_test" / f"epoch_{best_epoch:03d}.npz",
        allow_pickle=False,
    ) as archive:
        selected_source = {key: archive[key] for key in archive.files}

    other_dataset = "sprsound" if config.selection_dataset == "icbhi" else "icbhi"
    other_samples = (
        _spr_test_samples(config) if other_dataset == "sprsound" else _icbhi_test_samples(config)
    )
    other_waveforms = _prepare_waveforms(other_samples, base)
    other_label_free = _infer(
        model,
        {other_dataset: other_samples},
        other_waveforms,
        config.core,
        device,
        include_targets=other_dataset == "icbhi",
        variant_override=config.variant,
    )
    if other_dataset == "sprsound":
        _save_predictions(
            config.output_dir / "terminal/other_target_predictions_label_free.npz",
            other_label_free,
        )
        targets = load_terminal_spr_test_targets(
            list(other_samples), include_checksums=False
        )
        other_scored = _attach_targets(other_label_free, other_samples, targets)
    else:
        other_scored = other_label_free
    _save_predictions(
        config.output_dir / "terminal/other_target_predictions_scored.npz",
        other_scored,
    )

    combined = (
        _concat_predictions(selected_source, other_scored)
        if config.selection_dataset == "icbhi"
        else _concat_predictions(other_scored, selected_source)
    )
    _save_predictions(
        config.output_dir / "terminal/selected_predictions_scored.npz", combined
    )
    final_metrics = native_selection_scores(
        combined,
        variant=config.variant,
        active_datasets=CORE_DATASETS,
        thresholds=best_thresholds,
    )
    terminal = {
        "status": "benchmark_control_terminal_complete",
        "variant": config.variant,
        "seed": config.seed,
        "selected_epoch": best_epoch,
        "selection_dataset": config.selection_dataset,
        "selection_score": best_score,
        "thresholds": best_thresholds,
        "per_native_task": final_metrics["metrics"],
        "spr_cw": (
            {
                "status": "not_applicable",
                "reason": "C/W heads were retained for initialization parity but received no supervision",
            }
            if config.variant == "native_only"
            else spr_attribute_auroc(combined)
        ),
        **(
            {
                "attribute_metrics": {
                    "status": "not_applicable",
                    "reason": "explicit C/W supervision is disabled",
                },
                "hf": {
                    "status": "not_applicable",
                    "reason": "no supervised cross-dataset attribute readout exists",
                },
            }
            if config.variant == "native_only"
            else {}
        ),
        "evidence_boundary": "test-selected benchmark control; not clean causal evidence",
        "other_target_access": "after source-native checkpoint and thresholds were frozen",
    }
    write_json(config.output_dir / "terminal/native_metrics.json", terminal)
    summary = {
        "status": "complete_test_selected_benchmark_control",
        "variant": config.variant,
        "seed": config.seed,
        "completed_epochs": epoch,
        "updates": global_update,
        "selected_epoch": best_epoch,
        "selection_dataset": config.selection_dataset,
        "terminal": terminal,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    write_json(config.output_dir / "run_summary.json", summary)
    return summary


def default_config(
    repo_root: Path,
    variant: str,
    seed: int = SEED,
    *,
    device: str = "mps",
    cpu_threads: int = 4,
    output_root: Path | None = None,
) -> BenchmarkConfig:
    if output_root is None:
        output_root = (
            ATTRIBUTION_OUTPUT_ROOT_RELATIVE
            if variant == "native_only"
            else (OUTPUT_ROOT_RELATIVE if seed == SEED else MULTISEED_OUTPUT_ROOT_RELATIVE)
        )
    return BenchmarkConfig(
        repo_root=repo_root,
        variant=variant,
        output_dir=repo_root / output_root / variant / f"seed_{seed}",
        device=device,
        cpu_threads=cpu_threads,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--variant", choices=BENCHMARK_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS, default=SEED)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = default_config(
        args.repo_root,
        args.variant,
        args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        output_root=args.output_root,
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
            )
        )
        return
    print(json.dumps(run_benchmark_control(config), indent=2), flush=True)


if __name__ == "__main__":
    main()
