"""Local clean queue for four PAFA joint-hierarchy variants.

Training and validation are completed for all variants before any official test
data are loaded.  A fixed calibration/selection split is derived from the
existing JH1 epoch-6 validation artifact and is shared by every variant.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from acoustic.evaluation.sprsound_inter import resolve_biocas_root
from baseline.four_dataset_frozen_encoder.data import Sample, load_terminal_spr_test_targets
from baseline.multidataset_pipeline.beats_nal_protocol import (
    CORE_NODES,
    HierarchicalLossConfig,
    decode_icbhi_hierarchical_flat4,
    hierarchical_loss,
    mapped_targets,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.multidataset_pipeline.posthoc_native_readout import ICBHI_LABELS, native_metrics
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    PAFAJointHierarchyModel,
    _append_jsonl,
    _apply_author_ema,
    _batch,
    _build_components as _build_components_base,
    _icbhi_sample,
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


QUEUE_EVIDENCE_LABEL = "local_clean_queue_preregistered_single_seed"
REFERENCE_FILENAME = "reference_contract.json"
GROUP_LIST_FILENAME = "group_lists.json"
REFERENCE_VALIDATION_SOURCE = (
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH1_joint_hierarchy_seed42/"
    "validation/epoch_006.npz"
)
QUEUE_ROOT_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy")
REFERENCE_DIRNAME = "LocalCleanQueue_reference_seed42"
MAIN_SELECTION_FILENAME = "main_selection.json"
QUEUE_SUMMARY_FILENAME = "queue_summary.json"
EARLY_STOPPING_PATIENCE = 10
BETA_EFFECTIVE_NUMBER = 0.9999
MODES = ("J-Clean", "J-MVN", "J-SourceProp", "J-BalancedLoss")
INPUT_VARIANTS = {
    "J-Clean": "standard",
    "J-MVN": "unit_level_scalar_mvn",
    "J-SourceProp": "standard",
    "J-BalancedLoss": "standard",
}


def _queue_root(repo_root: Path) -> Path:
    return repo_root / QUEUE_ROOT_RELATIVE


def _reference_dir(repo_root: Path) -> Path:
    return _queue_root(repo_root) / REFERENCE_DIRNAME


def _run_dir(repo_root: Path, mode: str) -> Path:
    return _queue_root(repo_root) / f"LocalClean_{mode}_seed42"


def _main_selection_path(repo_root: Path) -> Path:
    return _queue_root(repo_root) / MAIN_SELECTION_FILENAME


def _queue_summary_path(repo_root: Path) -> Path:
    return _queue_root(repo_root) / QUEUE_SUMMARY_FILENAME


def _spr_root(repo_root: Path) -> Path:
    root = resolve_biocas_root(repo_root / "dataset/raw/sprsound")
    if SPRSOUND_COMMIT not in str(root):
        raise RuntimeError(f"SPRSound root is not pinned to {SPRSOUND_COMMIT}")
    return root


def _spr_sample(
    row: Mapping[str, object], partition: str, *, include_label: bool
) -> Sample:
    raw = str(row["raw_label"]) if include_label else None
    targets = (
        {
            "spr_binary": int(raw != "Normal"),
            "spr_seven": SPR_LABELS.index(raw),
        }
        if raw is not None
        else {}
    )
    metadata = {
        "event_id": str(row["event_id"]),
        "recording_id": str(row["recording_id"]),
        "patient_id": str(row["patient_id"]),
        "source_partition": str(row["partition"]),
        "event_index": int(row["event_index"]),
        "annotation_path": str(row["annotation_path"]),
    }
    if raw is not None:
        metadata["raw_label"] = raw
    return Sample(
        sample_id=f"spr:{row['event_id']}",
        dataset="sprsound",
        partition=partition,
        group_id=str(row["patient_id"]),
        audio_path=str(row["audio_path"]),
        crop_start_s=float(row["start_ms"]) / 1000.0,
        crop_end_s=float(row["end_ms"]) / 1000.0,
        targets=targets,
        metadata=metadata,
    )


def _load_spr_train_selection_samples(repo_root: Path) -> list[Sample]:
    root = _spr_root(repo_root)
    rows = spr_event_rows(
        root / "train2022_json",
        root / "train2022_wav",
        "train",
        True,
    )
    labels = np.asarray([str(row["raw_label"]) for row in rows])
    groups = np.asarray([str(row["patient_id"]) for row in rows])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPR_VALIDATION_SEED,
    )
    subtrain_indices, validation_indices = next(
        splitter.split(np.arange(len(rows)), labels, groups)
    )
    validation = set(np.asarray(validation_indices, dtype=np.int64).tolist())
    subtrain = set(np.asarray(subtrain_indices, dtype=np.int64).tolist())
    return [
        _spr_sample(
            row,
            "validation" if index in validation else "subtrain",
            include_label=True,
        )
        for index, row in enumerate(rows)
        if index in subtrain or index in validation
    ]


def _subset_predictions(
    predictions: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in predictions.items()}


def _select_calibration_thresholds(
    predictions: Mapping[str, np.ndarray], datasets: Sequence[str]
) -> tuple[dict[str, float], dict[str, object]]:
    thresholds: dict[str, float] = {}
    details: dict[str, object] = {}
    for node_index, node in enumerate(CORE_NODES[1:], start=1):
        eligible = predictions["eligible"][:, node_index]
        probabilities = predictions["attribute_probabilities"][:, node_index - 1]
        candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities[eligible])))
        best_objective = -1.0
        best_threshold = 0.5
        best_by_dataset: dict[str, float] = {}
        for threshold in candidates:
            by_dataset = {}
            for dataset in datasets:
                mask = (predictions["dataset_ids"] == dataset) & eligible
                by_dataset[dataset] = float(
                    f1_score(
                        predictions["targets"][mask, node_index].astype(int),
                        probabilities[mask] >= threshold,
                        zero_division=0,
                    )
                )
            objective = float(np.mean([by_dataset[dataset] for dataset in datasets]))
            if objective > best_objective or (
                objective == best_objective and threshold > best_threshold
            ):
                best_objective = objective
                best_threshold = float(threshold)
                best_by_dataset = by_dataset
        thresholds[node] = best_threshold
        details[node] = {
            "threshold": best_threshold,
            "equal_active_dataset_mean_f1": best_objective,
            "f1_by_dataset": best_by_dataset,
            "source": "calibration groups only",
        }
    return thresholds, details


def _icbhi_metrics(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    task: str,
) -> dict[str, object]:
    mask = predictions["dataset_ids"] == "icbhi"
    target = np.asarray(
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"][mask]],
        dtype=np.int64,
    )
    prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"][mask],
        predictions["attribute_probabilities"][mask],
        thresholds,
    )
    result = native_metrics(target, prediction, ICBHI_LABELS)
    result.update(
        {
            "task": task,
            "prediction_unit": "annotated respiratory cycle",
            "decoder": (
                "Level1 Normal->Normal; Level1 Abnormal->Crackle/Wheeze/Both "
                "from calibration thresholds; neither over threshold->larger "
                "probability-minus-threshold margin; tie=Crackle"
            ),
            "thresholds": dict(thresholds),
            "native_score": result["average_score"],
            "official_score": result["average_score"],
        }
    )
    return result


def _sprsound_metrics(
    predictions: Mapping[str, np.ndarray],
    *,
    task: str,
) -> dict[str, object]:
    mask = predictions["dataset_ids"] == "sprsound"
    target = predictions["targets"][mask, 0].astype(np.int64)
    prediction = predictions["level1_predictions"][mask].astype(np.int64)
    result = native_metrics(target, prediction, ("normal", "abnormal"))
    score = (result["average_score"] + result["harmonic_score"]) / 2.0
    result.update(
        {
            "task": task,
            "prediction_unit": "SPRSound BioCAS2022 respiratory event",
            "protocol": "official inter-subject Task1-1 Normal/Adventitious",
            "AS": result["average_score"],
            "HS": result["harmonic_score"],
            "native_score": score,
            "official_score": score,
        }
    )
    return result


def _validation_metrics(
    predictions: Mapping[str, np.ndarray], thresholds: Mapping[str, float]
) -> dict[str, object]:
    return {
        "icbhi_flat4": _icbhi_metrics(
            predictions,
            thresholds,
            task="ICBHI native validation flat4",
        ),
        "sprsound_task1_1": _sprsound_metrics(
            predictions,
            task="SPRSound native validation Task1-1",
        ),
    }


def _reference_from_jh1(repo_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    source = repo_root / REFERENCE_VALIDATION_SOURCE
    with np.load(source, allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    group_lists: dict[str, object] = {
        "source_validation_predictions": str(source),
        "seed": 42,
        "split": "GroupShuffleSplit test_size=0.5 on the JH1 epoch-6 validation groups",
        "datasets": {},
    }
    calibration_mask = np.zeros(len(predictions["sample_ids"]), dtype=bool)
    selection_mask = np.zeros(len(predictions["sample_ids"]), dtype=bool)
    for dataset in ("icbhi", "sprsound"):
        dataset_mask = predictions["dataset_ids"] == dataset
        group_values = predictions["group_ids"][dataset_mask]
        rows = np.arange(dataset_mask.sum(), dtype=np.int64)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        calibration_rows, selection_rows = next(splitter.split(rows, groups=group_values))
        dataset_indices = np.flatnonzero(dataset_mask)
        calibration_indices = dataset_indices[np.asarray(calibration_rows, dtype=np.int64)]
        selection_indices = dataset_indices[np.asarray(selection_rows, dtype=np.int64)]
        calibration_mask[calibration_indices] = True
        selection_mask[selection_indices] = True
        calibration_groups = sorted(
            set(predictions["group_ids"][calibration_indices].tolist())
        )
        selection_groups = sorted(
            set(predictions["group_ids"][selection_indices].tolist())
        )
        group_lists["datasets"][dataset] = {
            "calibration_groups": calibration_groups,
            "selection_groups": selection_groups,
            "calibration_units": int(len(calibration_indices)),
            "selection_units": int(len(selection_indices)),
        }
    calibration = _subset_predictions(predictions, calibration_mask)
    selection = _subset_predictions(predictions, selection_mask)
    thresholds, threshold_details = _select_calibration_thresholds(
        calibration,
        ("icbhi", "sprsound"),
    )
    selection_metrics = _validation_metrics(selection, thresholds)
    reference = {
        "status": "local_clean_queue_reference_ready",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "source": "JH1 selected epoch-6 validation NPZ only",
        "source_epoch": 6,
        "test_accessed": False,
        "threshold_source": "reference calibration groups only",
        "shared_attribute_thresholds": thresholds,
        "threshold_details": threshold_details,
        "selection_metrics": selection_metrics,
        "selected_validation_scores": {
            "icbhi": float(selection_metrics["icbhi_flat4"]["native_score"]),
            "sprsound": float(selection_metrics["sprsound_task1_1"]["native_score"]),
        },
        "group_lists_file": GROUP_LIST_FILENAME,
        "variants_must_reuse": list(MODES),
        "j2_curve_used": False,
    }
    return reference, group_lists


def prepare_reference(repo_root: Path) -> dict[str, object]:
    directory = _reference_dir(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    reference, group_lists = _reference_from_jh1(repo_root)
    _write_json(directory / REFERENCE_FILENAME, reference)
    _write_json(directory / GROUP_LIST_FILENAME, group_lists)
    return reference


def _load_reference(repo_root: Path) -> dict[str, object]:
    return json.loads((_reference_dir(repo_root) / REFERENCE_FILENAME).read_text())


def _load_clean_selection_samples(
    repo_root: Path, reference: Mapping[str, object], author_repo: Path, icbhi_audio_dir: Path
) -> list[Sample]:
    base = PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=author_repo,
        checkpoint=repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
        icbhi_audio_dir=icbhi_audio_dir,
        output_dir=_reference_dir(repo_root),
        device="mps",
    )
    icbhi = _icbhi_selection_samples(base)
    sprsound = _load_spr_train_selection_samples(repo_root)
    group_lists = json.loads(
        (_reference_dir(repo_root) / str(reference["group_lists_file"])).read_text()
    )
    samples: list[Sample] = []
    for dataset_samples in (icbhi, sprsound):
        dataset = dataset_samples[0].dataset
        dataset_groups = group_lists["datasets"][dataset]
        calibration_groups = set(dataset_groups["calibration_groups"])
        selection_groups = set(dataset_groups["selection_groups"])
        for sample in dataset_samples:
            if sample.partition == "subtrain":
                samples.append(sample)
            elif sample.group_id in calibration_groups:
                samples.append(replace(sample, partition="calibration"))
            elif sample.group_id in selection_groups:
                samples.append(replace(sample, partition="selection"))
            else:
                raise RuntimeError(
                    f"validation group missing from fixed reference: {sample.sample_id}"
                )
    return sorted(samples, key=lambda sample: sample.sample_id)


def _split_summary(samples: Sequence[Sample]) -> dict[str, object]:
    output: dict[str, object] = {"training_only": True, "datasets": {}}
    for dataset in ("icbhi", "sprsound"):
        output["datasets"][dataset] = {}
        for partition in ("subtrain", "calibration", "selection"):
            rows = [
                sample
                for sample in samples
                if sample.dataset == dataset and sample.partition == partition
            ]
            targets, eligible, _ = mapped_targets(rows)
            output["datasets"][dataset][partition] = {
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


def _partitioned(
    samples: Sequence[Sample], partition: str
) -> dict[str, tuple[Sample, ...]]:
    return {
        dataset: tuple(
            sample
            for sample in samples
            if sample.dataset == dataset and sample.partition == partition
        )
        for dataset in ("icbhi", "sprsound")
    }


def _unit_mvn_preprocess(
    self: torch.nn.Module,
    source: torch.Tensor,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
) -> torch.Tensor:
    del fbank_mean, fbank_std
    import torchaudio.compliance.kaldi as ta_kaldi

    fbanks = []
    for waveform in source:
        fbanks.append(
            ta_kaldi.fbank(
                waveform.unsqueeze(0),
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
        )
    fbank = torch.stack(fbanks, dim=0)
    mean = fbank.mean(dim=(1, 2), keepdim=True)
    std = fbank.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (fbank - mean) / std


def _build_components(
    config: "CleanQueueConfig", device: torch.device
) -> tuple[PAFAJointHierarchyModel, torch.nn.Module]:
    model, criterion = _build_components_base(config.base, device)
    if config.input_variant == "unit_level_scalar_mvn":
        model.beats.beats.preprocess = MethodType(
            _unit_mvn_preprocess,
            model.beats.beats,
        )
    return model, criterion


def _infer(
    model: PAFAJointHierarchyModel,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    waveform_store: Mapping[str, torch.Tensor],
    config: "CleanQueueConfig",
    device: torch.device,
    datasets: Sequence[str],
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
        for dataset in datasets:
            rows = samples_by_dataset[dataset]
            for start in range(0, len(rows), config.base.batch_size):
                current = list(rows[start : start + config.base.batch_size])
                waveform = torch.stack(
                    [waveform_store[row.sample_id] for row in current]
                ).to(device)
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
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _epoch_batches(
    sizes: Mapping[str, int],
    *,
    source_prop: bool,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed + epoch)
    if source_prop:
        target_batches = {
            dataset: sizes[dataset] // batch_size
            for dataset in ("icbhi", "sprsound")
        }
    else:
        equal_batches = max(sizes[dataset] // batch_size for dataset in ("icbhi", "sprsound"))
        target_batches = {dataset: equal_batches for dataset in ("icbhi", "sprsound")}
    batches: list[tuple[str, np.ndarray]] = []
    for dataset in ("icbhi", "sprsound"):
        needed = target_batches[dataset] * batch_size
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


def _balanced_binary_mean(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    parts = []
    positive = target >= 0.5
    negative = ~positive
    if bool(positive.any()):
        parts.append(values[positive].mean())
    if bool(negative.any()):
        parts.append(values[negative].mean())
    return torch.stack(parts).mean()


def _balanced_classification_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    eligible: torch.Tensor,
    positive_support: Mapping[str, int],
) -> torch.Tensor:
    level1_mask = eligible[:, 0]
    level1_values = torch.nn.functional.cross_entropy(
        logits["level1"][level1_mask],
        targets[level1_mask, 0].long(),
        reduction="none",
    )
    level1_loss = _balanced_binary_mean(
        level1_values,
        targets[level1_mask, 0],
    )
    level2_losses = {}
    for index, node in enumerate(("crackle", "wheeze"), start=1):
        mask = eligible[:, index]
        values = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[node][mask],
            targets[mask, index],
            reduction="none",
        )
        level2_losses[node] = _balanced_binary_mean(values, targets[mask, index])
    raw_weights = {
        node: (1.0 - BETA_EFFECTIVE_NUMBER)
        / (1.0 - BETA_EFFECTIVE_NUMBER ** int(positive_support[node]))
        for node in ("crackle", "wheeze")
    }
    denominator = sum(raw_weights.values())
    level2_loss = sum(
        (raw_weights[node] / denominator) * level2_losses[node]
        for node in ("crackle", "wheeze")
    )
    return 0.5 * level1_loss + 0.5 * level2_loss


def _positive_support(samples: Sequence[Sample]) -> dict[str, int]:
    targets, eligible, _ = mapped_targets(samples)
    return {
        node: int(targets[eligible[:, index], index].sum())
        for index, node in enumerate(CORE_NODES[1:], start=1)
    }


def _selection_state(
    metrics: Mapping[str, Mapping[str, object]], guardrail: float
) -> dict[str, object]:
    icbhi_score = float(metrics["icbhi_flat4"]["native_score"])
    spr_score = float(metrics["sprsound_task1_1"]["native_score"])
    if spr_score >= guardrail:
        key = (1, icbhi_score, spr_score)
        rule = "guardrail_met: (1, ICBHI Score, SPRSound Score)"
    else:
        key = (0, spr_score / guardrail, icbhi_score)
        rule = "guardrail_not_met: (0, SPRSound/guardrail, ICBHI Score)"
    return {
        "selection_rule": rule,
        "sprsound_guardrail": guardrail,
        "guardrail_met": spr_score >= guardrail,
        "icbhi_native_validation_score": icbhi_score,
        "sprsound_native_validation_score": spr_score,
        "selected_validation_native_scores": {
            "icbhi": icbhi_score,
            "sprsound": spr_score,
        },
        "selection_key": list(key),
        "strict_improvement": False,
    }


def _config_payload(config: "CleanQueueConfig") -> dict[str, object]:
    payload = config.base.to_dict()
    payload.update(
        {
            "condition": config.condition,
            "mode": config.mode,
            "input_variant": config.input_variant,
            "active_datasets": ["icbhi", "sprsound"],
            "evidence_label": QUEUE_EVIDENCE_LABEL,
            "clean_queue": True,
            "paper_writing": "separate; Local Result only",
            "reference_contract": str(_reference_dir(config.base.repo_root) / REFERENCE_FILENAME),
            "calibration_selection_split": "fixed from JH1 epoch-6 validation groups; calibration and selection group-disjoint",
            "threshold_policy": "shared Crackle/Wheeze thresholds fit on calibration groups only; tie=highest threshold",
            "selection": "SPRSound JH1-e6 validation Score guardrail; lexicographic key (1, ICBHI, SPR) if met, otherwise (0, SPR/guardrail, ICBHI); exact key tie retains earlier epoch",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": "same validation selection key",
            "early_stopping_policy": "epoch boundary; strict lexicographic key improvement; 50 epochs is the upper bound",
            "official_test_policy": "no official/outer test during training or validation; load both tests once only after all four variants are validation-complete and main role is validation-selected",
            "claim_boundary": "Local Result / clean queue diagnostic; not a paper result",
        }
    )
    if config.mode == "J-Clean":
        payload["sampler"] = "equal dataset batch count; ICBHI repeated to match SPRSound native batch count"
        payload["classification_objective"] = "equal-node eligible CE/BCE, unchanged JH1"
    elif config.mode == "J-MVN":
        payload["sampler"] = "equal dataset batch count; ICBHI repeated to match SPRSound native batch count"
        payload["classification_objective"] = "equal-node eligible CE/BCE, unchanged JH1"
        payload["input"] = "mono 16 kHz; 5 s repeat-pad/front-truncate; log-fbank unit-level scalar MVN before patch embedding replacing BEATs fixed affine normalization; no SpecAugment"
    elif config.mode == "J-SourceProp":
        payload["sampler"] = "natural source batch ratio from floor(native subtrain units/32): ICBHI 99 batches : SPRSound 163 batches"
        payload["dataset_loss_weighting"] = "none; each native batch contributes the same optimizer step"
        payload["classification_objective"] = "equal-node eligible CE/BCE, unchanged JH1"
    else:
        payload["sampler"] = "equal dataset batch count; ICBHI repeated to match SPRSound native batch count"
        payload["classification_objective"] = "per-node positive/negative means equally weighted; Level1 stage 0.5; Level2 stage 0.5; Crackle/Wheeze Level2 allocations inverse-effective-number from subtrain positive support, beta=0.9999"
    return payload


class CleanQueueConfig:
    def __init__(
        self,
        mode: str,
        base: PAFAJointHierarchyConfig,
        input_variant: str,
    ) -> None:
        self.mode = mode
        self.base = base
        self.input_variant = input_variant

    @property
    def condition(self) -> str:
        return f"PAFA_{self.mode}_clean_seed42"

    @property
    def selection_monitor(self) -> str:
        return "same validation lexicographic key"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown clean queue mode: {self.mode}")
        if self.input_variant != INPUT_VARIANTS[self.mode]:
            raise ValueError("clean queue frontend variant changed")
        self.base.validate()


def run_training(config: CleanQueueConfig) -> dict[str, object]:
    config.validate()
    reference = _load_reference(config.base.repo_root)
    if config.base.output_dir.exists() and any(config.base.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite clean run: {config.base.output_dir}")
    config.base.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.base.cpu_threads)
    _seed_everything(config.base.seed)
    samples = _load_clean_selection_samples(
        config.base.repo_root,
        reference,
        config.base.author_repo,
        config.base.icbhi_audio_dir,
    )
    _write_json(config.base.output_dir / "config.json", _config_payload(config))
    _write_json(config.base.output_dir / "selection_split_summary.json", _split_summary(samples))
    subtrain = _partitioned(samples, "subtrain")
    calibration = _partitioned(samples, "calibration")
    selection = _partitioned(samples, "selection")
    waveform_store = _prepare_waveforms(samples, config.base)
    device = torch.device(config.base.device)
    model, pafa_criterion = _build_components(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.base.learning_rate,
        weight_decay=config.base.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    patient_index = _patient_indices(
        [sample for rows in subtrain.values() for sample in rows]
    )
    positive_support = _positive_support(
        [sample for rows in subtrain.values() for sample in rows]
    )
    progress_path = config.base.output_dir / "progress.jsonl"
    train_log_path = config.base.output_dir / "train_log.jsonl"
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_state: dict[str, object] | None = None
    best_thresholds: dict[str, float] | None = None
    best_threshold_details: dict[str, object] | None = None
    best_validation_metrics: dict[str, object] | None = None
    no_improvement_epochs = 0
    completed_epochs = 0
    global_update = 0
    best_trajectory: list[dict[str, object]] = []
    started = time.perf_counter()
    source_prop = config.mode == "J-SourceProp"

    for epoch in range(1, config.base.epochs + 1):
        learning_rate = config.base.learning_rate * (
            config.base.cosine_eta_min_ratio
            + (1.0 - config.base.cosine_eta_min_ratio)
            * (1.0 + np.cos(np.pi * epoch / config.base.epochs))
            / 2.0
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        epoch_losses = {
            dataset: {"classification": [], "pafa": [], "total": []}
            for dataset in ("icbhi", "sprsound")
        }
        batches = _epoch_batches(
            {dataset: len(rows) for dataset, rows in subtrain.items()},
            source_prop=source_prop,
            batch_size=config.base.batch_size,
            seed=config.base.seed,
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
                if config.mode == "J-BalancedLoss":
                    classification_loss = _balanced_classification_loss(
                        logits,
                        targets,
                        eligible,
                        positive_support,
                    )
                else:
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
                    lambda_pcsl=config.base.lambda_pcsl,
                    lambda_gpal=config.base.lambda_gpal,
                )
                total_loss = config.base.classification_weight * classification_loss + config.base.pafa_weight * pafa_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _apply_author_ema(model, before, config.base.ema_beta)
            global_update += 1
            epoch_losses[dataset]["classification"].append(float(classification_loss.detach().cpu()))
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

        calibration_predictions = _infer(
            model,
            calibration,
            waveform_store,
            config,
            device,
            ("icbhi", "sprsound"),
            include_targets=True,
        )
        selection_predictions = _infer(
            model,
            selection,
            waveform_store,
            config,
            device,
            ("icbhi", "sprsound"),
            include_targets=True,
        )
        _save_predictions(
            config.base.output_dir / "calibration" / f"epoch_{epoch:03d}.npz",
            calibration_predictions,
        )
        _save_predictions(
            config.base.output_dir / "validation" / f"epoch_{epoch:03d}.npz",
            selection_predictions,
        )
        thresholds, threshold_details = _select_calibration_thresholds(
            calibration_predictions,
            ("icbhi", "sprsound"),
        )
        validation_metrics = _validation_metrics(selection_predictions, thresholds)
        state = _selection_state(
            validation_metrics,
            float(reference["selected_validation_scores"]["sprsound"]),
        )
        candidate_key = tuple(float(value) for value in state["selection_key"])
        improved = best_key is None or candidate_key > best_key
        state["selection_key"] = list(candidate_key)
        state["strict_improvement"] = improved
        next_no_improvement = 0 if improved else no_improvement_epochs + 1
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
            "calibration": {
                "thresholds": thresholds,
                "threshold_details": threshold_details,
                "groups_only": True,
            },
            "validation": validation_metrics,
            "selection": state,
            "early_stopping": {
                "monitor": "same validation selection key",
                "selection_key": list(candidate_key),
                "strict_improvement": improved,
                "no_improvement_epochs": next_no_improvement,
                "patience": EARLY_STOPPING_PATIENCE,
            },
            "outer_test_accessed": False,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        _append_jsonl(train_log_path, record)
        if improved:
            best_key = candidate_key
            best_epoch = epoch
            best_state = state
            best_thresholds = thresholds
            best_threshold_details = threshold_details
            best_validation_metrics = validation_metrics
            best_trajectory.append(
                {
                    "epoch": epoch,
                    "selection_key": list(candidate_key),
                    "icbhi_score": state["icbhi_native_validation_score"],
                    "sprsound_score": state["sprsound_native_validation_score"],
                }
            )
            no_improvement_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "update": global_update,
                    "model": copy.deepcopy(model.state_dict()),
                    "selection_state": state,
                    "thresholds": thresholds,
                    "config": _config_payload(config),
                },
                config.base.output_dir / "best_checkpoint.pt",
            )
        else:
            no_improvement_epochs = next_no_improvement
        completed_epochs = epoch
        print(
            json.dumps(
                {
                    "mode": config.mode,
                    "evidence_label": QUEUE_EVIDENCE_LABEL,
                    "epoch": epoch,
                    "update": global_update,
                    "selection_key": list(candidate_key),
                    "icbhi_validation_score": state["icbhi_native_validation_score"],
                    "sprsound_validation_score": state["sprsound_native_validation_score"],
                    "current_best_epoch": best_epoch,
                    "current_best_key": list(best_key),
                    "no_improvement_epochs": no_improvement_epochs,
                    "elapsed_minutes": record["elapsed_minutes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            break

    assert best_state is not None
    assert best_thresholds is not None
    assert best_threshold_details is not None
    assert best_validation_metrics is not None
    final_selection = {
        "status": "validation_complete_no_test",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "mode": config.mode,
        "selected_epoch": best_epoch,
        "selection_key": list(best_key),
        "selection_state": best_state,
        "selected_validation_metrics": best_validation_metrics,
        "shared_attribute_thresholds": best_thresholds,
        "threshold_details": best_threshold_details,
        "threshold_source": "selected epoch calibration groups only; no test tuning",
        "selection_phase_test_accessed": False,
        "official_test_accessed": False,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "same validation selection key",
        "early_stopped": completed_epochs < config.base.epochs,
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.base.epochs,
        "no_improvement_epochs_at_stop": no_improvement_epochs,
        "best_trajectory": best_trajectory,
        "reference_guardrail_source": REFERENCE_VALIDATION_SOURCE,
        "post_selection_test": "deferred until all four variants are validation-complete and main role is frozen",
    }
    _write_json(config.base.output_dir / "validation_selection.json", final_selection)
    summary = {
        "status": "validation_complete_no_test",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "claim_boundary": "Local Result / clean queue diagnostic; not a paper result",
        "condition": config.condition,
        "mode": config.mode,
        "input_variant": config.input_variant,
        "selected_epoch": best_epoch,
        "selection_key": list(best_key),
        "selection_state": best_state,
        "selected_validation_native_scores": best_state["selected_validation_native_scores"],
        "selected_validation_metrics": best_validation_metrics,
        "best_trajectory": best_trajectory,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "same validation selection key",
        "early_stopped": completed_epochs < config.base.epochs,
        "completed_training_epochs": completed_epochs,
        "max_epochs": config.base.epochs,
        "updates": global_update,
        "no_improvement_epochs_at_stop": no_improvement_epochs,
        "official_test_accessed": False,
        "terminal_test_accesses": {"icbhi": 0, "sprsound": 0},
        "reference_guardrail_score": reference["selected_validation_scores"]["sprsound"],
        "positive_support_subtrain": positive_support,
        "output_directory": str(config.base.output_dir),
    }
    _write_json(config.base.output_dir / "run_summary.json", summary)
    return summary


def repair_validation_summary(repo_root: Path, mode: str) -> dict[str, object]:
    run_dir = _run_dir(repo_root, mode)
    rows = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    best_row = max(
        rows,
        key=lambda row: (
            tuple(float(value) for value in row["selection"]["selection_key"]),
            -int(row["epoch"]),
        ),
    )
    best_state = {
        **best_row["selection"],
        "selected_validation_native_scores": {
            "icbhi": best_row["selection"]["icbhi_native_validation_score"],
            "sprsound": best_row["selection"]["sprsound_native_validation_score"],
        },
    }
    trajectory = [
        {
            "epoch": row["epoch"],
            "selection_key": row["selection"]["selection_key"],
            "icbhi_score": row["selection"]["icbhi_native_validation_score"],
            "sprsound_score": row["selection"]["sprsound_native_validation_score"],
        }
        for row in rows
        if row["selection"]["strict_improvement"]
    ]
    completed_epochs = int(rows[-1]["epoch"])
    best_thresholds = best_row["calibration"]["thresholds"]
    best_details = best_row["calibration"]["threshold_details"]
    validation_selection = {
        "status": "validation_complete_no_test",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "mode": mode,
        "selected_epoch": int(best_row["epoch"]),
        "selection_key": best_row["selection"]["selection_key"],
        "selection_state": best_state,
        "selected_validation_metrics": best_row["validation"],
        "shared_attribute_thresholds": best_thresholds,
        "threshold_details": best_details,
        "threshold_source": "selected epoch calibration groups only; no test tuning",
        "selection_phase_test_accessed": False,
        "official_test_accessed": False,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "same validation selection key",
        "early_stopped": True,
        "completed_training_epochs": completed_epochs,
        "max_epochs": 50,
        "no_improvement_epochs_at_stop": int(rows[-1]["early_stopping"]["no_improvement_epochs"]),
        "best_trajectory": trajectory,
        "reference_guardrail_source": REFERENCE_VALIDATION_SOURCE,
        "post_selection_test": "deferred until all four variants are validation-complete and main role is frozen",
        "repair_source": "existing train_log.jsonl and validation artifacts; no model inference",
    }
    _write_json(run_dir / "validation_selection.json", validation_selection)
    summary = {
        "status": "validation_complete_no_test",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "claim_boundary": "Local Result / clean queue diagnostic; not a paper result",
        "condition": f"PAFA_{mode}_clean_seed42",
        "mode": mode,
        "input_variant": INPUT_VARIANTS[mode],
        "selected_epoch": int(best_row["epoch"]),
        "selection_key": best_row["selection"]["selection_key"],
        "selection_state": best_state,
        "selected_validation_native_scores": best_state["selected_validation_native_scores"],
        "selected_validation_metrics": best_row["validation"],
        "best_trajectory": trajectory,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "same validation selection key",
        "early_stopped": True,
        "completed_training_epochs": completed_epochs,
        "max_epochs": 50,
        "updates": int(rows[-1]["update"]),
        "no_improvement_epochs_at_stop": int(rows[-1]["early_stopping"]["no_improvement_epochs"]),
        "official_test_accessed": False,
        "terminal_test_accesses": {"icbhi": 0, "sprsound": 0},
        "reference_guardrail_score": rows[0]["selection"]["sprsound_guardrail"],
        "output_directory": str(run_dir),
        "repair_source": "existing train_log.jsonl and validation artifacts; no model inference",
    }
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def _class_collapse(metric: Mapping[str, object]) -> bool:
    matrix = np.asarray(metric["confusion"], dtype=np.int64)
    return int(np.count_nonzero(matrix.sum(axis=0))) <= 1


def _load_terminal_samples(config: CleanQueueConfig) -> list[Sample]:
    icbhi_rows = read_official_cycles(
        config.base.icbhi_audio_dir,
        config.base.author_repo,
        ("test",),
    )
    icbhi = [_icbhi_sample(row, config.base.icbhi_audio_dir, "test") for row in icbhi_rows]
    root = _spr_root(config.base.repo_root)
    rows = spr_event_rows(
        root / "test2022_json/inter_test_json",
        root / "test2022_wav",
        "inter",
        False,
    )
    sprsound = [_spr_sample(row, "test", include_label=False) for row in rows]
    return sorted([*icbhi, *sprsound], key=lambda sample: sample.sample_id)


def evaluate_terminal(config: CleanQueueConfig) -> dict[str, object]:
    config.validate()
    reference = _load_reference(config.base.repo_root)
    main_selection = json.loads(_main_selection_path(config.base.repo_root).read_text())
    if set(main_selection["modes"]) != set(MODES):
        raise RuntimeError("main selection is incomplete")
    run_summary_path = config.base.output_dir / "run_summary.json"
    validation_path = config.base.output_dir / "validation_selection.json"
    summary = json.loads(run_summary_path.read_text())
    selection = json.loads(validation_path.read_text())
    if summary["official_test_accessed"]:
        raise RuntimeError("clean terminal test already accessed for this variant")
    device = torch.device(config.base.device)
    model, _ = _build_components(config, device)
    checkpoint = torch.load(config.base.output_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    terminal_samples = _load_terminal_samples(config)
    terminal_by_dataset = {
        dataset: tuple(sample for sample in terminal_samples if sample.dataset == dataset)
        for dataset in ("icbhi", "sprsound")
    }
    terminal_waveforms = _prepare_waveforms(terminal_samples, config.base)
    label_free = _infer(
        model,
        terminal_by_dataset,
        terminal_waveforms,
        config,
        device,
        ("icbhi", "sprsound"),
        include_targets=False,
    )
    terminal_dir = config.base.output_dir / "terminal"
    _save_predictions(terminal_dir / "selected_test_predictions_label_free.npz", label_free)
    spr_targets = load_terminal_spr_test_targets(terminal_samples, include_checksums=False)
    scored = _attach_targets(label_free, terminal_samples, spr_targets)
    _save_predictions(terminal_dir / "selected_test_predictions_scored.npz", scored)
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    icbhi = _icbhi_metrics(scored, thresholds, task="ICBHI official-test flat4")
    icbhi["protocol"] = "official recording split 60/40; 2756 respiratory cycles"
    sprsound = _sprsound_metrics(
        scored,
        task="SPRSound BioCAS2022 Task1-1 Normal/Adventitious",
    )
    sprsound["protocol"] = "official inter-subject test; 1429 respiratory events"
    terminal_metrics = {
        "status": "clean_queue_selected_terminal",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "mode": config.mode,
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_key": selection["selection_key"],
        "shared_attribute_thresholds": thresholds,
        "threshold_source": "selected epoch calibration groups only; no test tuning",
        "outer_test_accessed": True,
        "terminal_test_accesses": {"icbhi": 1, "sprsound": 1},
        "terminal_targets_loaded_after_label_free_prediction_write": True,
        "prediction_support": {"icbhi": 2756, "sprsound": 1429},
        "icbhi_flat4": icbhi,
        "sprsound_inter_task1_1": sprsound,
        "native_selected_metrics": {
            "icbhi_flat4": icbhi,
            "sprsound_task1_1": sprsound,
        },
        "cross_domain_metrics": {"status": "not_applicable_joint_training"},
        "test_access_order": [
            "all four variants validation-complete",
            "validation main-role selection frozen",
            "selected checkpoint loaded",
            "write terminal label-free predictions",
            "load SPRSound inter targets and score both terminal datasets",
        ],
        "class_collapse": {
            "icbhi": _class_collapse(icbhi),
            "sprsound": _class_collapse(sprsound),
        },
        "reference_guardrail_score": reference["selected_validation_scores"]["sprsound"],
    }
    _write_json(terminal_dir / "native_metrics.json", terminal_metrics)
    main_mode = main_selection["main_mode"]
    summary.update(
        {
            "status": "terminal_complete_clean_queue_run",
            "official_test_accessed": True,
            "terminal_test_accesses": {"icbhi": 1, "sprsound": 1},
            "selected_terminal_metrics": terminal_metrics,
            "main_role": config.mode == main_mode,
            "main_role_mode": main_mode,
            "no_class_collapse": not any(terminal_metrics["class_collapse"].values()),
            "clean_terminal_gate": {
                "icbhi_score_at_least_0_59": icbhi["official_score"] >= 0.59,
                "sprsound_score_at_least_0_90": sprsound["official_score"] >= 0.90,
                "no_class_collapse": not any(terminal_metrics["class_collapse"].values()),
                "numeric_gate_passed": (
                    config.mode == main_mode
                    and icbhi["official_score"] >= 0.59
                    and sprsound["official_score"] >= 0.90
                    and not any(terminal_metrics["class_collapse"].values())
                ),
            },
        }
    )
    _write_json(run_summary_path, summary)
    return summary


def select_main(repo_root: Path) -> dict[str, object]:
    rows = []
    for order, mode in enumerate(MODES):
        summary = json.loads((_run_dir(repo_root, mode) / "run_summary.json").read_text())
        if summary["official_test_accessed"]:
            raise RuntimeError("main role must be selected before terminal test access")
        rows.append(
            {
                "mode": mode,
                "run_directory": str(_run_dir(repo_root, mode)),
                "selected_epoch": summary["selected_epoch"],
                "selection_key": summary["selection_key"],
                "selected_validation_native_scores": summary["selected_validation_native_scores"],
                "order": order,
            }
        )
    winner = max(rows, key=lambda row: (tuple(row["selection_key"]), -row["order"]))
    payload = {
        "status": "validation_main_role_frozen_no_test",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "test_accessed": False,
        "selection_rule": "same lexicographic validation key; exact tie retains earlier preregistered variant order",
        "main_role_mode": winner["mode"],
        "main_role_claim_boundary": "validation-selected candidate for paper main role only; no paper claim",
        "modes": MODES,
        "runs": rows,
    }
    _write_json(_main_selection_path(repo_root), payload)
    return payload


def finalize_queue(repo_root: Path) -> dict[str, object]:
    main = json.loads(_main_selection_path(repo_root).read_text())
    summaries = []
    for mode in MODES:
        summary = json.loads((_run_dir(repo_root, mode) / "run_summary.json").read_text())
        if not summary["official_test_accessed"]:
            raise RuntimeError(f"terminal evaluation missing for {mode}")
        summaries.append(summary)
    main_summary = next(summary for summary in summaries if summary["mode"] == main["main_role_mode"])
    gate = main_summary["clean_terminal_gate"]
    payload = {
        "status": "local_clean_queue_complete",
        "evidence_label": QUEUE_EVIDENCE_LABEL,
        "main_role_mode": main["main_role_mode"],
        "main_role_selection": "validation-only lexicographic key; test not used",
        "variants": MODES,
        "runs": [
            {
                "mode": summary["mode"],
                "selected_epoch": summary["selected_epoch"],
                "selected_validation_native_scores": summary["selected_validation_native_scores"],
                "terminal_test_accesses": summary["terminal_test_accesses"],
                "icbhi_official_score": summary["selected_terminal_metrics"]["icbhi_flat4"]["official_score"],
                "sprsound_task1_1_official_score": summary["selected_terminal_metrics"]["sprsound_inter_task1_1"]["official_score"],
                "main_role": summary["main_role"],
            }
            for summary in summaries
        ],
        "clean_terminal_gate": gate,
        "test_access_boundary": "No official/outer test was read during any variant's training, calibration, selection, early stopping, or main-role choice; each variant then accessed ICBHI official test and SPRSound inter test once.",
        "next_step": "If gate passes, management may decide whether to open a separate HF/KAUH protocol round; no such experiment was started.",
    }
    _write_json(_queue_summary_path(repo_root), payload)
    return payload


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    return (
        repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        repo_root / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
        repo_root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
    )


def _cli_config(args: argparse.Namespace) -> CleanQueueConfig:
    if args.mode is None:
        raise ValueError("--mode is required for a variant operation")
    author_repo, checkpoint, icbhi_audio_dir = _default_paths(args.repo_root)
    base = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio_dir,
        output_dir=args.output_dir or _run_dir(args.repo_root, args.mode),
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    return CleanQueueConfig(args.mode, base, INPUT_VARIANTS[args.mode])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prepare-reference", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--select-main", action="store_true")
    parser.add_argument("--evaluate-terminal", action="store_true")
    parser.add_argument("--finalize-queue", action="store_true")
    parser.add_argument("--repair-validation-summary", action="store_true")
    args = parser.parse_args()
    operations = sum(
        int(value)
        for value in (
            args.prepare_reference,
            args.run,
            args.select_main,
            args.evaluate_terminal,
            args.finalize_queue,
            args.repair_validation_summary,
        )
    )
    if operations != 1:
        raise ValueError("choose exactly one queue operation")
    if args.prepare_reference:
        print(json.dumps(prepare_reference(args.repo_root), indent=2, sort_keys=True))
    elif args.select_main:
        print(json.dumps(select_main(args.repo_root), indent=2, sort_keys=True))
    elif args.finalize_queue:
        print(json.dumps(finalize_queue(args.repo_root), indent=2, sort_keys=True))
    elif args.repair_validation_summary:
        if args.mode is None:
            raise ValueError("--mode is required for validation summary repair")
        print(json.dumps(repair_validation_summary(args.repo_root, args.mode), indent=2, sort_keys=True))
    elif args.run:
        print(json.dumps(run_training(_cli_config(args)), indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps(evaluate_terminal(_cli_config(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
