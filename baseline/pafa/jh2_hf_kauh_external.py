"""Fixed-checkpoint HF-test and KAUH external transfer diagnostics for JH2."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline.four_dataset_frozen_encoder.data import Sample, _parse_hms
from baseline.multidataset_pipeline.beats_nal_protocol import decode_icbhi_hierarchical_flat4
from baseline.multidataset_pipeline.posthoc_native_readout import ICBHI_LABELS, native_metrics
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _build_components,
    _prepare_waveforms,
    _save_predictions,
    _write_json,
)


# The external runner intentionally does not import the ICBHI/SPRSound terminal
# target loader.  JH2's fixed thresholds are the only values read from its
# selected validation/summary artifacts.
EVIDENCE_LABEL = "fixed_JH2_test_selected_checkpoint_external_transfer_diagnostic"
JH2_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2")
OUTPUT_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42")
HF_TEST_RELATIVE = Path("dataset/raw/hf_lung_v1/source_original/test/test")
KAUH_RELATIVE = Path("dataset/raw/kauh_fraiwan/source_original/audio_files")
KAUH_RAW_LABELS = ("N", "E W", "I E W", "C", "I C", "I C E W", "Crep", "Bronchial", "I C B")
HF_TOKENS = ("I", "E", "D", "Wheeze", "Rhonchi", "Stridor")
KAUH_COMPATIBLE = {
    "N": (0, "normal"),
    "E W": (2, "wheeze"),
    "I E W": (2, "wheeze"),
    "C": (1, "crackle"),
    "I C": (1, "crackle"),
    "I C E W": (3, "both"),
}


def _fixed_jh2(repo_root: Path) -> dict[str, object]:
    run_dir = repo_root / JH2_RELATIVE
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    selection = json.loads((run_dir / "validation_selection.json").read_text())
    selected = json.loads((run_dir / "terminal/selected_native_metrics.json").read_text())
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    selected_thresholds = {
        key: float(value)
        for key, value in selected["shared_attribute_thresholds"].items()
    }
    if summary["selected_epoch"] != 19 or thresholds != selected_thresholds:
        raise RuntimeError("JH2 selected checkpoint or fixed threshold artifact mismatch")
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "backbone_checkpoint": Path(str(config["checkpoint"])),
        "selected_epoch": 19,
        "thresholds": thresholds,
        "threshold_source": "JH2 validation_selection.json and selected_native_metrics.json",
    }


def _base_config(repo_root: Path, jh2: Mapping[str, object], output_dir: Path) -> PAFAJointHierarchyConfig:
    return PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        checkpoint=jh2["backbone_checkpoint"],
        icbhi_audio_dir=repo_root / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
        output_dir=output_dir,
        device="mps",
        cpu_threads=4,
    )


def _parse_kauh_identity(path: Path) -> tuple[str, str, str]:
    match = re.match(r"^([BDE])P(\d+)_(.*)$", path.stem)
    if not match:
        raise RuntimeError(f"invalid KAUH filename: {path.name}")
    filter_mode, patient_number, rest = match.groups()
    if len(rest.split(",")) != 5:
        raise RuntimeError(f"invalid KAUH filename fields: {path.name}")
    return f"P{patient_number}", filter_mode, rest


def _load_kauh_samples(repo_root: Path) -> list[Sample]:
    root = repo_root / KAUH_RELATIVE
    samples = []
    for path in sorted(root.glob("*.wav")):
        patient_id, filter_mode, rest = _parse_kauh_identity(path)
        samples.append(
            Sample(
                sample_id=f"kauh:{patient_id}:{filter_mode}",
                dataset="kauh",
                partition="test",
                group_id=patient_id,
                audio_path=str(path.resolve()),
                crop_start_s=None,
                crop_end_s=None,
                targets={},
                metadata={
                    "patient_id": patient_id,
                    "filter_mode": filter_mode,
                    "filename_rest": rest,
                },
            )
        )
    by_patient = Counter(sample.group_id for sample in samples)
    by_filter = Counter(sample.metadata["filter_mode"] for sample in samples)
    if len(samples) != 336 or len(by_patient) != 112 or any(by_patient[value] != 3 for value in by_patient):
        raise RuntimeError("KAUH external set must contain 336 recordings and 112 B/D/E patients")
    if set(by_filter) != {"B", "D", "E"} or any(value != 112 for value in by_filter.values()):
        raise RuntimeError("KAUH external set must retain every B/D/E sibling")
    siblings = {
        patient: {sample.metadata["filter_mode"] for sample in samples if sample.group_id == patient}
        for patient in by_patient
    }
    if any(modes != {"B", "D", "E"} for modes in siblings.values()):
        raise RuntimeError("KAUH B/D/E sibling association failed")
    return samples


def _load_hf_test_samples(repo_root: Path) -> list[Sample]:
    root = repo_root / HF_TEST_RELATIVE
    samples = []
    for path in sorted(root.rglob("*.wav")):
        samples.append(
            Sample(
                sample_id=f"hf:test:{path.stem}",
                dataset="hf_lung",
                partition="test",
                group_id=path.stem,
                audio_path=str(path.resolve()),
                crop_start_s=None,
                crop_end_s=None,
                targets={},
                metadata={
                    "recording_id": path.stem,
                    "label_path": str(path.with_name(path.stem + "_label.txt").resolve()),
                },
            )
        )
    if len(samples) != 1956:
        raise RuntimeError(f"HF source-test recording count changed: {len(samples)}")
    return samples


def _predict_kauh(
    model: torch.nn.Module,
    samples: Sequence[Sample],
    waveform_store: Mapping[str, torch.Tensor],
    config: PAFAJointHierarchyConfig,
    device: torch.device,
    thresholds: Mapping[str, float],
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "patient_ids": [],
        "filter_modes": [],
        "file_names": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
        "flat4_predictions": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), config.batch_size):
            current = list(samples[start : start + config.batch_size])
            waveform = torch.stack([waveform_store[row.sample_id] for row in current]).to(device)
            output, _ = model(waveform, training=False)
            level1 = output["level1"].float().cpu()
            attributes = torch.stack((output["crackle"], output["wheeze"]), dim=-1).float().cpu()
            level1_predictions = level1.argmax(dim=-1).numpy()
            attribute_probabilities = torch.sigmoid(attributes).numpy()
            flat4 = decode_icbhi_hierarchical_flat4(
                level1_predictions,
                attribute_probabilities,
                thresholds,
            )
            fields["prediction_ids"].append(np.asarray([row.sample_id for row in current]))
            fields["sample_ids"].append(np.asarray([row.sample_id for row in current]))
            fields["patient_ids"].append(np.asarray([row.group_id for row in current]))
            fields["filter_modes"].append(
                np.asarray([str(row.metadata["filter_mode"]) for row in current])
            )
            fields["file_names"].append(np.asarray([Path(row.audio_path).name for row in current]))
            fields["level1_logits"].append(level1.numpy())
            fields["level1_probabilities"].append(torch.softmax(level1, dim=-1).numpy())
            fields["level1_predictions"].append(level1_predictions)
            fields["attribute_logits"].append(attributes.numpy())
            fields["attribute_probabilities"].append(attribute_probabilities)
            fields["flat4_predictions"].append(flat4)
            if start == 0 or (start + len(current)) % 100 == 0 or start + len(current) == len(samples):
                print(f"KAUH label-free recordings: {start + len(current)}/{len(samples)}", flush=True)
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _load_hf_windows(path: Path, sample_rate: int) -> tuple[list[torch.Tensor], list[list[float]]]:
    import torchaudio
    from torchaudio import transforms as T

    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = T.Resample(source_rate, sample_rate)(waveform)
    fade_samples = int(sample_rate / 16)
    waveform = T.Fade(
        fade_in_len=fade_samples,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )(waveform)
    values = waveform.squeeze(0).to(torch.float32).contiguous()
    width = sample_rate * 5
    required = width * 3
    if values.shape[-1] < required:
        raise RuntimeError(f"HF source-test recording shorter than 15 seconds: {path.name}")
    values = values[:required]
    windows = [values[index * width : (index + 1) * width] for index in range(3)]
    time_map = [[float(index * 5), float((index + 1) * 5)] for index in range(3)]
    return windows, time_map


def _predict_hf(
    model: torch.nn.Module,
    samples: Sequence[Sample],
    config: PAFAJointHierarchyConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "recording_ids": [],
        "window_indices": [],
        "source_start_s": [],
        "source_end_s": [],
        "file_names": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), config.batch_size):
            current = list(samples[start : start + config.batch_size])
            windows: list[torch.Tensor] = []
            recording_ids: list[str] = []
            window_indices: list[int] = []
            starts: list[float] = []
            ends: list[float] = []
            file_names: list[str] = []
            for sample in current:
                current_windows, time_map = _load_hf_windows(Path(sample.audio_path), config.sample_rate)
                for window_index, (window, (source_start, source_end)) in enumerate(zip(current_windows, time_map)):
                    windows.append(window)
                    recording_ids.append(str(sample.metadata["recording_id"]))
                    window_indices.append(window_index)
                    starts.append(source_start)
                    ends.append(source_end)
                    file_names.append(Path(sample.audio_path).name)
            output, _ = model(torch.stack(windows).to(device), training=False)
            level1 = output["level1"].float().cpu()
            attributes = torch.stack((output["crackle"], output["wheeze"]), dim=-1).float().cpu()
            level1_predictions = level1.argmax(dim=-1).numpy()
            ids = np.asarray([f"hf:test:{recording}::window_{index:02d}" for recording, index in zip(recording_ids, window_indices)])
            fields["prediction_ids"].append(ids)
            fields["sample_ids"].append(np.asarray([f"hf:test:{recording}" for recording in recording_ids]))
            fields["recording_ids"].append(np.asarray(recording_ids))
            fields["window_indices"].append(np.asarray(window_indices, dtype=np.int64))
            fields["source_start_s"].append(np.asarray(starts, dtype=np.float32))
            fields["source_end_s"].append(np.asarray(ends, dtype=np.float32))
            fields["file_names"].append(np.asarray(file_names))
            fields["level1_logits"].append(level1.numpy())
            fields["level1_probabilities"].append(torch.softmax(level1, dim=-1).numpy())
            fields["level1_predictions"].append(level1_predictions)
            fields["attribute_logits"].append(attributes.numpy())
            fields["attribute_probabilities"].append(torch.sigmoid(attributes).numpy())
            completed = min(start + len(current), len(samples))
            if start == 0 or completed % 100 == 0 or completed == len(samples):
                print(f"HF label-free recordings: {completed}/{len(samples)}", flush=True)
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _parse_hf_annotations(samples: Sequence[Sample]) -> dict[str, dict[str, object]]:
    annotations: dict[str, dict[str, object]] = {}
    for sample in samples:
        recording_id = str(sample.metadata["recording_id"])
        path = Path(str(sample.metadata["label_path"]))
        if not path.is_file():
            annotations[recording_id] = {
                "status": "NA_recording",
                "tokens": [],
                "intervals": [],
            }
            continue
        intervals = []
        invalid = False
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3:
                invalid = True
                break
            token = parts[0]
            if token not in HF_TOKENS:
                raise RuntimeError(f"unknown HF annotation token: {token} in {path}")
            start, end = _parse_hms(parts[1]), _parse_hms(parts[2])
            intervals.append((token, float(start), float(end)))
        if invalid:
            annotations[recording_id] = {"status": "NA_recording", "tokens": [], "intervals": []}
        elif not intervals:
            annotations[recording_id] = {"status": "empty_annotation", "tokens": [], "intervals": []}
        else:
            tokens = sorted({token for token, _, _ in intervals})
            status = "eligible_presence" if set(tokens) & {"D", "Wheeze", "Rhonchi", "Stridor"} else "phase_only_or_not_annotated"
            annotations[recording_id] = {
                "status": status,
                "tokens": tokens,
                "intervals": intervals,
            }
    return annotations


def _binary_threshold_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, object]:
    prediction = (probability >= threshold).astype(np.int64)
    result = native_metrics(target.astype(np.int64), prediction, ("negative", "positive"))
    result.update({"threshold": threshold, "official_score": result["average_score"]})
    return result


def _curve_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    if len(np.unique(target)) < 2:
        return {"status": "HOLD_single_class_support", "auroc": None, "auprc": None, "support": int(len(target))}
    return {
        "status": "complete",
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "support": int(len(target)),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
    }


def _hf_scored_predictions(
    predictions: Mapping[str, np.ndarray],
    annotations: Mapping[str, Mapping[str, object]],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    recording_ids = sorted(set(predictions["recording_ids"].tolist()))
    statuses = []
    d_targets = []
    w_targets = []
    eligible = []
    for recording_id in predictions["recording_ids"]:
        row = annotations[str(recording_id)]
        tokens = set(row["tokens"])
        statuses.append(str(row["status"]))
        is_eligible = bool(tokens & {"D", "Wheeze", "Rhonchi", "Stridor"})
        eligible.append(is_eligible)
        d_targets.append(int("D" in tokens) if is_eligible else -1)
        w_targets.append(int("Wheeze" in tokens) if is_eligible else -1)
    scored = {
        **predictions,
        "annotation_status": np.asarray(statuses),
        "recording_eligible": np.asarray(eligible, dtype=bool),
        "d_recording_target": np.asarray(d_targets, dtype=np.int64),
        "wheeze_recording_target": np.asarray(w_targets, dtype=np.int64),
    }
    presence_rows = []
    for recording_id in recording_ids:
        indices = np.flatnonzero(predictions["recording_ids"] == recording_id)
        row = annotations[recording_id]
        tokens = set(row["tokens"])
        is_eligible = bool(tokens & {"D", "Wheeze", "Rhonchi", "Stridor"})
        if not is_eligible:
            continue
        attr_max = predictions["attribute_probabilities"][indices].max(axis=0)
        presence_rows.append((int("D" in tokens), int("Wheeze" in tokens), float(attr_max[0]), float(attr_max[1])))
    presence = np.asarray(presence_rows, dtype=np.float64)
    interval_support = Counter()
    interval_hits = Counter()
    interval_values: dict[str, list[float]] = {"D": [], "Wheeze": []}
    for recording_id in recording_ids:
        indices = np.flatnonzero(predictions["recording_ids"] == recording_id)
        row = annotations[recording_id]
        for token, interval_start, interval_end in row["intervals"]:
            if token not in {"D", "Wheeze"}:
                continue
            overlap = indices[
                (predictions["source_start_s"][indices] < interval_end)
                & (predictions["source_end_s"][indices] > interval_start)
            ]
            if len(overlap) == 0:
                continue
            column = 0 if token == "D" else 1
            value = float(predictions["attribute_probabilities"][overlap, column].max())
            interval_support[token] += 1
            interval_hits[token] += int(value >= thresholds["crackle" if token == "D" else "wheeze"])
            interval_values[token].append(value)
    interval_coverage = {
        token: {
            "positive_interval_support": int(interval_support[token]),
            "thresholded_hits": int(interval_hits[token]),
            "positive_recall": float(interval_hits[token] / interval_support[token]) if interval_support[token] else None,
            "threshold": thresholds["crackle" if token == "D" else "wheeze"],
            "max_probability_mean": float(np.mean(interval_values[token])) if interval_values[token] else None,
        }
        for token in ("D", "Wheeze")
    }
    recording_presence = {}
    if len(presence):
        for column, token, threshold_name in ((0, "D", "crackle"), (1, "Wheeze", "wheeze")):
            recording_presence[token] = {
                "eligible_recordings": int(len(presence)),
                "curve": _curve_metrics(presence[:, column].astype(np.int64), presence[:, column + 2]),
                "thresholded": _binary_threshold_metrics(
                    presence[:, column].astype(np.int64),
                    presence[:, column + 2],
                    thresholds[threshold_name],
                ),
            }
    else:
        recording_presence = {
            token: {"eligible_recordings": 0, "curve": {"status": "HOLD_no_eligible_recordings"}, "thresholded": None}
            for token in ("D", "Wheeze")
        }
    token_interval_support = Counter()
    token_recording_support = Counter()
    status_counts = Counter()
    for row in annotations.values():
        status_counts[str(row["status"])] += 1
        token_interval_support.update(token for token, _, _ in row["intervals"])
        token_recording_support.update(set(row["tokens"]))
    level1_counts = np.bincount(predictions["level1_predictions"].astype(np.int64), minlength=2)
    attribute_counts = {
        token: int((predictions["attribute_probabilities"][:, index] >= thresholds[token]).sum())
        for index, token in enumerate(("crackle", "wheeze"))
    }
    metrics = {
        "status": "hf_external_diagnostic_complete",
        "evidence_label": EVIDENCE_LABEL,
        "selected_epoch": 19,
        "threshold_source": "fixed JH2 selected validation thresholds; no HF tuning",
        "shared_attribute_thresholds": dict(thresholds),
        "source_test_recordings": len(recording_ids),
        "exact_windows_per_recording": 3,
        "window_geometry": "[0,5], [5,10], [10,15] seconds; continuous and non-overlapping",
        "window_source_time_map_preserved": True,
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "annotation_interval_support": dict(sorted(token_interval_support.items())),
        "annotation_recording_support": dict(sorted(token_recording_support.items())),
        "positive_interval_coverage": interval_coverage,
        "recording_presence_pool": {
            "eligible_definition": "recording contains at least one D/Wheeze/Rhonchi/Stridor interval",
            "gap_unknown_not_in_denominator": True,
            "empty_and_NA_excluded": True,
            "metrics": recording_presence,
        },
        "unsupported_native_tasks": {
            "I_E_phase": {"status": "HOLD_not_a_JH2_attribute_task"},
            "CAS_DAS": {"status": "HOLD_not_a_JH2_head"},
            "Rhonchi_Stridor": {
                "status": "HOLD_not_a_JH2_compatible_attribute",
                "interval_support": int(token_interval_support["Rhonchi"] + token_interval_support["Stridor"]),
                "recording_support": int(token_recording_support["Rhonchi"] + token_recording_support["Stridor"]),
            },
            "Level1_normal_detection": {"status": "HOLD_no_HF_normal_target_mapping"},
        },
        "prediction_distribution": {
            "window_level1_argmax": {"normal": int(level1_counts[0]), "abnormal": int(level1_counts[1])},
            "window_attribute_thresholded": attribute_counts,
            "probability_summary": {
                name: {
                    "mean": float(np.mean(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for name, values in (
                    ("level1_abnormal", predictions["level1_probabilities"][:, 1]),
                    ("crackle", predictions["attribute_probabilities"][:, 0]),
                    ("wheeze", predictions["attribute_probabilities"][:, 1]),
                )
            },
        },
        "native_temporal_IE_CAS_DAS_reproduction": False,
        "event_localization_claim": False,
        "normal_detection_claim": False,
    }
    return scored, metrics


def _parse_kauh_target(sample: Sample) -> dict[str, object]:
    patient_id, filter_mode, rest = _parse_kauh_identity(Path(sample.audio_path))
    fields = [value.strip() for value in rest.split(",")]
    diagnosis, raw_sound, location, age, gender = fields
    if raw_sound not in KAUH_RAW_LABELS:
        raise RuntimeError(f"unknown KAUH raw sound label: {raw_sound}")
    if raw_sound in KAUH_COMPATIBLE:
        flat4, label = KAUH_COMPATIBLE[raw_sound]
        return {
            "patient_id": patient_id,
            "filter_mode": filter_mode,
            "raw_sound": raw_sound,
            "raw_diagnosis": diagnosis,
            "location": location,
            "age": age,
            "gender": gender,
            "compatible": True,
            "mapping_status": "source_defined_shared_overlay",
            "level1_target": int(flat4 != 0),
            "flat4_target": int(flat4),
            "overlay_label": label,
        }
    return {
        "patient_id": patient_id,
        "filter_mode": filter_mode,
        "raw_sound": raw_sound,
        "raw_diagnosis": diagnosis,
        "location": location,
        "age": age,
        "gender": gender,
        "compatible": False,
        "mapping_status": "raw_unresolved_not_in_shared_overlay",
        "level1_target": -1,
        "flat4_target": -1,
        "overlay_label": "unresolved",
    }


def _kauh_scored_predictions(
    predictions: Mapping[str, np.ndarray],
    samples: Sequence[Sample],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    targets = [_parse_kauh_target(sample) for sample in samples]
    scored = {
        **predictions,
        "raw_ground_truth": np.asarray([row["raw_sound"] for row in targets]),
        "raw_diagnosis": np.asarray([row["raw_diagnosis"] for row in targets]),
        "locations": np.asarray([row["location"] for row in targets]),
        "ages": np.asarray([row["age"] for row in targets]),
        "genders": np.asarray([row["gender"] for row in targets]),
        "mapping_status": np.asarray([row["mapping_status"] for row in targets]),
        "overlay_labels": np.asarray([row["overlay_label"] for row in targets]),
        "compatible": np.asarray([row["compatible"] for row in targets], dtype=bool),
        "level1_targets": np.asarray([row["level1_target"] for row in targets], dtype=np.int64),
        "flat4_targets": np.asarray([row["flat4_target"] for row in targets], dtype=np.int64),
    }
    return scored, targets


def _binary_metric(target: np.ndarray, prediction: np.ndarray, task: str) -> dict[str, object]:
    result = native_metrics(target.astype(np.int64), prediction.astype(np.int64), ("normal", "abnormal"))
    result.update({"task": task, "official_score": result["average_score"]})
    return result


def _flat4_metric(target: np.ndarray, prediction: np.ndarray, task: str) -> dict[str, object]:
    result = native_metrics(target.astype(np.int64), prediction.astype(np.int64), ICBHI_LABELS)
    result.update({"task": task, "official_score": result["average_score"]})
    return result


def _patient_predictions(
    scored: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    patient_ids = sorted(set(scored["patient_ids"].tolist()), key=lambda value: int(str(value).lstrip("P")))
    fields: dict[str, list[object]] = {
        "patient_ids": [],
        "raw_ground_truth": [],
        "compatible": [],
        "level1_probabilities": [],
        "attribute_probabilities": [],
        "level1_predictions": [],
        "flat4_predictions": [],
    }
    for patient_id in patient_ids:
        indices = np.flatnonzero(scored["patient_ids"] == patient_id)
        level1_prob = scored["level1_probabilities"][indices].mean(axis=0)
        attribute_prob = scored["attribute_probabilities"][indices].mean(axis=0)
        level1_prediction = int(level1_prob.argmax())
        flat4_prediction = int(
            decode_icbhi_hierarchical_flat4(
                np.asarray([level1_prediction]),
                attribute_prob.reshape(1, 2),
                thresholds,
            )[0]
        )
        fields["patient_ids"].append(patient_id)
        fields["raw_ground_truth"].append(str(scored["raw_ground_truth"][indices[0]]))
        fields["compatible"].append(bool(scored["compatible"][indices].all()))
        fields["level1_probabilities"].append(level1_prob)
        fields["attribute_probabilities"].append(attribute_prob)
        fields["level1_predictions"].append(level1_prediction)
        fields["flat4_predictions"].append(flat4_prediction)
    patient_arrays = {
        "patient_ids": np.asarray(fields["patient_ids"]),
        "raw_ground_truth": np.asarray(fields["raw_ground_truth"]),
        "compatible": np.asarray(fields["compatible"], dtype=bool),
        "level1_probabilities": np.asarray(fields["level1_probabilities"], dtype=np.float32),
        "attribute_probabilities": np.asarray(fields["attribute_probabilities"], dtype=np.float32),
        "level1_predictions": np.asarray(fields["level1_predictions"], dtype=np.int64),
        "flat4_predictions": np.asarray(fields["flat4_predictions"], dtype=np.int64),
    }
    return patient_arrays, np.asarray(patient_ids)


def _kauh_metrics(
    scored: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    compatible = scored["compatible"]
    record_level1 = _binary_metric(
        scored["level1_targets"][compatible],
        scored["level1_predictions"][compatible],
        "KAUH compatible recording-level Level1 binary",
    )
    record_flat4 = _flat4_metric(
        scored["flat4_targets"][compatible],
        scored["flat4_predictions"][compatible],
        "KAUH compatible recording-level shared-overlay flat4",
    )
    patient_scored, _ = _patient_predictions(scored, thresholds)
    patient_compatible = patient_scored["compatible"]
    patient_target_flat4 = np.asarray(
        [KAUH_COMPATIBLE[str(value)][0] for value in patient_scored["raw_ground_truth"][patient_compatible]],
        dtype=np.int64,
    )
    patient_target_level1 = (patient_target_flat4 != 0).astype(np.int64)
    patient_level1 = _binary_metric(
        patient_target_level1,
        patient_scored["level1_predictions"][patient_compatible],
        "KAUH compatible patient-level Level1 binary after B/D/E probability mean",
    )
    patient_flat4 = _flat4_metric(
        patient_target_flat4,
        patient_scored["flat4_predictions"][patient_compatible],
        "KAUH compatible patient-level shared-overlay flat4 after B/D/E probability mean",
    )
    raw_support = Counter(scored["raw_ground_truth"].tolist())
    unresolved = {
        raw: {
            "recording_support": int(raw_support[raw]),
            "prediction_distribution_level1": {
                "normal": int(scored["level1_predictions"][scored["raw_ground_truth"] == raw].tolist().count(0)),
                "abnormal": int(scored["level1_predictions"][scored["raw_ground_truth"] == raw].tolist().count(1)),
            },
            "status": "HOLD_raw_unresolved_not_scored",
        }
        for raw in ("Crep", "Bronchial", "I C B")
    }
    per_filter = {}
    for filter_mode in ("B", "D", "E"):
        mask = (scored["filter_modes"] == filter_mode) & compatible
        per_filter[filter_mode] = {
            "recordings": int((scored["filter_modes"] == filter_mode).sum()),
            "compatible_recordings": int(mask.sum()),
            "raw_sound_support": dict(
                sorted(
                    Counter(
                        scored["raw_ground_truth"][scored["filter_modes"] == filter_mode].tolist()
                    ).items()
                )
            ),
            "level1_binary": _binary_metric(
                scored["level1_targets"][mask],
                scored["level1_predictions"][mask],
                f"KAUH {filter_mode}-filter compatible Level1 binary",
            ),
            "flat4": _flat4_metric(
                scored["flat4_targets"][mask],
                scored["flat4_predictions"][mask],
                f"KAUH {filter_mode}-filter compatible shared-overlay flat4",
            ),
        }
    metrics = {
        "status": "kauh_external_diagnostic_complete",
        "evidence_label": EVIDENCE_LABEL,
        "selected_epoch": 19,
        "threshold_source": "fixed JH2 selected validation thresholds; no KAUH tuning",
        "shared_attribute_thresholds": dict(thresholds),
        "recording_support": 336,
        "patient_support": 112,
        "filter_support": {filter_mode: 112 for filter_mode in ("B", "D", "E")},
        "raw_sound_support_recordings": dict(sorted(raw_support.items())),
        "compatible_raw_sounds": sorted(KAUH_COMPATIBLE),
        "compatible_recording_support": int(compatible.sum()),
        "unresolved_raw_sounds": unresolved,
        "recording_level": {
            "level1_binary": record_level1,
            "flat4": record_flat4,
        },
        "patient_level_after_BDE_probability_mean": {
            "level1_binary": patient_level1,
            "flat4": patient_flat4,
        },
        "per_filter_BDE_audit": per_filter,
        "prediction_distribution_all_recordings": {
            "level1": {
                "normal": int((scored["level1_predictions"] == 0).sum()),
                "abnormal": int((scored["level1_predictions"] == 1).sum()),
            },
            "flat4": {
                label: int((scored["flat4_predictions"] == index).sum())
                for index, label in enumerate(ICBHI_LABELS)
            },
        },
        "native_raw9_reproduction": False,
        "compatible_overlay_only": True,
        "unsupported_labels_are_not_mapped": True,
    }
    return metrics, patient_scored


def _config_payload(repo_root: Path, output_dir: Path, jh2: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "fixed_checkpoint_external_transfer_inference",
        "evidence_label": EVIDENCE_LABEL,
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "checkpoint": str(jh2["checkpoint"]),
        "backbone_checkpoint_for_model_construction": str(jh2["backbone_checkpoint"]),
        "selected_epoch": jh2["selected_epoch"],
        "frozen_shared_thresholds": jh2["thresholds"],
        "threshold_source": jh2["threshold_source"],
        "model": "BEATs iter3+ AS2M full fine-tuned PAFA-JH2 hierarchy",
        "device": "mps",
        "precision": "FP32",
        "input": "mono 16 kHz; JH2 5 s repeat-pad/front-truncate for KAUH; HF three continuous non-overlapping 5 s windows over 15 s source-test recording",
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "icbhi_sprsound_official_test_accessed": False,
        "external_sources": {
            "hf": str(repo_root / HF_TEST_RELATIVE),
            "kauh": str(repo_root / KAUH_RELATIVE),
        },
        "test_accessed_datasets": ["hf_lung_source_test", "kauh_v3_all_filtered"],
        "hf_policy": {
            "source_split": "test only",
            "windows": "exactly [0,5], [5,10], [10,15] seconds",
            "compatible_attributes": {"D": "Crackle", "Wheeze": "Wheeze"},
            "unsupported": ["Rhonchi", "Stridor", "I", "E", "CAS", "DAS"],
        },
        "kauh_policy": {
            "recordings": 336,
            "patients": 112,
            "filters": ["B", "D", "E"],
            "compatible_overlay": {
                "N": "Normal",
                "E W": "Wheeze",
                "I E W": "Wheeze",
                "C": "Crackle",
                "I C": "Crackle",
                "I C E W": "Both",
            },
            "unresolved": ["Crep", "Bronchial", "I C B"],
        },
        "label_free_written_before_external_targets": True,
    }


def run(repo_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite external evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jh2 = _fixed_jh2(repo_root)
    config = _base_config(repo_root, jh2, output_dir)
    _write_json(output_dir / "config.json", _config_payload(repo_root, output_dir, jh2))
    _write_json(
        output_dir / "protocol_summary.json",
        {
            "evidence_label": EVIDENCE_LABEL,
            "checkpoint_source": str(jh2["checkpoint"]),
            "backbone_checkpoint_for_model_construction": str(jh2["backbone_checkpoint"]),
            "selected_epoch": 19,
            "frozen_thresholds": jh2["thresholds"],
            "training": False,
            "selection": False,
            "test_accessed": ["HF source-test", "KAUH all filtered recordings"],
            "not_accessed": ["ICBHI official test", "SPRSound inter test"],
            "claim_boundary": "external transfer diagnostic; no HF/KAUH native reproduction claim",
        },
    )
    kauh_samples = _load_kauh_samples(repo_root)
    hf_samples = _load_hf_test_samples(repo_root)
    device = torch.device("mps")
    model, _ = _build_components(config, device)
    checkpoint = torch.load(jh2["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    thresholds = jh2["thresholds"]

    kauh_waveforms = _prepare_waveforms(kauh_samples, config)
    kauh_label_free = _predict_kauh(
        model,
        kauh_samples,
        kauh_waveforms,
        config,
        device,
        thresholds,
    )
    _save_predictions(output_dir / "kauh_predictions_label_free.npz", kauh_label_free)
    hf_label_free = _predict_hf(model, hf_samples, config, device)
    _save_predictions(output_dir / "hf_predictions_label_free.npz", hf_label_free)

    # External annotation loading begins only after both label-free artifacts exist.
    hf_annotations = _parse_hf_annotations(hf_samples)
    hf_scored, hf_metrics = _hf_scored_predictions(hf_label_free, hf_annotations, thresholds)
    _save_predictions(output_dir / "hf_predictions_scored.npz", hf_scored)
    kauh_scored, _ = _kauh_scored_predictions(kauh_label_free, kauh_samples, thresholds)
    kauh_metrics, patient_scored = _kauh_metrics(kauh_scored, thresholds)
    _save_predictions(output_dir / "kauh_predictions_scored.npz", kauh_scored)
    _save_predictions(output_dir / "kauh_patient_predictions_scored.npz", patient_scored)
    _write_json(
        output_dir / "hf_metrics.json",
        {
            **hf_metrics,
            "checkpoint_source": str(jh2["checkpoint"]),
            "label_free_predictions": str(output_dir / "hf_predictions_label_free.npz"),
            "scored_predictions": str(output_dir / "hf_predictions_scored.npz"),
        },
    )
    _write_json(
        output_dir / "kauh_metrics.json",
        {
            **kauh_metrics,
            "checkpoint_source": str(jh2["checkpoint"]),
            "label_free_predictions": str(output_dir / "kauh_predictions_label_free.npz"),
            "scored_predictions": str(output_dir / "kauh_predictions_scored.npz"),
            "patient_predictions": str(output_dir / "kauh_patient_predictions_scored.npz"),
        },
    )
    summary = {
        "status": "fixed_checkpoint_external_transfer_diagnostic_complete",
        "evidence_label": EVIDENCE_LABEL,
        "checkpoint_source": str(jh2["checkpoint"]),
        "selected_epoch": 19,
        "frozen_shared_thresholds": thresholds,
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "test_accessed_datasets": ["hf_lung_source_test", "kauh_v3_all_filtered"],
        "icbhi_sprsound_official_test_accessed": False,
        "hf_metrics_path": str(output_dir / "hf_metrics.json"),
        "kauh_metrics_path": str(output_dir / "kauh_metrics.json"),
        "label_free_written_before_external_targets": True,
        "claim_boundary": "external transfer diagnostic; not HF/KAUH native reproduction, not JH2 reselection, not a paper claim",
        "outputs": {
            "hf_label_free": str(output_dir / "hf_predictions_label_free.npz"),
            "hf_scored": str(output_dir / "hf_predictions_scored.npz"),
            "kauh_label_free": str(output_dir / "kauh_predictions_label_free.npz"),
            "kauh_scored": str(output_dir / "kauh_predictions_scored.npz"),
            "kauh_patient_scored": str(output_dir / "kauh_patient_predictions_scored.npz"),
        },
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / OUTPUT_RELATIVE
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "evidence_label": EVIDENCE_LABEL, "output_dir": str(output_dir)}, indent=2))
        return
    print(json.dumps(run(args.repo_root, output_dir), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
