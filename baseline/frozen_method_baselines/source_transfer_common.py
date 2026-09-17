"""Shared data and metrics for ICBHI-source fixed-transfer baselines.

The module deliberately avoids sklearn and legacy identity gates.  Audio
dependencies are imported only when an authorized run actually loads audio.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .pcmcl_source_transfer import build_pcmcl_single_5s


ICBHI_LABELS = ("normal", "crackle", "wheeze", "both")
SPR_LABELS = (
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
)
KAUH_COMPATIBLE = {
    "N": 0,
    "E W": 1,
    "I E W": 1,
    "C": 1,
    "I C": 1,
    "I C E W": 1,
}
KAUH_EXCLUDED = {"Crep", "Bronchial", "I C B"}


def load_beats_transfer_class(source: Path) -> type[torch.nn.Module]:
    """Load the author BEATs wrapper without importing unrelated backbones."""

    sys.path.insert(0, str(source.resolve()))
    spec = importlib.util.spec_from_file_location(
        "_acoustic_author_beats", source / "models" / "beats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BEATsTransferLearningModel


@dataclass(frozen=True)
class AudioUnit:
    sample_id: str
    dataset: str
    audio_path: Path
    group_id: str
    start_s: float | None = None
    end_s: float | None = None
    target: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def update_early_stopping(
    *,
    score: float,
    best_score: float,
    no_improvement_epochs: int,
    patience: int,
    min_delta: float,
    eligible: bool = True,
    enabled: bool = True,
) -> dict[str, object]:
    """Update strict best selection; apply patience only when enabled."""

    improved = eligible and score > best_score + min_delta
    next_best = score if improved else best_score
    next_count = 0 if improved else no_improvement_epochs + 1
    return {
        "improved": improved,
        "best_score": next_best,
        "no_improvement_epochs": next_count,
        "stopped_early": enabled and next_count >= patience,
    }


def prepare_run_directory(
    result_dir: Path,
    config: Mapping[str, object],
    resume: Path | None,
) -> None:
    config_path = result_dir / "config.json"
    if resume is None:
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(
                f"fresh run refuses non-empty result directory: {result_dir}"
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
        return
    expected = (result_dir / "last_checkpoint.pt").resolve()
    if resume.resolve() != expected or not expected.is_file():
        raise ValueError("resume must use this seed directory's last_checkpoint.pt")
    if not config_path.is_file() or json.loads(config_path.read_text()) != dict(config):
        raise RuntimeError("resume config does not match the existing seed config")
    summary = result_dir / "run_summary.json"
    if summary.is_file() and json.loads(summary.read_text()).get("status") in {
        "complete_test_selected_source_transfer",
        "complete_joint_native_union",
    }:
        raise RuntimeError("completed seed must be skipped, not resumed")


def load_icbhi_units(repo_root: Path, official_split: str) -> list[AudioUnit]:
    manifest = repo_root / "dataset/processed/manifests/icbhi_2017_cycles.csv"
    rows: list[AudioUnit] = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["official_split"] != official_split:
                continue
            rows.append(
                AudioUnit(
                    sample_id=f"icbhi:{row['cycle_id']}",
                    dataset="icbhi",
                    audio_path=(repo_root / row["audio_path"]).resolve(),
                    group_id=row["patient_id"],
                    start_s=float(row["cycle_start_s"]),
                    end_s=float(row["cycle_end_s"]),
                    target=ICBHI_LABELS.index(row["native_four_class_label"]),
                    metadata={
                        "recording_id": row["recording_id"],
                        "internal_partition": row.get("partition"),
                    },
                )
            )
    return sorted(rows, key=lambda row: row.sample_id)


def _spr_root(repo_root: Path) -> Path:
    values = sorted(
        (repo_root / "dataset/raw/sprsound/source_original").glob("SPRSound-*")
    )
    if len(values) != 1:
        raise RuntimeError("expected one pinned SPRSound source directory")
    return values[0] / "BioCAS2022"


def load_spr_inter_units(repo_root: Path, *, include_targets: bool) -> list[AudioUnit]:
    root = _spr_root(repo_root)
    json_dir = root / "test2022_json/inter_test_json"
    wav_dir = root / "test2022_wav"
    rows: list[AudioUnit] = []
    for annotation in sorted(json_dir.glob("*.json")):
        payload = json.loads(annotation.read_text())
        recording_id = annotation.stem
        wav = (wav_dir / f"{recording_id}.wav").resolve()
        patient_id = recording_id.split("_", 1)[0]
        for index, event in enumerate(payload["event_annotation"]):
            raw = str(event["type"])
            rows.append(
                AudioUnit(
                    sample_id=f"spr:inter:{recording_id}:event_{index:03d}",
                    dataset="sprsound",
                    audio_path=wav,
                    group_id=patient_id,
                    start_s=int(event["start"]) / 1000,
                    end_s=int(event["end"]) / 1000,
                    target=(int(raw != "Normal") if include_targets else None),
                    metadata={
                        "annotation_path": str(annotation.resolve()),
                        "event_index": index,
                        **({"raw_label": raw} if include_targets else {}),
                    },
                )
            )
    return sorted(rows, key=lambda row: row.sample_id)


def load_spr_train_units(
    repo_root: Path,
    partition_by_group: Mapping[str, str],
) -> list[AudioUnit]:
    root = _spr_root(repo_root)
    json_dir = root / "train2022_json"
    wav_dir = root / "train2022_wav"
    rows: list[AudioUnit] = []
    for annotation in sorted(json_dir.glob("*.json")):
        payload = json.loads(annotation.read_text())
        recording_id = annotation.stem
        wav = (wav_dir / f"{recording_id}.wav").resolve()
        patient_id = recording_id.split("_", 1)[0]
        events = payload["event_annotation"]
        if not events:
            continue
        partition = partition_by_group[patient_id]
        for index, event in enumerate(events):
            raw = str(event["type"])
            rows.append(
                AudioUnit(
                    sample_id=f"spr:train:{recording_id}:event_{index:03d}",
                    dataset="sprsound",
                    audio_path=wav,
                    group_id=patient_id,
                    start_s=int(event["start"]) / 1000,
                    end_s=int(event["end"]) / 1000,
                    target=SPR_LABELS.index(raw),
                    metadata={
                        "raw_label": raw,
                        "internal_partition": partition,
                    },
                )
            )
    return sorted(rows, key=lambda row: row.sample_id)


def load_spr_targets(units: Sequence[AudioUnit]) -> dict[str, dict[str, object]]:
    cache: dict[str, Mapping[str, object]] = {}
    output = {}
    for unit in units:
        path = str(unit.metadata["annotation_path"])
        if path not in cache:
            cache[path] = json.loads(Path(path).read_text())
        event = cache[path]["event_annotation"][int(unit.metadata["event_index"])]
        raw = str(event["type"])
        output[unit.sample_id] = {"raw_label": raw, "binary": int(raw != "Normal")}
    return output


def load_hf_test_units(repo_root: Path) -> list[AudioUnit]:
    root = repo_root / "dataset/raw/hf_lung_v1/source_original/test"
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        label = wav.with_name(f"{wav.stem}_label.txt")
        rows.append(
            AudioUnit(
                sample_id=f"hf:test:{wav.stem}",
                dataset="hf_lung",
                audio_path=wav.resolve(),
                group_id=wav.stem,
                metadata={"label_path": str(label.resolve())},
            )
        )
    return rows


def _hms(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_hf_cas_targets(units: Sequence[AudioUnit]) -> dict[str, dict[str, object]]:
    output = {}
    for unit in units:
        tokens = []
        intervals = []
        for line in Path(str(unit.metadata["label_path"])).read_text().splitlines():
            if not line.strip():
                continue
            token, start, end = line.split()
            tokens.append(token)
            intervals.append((token, _hms(start), _hms(end)))
        eligible = any(value in {"D", "Wheeze", "Rhonchi", "Stridor"} for value in tokens)
        if eligible:
            output[unit.sample_id] = {
                "positive": int(any(value in {"Wheeze", "Rhonchi", "Stridor"} for value in tokens)),
                "intervals": intervals,
            }
    return output


def load_kauh_units(repo_root: Path) -> list[AudioUnit]:
    root = repo_root / "dataset/raw/kauh_fraiwan/source_original/audio_files"
    pattern = re.compile(r"^([BDE])P(\d+)_(.*)$")
    rows = []
    for wav in sorted(root.glob("*.wav")):
        match = pattern.match(wav.stem)
        if not match:
            continue
        view, patient, rest = match.groups()
        fields = [value.strip() for value in rest.split(",")]
        sound = fields[1]
        rows.append(
            AudioUnit(
                sample_id=f"kauh:P{patient}:{view}",
                dataset="kauh",
                audio_path=wav.resolve(),
                group_id=f"P{patient}",
                target=KAUH_COMPATIBLE.get(sound),
                metadata={"view": view, "raw_sound": sound},
            )
        )
    return sorted(rows, key=lambda row: (int(row.group_id[1:]), str(row.metadata["view"])))


def load_waveform(unit: AudioUnit, *, sample_rate: int = 16_000) -> torch.Tensor:
    import torchaudio

    waveform, source_rate = torchaudio.load(unit.audio_path)
    waveform = waveform.to(torch.float32).mean(dim=0)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if unit.start_s is not None and unit.end_s is not None:
        waveform = waveform[
            int(round(unit.start_s * sample_rate)) : int(round(unit.end_s * sample_rate))
        ]
    return waveform


def load_single_5s(unit: AudioUnit) -> torch.Tensor:
    return build_pcmcl_single_5s(load_waveform(unit))


def load_hf_windows(unit: AudioUnit) -> torch.Tensor:
    waveform = load_waveform(unit)
    target = 15 * 16_000
    if waveform.numel() != target:
        raise RuntimeError("HF source-test recording is not exactly 15 s after resampling")
    return waveform.reshape(3, 5 * 16_000)


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for truth, guess in zip(target.astype(int), prediction.astype(int)):
        matrix[truth, guess] += 1
    return matrix


def _recall(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix), denominator, out=np.zeros(len(matrix), dtype=float), where=denominator > 0
    )


def _f1(matrix: np.ndarray) -> np.ndarray:
    tp = np.diag(matrix).astype(float)
    precision_den = matrix.sum(axis=0)
    recall_den = matrix.sum(axis=1)
    precision = np.divide(tp, precision_den, out=np.zeros_like(tp), where=precision_den > 0)
    recall = np.divide(tp, recall_den, out=np.zeros_like(tp), where=recall_den > 0)
    return np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )


def icbhi_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(target, prediction, 4)
    recalls = _recall(matrix)
    specificity = float(recalls[0])
    abnormal_support = matrix[1:].sum()
    sensitivity = float(np.diag(matrix)[1:].sum() / abnormal_support)
    return {
        "rows": int(len(target)),
        "confusion": matrix.tolist(),
        "per_class_recall": dict(zip(ICBHI_LABELS, recalls.tolist())),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": (specificity + sensitivity) / 2,
        "macro_f1": float(_f1(matrix).mean()),
        "uar": float(recalls.mean()),
    }


def spr_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(target, prediction, 2)
    recalls = _recall(matrix)
    specificity, sensitivity = float(recalls[0]), float(recalls[1])
    average = (specificity + sensitivity) / 2
    harmonic = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    return {
        "rows": int(len(target)),
        "confusion": matrix.tolist(),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "average_score": average,
        "harmonic_score": harmonic,
        "official_score": (average + harmonic) / 2,
        "macro_f1": float(_f1(matrix).mean()),
        "uar": float(recalls.mean()),
    }


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(target, prediction, 2)
    recalls = _recall(matrix)
    return {
        "rows": int(len(target)),
        "confusion": matrix.tolist(),
        "balanced_accuracy": float(recalls.mean()),
        "normal_recall": float(recalls[0]),
        "abnormal_recall": float(recalls[1]),
    }


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(int)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positives = target == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aggregate_seed_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def flatten(prefix: str, value: object, output: dict[str, float]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), child, output)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            output[prefix] = float(value)

    flattened = []
    for row in rows:
        current: dict[str, float] = {}
        flatten("", row, current)
        flattened.append(current)
    common = set.intersection(*(set(row) for row in flattened)) if flattened else set()
    summary = {}
    for key in sorted(common):
        values = np.asarray([row[key] for row in flattened], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "values": values.tolist(),
        }
    return summary
