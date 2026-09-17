"""Cache alignment and Table 1 target construction without feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from baseline.four_dataset_frozen_encoder.data import Sample


KAUH_COMPATIBLE = {"N": 0, "E W": 1, "I E W": 1, "C": 1, "I C": 1, "I C E W": 1}


@dataclass(frozen=True)
class TaskArrays:
    sample_ids: np.ndarray
    group_ids: np.ndarray
    embeddings: np.ndarray
    targets: np.ndarray


def load_aligned_embeddings(
    cache_path: Path, samples: Sequence[Sample]
) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as archive:
        cache_ids = archive["sample_ids"].astype(str)
        embeddings = archive["embeddings"].astype(np.float32, copy=False)
    index = {sample_id: position for position, sample_id in enumerate(cache_ids.tolist())}
    return np.stack([embeddings[index[sample.sample_id]] for sample in samples])


def task_arrays(
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    *,
    task: str,
    partition: str,
    terminal_spr_targets: dict[str, dict[str, int]] | None = None,
) -> TaskArrays:
    if task == "kauh_binary":
        return _kauh_patient_arrays(samples, embeddings, partition)
    selected = [
        index
        for index, sample in enumerate(samples)
        if sample.partition == partition
        and _target(sample, task, terminal_spr_targets) is not None
    ]
    return TaskArrays(
        sample_ids=np.asarray([samples[index].sample_id for index in selected]),
        group_ids=np.asarray([samples[index].group_id for index in selected]),
        embeddings=embeddings[selected],
        targets=np.asarray(
            [
                _target(samples[index], task, terminal_spr_targets)
                for index in selected
            ]
        ),
    )


def _target(
    sample: Sample,
    task: str,
    terminal_spr_targets: dict[str, dict[str, int]] | None = None,
) -> int | None:
    if task == "icbhi_flat4":
        return int(sample.targets[task]) if task in sample.targets else None
    if task == "spr_binary":
        if task in sample.targets:
            return int(sample.targets[task])
        if terminal_spr_targets and sample.sample_id in terminal_spr_targets:
            return int(terminal_spr_targets[sample.sample_id][task])
        return None
    if task == "hf_cas":
        labels = sample.targets.get("hf_adventitious_presence")
        if labels is None:
            return None
        # Source order is D, Wheeze, Rhonchi, Stridor.  D-only is the eligible negative.
        return int(any(int(value) for value in labels[1:]))
    raise ValueError(task)


def _kauh_patient_arrays(
    samples: Sequence[Sample], embeddings: np.ndarray, partition: str
) -> TaskArrays:
    by_group: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.dataset == "kauh" and sample.partition == partition:
            raw = str(sample.metadata["raw_sound"])
            if raw in KAUH_COMPATIBLE:
                by_group.setdefault(sample.group_id, []).append(index)
    group_ids = sorted(by_group, key=lambda value: int(value.removeprefix("P")))
    values = []
    targets = []
    for group_id in group_ids:
        indices = by_group[group_id]
        values.append(embeddings[indices].mean(axis=0))
        raw = str(samples[indices[0]].metadata["raw_sound"])
        targets.append(KAUH_COMPATIBLE[raw])
    return TaskArrays(
        sample_ids=np.asarray(group_ids),
        group_ids=np.asarray(group_ids),
        embeddings=np.asarray(values, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
    )
