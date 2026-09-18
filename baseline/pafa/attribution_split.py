"""Frozen source partitions for the approved LSAA attribution runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from acoustic.evaluation.sprsound_inter import resolve_biocas_root
from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.shared_encoder_native_heads.protocol import SPR_LABELS, spr_event_rows
from baseline.pafa.beats_ce_reproduction import read_official_cycles
from baseline.pafa.joint_hierarchy import _icbhi_sample


SPLIT_CONTRACT_RELATIVE = Path("baseline/pafa/lsaa_attribution_source_splits.json")


def _partition_samples(
    samples: Sequence[Sample], validation_groups: set[str]
) -> list[Sample]:
    return [
        replace(
            sample,
            partition=(
                "validation" if str(sample.group_id) in validation_groups else "subtrain"
            ),
        )
        for sample in samples
    ]


def load_attribution_source_samples(config: object) -> list[Sample]:
    contract_path = Path(config.repo_root) / SPLIT_CONTRACT_RELATIVE
    contract = json.loads(contract_path.read_text())
    seed_contract = contract["seeds"][str(config.seed)]["datasets"]

    icbhi_rows = read_official_cycles(
        Path(config.icbhi_audio_dir), Path(config.author_repo), ("train",)
    )
    icbhi = [
        _icbhi_sample(row, Path(config.icbhi_audio_dir), "subtrain")
        for row in icbhi_rows
    ]
    spr_root = resolve_biocas_root(Path(config.repo_root) / "dataset/raw/sprsound")
    spr_rows = spr_event_rows(
        spr_root / "train2022_json", spr_root / "train2022_wav", "train", True
    )
    sprsound = [
        Sample(
            sample_id=f"spr:{row['event_id']}",
            dataset="sprsound",
            partition="subtrain",
            group_id=str(row["patient_id"]),
            audio_path=str(row["audio_path"]),
            crop_start_s=float(row["start_ms"]) / 1000,
            crop_end_s=float(row["end_ms"]) / 1000,
            targets={
                "spr_binary": int(str(row["raw_label"]) != "Normal"),
                "spr_seven": SPR_LABELS.index(str(row["raw_label"])),
            },
            metadata={
                "event_id": str(row["event_id"]),
                "recording_id": str(row["recording_id"]),
                "patient_id": str(row["patient_id"]),
                "source_partition": "train",
                "event_index": int(row["event_index"]),
                "annotation_path": str(row["annotation_path"]),
                "raw_label": str(row["raw_label"]),
            },
        )
        for row in spr_rows
    ]

    samples = []
    for dataset, rows in (("icbhi", icbhi), ("sprsound", sprsound)):
        frozen = _partition_samples(
            rows, set(seed_contract[dataset]["validation_groups"])
        )
        expected = seed_contract[dataset]["expected_support"]
        for partition in ("subtrain", "validation"):
            selected = [row for row in frozen if row.partition == partition]
            observed = {
                "units": len(selected),
                "groups": len({str(row.group_id) for row in selected}),
            }
            if observed != expected[partition]:
                raise RuntimeError(
                    f"frozen {dataset} {partition} support changed: "
                    f"{observed} != {expected[partition]}"
                )
        samples.extend(frozen)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise RuntimeError("duplicate sample ID in frozen attribution source split")
    return sorted(samples, key=lambda sample: sample.sample_id)


def split_contract_path(repo_root: Path) -> Path:
    return repo_root / SPLIT_CONTRACT_RELATIVE


def frozen_split_reference(
    samples: Sequence[Sample], seed: int
) -> dict[str, object]:
    return {
        "schema": "lsaa_attribution_applied_split_v1",
        "seed": seed,
        "contract": str(SPLIT_CONTRACT_RELATIVE),
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
                                str(row.group_id)
                                for row in samples
                                if row.dataset == dataset
                                and row.partition == partition
                            }
                        ),
                    }
                    for partition in ("subtrain", "validation")
                },
                "records": [
                    {
                        "sample_id": row.sample_id,
                        "group_id": str(row.group_id),
                        "partition": row.partition,
                    }
                    for row in samples
                    if row.dataset == dataset
                ],
            }
            for dataset in ("icbhi", "sprsound")
        },
    }
