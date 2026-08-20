"""Canonical sample construction for the four-dataset frozen-encoder demo."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from baseline.shared_encoder_native_heads.protocol import (
    EXPECTED_SPR_SUPPORT,
    ICBHI_LABELS,
    SPRSOUND_COMMIT,
    SPR_LABELS,
    SPR_VALIDATION_SEED,
    load_icbhi_rows,
    spr_event_rows,
)
from acoustic.evaluation.sprsound_inter import resolve_biocas_root


SEED = 20260728
KAUH_LABELS = ["N", "E W", "I E W", "C", "I C", "I C E W", "Crep", "Bronchial", "I C B"]
HF_PHASE_LABELS = ["I", "E"]
HF_ADVENTITIOUS_LABELS = ["D", "Wheeze", "Rhonchi", "Stridor"]
EXPECTED_SPR_INTER_ID_SHA256 = (
    "81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b"
)
EXPECTED_HF_SOURCE_PROXY_COUNTS = {"train": 118, "test": 39}
EXPECTED_HF_PARTITION_PROXY_COUNTS = {
    "subtrain": 94,
    "validation": 24,
    "test": 39,
}
EXPECTED_HF_ASSIGNMENT_SHA256 = (
    "33387aa62ebcb8adbc1fba626e6d27f27a3121d3a686cda6fc075a7da106943e"
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    partition: str
    group_id: str
    audio_path: str
    crop_start_s: float | None
    crop_end_s: float | None
    targets: dict[str, object]
    metadata: dict[str, object]


def _sha(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _parse_hms(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _hf_date_proxy(name: str) -> str:
    if name.startswith("steth_"):
        value = name.split("_")[1]
        if not re.fullmatch(r"\d{8}", value):
            raise RuntimeError(f"cannot parse HF date proxy: {name}")
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    match = re.match(r"trunc_(\d{4}-\d{2}-\d{2})-", name)
    if not match:
        raise RuntimeError(f"cannot parse HF date proxy: {name}")
    return match.group(1)


def hf_assignment_sha256(samples: list[Sample]) -> str:
    """Reproduce the accepted HF assignment digest without calling proxies patients."""
    rows = []
    for sample in samples:
        if sample.dataset != "hf_lung":
            continue
        source_split = str(sample.metadata["source_split"])
        stem = sample.sample_id.split(":", 2)[2]
        compact_proxy = str(sample.metadata["date_proxy"]).replace("-", "")
        rows.append(
            f"{source_split}:{stem}\t{sample.partition}\t{compact_proxy}"
        )
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _load_hf(
    dataset_root: Path, *, include_checksums: bool = True
) -> tuple[list[Sample], dict[str, object]]:
    root = dataset_root / "hf_lung_v1/source_original"
    raw: list[dict[str, object]] = []
    for wav in sorted(root.rglob("*.wav")):
        source_split = "train" if "/train/" in str(wav) else "test"
        label_path = wav.with_name(wav.stem + "_label.txt")
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        intervals = []
        for line in label_path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError(f"invalid HF label row: {label_path}: {line}")
            token = parts[0]
            if token not in {*HF_PHASE_LABELS, *HF_ADVENTITIOUS_LABELS}:
                raise RuntimeError(f"unknown HF token: {token}")
            start, end = _parse_hms(parts[1]), _parse_hms(parts[2])
            if not (0 <= start < end <= 15.001):
                raise RuntimeError(f"invalid HF interval: {label_path}: {line}")
            intervals.append((token, start, end))
        raw.append(
            {
                "wav": wav.resolve(),
                "stem": wav.stem,
                "source_split": source_split,
                "date_proxy": _hf_date_proxy(wav.name),
                "intervals": intervals,
            }
        )
    if len(raw) != 9765 or Counter(row["source_split"] for row in raw) != {"train": 7809, "test": 1956}:
        raise RuntimeError("HF recording/split count gate failed")

    train_indices = [i for i, row in enumerate(raw) if row["source_split"] == "train"]
    groups = np.asarray([raw[i]["date_proxy"] for i in train_indices])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    sub_rel, val_rel = next(splitter.split(train_indices, groups=groups))
    validation = {train_indices[i] for i in val_rel}
    samples: list[Sample] = []
    for index, row in enumerate(raw):
        partition = "test" if row["source_split"] == "test" else ("validation" if index in validation else "subtrain")
        tokens = [str(interval[0]) for interval in row["intervals"]]
        targets: dict[str, object] = {}
        if any(token in HF_PHASE_LABELS for token in tokens):
            targets["hf_phase_presence"] = [int(label in tokens) for label in HF_PHASE_LABELS]
        if any(token in HF_ADVENTITIOUS_LABELS for token in tokens):
            targets["hf_adventitious_presence"] = [
                int(label in tokens) for label in HF_ADVENTITIOUS_LABELS
            ]
        samples.append(
            Sample(
                sample_id=f"hf:{row['source_split']}:{row['stem']}",
                dataset="hf_lung",
                partition=partition,
                group_id=f"date_proxy:{row['date_proxy']}",
                audio_path=str(row["wav"]),
                crop_start_s=None,
                crop_end_s=None,
                targets=targets,
                metadata={
                    "source_split": row["source_split"],
                    "date_proxy": row["date_proxy"],
                    "patient_id": None,
                    "raw_intervals": [
                        [str(token), float(start), float(end)]
                        for token, start, end in row["intervals"]
                    ],
                    "recording_state": "observed" if row["intervals"] else "empty",
                    "annotation_state": "observed_positive_presence" if targets else "not_annotated",
                },
            )
        )
    sub_groups = {sample.group_id for sample in samples if sample.partition == "subtrain"}
    val_groups = {sample.group_id for sample in samples if sample.partition == "validation"}
    test_groups = {sample.group_id for sample in samples if sample.partition == "test"}
    source_groups = {
        source_split: {
            str(row["date_proxy"])
            for row in raw
            if row["source_split"] == source_split
        }
        for source_split in ("train", "test")
    }
    if sub_groups & val_groups or (sub_groups | val_groups) & test_groups:
        raise RuntimeError("HF date-proxy split overlap")
    source_proxy_counts = {
        source_split: len(values)
        for source_split, values in source_groups.items()
    }
    partition_proxy_counts = {
        "subtrain": len(sub_groups),
        "validation": len(val_groups),
        "test": len(test_groups),
    }
    if (
        source_proxy_counts != EXPECTED_HF_SOURCE_PROXY_COUNTS
        or partition_proxy_counts != EXPECTED_HF_PARTITION_PROXY_COUNTS
        or source_groups["train"] & source_groups["test"]
    ):
        raise RuntimeError("HF canonical date-proxy assignment gate failed")
    receipt = {
        "rows": len(samples),
        "partition": dict(sorted(Counter(sample.partition for sample in samples).items())),
        "source_split": {"train": 7809, "test": 1956},
        "source_date_proxy_counts": source_proxy_counts,
        "date_proxy_counts": partition_proxy_counts,
        "date_proxy_format": "YYYY-MM-DD",
        "date_proxy_identity": "deidentified grouping proxy; not patient_id",
        "group_overlap": 0,
        "empty_annotation_files": sum(not sample.targets for sample in samples),
        "task_eligible": {
            task: dict(
                sorted(
                    Counter(sample.partition for sample in samples if task in sample.targets).items()
                )
            )
            for task in ("hf_phase_presence", "hf_adventitious_presence")
        },
        "task_positive_support": {
            task: {
                partition: [
                    sum(
                        int(sample.targets[task][label_index])
                        for sample in samples
                        if sample.partition == partition and task in sample.targets
                    )
                    for label_index in range(
                        len(HF_PHASE_LABELS)
                        if task == "hf_phase_presence"
                        else len(HF_ADVENTITIOUS_LABELS)
                    )
                ]
                for partition in ("subtrain", "validation", "test")
            }
            for task in ("hf_phase_presence", "hf_adventitious_presence")
        },
        "label_policy": (
            "recording-level positive-presence diagnostic within each source-annotated "
            "label pool; a recording is eligible only when at least one label in that "
            "pool is observed; peer-label absence is negative only inside that eligible "
            "pool; unannotated gaps and empty annotation files are never normal/negative"
        ),
        "input_alignment": (
            "all 15 s recordings are represented by three non-overlapping author-length "
            "5 s windows and mean-pooled after the frozen encoder"
        ),
    }
    if include_checksums:
        assignment_sha256 = hf_assignment_sha256(samples)
        if assignment_sha256 != EXPECTED_HF_ASSIGNMENT_SHA256:
            raise RuntimeError("HF canonical assignment SHA256 gate failed")
        receipt["assignment_sha256"] = assignment_sha256
        receipt["assignment_digest_serialization"] = (
            "raw sorted(Path.rglob('*.wav')) order; "
            "source_split:stem<TAB>partition<TAB>YYYYMMDD<LF>"
        )
    return samples, receipt


def _kauh_patient_rows(audio_dir: Path) -> list[dict[str, object]]:
    patient_rows: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"^([BDE])P(\d+)_(.*)$")
    for wav in sorted(audio_dir.glob("*.wav")):
        match = pattern.match(wav.stem)
        if not match:
            raise RuntimeError(f"invalid KAUH filename: {wav.name}")
        filter_mode, patient_id, rest = match.groups()
        fields = rest.split(",")
        if len(fields) != 5:
            raise RuntimeError(f"invalid KAUH fields: {wav.name}")
        diagnosis, sound, location, age, gender = [value.strip() for value in fields]
        if sound not in KAUH_LABELS:
            raise RuntimeError(f"unknown KAUH raw sound label: {sound}")
        row = patient_rows.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "sound": sound,
                "diagnosis": diagnosis,
                "files": [],
            },
        )
        if row["sound"] != sound:
            raise RuntimeError(f"KAUH sibling label mismatch for P{patient_id}")
        row["files"].append((filter_mode, wav.resolve(), location, age, gender))
    rows = sorted(patient_rows.values(), key=lambda row: int(str(row["patient_id"])))
    if len(rows) != 112 or any(sorted(mode for mode, *_ in row["files"]) != ["B", "D", "E"] for row in rows):
        raise RuntimeError("KAUH 112-patient B/D/E gate failed")
    return rows


def _load_kauh(
    dataset_root: Path, outer_fold: int, *, include_checksums: bool = True
) -> tuple[list[Sample], dict[str, object]]:
    patients = _kauh_patient_rows(dataset_root / "kauh_fraiwan/source_original/audio_files")
    labels = np.asarray([str(row["sound"]) for row in patients])
    groups = np.asarray([str(row["patient_id"]) for row in patients])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = list(outer.split(np.arange(len(patients)), labels, groups))
    dev_idx, test_idx = folds[outer_fold]
    dev_labels, dev_groups = labels[dev_idx], groups[dev_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + outer_fold)
    with warnings.catch_warnings(record=True) as inner_caught:
        warnings.simplefilter("always")
        sub_rel, val_rel = next(
            inner.split(np.arange(len(dev_idx)), dev_labels, dev_groups)
        )
    partition_by_patient = {groups[index]: "test" for index in test_idx}
    partition_by_patient.update({dev_groups[index]: "subtrain" for index in sub_rel})
    partition_by_patient.update({dev_groups[index]: "validation" for index in val_rel})
    samples: list[Sample] = []
    for row in patients:
        patient_id = str(row["patient_id"])
        partition = partition_by_patient[patient_id]
        label_index = KAUH_LABELS.index(str(row["sound"]))
        for mode, wav, location, age, gender in row["files"]:
            samples.append(
                Sample(
                    sample_id=f"kauh:P{patient_id}:{mode}",
                    dataset="kauh",
                    partition=partition,
                    group_id=f"P{patient_id}",
                    audio_path=str(wav),
                    crop_start_s=None,
                    crop_end_s=None,
                    targets={"kauh_raw9": label_index},
                    metadata={
                        "patient_id": patient_id,
                        "filter_mode": mode,
                        "raw_sound": row["sound"],
                        "raw_diagnosis": row["diagnosis"],
                        "location": location,
                        "age": age,
                        "gender": gender,
                    },
                )
            )
    by_partition = {
        partition: {sample.group_id for sample in samples if sample.partition == partition}
        for partition in ("subtrain", "validation", "test")
    }
    if any(by_partition[a] & by_partition[b] for a, b in (("subtrain", "validation"), ("subtrain", "test"), ("validation", "test"))):
        raise RuntimeError("KAUH patient leakage")
    receipt = {
        "rows": len(samples),
        "patients": 112,
        "outer_fold": outer_fold,
        "partition": dict(sorted(Counter(sample.partition for sample in samples).items())),
        "partition_patients": {key: len(value) for key, value in by_partition.items()},
        "patient_overlap": 0,
        "raw_label_support_recordings": dict(
            sorted(Counter(str(sample.metadata["raw_sound"]) for sample in samples).items())
        ),
        "raw_label_support_patients": dict(
            sorted(Counter(str(row["sound"]) for row in patients).items())
        ),
        "test_support_recordings": dict(
            sorted(
                Counter(
                    str(sample.metadata["raw_sound"])
                    for sample in samples
                    if sample.partition == "test"
                ).items()
            )
        ),
        "stratification_caveat": (
            "best-effort StratifiedGroupKFold; at least one raw class has only one "
            "patient, so every fold cannot contain every class; report aggregate "
            "five-fold patient-grouped OOF in the fixed nine-label space"
        ),
        "suppressed_expected_split_warnings": [
            str(item.message) for item in [*caught, *inner_caught]
        ],
    }
    if include_checksums:
        receipt["outer_test_ordered_id_sha256"] = _sha(
            [sample.sample_id for sample in samples if sample.partition == "test"]
        )
    return samples, receipt


def _load_icbhi(
    dataset_root: Path, *, include_checksums: bool = True
) -> tuple[list[Sample], dict[str, object]]:
    rows, receipt = load_icbhi_rows(
        dataset_root / "icbhi_2017", include_checksums=include_checksums
    )
    samples = [
        Sample(
            sample_id=f"icbhi:{row['cycle_id']}",
            dataset="icbhi",
            partition=str(row["partition"]),
            group_id=str(row["patient_id"]),
            audio_path=str(row["audio_path"]),
            crop_start_s=float(row["cycle_start_s"]),
            crop_end_s=float(row["cycle_end_s"]),
            targets={"icbhi_flat4": ICBHI_LABELS.index(str(row["native_four_class_label"]))},
            metadata={
                "cycle_id": str(row["cycle_id"]),
                "recording_id": str(row["recording_id"]),
                "patient_id": str(row["patient_id"]),
                "official_split": str(row["official_split"]),
            },
        )
        for row in rows
    ]
    return samples, receipt


def _load_spr(
    dataset_root: Path, *, include_checksums: bool = True
) -> tuple[list[Sample], dict[str, object]]:
    root = resolve_biocas_root(dataset_root / "sprsound")
    if SPRSOUND_COMMIT not in str(root):
        raise RuntimeError(f"SPRSound root is not pinned to {SPRSOUND_COMMIT}")
    train = spr_event_rows(
        root / "train2022_json", root / "train2022_wav", "train", True
    )
    inter = spr_event_rows(
        root / "test2022_json/inter_test_json",
        root / "test2022_wav",
        "inter",
        False,
    )
    if (
        len(train) != 6656
        or len(inter) != 1429
        or dict(sorted(Counter(str(row["raw_label"]) for row in train).items()))
        != EXPECTED_SPR_SUPPORT["train"]
    ):
        raise RuntimeError("SPRSound label-free row/support/ID gate failed")
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SPR_VALIDATION_SEED
    )
    labels = np.asarray([str(row["raw_label"]) for row in train])
    groups = np.asarray([str(row["patient_id"]) for row in train])
    subtrain_index, validation_index = next(
        splitter.split(np.arange(len(train)), labels, groups)
    )
    validation_set = set(validation_index.tolist())
    for index, row in enumerate(train):
        row["partition"] = "validation" if index in validation_set else "subtrain"
    subtrain_patients = {str(train[index]["patient_id"]) for index in subtrain_index}
    validation_patients = {
        str(train[index]["patient_id"]) for index in validation_index
    }
    inter_patients = {str(row["patient_id"]) for row in inter}
    if (
        (len(subtrain_index), len(validation_index)) != (5219, 1437)
        or (len(subtrain_patients), len(validation_patients)) != (194, 49)
        or subtrain_patients & validation_patients
        or (subtrain_patients | validation_patients) & inter_patients
    ):
        raise RuntimeError("SPRSound grouped split gate failed")
    rows = train + inter
    receipt = {
        "source_commit": SPRSOUND_COMMIT,
        "prediction_unit": "official BioCAS2022 event",
        "train_events": len(train),
        "subtrain_events": len(subtrain_index),
        "validation_events": len(validation_index),
        "inter_events": len(inter),
        "subtrain_patients": len(subtrain_patients),
        "validation_patients": len(validation_patients),
        "subtrain_validation_patient_overlap": 0,
        "train_inter_patient_overlap": 0,
        "train_support": EXPECTED_SPR_SUPPORT["train"],
        "validation": (
            "StratifiedGroupKFold fold 0 inside official train; patient_id; "
            f"seed {SPR_VALIDATION_SEED}"
        ),
        "test_policy": "inter primary; intra excluded from this matrix",
        "test_manifest_label_free": True,
        "terminal_scoring_labels_loaded_after_prediction_write": True,
    }
    samples = []
    for row in rows:
        raw = str(row["raw_label"]) if "raw_label" in row else None
        if raw is not None and raw not in SPR_LABELS:
            raise RuntimeError(f"unknown SPR label: {raw}")
        is_test = row["partition"] == "inter"
        partition = "test" if is_test else str(row["partition"])
        targets = (
            {}
            if is_test
            else {
                "spr_binary": int(raw != "Normal"),
                "spr_seven": SPR_LABELS.index(str(raw)),
            }
        )
        metadata = {
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
            "patient_id": str(row["patient_id"]),
            "source_partition": str(row["partition"]),
            "event_index": int(row["event_index"]),
            "annotation_path": str(row["annotation_path"]),
        }
        if not is_test:
            metadata["raw_label"] = raw
        samples.append(
            Sample(
                sample_id=f"spr:{row['event_id']}",
                dataset="sprsound",
                partition=partition,
                group_id=str(row["patient_id"]),
                audio_path=str(row["audio_path"]),
                crop_start_s=float(row["start_ms"]) / 1000,
                crop_end_s=float(row["end_ms"]) / 1000,
                targets=targets,
                metadata=metadata,
            )
        )
    receipt = {
        **receipt,
        "main_matrix_rows": len(samples),
        "inter_targets_in_training_manifest": 0,
        "intra_excluded_from_main_matrix": True,
    }
    if include_checksums:
        if _sha([str(row["event_id"]) for row in inter]) != EXPECTED_SPR_INTER_ID_SHA256:
            raise RuntimeError("SPRSound label-free row/support/ID gate failed")
        receipt["inter_ordered_id_sha256"] = EXPECTED_SPR_INTER_ID_SHA256
    return samples, receipt


def load_terminal_spr_test_targets(
    samples: list[Sample], *, include_checksums: bool = True
) -> dict[str, dict[str, int]]:
    """Load SPR inter labels only after label-free predictions are durably written."""
    targets: dict[str, dict[str, int]] = {}
    payload_cache: dict[str, dict[str, object]] = {}
    for sample in samples:
        if sample.dataset != "sprsound" or sample.partition != "test":
            continue
        if sample.targets or "raw_label" in sample.metadata:
            raise RuntimeError("SPRSound test label leaked into inference sample")
        annotation_path = str(sample.metadata["annotation_path"])
        if annotation_path not in payload_cache:
            payload_cache[annotation_path] = json.loads(
                Path(annotation_path).read_text()
            )
        event = payload_cache[annotation_path]["event_annotation"][
            int(sample.metadata["event_index"])
        ]
        raw = str(event["type"])
        if raw not in SPR_LABELS:
            raise RuntimeError(f"unknown terminal SPR label: {raw}")
        targets[sample.sample_id] = {
            "spr_binary": int(raw != "Normal"),
            "spr_seven": SPR_LABELS.index(raw),
        }
    expected_ids = sorted(
        sample.sample_id
        for sample in samples
        if sample.dataset == "sprsound" and sample.partition == "test"
    )
    if sorted(targets) != expected_ids:
        raise RuntimeError("terminal SPR test target identity gate failed")
    if include_checksums and len(expected_ids) == 1429:
        event_ids = sorted(
            str(sample.metadata["event_id"])
            for sample in samples
            if sample.dataset == "sprsound" and sample.partition == "test"
        )
        if _sha(event_ids) != EXPECTED_SPR_INTER_ID_SHA256:
            raise RuntimeError("full terminal SPR test target digest failed")
    return targets


def build_samples(
    dataset_root: Path,
    kauh_outer_fold: int = 0,
    *,
    include_checksums: bool = True,
) -> tuple[list[Sample], dict[str, object]]:
    loaders = {
        "icbhi": lambda: _load_icbhi(
            dataset_root, include_checksums=include_checksums
        ),
        "sprsound": lambda: _load_spr(
            dataset_root, include_checksums=include_checksums
        ),
        "hf_lung": lambda: _load_hf(
            dataset_root, include_checksums=include_checksums
        ),
        "kauh": lambda: _load_kauh(
            dataset_root, kauh_outer_fold, include_checksums=include_checksums
        ),
    }
    samples: list[Sample] = []
    receipts: dict[str, object] = {}
    for name, loader in loaders.items():
        current, receipt = loader()
        samples.extend(current)
        receipts[name] = receipt
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate four-dataset sample ID")
    receipt = {
        "status": "four_dataset_sample_contract_passed",
        "rows": len(samples),
        "unique_ids": len(ids),
        "dataset_rows": dict(sorted(Counter(sample.dataset for sample in samples).items())),
        "datasets": receipts,
    }
    if include_checksums:
        receipt["ordered_id_sha256"] = _sha(ids)
    return samples, receipt


def sample_to_row(sample: Sample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "partition": sample.partition,
        "group_id": sample.group_id,
        "audio_path": sample.audio_path,
        "crop_start_s": sample.crop_start_s,
        "crop_end_s": sample.crop_end_s,
        "targets_json": json.dumps(sample.targets, sort_keys=True),
        "metadata_json": json.dumps(sample.metadata, sort_keys=True),
    }
