"""Build receipted SPRSound train/validation/inter event manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .common import (
    EXPECTED_INTER_EVENTS,
    EXPECTED_INTER_ID_SHA256,
    RAW_LABELS,
    class_counts,
    id_sha256,
    resolve_biocas_root,
    task_label,
    update_run_manifest,
    validate_result_root,
    write_csv,
)


EXPECTED_TRAIN_EVENTS = 6656
EXPECTED_TRAIN_ANNOTATION_FILES = 1949
EXPECTED_TRAIN_ARCHIVE_PATIENTS = 251
EXPECTED_TRAIN_EVENT_RECORDINGS = 1772
EXPECTED_TRAIN_EVENT_PATIENTS = 243
EXPECTED_TRAIN_RAW = {
    "Coarse Crackle": 49,
    "Fine Crackle": 912,
    "Normal": 5159,
    "Rhonchi": 39,
    "Stridor": 15,
    "Wheeze": 452,
    "Wheeze+Crackle": 30,
}
EXPECTED_SUBTRAIN_EVENTS = 5219
EXPECTED_VALIDATION_EVENTS = 1437
EXPECTED_SUBTRAIN_PATIENTS = 194
EXPECTED_VALIDATION_PATIENTS = 49
EXPECTED_SUBTRAIN_RAW = {
    "Coarse Crackle": 46,
    "Fine Crackle": 593,
    "Normal": 4114,
    "Rhonchi": 34,
    "Stridor": 15,
    "Wheeze": 396,
    "Wheeze+Crackle": 21,
}
EXPECTED_VALIDATION_RAW = {
    "Coarse Crackle": 3,
    "Fine Crackle": 319,
    "Normal": 1045,
    "Rhonchi": 5,
    "Wheeze": 56,
    "Wheeze+Crackle": 9,
}


def event_rows(json_dir: Path, wav_dir: Path, partition: str, include_labels: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for annotation_path in sorted(json_dir.glob("*.json")):
        recording_id = annotation_path.stem
        audio_path = wav_dir / f"{recording_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        payload = json.loads(annotation_path.read_text())
        for index, event in enumerate(payload.get("event_annotation", [])):
            start_ms, end_ms = int(event["start"]), int(event["end"])
            if start_ms < 0 or end_ms <= start_ms:
                raise RuntimeError(f"invalid event boundary: {annotation_path} {event}")
            row: dict[str, object] = {
                "event_id": f"{partition}:{recording_id}:event_{index:03d}",
                "partition": partition,
                "recording_id": recording_id,
                "patient_id": recording_id.split("_", 1)[0],
                "event_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "audio_path": str(audio_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
            }
            if include_labels:
                raw = str(event["type"])
                if raw not in RAW_LABELS:
                    raise RuntimeError(f"unknown event label: {raw}")
                row["raw_label"] = raw
            rows.append(row)
    if len({str(row["event_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate event IDs in {partition}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = resolve_biocas_root(args.dataset_root)
    result_root = validate_result_root(args.result_root)
    output = result_root / "data"
    if output.exists():
        raise FileExistsError(f"data contract is immutable once written: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = event_rows(root / "train2022_json", root / "train2022_wav", "train", True)
    inter = event_rows(
        root / "test2022_json" / "inter_test_json", root / "test2022_wav", "inter", False
    )
    train_annotation_paths = sorted((root / "train2022_json").glob("*.json"))
    train_archive_patients = {
        path.stem.split("_", 1)[0] for path in train_annotation_paths
    }
    if (
        len(train) != EXPECTED_TRAIN_EVENTS
        or len(train_annotation_paths) != EXPECTED_TRAIN_ANNOTATION_FILES
        or len(train_archive_patients) != EXPECTED_TRAIN_ARCHIVE_PATIENTS
        or len({row["recording_id"] for row in train}) != EXPECTED_TRAIN_EVENT_RECORDINGS
        or len({row["patient_id"] for row in train}) != EXPECTED_TRAIN_EVENT_PATIENTS
        or class_counts(train, "raw_label") != EXPECTED_TRAIN_RAW
        or len(inter) != EXPECTED_INTER_EVENTS
    ):
        raise RuntimeError("SPRSound canonical count gate failed")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260722)
    y = np.asarray([str(row["raw_label"]) for row in train])
    groups = np.asarray([str(row["patient_id"]) for row in train])
    train_indices, validation_indices = next(splitter.split(np.arange(len(train)), y, groups))
    validation_set = set(validation_indices.tolist())
    for index, row in enumerate(train):
        row["inner_split"] = "validation" if index in validation_set else "subtrain"
    subtrain_patients = {str(train[index]["patient_id"]) for index in train_indices}
    validation_patients = {str(train[index]["patient_id"]) for index in validation_indices}
    if subtrain_patients & validation_patients:
        raise RuntimeError("patient leakage across inner split")
    subtrain_raw = dict(sorted(Counter(y[train_indices]).items()))
    validation_raw = dict(sorted(Counter(y[validation_indices]).items()))
    if (
        len(train_indices) != EXPECTED_SUBTRAIN_EVENTS
        or len(validation_indices) != EXPECTED_VALIDATION_EVENTS
        or len(subtrain_patients) != EXPECTED_SUBTRAIN_PATIENTS
        or len(validation_patients) != EXPECTED_VALIDATION_PATIENTS
        or subtrain_raw != EXPECTED_SUBTRAIN_RAW
        or validation_raw != EXPECTED_VALIDATION_RAW
    ):
        raise RuntimeError("patient-grouped validation assignment drift")

    train.sort(key=lambda row: str(row["event_id"]))
    inter.sort(key=lambda row: str(row["event_id"]))
    inter_ids = [str(row["event_id"]) for row in inter]
    if id_sha256(inter_ids) != EXPECTED_INTER_ID_SHA256:
        raise RuntimeError("inter event IDs differ from the frozen B0 target")
    write_csv(output / "train_events.csv", train)
    write_csv(output / "inter_events_label_free.csv", inter)
    assignments = [
        {
            "patient_id": patient,
            "inner_split": "validation" if patient in validation_patients else "subtrain",
            "split_seed": 20260722,
            "outer_fold": 0,
        }
        for patient in sorted(subtrain_patients | validation_patients)
    ]
    write_csv(output / "patient_validation_assignment.csv", assignments)

    task_receipt: dict[str, object] = {}
    for task in ("binary_broad", "narrow_four"):
        task_rows: list[dict[str, object]] = []
        excluded = Counter()
        for row in train:
            mapped = task_label(str(row["raw_label"]), task)
            if mapped is None:
                excluded[str(row["raw_label"])] += 1
                continue
            task_rows.append({**row, "task": task, "task_label": mapped})
        write_csv(output / f"train_{task}.csv", task_rows)
        task_receipt[task] = {
            "included": len(task_rows),
            "excluded": dict(sorted(excluded.items())),
            "subtrain": sum(row["inner_split"] == "subtrain" for row in task_rows),
            "validation": sum(row["inner_split"] == "validation" for row in task_rows),
            "subtrain_class_counts": class_counts(
                [row for row in task_rows if row["inner_split"] == "subtrain"], "task_label"
            ),
            "validation_class_counts": class_counts(
                [row for row in task_rows if row["inner_split"] == "validation"], "task_label"
            ),
        }

    receipt = {
        "status": "data_and_grouped_validation_verified",
        "biocas_root": str(root),
        "train": {
            "events": len(train),
            "archive_annotation_files": len(train_annotation_paths),
            "archive_patients": len(train_archive_patients),
            "event_bearing_recordings": len({row["recording_id"] for row in train}),
            "event_bearing_patients": len({row["patient_id"] for row in train}),
            "recordings_without_events": len(train_annotation_paths)
            - len({row["recording_id"] for row in train}),
            "raw_class_counts": class_counts(train, "raw_label"),
        },
        "validation": {
            "method": "StratifiedGroupKFold",
            "n_splits": 5,
            "fold": 0,
            "random_state": 20260722,
            "subtrain_events": len(train_indices),
            "validation_events": len(validation_indices),
            "subtrain_patients": len(subtrain_patients),
            "validation_patients": len(validation_patients),
            "patient_overlap": 0,
            "subtrain_raw_counts": subtrain_raw,
            "validation_raw_counts": validation_raw,
        },
        "inter": {
            "events": len(inter),
            "ordered_event_id_sha256": id_sha256(inter_ids),
            "manifest_is_label_free": True,
            "selection_use": "forbidden",
            "evaluation_use": "single final evaluation after best inner-validation checkpoint is fixed",
        },
        "tasks": task_receipt,
    }
    update_run_manifest(result_root, "data_protocol", receipt)
    print(
        "c0_data_ok train=6656 subtrain={} validation={} inter=1429 overlap=0".format(
            len(train_indices), len(validation_indices)
        )
    )


if __name__ == "__main__":
    main()
