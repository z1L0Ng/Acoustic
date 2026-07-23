"""Frozen SPRSound inter target contract shared without coupling model environments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


FOUR_LABELS = ["normal", "crackle", "wheeze", "both"]
BINARY_LABELS = ["normal", "abnormal"]
RAW_TO_FOUR = {
    "Normal": "normal",
    "Fine Crackle": "crackle",
    "Coarse Crackle": "crackle",
    "Wheeze": "wheeze",
    "Wheeze+Crackle": "both",
}
EXPECTED_EVENTS = 1429
EXPECTED_ID_SHA256 = "81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b"
EXPECTED_RAW_COUNTS = {
    "Coarse Crackle": 3,
    "Fine Crackle": 80,
    "Normal": 1040,
    "Wheeze": 305,
    "Wheeze+Crackle": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def resolve_biocas_root(dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    if (root / "test2022_json" / "inter_test_json").is_dir():
        return root
    matches = list(root.glob("source_original/*/BioCAS2022"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not resolve one BioCAS2022 root under {root}: {matches}")
    return matches[0].resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_label_free_inter_manifest(dataset_root: Path) -> list[dict[str, object]]:
    root = resolve_biocas_root(dataset_root)
    json_dir = root / "test2022_json" / "inter_test_json"
    wav_dir = root / "test2022_wav"
    rows: list[dict[str, object]] = []
    for annotation_path in sorted(json_dir.glob("*.json")):
        recording_id = annotation_path.stem
        audio_path = wav_dir / f"{recording_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        payload = json.loads(annotation_path.read_text())
        for event_index, event in enumerate(payload.get("event_annotation", [])):
            start_ms, end_ms = int(event["start"]), int(event["end"])
            if start_ms < 0 or end_ms <= start_ms:
                raise RuntimeError(f"invalid boundary: {annotation_path} {event}")
            rows.append(
                {
                    "event_id": f"inter:{recording_id}:event_{event_index:03d}",
                    "source_partition": "BioCAS2022_inter_subject",
                    "recording_id": recording_id,
                    "patient_id": recording_id.split("_", 1)[0],
                    "event_index": event_index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "audio_path": str(audio_path.resolve()),
                    "annotation_path": str(annotation_path.resolve()),
                }
            )
    rows.sort(key=lambda row: str(row["event_id"]))
    ids = [str(row["event_id"]) for row in rows]
    if len(rows) != EXPECTED_EVENTS or len(set(ids)) != EXPECTED_EVENTS:
        raise RuntimeError("SPRSound inter event coverage mismatch")
    if id_sha256(ids) != EXPECTED_ID_SHA256:
        raise RuntimeError("SPRSound inter event ID set/order differs from verified Patch-Mix B0")
    return rows


def load_scoring_labels(manifest_rows: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        payload = json.loads(Path(str(row["annotation_path"])).read_text())
        raw = str(payload["event_annotation"][int(row["event_index"])]["type"])
        if raw not in RAW_TO_FOUR:
            raise RuntimeError(f"unsupported inter label in frozen narrow-four target: {raw}")
        labels[str(row["event_id"])] = {
            "raw_event_label": raw,
            "binary_broad_label": "normal" if raw == "Normal" else "abnormal",
            "narrow_four_label": RAW_TO_FOUR[raw],
            "narrow_four_mapping_status": "included",
        }
    counts = dict(sorted(Counter(row["raw_event_label"] for row in labels.values()).items()))
    if len(labels) != EXPECTED_EVENTS or counts != EXPECTED_RAW_COUNTS:
        raise RuntimeError(f"frozen scoring-label receipt mismatch: {counts}")
    return labels


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    predicted = np.bincount(y_pred, minlength=len(labels))
    payload: dict[str, object] = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": (specificity + sensitivity) / 2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
            }
            for index, label in enumerate(labels)
        },
        "collapse_diagnostics": {
            "predicted_counts": {label: int(predicted[index]) for index, label in enumerate(labels)},
            "single_class_collapse": bool(np.count_nonzero(predicted) == 1),
        },
    }
    if labels == FOUR_LABELS:
        payload["both_recall"] = float(recall[3])
        payload["both_support_warning"] = "support=1; no minority conclusion"
    return payload, matrix


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def _read_confusion(path: Path, labels: list[str]) -> np.ndarray:
    rows = read_csv(path)
    if [row["true/pred"] for row in rows] != labels:
        raise RuntimeError(f"confusion label order mismatch: {path}")
    return np.asarray([[int(row[label]) for label in labels] for row in rows], dtype=np.int64)


def write_confusion(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    rows = [
        {"true/pred": label, **{pred: int(value) for pred, value in zip(labels, values)}}
        for label, values in zip(labels, matrix)
    ]
    write_csv(path, rows)


def finalize_full_result(
    result_root: Path,
    manifest_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    method_receipt: dict[str, object],
) -> dict[str, object]:
    """Join labels only after all logits exist; retain original label-free artifact."""
    prediction_rows.sort(key=lambda row: str(row["event_id"]))
    ids = [str(row["event_id"]) for row in prediction_rows]
    if len(ids) != EXPECTED_EVENTS or len(set(ids)) != EXPECTED_EVENTS or id_sha256(ids) != EXPECTED_ID_SHA256:
        raise RuntimeError("prediction ID gate failed")
    write_csv(result_root / "predictions_label_free.csv", prediction_rows)
    scoring = load_scoring_labels(manifest_rows)
    merged = []
    for row in prediction_rows:
        merged.append({**row, **scoring[str(row["event_id"])]})
    write_csv(result_root / "predictions.csv", merged)

    y_four = np.asarray([FOUR_LABELS.index(str(row["narrow_four_label"])) for row in merged])
    p_four = np.asarray([int(row["pred_narrow_four_index"]) for row in merged])
    y_binary = np.asarray([BINARY_LABELS.index(str(row["binary_broad_label"])) for row in merged])
    p_binary = np.asarray([int(row["pred_binary_broad_index"]) for row in merged])
    four_metrics, four_matrix = metrics(y_four, p_four, FOUR_LABELS)
    binary_metrics, binary_matrix = metrics(y_binary, p_binary, BINARY_LABELS)
    floor_four = np.zeros_like(y_four)
    floor_binary = np.zeros_like(y_binary)
    floor_four_metrics, floor_four_matrix = metrics(y_four, floor_four, FOUR_LABELS)
    floor_binary_metrics, floor_binary_matrix = metrics(y_binary, floor_binary, BINARY_LABELS)
    floor_rows = [
        {
            "event_id": row["event_id"],
            "raw_event_label": row["raw_event_label"],
            "binary_broad_label": row["binary_broad_label"],
            "narrow_four_label": row["narrow_four_label"],
            "floor_pred_binary_broad_label": "normal",
            "floor_pred_binary_broad_index": 0,
            "floor_pred_narrow_four_label": "normal",
            "floor_pred_narrow_four_index": 0,
        }
        for row in merged
    ]
    write_csv(result_root / "all_normal_floor_predictions.csv", floor_rows)
    for name, labels, matrix in (
        ("binary_broad", BINARY_LABELS, binary_matrix),
        ("narrow_four", FOUR_LABELS, four_matrix),
        ("floor_binary_broad", BINARY_LABELS, floor_binary_matrix),
        ("floor_narrow_four", FOUR_LABELS, floor_four_matrix),
    ):
        write_confusion(result_root / f"{name}_confusion.csv", labels, matrix)
    payload = {
        "status": "completed_pending_independent_verification",
        "claim_boundary": "frozen source checkpoint exploratory transfer; no target training/tuning/selection",
        "method": method_receipt,
        "target": {
            "dataset": "SPRSound BioCAS2022",
            "partition": "official inter-subject",
            "unit": "event",
            "events": EXPECTED_EVENTS,
            "event_id_sha256": EXPECTED_ID_SHA256,
            "raw_label_counts": EXPECTED_RAW_COUNTS,
            "both_support_warning": "support=1; no minority conclusion",
        },
        "label_isolation": "predictions_label_free.csv written before scoring-label load",
        "binary_decision_rule": "normal iff four-class argmax is normal; otherwise abnormal",
        "transfer": {"binary_broad": binary_metrics, "narrow_four": four_metrics},
        "all_normal_floor": {
            "binary_broad": floor_binary_metrics,
            "narrow_four": floor_four_metrics,
        },
    }
    write_json(result_root / "metrics.json", payload)
    return payload


def verify_smoke_result(result_root: Path, dataset_root: Path, expected_method: str) -> dict[str, object]:
    """Verify the fixed eight-event smoke without loading any target labels."""
    expected = build_label_free_inter_manifest(dataset_root)[:8]
    expected_ids = [str(row["event_id"]) for row in expected]
    rows = read_csv(result_root / "predictions_label_free.csv")
    ids = [row["event_id"] for row in rows]
    if ids != expected_ids or len(set(ids)) != 8:
        raise RuntimeError("smoke event-ID/order mismatch")
    forbidden = {"raw_event_label", "binary_broad_label", "narrow_four_label"}
    if forbidden & set(rows[0]):
        raise RuntimeError("smoke label-free artifact contains target labels")
    if any((result_root / name).exists() for name in ("metrics.json", "predictions.csv")):
        raise RuntimeError("smoke must not join labels or calculate target metrics")
    max_probability_error = 0.0
    for row in rows:
        logits = np.asarray([float(row[f"logit_{label}"]) for label in FOUR_LABELS])
        probabilities = np.asarray([float(row[f"prob_{label}"]) for label in FOUR_LABELS])
        binary = np.asarray(
            [float(row["prob_binary_normal"]), float(row["prob_binary_abnormal"])]
        )
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise RuntimeError("non-finite smoke output")
        max_probability_error = max(
            max_probability_error,
            abs(float(probabilities.sum()) - 1.0),
            float(np.max(np.abs(probabilities - _softmax(logits)))),
            float(np.max(np.abs(binary - np.asarray([probabilities[0], probabilities[1:].sum()])))),
        )
        prediction = int(row["pred_narrow_four_index"])
        if prediction != int(np.argmax(probabilities)):
            raise RuntimeError("smoke four-class argmax mismatch")
        if int(row["pred_binary_broad_index"]) != (0 if prediction == 0 else 1):
            raise RuntimeError("smoke binary-collapse mismatch")
    if max_probability_error > 1e-5:
        raise RuntimeError(f"smoke probability gate failed: {max_probability_error}")
    run_receipt = json.loads((result_root / "smoke_receipt.json").read_text())
    if run_receipt["method_id"] != expected_method or int(run_receipt["events"]) != 8:
        raise RuntimeError("smoke receipt method/coverage mismatch")
    receipt = {
        "status": "verified_label_free_smoke",
        "method_id": expected_method,
        "events": 8,
        "event_id_sha256": id_sha256(ids),
        "target_labels_accessed": False,
        "target_metrics_calculated": False,
        "probability_max_abs_error": max_probability_error,
    }
    write_json(result_root / "smoke_verification.json", receipt)
    return receipt


def verify_full_result(result_root: Path, dataset_root: Path, expected_method: str) -> dict[str, object]:
    manifest = build_label_free_inter_manifest(dataset_root)
    label_free = read_csv(result_root / "predictions_label_free.csv")
    scored = read_csv(result_root / "predictions.csv")
    floor = read_csv(result_root / "all_normal_floor_predictions.csv")
    ids = [row["event_id"] for row in label_free]
    scored_ids = [row["event_id"] for row in scored]
    floor_ids = [row["event_id"] for row in floor]
    if (
        len(ids) != EXPECTED_EVENTS
        or len(set(ids)) != EXPECTED_EVENTS
        or id_sha256(ids) != EXPECTED_ID_SHA256
        or scored_ids != ids
        or floor_ids != ids
    ):
        raise RuntimeError("verification coverage gate failed")
    forbidden = {"raw_event_label", "binary_broad_label", "narrow_four_label"}
    if forbidden & set(label_free[0]):
        raise RuntimeError("label-free artifact contains target labels")
    scoring = load_scoring_labels(manifest)
    max_prob_error = 0.0
    max_softmax_error = 0.0
    max_binary_prob_error = 0.0
    for row in scored:
        expected = scoring[row["event_id"]]
        if any(row[key] != value for key, value in expected.items()):
            raise RuntimeError(f"scoring lineage mismatch: {row['event_id']}")
        probs = np.asarray([float(row[f"prob_{label}"]) for label in FOUR_LABELS])
        logits = np.asarray([float(row[f"logit_{label}"]) for label in FOUR_LABELS])
        if not np.isfinite(probs).all() or not np.isfinite(logits).all():
            raise RuntimeError("non-finite model output")
        max_prob_error = max(max_prob_error, abs(float(probs.sum()) - 1.0))
        max_softmax_error = max(max_softmax_error, float(np.max(np.abs(probs - _softmax(logits)))))
        if int(row["pred_narrow_four_index"]) != int(np.argmax(probs)):
            raise RuntimeError("argmax mismatch")
        expected_binary = 0 if int(row["pred_narrow_four_index"]) == 0 else 1
        if int(row["pred_binary_broad_index"]) != expected_binary:
            raise RuntimeError("binary collapse mismatch")
        binary_probs = np.asarray(
            [float(row["prob_binary_normal"]), float(row["prob_binary_abnormal"])]
        )
        expected_binary_probs = np.asarray([probs[0], probs[1:].sum()])
        max_binary_prob_error = max(
            max_binary_prob_error,
            float(np.max(np.abs(binary_probs - expected_binary_probs))),
        )
    if max_prob_error > 1e-5 or max_softmax_error > 1e-5 or max_binary_prob_error > 1e-5:
        raise RuntimeError(
            "probability gate failed: "
            f"sum={max_prob_error} softmax={max_softmax_error} binary={max_binary_prob_error}"
        )

    stored = json.loads((result_root / "metrics.json").read_text())
    if stored["method"]["method_id"] != expected_method:
        raise RuntimeError("method identity mismatch")
    y_four = np.asarray([FOUR_LABELS.index(row["narrow_four_label"]) for row in scored])
    p_four = np.asarray([int(row["pred_narrow_four_index"]) for row in scored])
    y_binary = np.asarray([BINARY_LABELS.index(row["binary_broad_label"]) for row in scored])
    p_binary = np.asarray([int(row["pred_binary_broad_index"]) for row in scored])
    recomputed = {
        "binary_broad": metrics(y_binary, p_binary, BINARY_LABELS)[0],
        "narrow_four": metrics(y_four, p_four, FOUR_LABELS)[0],
    }
    floor_recomputed = {
        "binary_broad": metrics(y_binary, np.zeros_like(y_binary), BINARY_LABELS)[0],
        "narrow_four": metrics(y_four, np.zeros_like(y_four), FOUR_LABELS)[0],
    }
    keys = ["specificity", "sensitivity", "icbhi_score", "macro_f1", "weighted_f1", "uar"]
    differences = {
        task: max(abs(float(stored["transfer"][task][key]) - float(values[key])) for key in keys)
        for task, values in recomputed.items()
    }
    floor_differences = {
        task: max(
            abs(float(stored["all_normal_floor"][task][key]) - float(values[key]))
            for key in keys
        )
        for task, values in floor_recomputed.items()
    }
    if max(differences.values()) > 1e-12 or max(floor_differences.values()) > 1e-12:
        raise RuntimeError(f"metric mismatch: model={differences} floor={floor_differences}")

    expected_confusions = {
        "binary_broad": metrics(y_binary, p_binary, BINARY_LABELS)[1],
        "narrow_four": metrics(y_four, p_four, FOUR_LABELS)[1],
        "floor_binary_broad": metrics(y_binary, np.zeros_like(y_binary), BINARY_LABELS)[1],
        "floor_narrow_four": metrics(y_four, np.zeros_like(y_four), FOUR_LABELS)[1],
    }
    confusion_totals: dict[str, int] = {}
    for name, expected in expected_confusions.items():
        labels = BINARY_LABELS if "binary_broad" in name else FOUR_LABELS
        observed = _read_confusion(result_root / f"{name}_confusion.csv", labels)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"confusion matrix mismatch: {name}")
        confusion_totals[name] = int(observed.sum())
        if confusion_totals[name] != EXPECTED_EVENTS:
            raise RuntimeError(f"confusion total mismatch: {name}")

    for row in floor:
        if (
            row["floor_pred_binary_broad_label"] != "normal"
            or int(row["floor_pred_binary_broad_index"]) != 0
            or row["floor_pred_narrow_four_label"] != "normal"
            or int(row["floor_pred_narrow_four_index"]) != 0
        ):
            raise RuntimeError("all-normal floor prediction mismatch")
    receipt = {
        "status": "verified_frozen_source_checkpoint_exploratory_transfer",
        "method_id": expected_method,
        "events": EXPECTED_EVENTS,
        "unique_event_ids": len(set(ids)),
        "event_id_sha256": id_sha256(ids),
        "probability_sum_max_abs_error": max_prob_error,
        "softmax_max_abs_error": max_softmax_error,
        "binary_probability_max_abs_error": max_binary_prob_error,
        "metric_max_abs_difference": differences,
        "floor_metric_max_abs_difference": floor_differences,
        "confusion_totals": confusion_totals,
        "both_support_warning": "support=1; no minority conclusion",
        "degradation_claim_allowed_without_target_native_reference": False,
    }
    write_json(result_root / "verification.json", receipt)
    return receipt


def group_by_recording(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)
    return grouped
