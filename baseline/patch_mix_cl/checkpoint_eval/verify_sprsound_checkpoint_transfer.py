"""Independent verifier for Patch-Mix B0 and all-normal B0-floor."""

from __future__ import annotations

import argparse
import csv
import json
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rebuild_source_rows(dataset_root: Path) -> dict[str, dict[str, str]]:
    json_dir = dataset_root / "test2022_json" / "inter_test_json"
    rows = {}
    for json_path in sorted(json_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        for index, event in enumerate(payload.get("event_annotation", [])):
            event_id = f"inter:{json_path.stem}:event_{index:03d}"
            raw_label = str(event["type"])
            if raw_label not in RAW_TO_FOUR:
                raise RuntimeError(f"unexpected inter label: {raw_label}")
            rows[event_id] = {
                "recording_id": json_path.stem,
                "start_ms": str(int(event["start"])),
                "end_ms": str(int(event["end"])),
                "raw_event_label": raw_label,
                "binary_broad_label": "normal" if raw_label == "Normal" else "abnormal",
                "narrow_four_label": RAW_TO_FOUR[raw_label],
            }
    if len(rows) != 1429:
        raise RuntimeError(f"source JSON has {len(rows)} events, expected 1429")
    return rows


def recompute(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[dict, np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    metrics = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": (specificity + sensitivity) / 2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
    }
    if labels == FOUR_LABELS:
        metrics["both_recall"] = float(recall[3])
    classes = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {"metrics": metrics, "per_class": classes}, matrix


def compare_metrics(recomputed: dict, recorded: dict) -> float:
    differences = {
        key: abs(float(value) - float(recorded[key]))
        for key, value in recomputed["metrics"].items()
    }
    maximum = max(differences.values())
    if maximum > 1e-10:
        raise RuntimeError(f"metric mismatch: {differences}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    source = rebuild_source_rows(args.dataset_root)
    manifest = read_csv(args.result_root / "inference_manifest_label_free.csv")
    predictions = read_csv(args.result_root / "b0_predictions.csv")
    floor = read_csv(args.result_root / "b0_floor_predictions.csv")
    recorded = json.loads((args.result_root / "metrics.json").read_text())
    protocol = json.loads((args.result_root / "protocol.json").read_text())
    for name, rows in (("manifest", manifest), ("predictions", predictions), ("floor", floor)):
        ids = [row["event_id"] for row in rows]
        if len(ids) != 1429 or len(set(ids)) != 1429 or set(ids) != set(source):
            raise RuntimeError(f"{name} event-ID coverage failed")
    forbidden_manifest_fields = {
        "raw_event_label",
        "binary_broad_label",
        "narrow_four_label",
        "narrow_four_mapping_status",
    }
    if forbidden_manifest_fields.intersection(manifest[0]):
        raise RuntimeError("inference manifest contains target-label fields")

    for row in predictions:
        expected = source[row["event_id"]]
        for key, value in expected.items():
            if row[key] != value:
                raise RuntimeError(f"lineage/mapping mismatch for {row['event_id']}: {key}")
    logits = np.asarray(
        [[float(row[f"logit_{label}"]) for label in FOUR_LABELS] for row in predictions]
    )
    probabilities = np.asarray(
        [[float(row[f"prob_{label}"]) for label in FOUR_LABELS] for row in predictions]
    )
    binary_probabilities = np.asarray(
        [[float(row["prob_binary_normal"]), float(row["prob_binary_abnormal"])] for row in predictions]
    )
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or not np.isfinite(binary_probabilities).all():
        raise RuntimeError("non-finite logits/probabilities")
    probability_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
    binary_probability_error = float(np.max(np.abs(binary_probabilities.sum(axis=1) - 1)))
    aggregation_error = float(np.max(np.abs(binary_probabilities[:, 1] - probabilities[:, 1:].sum(axis=1))))
    if max(probability_error, binary_probability_error, aggregation_error) > 1e-5:
        raise RuntimeError("probability normalization/aggregation failed")
    pred_four = np.asarray([int(row["pred_narrow_four_index"]) for row in predictions])
    pred_binary = np.asarray([int(row["pred_binary_broad_index"]) for row in predictions])
    if not np.array_equal(pred_four, probabilities.argmax(axis=1)):
        raise RuntimeError("four-class argmax mismatch")
    if not np.array_equal(pred_binary, (pred_four != 0).astype(int)):
        raise RuntimeError("binary prediction is not the author-style collapsed four-class argmax")

    four_index = {label: index for index, label in enumerate(FOUR_LABELS)}
    y_four = np.asarray([four_index[row["narrow_four_label"]] for row in predictions])
    y_binary = np.asarray([0 if row["binary_broad_label"] == "normal" else 1 for row in predictions])
    b0_binary, b0_binary_matrix = recompute(y_binary, pred_binary, BINARY_LABELS)
    b0_four, b0_four_matrix = recompute(y_four, pred_four, FOUR_LABELS)
    floor_binary, floor_binary_matrix = recompute(y_binary, np.zeros(1429, dtype=int), BINARY_LABELS)
    floor_four, floor_four_matrix = recompute(y_four, np.zeros(1429, dtype=int), FOUR_LABELS)
    max_differences = {
        "b0_binary_broad": compare_metrics(b0_binary, recorded["b0"]["binary_broad"]["metrics"]),
        "b0_narrow_four": compare_metrics(b0_four, recorded["b0"]["narrow_four"]["metrics"]),
        "floor_binary_broad": compare_metrics(
            floor_binary, recorded["b0_floor_all_normal"]["binary_broad"]["metrics"]
        ),
        "floor_narrow_four": compare_metrics(
            floor_four, recorded["b0_floor_all_normal"]["narrow_four"]["metrics"]
        ),
    }
    expected_matrices = {
        "b0_binary_broad": b0_binary_matrix,
        "b0_narrow_four": b0_four_matrix,
        "floor_binary_broad": floor_binary_matrix,
        "floor_narrow_four": floor_four_matrix,
    }
    recorded_matrices = {
        "b0_binary_broad": recorded["b0"]["binary_broad"]["confusion_matrix"],
        "b0_narrow_four": recorded["b0"]["narrow_four"]["confusion_matrix"],
        "floor_binary_broad": recorded["b0_floor_all_normal"]["binary_broad"]["confusion_matrix"],
        "floor_narrow_four": recorded["b0_floor_all_normal"]["narrow_four"]["confusion_matrix"],
    }
    for key, matrix in expected_matrices.items():
        if int(matrix.sum()) != 1429 or matrix.tolist() != recorded_matrices[key]:
            raise RuntimeError(f"{key} confusion mismatch")
    if protocol["intra"] != "not executed" or protocol["c0"] != "not executed":
        raise RuntimeError("protocol scope violation")
    if recorded["coverage"]["narrow_four_counts"].get("both") != 1:
        raise RuntimeError("expected narrow-four Both support=1")

    receipt = {
        "status": "verified_published_model_author_checkpoint_icbhi_test_selected_zero_target_tuning_exploratory_transfer",
        "events": 1429,
        "unique_event_ids": 1429,
        "source_json_lineage_and_mapping_match": True,
        "inference_manifest_label_free": True,
        "logits_probabilities_finite": True,
        "probability_sum_max_abs_error": probability_error,
        "binary_probability_sum_max_abs_error": binary_probability_error,
        "binary_probability_aggregation_max_abs_error": aggregation_error,
        "four_class_argmax_match": True,
        "binary_author_collapse_rule_match": True,
        "metrics": {
            "b0_binary_broad": b0_binary,
            "b0_narrow_four": b0_four,
            "floor_binary_broad": floor_binary,
            "floor_narrow_four": floor_four,
        },
        "confusion_totals": {key: int(matrix.sum()) for key, matrix in expected_matrices.items()},
        "metric_max_abs_differences": max_differences,
        "coverage": recorded["coverage"],
        "both_support_warning": "support=1; no minority conclusion is valid",
        "intra_executed": False,
        "c0_executed": False,
        "degradation_conclusion_allowed": False,
    }
    (args.result_root / "verification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print("sprsound_b0_verification_ok events=1429 tasks=2 floor=all_normal")


if __name__ == "__main__":
    main()
