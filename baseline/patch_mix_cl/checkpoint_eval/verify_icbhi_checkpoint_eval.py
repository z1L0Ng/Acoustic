"""Independently verify exported Patch-Mix checkpoint-only predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


LABELS = ["normal", "crackle", "wheeze", "both"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = [
        row for row in load_rows(args.manifest) if row["official_split"] == "test"
    ]
    prediction_rows = load_rows(args.predictions)
    expected_ids = {row["cycle_id"] for row in manifest_rows}
    observed_ids = {row["cycle_id"] for row in prediction_rows}
    if len(manifest_rows) != 2756 or len(expected_ids) != 2756:
        raise RuntimeError("manifest does not contain 2,756 unique test cycles")
    if len(prediction_rows) != 2756 or len(observed_ids) != 2756:
        raise RuntimeError("predictions do not contain 2,756 unique cycles")
    if observed_ids != expected_ids:
        raise RuntimeError("prediction cycle IDs do not match the official test manifest")

    manifest_labels = {row["cycle_id"]: row["native_four_class_label"] for row in manifest_rows}
    for row in prediction_rows:
        if row["true_label"] != manifest_labels[row["cycle_id"]]:
            raise RuntimeError(f"label mismatch for {row['cycle_id']}")

    logits = np.asarray(
        [[float(row[f"logit_{label}"]) for label in LABELS] for row in prediction_rows]
    )
    probabilities = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in prediction_rows]
    )
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite logits or probabilities")
    probability_sum_max_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if probability_sum_max_error > 1e-5:
        raise RuntimeError(f"probabilities do not sum to one: {probability_sum_max_error}")

    y_true = np.asarray([int(row["true_index"]) for row in prediction_rows])
    y_pred = np.asarray([int(row["pred_index"]) for row in prediction_rows])
    if not np.array_equal(y_pred, probabilities.argmax(axis=1)):
        raise RuntimeError("exported predictions differ from probability argmax")
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(4))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(4), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    score = (specificity + sensitivity) / 2
    recomputed = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": score,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "both_recall": float(recall[3]),
    }

    recorded = json.loads(args.metrics.read_text())
    differences = {
        key: abs(value - float(recorded["metrics"][key])) for key, value in recomputed.items()
    }
    max_metric_difference = max(differences.values())
    if max_metric_difference > 1e-10:
        raise RuntimeError(f"recorded metrics do not match predictions: {differences}")
    if int(matrix.sum()) != 2756 or int(recorded["confusion_total"]) != 2756:
        raise RuntimeError("confusion total mismatch")
    if not all(math.isfinite(value) for value in recomputed.values()):
        raise RuntimeError("non-finite recomputed metric")

    receipt = {
        "status": "verified",
        "test_cycles": len(prediction_rows),
        "unique_cycle_ids": len(observed_ids),
        "cycle_id_set_matches_manifest": True,
        "labels_match_manifest": True,
        "logits_finite": True,
        "probabilities_finite": True,
        "probability_sum_max_abs_error": probability_sum_max_error,
        "prediction_matches_probability_argmax": True,
        "confusion_total": int(matrix.sum()),
        "confusion_matrix": matrix.tolist(),
        "metrics": recomputed,
        "metric_max_abs_difference": max_metric_difference,
        "per_class_support": {label: int(support[index]) for index, label in enumerate(LABELS)},
        "per_class_precision": {
            label: float(precision[index]) for index, label in enumerate(LABELS)
        },
        "per_class_recall": {label: float(recall[index]) for index, label in enumerate(LABELS)},
        "per_class_f1": {label: float(class_f1[index]) for index, label in enumerate(LABELS)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "checkpoint_eval_verification_ok "
        f"cycles={len(prediction_rows)} metric_max_diff={max_metric_difference:.3g}"
    )


if __name__ == "__main__":
    main()
