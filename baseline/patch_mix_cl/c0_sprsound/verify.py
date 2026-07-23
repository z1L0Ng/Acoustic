"""Independent structural and metric verification for C0 smoke/full artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import TASK_LABELS, classification_metrics, read_csv, task_label


def compare_metrics(observed: dict, expected: dict) -> float:
    keys = ["specificity", "sensitivity", "icbhi_score", "macro_f1", "weighted_f1", "uar"]
    if "both_recall" in expected:
        keys.append("both_recall")
    return max(abs(float(observed[key]) - float(expected[key])) for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-name", default="full")
    args = parser.parse_args()
    root = args.result_root.resolve()

    bootstrap = json.loads((root / "receipts" / "bootstrap.json").read_text())
    data_receipt = json.loads((root / "receipts" / "data_protocol.json").read_text())
    cache_receipt = json.loads((root / "receipts" / f"cache_{args.cache_name}.json").read_text())
    if bootstrap["status"] != "bootstrap_verified":
        raise RuntimeError("bootstrap receipt failed")
    if data_receipt["validation"]["patient_overlap"] != 0:
        raise RuntimeError("inner split patient leakage")
    assignments = read_csv(root / "data" / "patient_validation_assignment.csv")
    if len(assignments) != 243 or len({row["patient_id"] for row in assignments}) != 243:
        raise RuntimeError("patient assignment mismatch")
    if not cache_receipt["train"]["finite"] or not cache_receipt["inter"]["finite"]:
        raise RuntimeError("cache finite gate failed")

    verified = {}
    inter_manifest = {
        row["event_id"]: row
        for row in read_csv(root / "data" / "inter_events_label_free.csv")
    }
    for task, labels in TASK_LABELS.items():
        task_dir = root / args.mode / task
        receipt = json.loads((task_dir / "run_receipt.json").read_text())
        label_free = read_csv(task_dir / "inter_predictions_label_free.csv")
        ids = [row["event_id"] for row in label_free]
        if len(ids) != len(set(ids)) or any("true_" in key or "raw_label" in key for key in label_free[0]):
            raise RuntimeError(f"label-free prediction contract failed: {task}")
        numeric_keys = [key for key in label_free[0] if key.startswith(("logit_", "prob_"))]
        if not numeric_keys or not all(
            math.isfinite(float(row[key])) for row in label_free for key in numeric_keys
        ):
            raise RuntimeError(f"prediction finite gate failed: {task}")
        if not receipt["finite_loss_and_gradient"] or receipt["inter_metrics_used_for_selection"]:
            raise RuntimeError(f"training boundary failed: {task}")
        if args.mode == "smoke":
            if receipt["inter_label_access"] != "none" or (task_dir / "inter_metrics.json").exists():
                raise RuntimeError(f"smoke accessed inter labels: {task}")
            verified[task] = {"inter_forward_rows": len(ids), "metrics": "not_computed"}
            continue

        if len(ids) != 1429:
            raise RuntimeError(f"full inter coverage mismatch: {task}")
        scored = read_csv(task_dir / "inter_predictions.csv")
        if len(scored) != 1429 or {row["event_id"] for row in scored} != set(ids):
            raise RuntimeError(f"scored prediction coverage mismatch: {task}")
        for row in scored:
            annotation = inter_manifest[row["event_id"]]
            raw_payload = json.loads(Path(annotation["annotation_path"]).read_text())
            raw = str(raw_payload["event_annotation"][int(annotation["event_index"])]["type"])
            expected = task_label(raw, task)
            if row["raw_label"] != raw or row["task_label"] != (expected or ""):
                raise RuntimeError(f"independent label rebuild mismatch: {row['event_id']}")
        included = [row for row in scored if row["mapping_status"] == "included"]
        y_true = np.asarray([int(row["true_index"]) for row in included], dtype=np.int64)
        y_pred = np.asarray([int(row["pred_index"]) for row in included], dtype=np.int64)
        expected_metrics, matrix = classification_metrics(y_true, y_pred, labels)
        observed_metrics = json.loads((task_dir / "inter_metrics.json").read_text())
        difference = compare_metrics(observed_metrics, expected_metrics)
        if difference > 1e-12 or int(matrix.sum()) != len(included):
            raise RuntimeError(f"independent metric mismatch: {task} {difference}")
        verified[task] = {
            "inter_forward_rows": len(ids),
            "inter_scored_rows": len(included),
            "confusion_total": int(matrix.sum()),
            "max_metric_difference": difference,
        }

    print(
        f"c0_{args.mode}_verification_ok tasks=2 cache={args.cache_name} "
        f"details={json.dumps(verified, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
