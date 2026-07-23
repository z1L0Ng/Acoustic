"""Evaluation-only grouped uncertainty and descriptive chance references."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import (
    TASK_LABELS,
    classification_metrics,
    read_csv,
    validate_result_root,
    write_csv,
    write_json,
)


BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 5000
MONTE_CARLO_SEED = 20260723
MONTE_CARLO_REPLICATES = 5000
REQUESTED = {
    "binary_broad": ["icbhi_score", "macro_f1", "uar"],
    "narrow_four": ["icbhi_score", "macro_f1"],
}


def vectors(rows: list[dict[str, str]], task: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    labels = TASK_LABELS[task]
    true_key = "binary_broad_label" if task == "binary_broad" else "narrow_four_label"
    if prefix == "b0":
        pred_key = "pred_binary_broad_label" if task == "binary_broad" else "pred_narrow_four_label"
    elif prefix == "floor":
        pred_key = "floor_pred_binary_broad_label" if task == "binary_broad" else "floor_pred_narrow_four_label"
    else:
        pred_key = "pred_label"
        true_key = "task_label"
    included = [row for row in rows if row.get(true_key, "") in labels]
    return (
        np.asarray([labels.index(row[true_key]) for row in included], dtype=np.int64),
        np.asarray([labels.index(row[pred_key]) for row in included], dtype=np.int64),
    )


def patient_groups(rows: list[dict[str, str]], task: str) -> tuple[list[str], dict[str, np.ndarray]]:
    labels = TASK_LABELS[task]
    true_key = "binary_broad_label" if task == "binary_broad" else "narrow_four_label"
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get(true_key, "") in labels:
            grouped[row["patient_id"]].append(index)
    patients = sorted(grouped)
    return patients, {patient: np.asarray(indices, dtype=np.int64) for patient, indices in grouped.items()}


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def grouped_bootstrap(
    rows: list[dict[str, str]], task: str, pred_kind: str, seed: int, repeats: int
) -> dict[str, dict[str, float]]:
    labels = TASK_LABELS[task]
    y_true, y_pred = vectors(rows, task, pred_kind)
    patients, groups = patient_groups(rows, task)
    # vectors() preserves all current B0 rows, so row indices are shared here.
    if len(y_true) != len(rows):
        raise RuntimeError("current inter task unexpectedly excludes rows; align indices before bootstrap")
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(repeats, dtype=np.float64) for metric in REQUESTED[task]}
    for repeat in range(repeats):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([groups[patient] for patient in sampled])
        metrics, _ = classification_metrics(y_true[indices], y_pred[indices], labels)
        for metric in values:
            values[metric][repeat] = float(metrics[metric])
    point, _ = classification_metrics(y_true, y_pred, labels)
    return {
        metric: {"point_estimate": float(point[metric]), **percentile_summary(samples)}
        for metric, samples in values.items()
    }


def metrics_from_expected_confusion(support: np.ndarray, q: np.ndarray, labels: list[str]) -> dict[str, float]:
    matrix = support[:, None] * q[None, :]
    recall = np.diag(matrix) / matrix.sum(axis=1)
    precision = np.divide(
        np.diag(matrix), matrix.sum(axis=0), out=np.zeros(len(labels)), where=matrix.sum(axis=0) > 0
    )
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=(precision + recall) > 0)
    specificity = matrix[0, 0] / matrix[0].sum() * 100
    sensitivity = np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100
    return {
        "icbhi_score": float((specificity + sensitivity) / 2),
        "macro_f1": float(f1.mean()),
        "uar": float(recall.mean()),
    }


def chance_reference(
    y_true: np.ndarray,
    labels: list[str],
    probabilities: np.ndarray,
    seed: int,
    repeats: int,
) -> dict[str, object]:
    support = np.bincount(y_true, minlength=len(labels)).astype(float)
    expected = metrics_from_expected_confusion(support, probabilities, labels)
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(repeats, dtype=np.float64) for metric in expected}
    for repeat in range(repeats):
        prediction = rng.choice(len(labels), size=len(y_true), p=probabilities)
        metrics, _ = classification_metrics(y_true, prediction, labels)
        for metric in values:
            values[metric][repeat] = float(metrics[metric])
    return {
        "prediction_probabilities": {
            label: float(probabilities[index]) for index, label in enumerate(labels)
        },
        "expected_confusion_metric": expected,
        "monte_carlo": {metric: percentile_summary(samples) for metric, samples in values.items()},
    }


def paired_c0_minus_b0(
    b0_rows: list[dict[str, str]], c0_rows: list[dict[str, str]], task: str
) -> dict[str, object]:
    labels = TASK_LABELS[task]
    c0_by_id = {row["event_id"]: row for row in c0_rows if row.get("mapping_status") == "included"}
    if len(c0_by_id) != len(c0_rows) or set(c0_by_id) != {row["event_id"] for row in b0_rows}:
        raise RuntimeError("paired C0/B0 requires identical included event IDs")
    y_true, b0_pred = vectors(b0_rows, task, "b0")
    c0_pred = np.asarray([int(c0_by_id[row["event_id"]]["pred_index"]) for row in b0_rows])
    patients, groups = patient_groups(b0_rows, task)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in REQUESTED[task]}
    for repeat in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([groups[patient] for patient in sampled])
        b0_metrics, _ = classification_metrics(y_true[indices], b0_pred[indices], labels)
        c0_metrics, _ = classification_metrics(y_true[indices], c0_pred[indices], labels)
        for metric in differences:
            differences[metric][repeat] = float(c0_metrics[metric]) - float(b0_metrics[metric])
    b0_point, _ = classification_metrics(y_true, b0_pred, labels)
    c0_point, _ = classification_metrics(y_true, c0_pred, labels)
    return {
        metric: {
            "b0": float(b0_point[metric]),
            "c0": float(c0_point[metric]),
            "c0_minus_b0": float(c0_point[metric]) - float(b0_point[metric]),
            "effect_size_percentage_points": (
                float(c0_point[metric]) - float(b0_point[metric])
            )
            * (1.0 if metric == "icbhi_score" else 100.0),
            "effect_scale": "percentage_points",
            "bootstrap_delta_ci_native_scale": True,
            **percentile_summary(samples),
        }
        for metric, samples in differences.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c0-result-root", type=Path, required=True)
    args = parser.parse_args()
    b0_root = args.b0_result_root.resolve()
    output = args.output_dir.resolve()
    if output.name != "comparison" or validate_result_root(output.parent) != output.parent:
        raise ValueError("statistical output must be result/sprsound_patchmix_target_training/comparison")
    if output.exists():
        raise FileExistsError(f"comparison output is immutable once written: {output}")
    output.mkdir(parents=True, exist_ok=True)
    b0_rows = read_csv(b0_root / "b0_predictions.csv")
    floor_rows = read_csv(b0_root / "b0_floor_predictions.csv")
    floor_by_id = {row["event_id"]: row for row in floor_rows}
    for row in b0_rows:
        row.update(floor_by_id[row["event_id"]])
    if len(b0_rows) != 1429 or len({row["event_id"] for row in b0_rows}) != 1429:
        raise RuntimeError("B0 verified prediction coverage gate failed")
    patient_count = len({row["patient_id"] for row in b0_rows})

    payload: dict[str, object] = {
        "status": "evaluation_only_statistical_supplement_complete",
        "claim_boundary": "descriptive uncertainty and floors only; no model, threshold, row, mapping, or preprocessing selection",
        "grouping": {
            "unit": "patient_id",
            "patient_count": patient_count,
            "evidence": "official SPRSound README defines the first filename element as patient number",
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95%",
        },
        "monte_carlo": {"seed": MONTE_CARLO_SEED, "replicates": MONTE_CARLO_REPLICATES},
        "practical_evidence_rule": {
            "primary": "paired C0 minus B0 gap on identical inter event IDs",
            "materiality_reference": ">=5 percentage-point absolute gap",
            "boundary": "project reference only; not a community standard or statistical significance test",
            "both_warning": "support=1; Both recall and its UAR contribution cannot support evidence",
        },
        "tasks": {},
    }
    flat_rows: list[dict[str, object]] = []
    for task, labels in TASK_LABELS.items():
        y_true, _ = vectors(b0_rows, task, "b0")
        prevalence = np.bincount(y_true, minlength=len(labels)).astype(float)
        prevalence /= prevalence.sum()
        task_payload: dict[str, object] = {
            "b0_grouped_bootstrap": grouped_bootstrap(
                b0_rows, task, "b0", BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES
            ),
            "all_normal_grouped_bootstrap": grouped_bootstrap(
                b0_rows, task, "floor", BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES
            ),
            "uniform_random": chance_reference(
                y_true,
                labels,
                np.full(len(labels), 1 / len(labels)),
                MONTE_CARLO_SEED,
                MONTE_CARLO_REPLICATES,
            ),
            "target_prevalence_matched_random": chance_reference(
                y_true, labels, prevalence, MONTE_CARLO_SEED + 1, MONTE_CARLO_REPLICATES
            ),
            "target_prevalence_use": "descriptive floor only; computed after labels and forbidden for selection",
        }
        c0_path = validate_result_root(args.c0_result_root) / "full" / task / "inter_predictions.csv"
        task_payload["paired_c0_minus_b0"] = paired_c0_minus_b0(
            b0_rows, read_csv(c0_path), task
        )
        payload["tasks"][task] = task_payload
        for source in ("b0_grouped_bootstrap", "all_normal_grouped_bootstrap"):
            for metric, values in task_payload[source].items():
                flat_rows.append({"task": task, "reference": source, "metric": metric, **values})
        for source in ("uniform_random", "target_prevalence_matched_random"):
            reference = task_payload[source]
            for metric, values in reference["monte_carlo"].items():
                flat_rows.append(
                    {
                        "task": task,
                        "reference": source,
                        "metric": metric,
                        "point_estimate": reference["expected_confusion_metric"][metric],
                        **values,
                    }
                )
    write_json(output / "statistical_comparison_receipt.json", payload)
    write_csv(output / "statistical_comparison_summary.csv", flat_rows)
    print(
        f"c0_statistical_supplement_ok events=1429 patients={patient_count} "
        f"bootstrap={BOOTSTRAP_REPLICATES} monte_carlo={MONTE_CARLO_REPLICATES}"
    )


if __name__ == "__main__":
    main()
