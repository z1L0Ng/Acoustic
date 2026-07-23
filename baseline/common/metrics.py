from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    true_abnormal = y_true != "normal"
    pred_abnormal = y_pred != "normal"
    sensitivity = float(pred_abnormal[true_abnormal].mean()) if true_abnormal.any() else float("nan")
    specificity = float((~pred_abnormal[~true_abnormal]).mean()) if (~true_abnormal).any() else float("nan")
    weights = support / support.sum()
    result = {
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * weights)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "abnormal_sensitivity": sensitivity,
        "normal_specificity": specificity,
        "icbhi_score": float((sensitivity + specificity) / 2.0),
        "both_recall": float(recall[labels.index("both")]) if "both" in labels else float("nan"),
        "class_metrics": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": labels,
    }
    return result


def flatten_metrics(metrics: dict) -> dict:
    row = {key: value for key, value in metrics.items() if key not in {"class_metrics", "confusion_matrix", "confusion_matrix_labels"}}
    for label, values in metrics["class_metrics"].items():
        for metric, value in values.items():
            row[f"{label}_{metric}"] = value
    return row
