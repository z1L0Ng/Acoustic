"""Independent readback verifier for persisted terminal native-task predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import torch


NATIVE_TASKS = (
    "ICBHI_flat4",
    "SPRSound_binary",
    "SPRSound_raw7",
    "HF_temporal4",
    "KAUH_raw9",
)
FORBIDDEN_KEYS = {
    "pooled_score",
    "global_score",
    "cross_dataset_score",
    "ranking",
}
ATOL = 1e-12


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden_keys(value: object, location: str = "receipt") -> None:
    if isinstance(value, Mapping):
        found = FORBIDDEN_KEYS & set(value)
        if found:
            raise RuntimeError(f"forbidden aggregate keys at {location}: {sorted(found)}")
        for key, child in value.items():
            _reject_forbidden_keys(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{location}[{index}]")


def _confusion(target: Sequence[int], predicted: Sequence[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    if len(target) != len(predicted) or not target:
        raise RuntimeError("multiclass prediction/target rows must be aligned and non-empty")
    for truth, guess in zip(target, predicted):
        if not 0 <= truth < classes or not 0 <= guess < classes:
            raise RuntimeError("multiclass value is outside the frozen label space")
        matrix[truth][guess] += 1
    return matrix


def _multiclass_metrics(
    target: Sequence[int], predicted: Sequence[int], labels: Sequence[str], task: str
) -> dict[str, object]:
    matrix = _confusion(target, predicted, len(labels))
    supports = [sum(row) for row in matrix]
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(len(labels)):
        tp = matrix[index][index]
        column = sum(row[index] for row in matrix)
        support = supports[index]
        precision = tp / column if column else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    rows = len(target)
    output: dict[str, object] = {
        "rows": rows,
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": sum(f1 * support for f1, support in zip(f1s, supports)) / rows,
        "uar": sum(recalls) / len(recalls),
        "confusion": matrix,
        "per_class": {
            label: {
                "precision": precisions[index],
                "recall": recalls[index],
                "f1": f1s[index],
                "support": supports[index],
            }
            for index, label in enumerate(labels)
        },
    }
    if task in {"ICBHI_flat4", "SPRSound_binary", "SPRSound_raw7"}:
        specificity = matrix[0][0] / supports[0] if supports[0] else 0.0
        abnormal_total = sum(supports[1:])
        sensitivity = (
            sum(matrix[index][index] for index in range(1, len(labels))) / abnormal_total
            if abnormal_total
            else 0.0
        )
        average = (specificity + sensitivity) / 2
        harmonic = (
            2 * specificity * sensitivity / (specificity + sensitivity)
            if specificity + sensitivity
            else 0.0
        )
        output.update(
            {
                "specificity": specificity,
                "sensitivity": sensitivity,
                "average_score": average,
                "harmonic_score": harmonic,
                "native_score": (
                    (average + harmonic) / 2
                    if task.startswith("SPRSound")
                    else average
                ),
            }
        )
    return output


def _binary_auc(target: Sequence[int], score: Sequence[float]) -> float:
    pairs = sorted(zip(score, target), key=lambda item: item[0])
    positives = sum(target)
    negatives = len(target) - positives
    if not positives or not negatives:
        raise RuntimeError("ROC-AUC requires positive and negative support")
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(target: Sequence[int], score: Sequence[float]) -> float:
    pairs = sorted(zip(score, target), key=lambda item: item[0], reverse=True)
    positives = sum(target)
    if not positives:
        raise RuntimeError("average precision requires positive support")
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        group_positive = sum(label for _, label in pairs[start:end])
        true_positive += group_positive
        false_positive += end - start - group_positive
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return result


def _assert_close(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"{label} mapping keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{label} list shape differs")
        for index, item in enumerate(expected):
            _assert_close(actual[index], item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=ATOL
        ):
            raise RuntimeError(f"{label} differs: {actual} != {expected}")
    elif actual != expected:
        raise RuntimeError(f"{label} differs: {actual} != {expected}")


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_path(path)}


def verify_terminal_result(
    pipeline: str,
    terminal_receipt_path: Path,
    approval_path: Path,
    selection_path: Path,
    threshold_path: Path,
    *,
    subject_code_commit: str,
    verifier_code_commit: str,
) -> dict[str, object]:
    terminal = json.loads(terminal_receipt_path.read_text())
    approval = json.loads(approval_path.read_text())
    selection = json.loads(selection_path.read_text())
    threshold = json.loads(threshold_path.read_text())
    _reject_forbidden_keys(terminal)
    scorer = terminal["native_metrics_by_dataset_task"]
    selection_artifact = _artifact(selection_path)
    checkpoint_artifact = _artifact(Path(selection["selected_checkpoint"]["path"]))
    threshold_artifact = _artifact(threshold_path)
    approval_artifact = _artifact(approval_path)
    if (
        terminal["status"] != "terminal_native_task_scoring_complete"
        or scorer["outer_test_accessed"] is not True
        or scorer["terminal_targets_loaded"] is not True
        or scorer["cross_dataset_pooling"] is not False
        or terminal["cross_dataset_pooled_performance"] is not False
        or tuple(scorer["native_task_names"]) != NATIVE_TASKS
        or set(scorer["native_tasks"]) != set(NATIVE_TASKS)
    ):
        raise RuntimeError("terminal task/isolation schema failed independent readback")
    if (
        terminal["selection_receipt_artifact"] != selection_artifact
        or terminal["selected_checkpoint"]["path"] != checkpoint_artifact["path"]
        or terminal["selected_checkpoint"]["size_bytes"] != checkpoint_artifact["size_bytes"]
        or terminal["selected_checkpoint"]["sha256"] != checkpoint_artifact["sha256"]
        or terminal["hf_threshold_receipt_artifact"]["path"] != threshold_artifact["path"]
        or terminal["hf_threshold_receipt_artifact"]["size_bytes"] != threshold_artifact["size_bytes"]
        or terminal["hf_threshold_receipt_artifact"]["sha256"] != threshold_artifact["sha256"]
        or terminal["terminal_approval_receipt_sha256"] != approval_artifact["sha256"]
        or approval["selection_receipt_sha256"] != selection_artifact["sha256"]
        or approval["selected_checkpoint_path"] != checkpoint_artifact["path"]
        or approval["selected_checkpoint_size_bytes"] != checkpoint_artifact["size_bytes"]
        or approval["selected_checkpoint_sha256"] != checkpoint_artifact["sha256"]
        or approval["hf_threshold_receipt_sha256"] != threshold_artifact["sha256"]
        or threshold["validation_selection_receipt_sha256"] != selection_artifact["sha256"]
        or threshold["selected_checkpoint_sha256"] != checkpoint_artifact["sha256"]
    ):
        raise RuntimeError("approval/selection/checkpoint/threshold identity binding failed")
    joined_receipt = scorer["prediction_artifacts"]["terminal_joined_predictions"]
    joined_path = Path(joined_receipt["path"])
    if _artifact(joined_path) != joined_receipt:
        raise RuntimeError("joined prediction artifact byte identity failed")
    joined = torch.load(joined_path, map_location="cpu", weights_only=False)
    if (
        joined["outer_test_accessed"] is not True
        or joined["prediction_before_spr_label_join"] is not True
        or tuple(joined["native_tasks"]) != NATIVE_TASKS
        or set(joined["tasks"]) != set(NATIVE_TASKS)
        or joined["selected_checkpoint_sha256"]
        != selection["selected_checkpoint"]["sha256"]
    ):
        raise RuntimeError("joined prediction identity/isolation binding failed")
    task_checks: dict[str, object] = {}
    for task in NATIVE_TASKS:
        recorded = scorer["native_tasks"][task]
        persisted = joined["tasks"][task]
        ids = list(persisted["prediction_ids"])
        if len(ids) != len(set(ids)) or len(ids) != recorded["prediction_unit_count"]:
            raise RuntimeError(f"{task} ordered prediction IDs failed")
        if task != "HF_temporal4":
            target = persisted["targets"].tolist()
            predicted = persisted["predicted_classes"].tolist()
            recomputed = _multiclass_metrics(target, predicted, recorded["labels"], task)
            _assert_close(recorded["metrics"], recomputed, f"{task}.metrics")
            if recorded["denominator"] != len(target):
                raise RuntimeError(f"{task} denominator differs")
            task_checks[task] = {
                "ordered_prediction_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "denominator": len(target),
                "metrics_recomputed": True,
            }
            continue
        probabilities = torch.cat(persisted["probabilities"])
        targets = torch.cat(persisted["targets"])
        window_mask = torch.cat(persisted["window_mask"])
        annotation_mask = torch.cat(persisted["annotation_mask"])
        valid_mask = torch.cat(persisted["valid_mask"])
        thresholds = persisted["thresholds"].to(torch.float64)
        if (
            persisted["threshold_receipt_sha256"] != sha256_path(threshold_path)
            or threshold["thresholds"] != thresholds.tolist()
        ):
            raise RuntimeError("HF frozen threshold binding failed")
        per_channel: dict[str, object] = {}
        total = 0
        effective = window_mask.unsqueeze(-1) & annotation_mask & valid_mask
        for index, channel in enumerate(recorded["channel_order"]):
            current_target = targets[..., index][effective[..., index]].to(torch.int64).tolist()
            current_score = probabilities[..., index][effective[..., index]].to(torch.float64).tolist()
            predicted = [int(value >= thresholds[index]) for value in current_score]
            matrix = _confusion(current_target, predicted, 2)
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            denominator = len(current_target)
            total += denominator
            per_channel[channel] = {
                "denominator": denominator,
                "positive_support": sum(current_target),
                "constructed_negative_support": denominator - sum(current_target),
                "threshold": float(thresholds[index]),
                "confusion": matrix,
                "accuracy": sum(a == b for a, b in zip(current_target, predicted)) / denominator,
                "roc_auc": _binary_auc(current_target, current_score),
                "average_precision": _average_precision(current_target, current_score),
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "positive_predictive_value": tp / (tp + fp) if tp + fp else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            }
        _assert_close(recorded["per_channel"], per_channel, "HF_temporal4.per_channel")
        if recorded["channel_denominator_total"] != total:
            raise RuntimeError("HF channel denominator differs")
        task_checks[task] = {
            "ordered_prediction_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "recordings": len(ids),
            "channel_denominator_total": total,
            "metrics_recomputed": True,
            "thresholds_selected_on_outer_test": False,
        }
    label_free_path = Path(
        scorer["prediction_artifacts"]["sprsound_label_free_predictions"]["path"]
    )
    if _artifact(label_free_path) != scorer["prediction_artifacts"]["sprsound_label_free_predictions"]:
        raise RuntimeError("SPRSound label-free artifact byte identity failed")
    label_free = [json.loads(line) for line in label_free_path.read_text().splitlines()]
    spr_binary = joined["tasks"]["SPRSound_binary"]
    spr_raw7 = joined["tasks"]["SPRSound_raw7"]
    if len(label_free) != len(spr_binary["prediction_ids"]):
        raise RuntimeError("SPRSound label-free row count differs")
    for index, row in enumerate(label_free):
        if set(row) != {"sample_id", "spr_binary_prediction", "spr_raw7_prediction"}:
            raise RuntimeError("SPRSound label-free schema contains labels or extra fields")
        if (
            row["sample_id"] != spr_binary["prediction_ids"][index]
            or row["sample_id"] != spr_raw7["prediction_ids"][index]
            or row["spr_binary_prediction"] != int(spr_binary["predicted_classes"][index])
            or row["spr_raw7_prediction"] != int(spr_raw7["predicted_classes"][index])
        ):
            raise RuntimeError("SPRSound prediction-before-label-join binding failed")
    if scorer["native_tasks"]["KAUH_raw9"]["denominator"] != 69:
        raise RuntimeError("KAUH terminal result is not the frozen fold0 support")
    return {
        "schema_version": "shared_window_terminal_independent_verifier_v1",
        "status": "verified_terminal_native_tasks",
        "pipeline_id": pipeline,
        "verifier_identity": "independent_persisted_prediction_recompute",
        "verifier_code_commit": verifier_code_commit,
        "subject_code_commit": subject_code_commit,
        "terminal_receipt": _artifact(terminal_receipt_path),
        "terminal_approval": approval_artifact,
        "validation_selection_receipt": selection_artifact,
        "selected_checkpoint": checkpoint_artifact,
        "hf_threshold_receipt": threshold_artifact,
        "provider_identity_sha256": scorer["provider_identity_sha256"],
        "scorer_schema_version": scorer["schema_version"],
        "prediction_artifacts": scorer["prediction_artifacts"],
        "task_checks": task_checks,
        "sprsound_prediction_before_label_join_verified": True,
        "outer_test_accessed": True,
        "no_pooled_global_cross_dataset_score_or_ranking": True,
        "kauh_scope": "fold0 terminal only, not 5-fold OOF",
        "warnings": ["KAUH result covers frozen fold0 only; it is not 5-fold OOF."],
    }


def _write_json_no_replace(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        temporary.write_bytes(raw)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"receipt already exists: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return _artifact(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("P1", "P2", "P3", "P5"), required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--terminal-approval", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--hf-threshold-receipt", type=Path, required=True)
    parser.add_argument("--subject-code-commit", required=True)
    parser.add_argument("--verifier-code-commit", required=True)
    parser.add_argument("--verifier-receipt", type=Path, required=True)
    parser.add_argument("--decision-receipt", type=Path, required=True)
    args = parser.parse_args()
    verified = verify_terminal_result(
        args.pipeline,
        args.terminal_receipt,
        args.terminal_approval,
        args.selection_receipt,
        args.hf_threshold_receipt,
        subject_code_commit=args.subject_code_commit,
        verifier_code_commit=args.verifier_code_commit,
    )
    verifier_artifact = _write_json_no_replace(args.verifier_receipt, verified)
    decision = {
        "schema_version": "shared_window_terminal_decision_v1",
        "status": "GO_verified_terminal_native_tasks",
        "pipeline_id": args.pipeline,
        "verifier_receipt": verifier_artifact,
        "five_native_tasks_verified": list(NATIVE_TASKS),
        "no_pooled_global_cross_dataset_score_or_ranking": True,
        "kauh_scope": "fold0 terminal only, not 5-fold OOF",
    }
    decision_artifact = _write_json_no_replace(args.decision_receipt, decision)
    print(json.dumps({"verifier_receipt": verifier_artifact, "decision_receipt": decision_artifact}, sort_keys=True))


if __name__ == "__main__":
    main()
