"""Independent gate and full-result verification for the D0-D3 matrix."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from baseline.four_dataset_frozen_encoder.data import (
    KAUH_LABELS,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import sha256_file
from baseline.four_dataset_frozen_encoder.run import EXPERIMENT_ID
from baseline.four_dataset_frozen_encoder.train import CONDITIONS, TASK_SPECS


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DATASET_ROWS = {
    "icbhi": 6898,
    "sprsound": 8085,
    "hf_lung": 9765,
    "kauh": 336,
}


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"metric mismatch {name}: {actual} != {expected}")


def _multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], task: str
) -> dict[str, object]:
    indices = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    output: dict[str, object] = {
        "rows": len(y_true),
        "macro_f1": f1_score(
            y_true, y_pred, labels=indices, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=indices, average="weighted", zero_division=0
        ),
        "uar": float(np.mean(recall)),
        "confusion": matrix.astype(int).tolist(),
        "per_class": {
            label: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    if task in {"icbhi_flat4", "spr_binary", "spr_seven"}:
        specificity = (
            float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
        )
        abnormal_total = matrix[1:].sum()
        sensitivity = (
            float(np.trace(matrix[1:, 1:]) / abnormal_total)
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
                    if task.startswith("spr_")
                    else average
                ),
            }
        )
    return output


def _multilabel_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, average=None, zero_division=0
    )
    return {
        "rows": len(y_true),
        "macro_f1": f1_score(
            y_true, predicted, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_true, predicted, average="micro", zero_division=0
        ),
        "subset_accuracy": float(np.mean(np.all(y_true == predicted, axis=1))),
        "per_class": {
            label: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
                "average_precision": (
                    average_precision_score(
                        y_true[:, index], probabilities[:, index]
                    )
                    if bool(y_true[:, index].sum())
                    else None
                ),
                "roc_auc": (
                    roc_auc_score(y_true[:, index], probabilities[:, index])
                    if len(np.unique(y_true[:, index])) == 2
                    else None
                ),
            }
            for index, label in enumerate(labels)
        },
    }


def _compare_metrics(
    actual: dict[str, object], expected: dict[str, object], task: str
) -> None:
    for key in (
        "rows",
        "macro_f1",
        "weighted_f1",
        "micro_f1",
        "subset_accuracy",
        "uar",
        "specificity",
        "sensitivity",
        "average_score",
        "harmonic_score",
        "native_score",
    ):
        if key in actual:
            if key == "rows":
                if int(actual[key]) != int(expected[key]):
                    raise RuntimeError(f"{task} row metric mismatch")
            else:
                _assert_close(float(actual[key]), float(expected[key]), f"{task}.{key}")
    if "confusion" in actual and actual["confusion"] != expected["confusion"]:
        raise RuntimeError(f"{task} confusion mismatch")
    for label, values in actual["per_class"].items():
        for key, value in values.items():
            expected_value = expected["per_class"][label][key]
            if value is None or expected_value is None:
                if value is not None or expected_value is not None:
                    raise RuntimeError(f"{task}.{label}.{key} null mismatch")
            elif key == "support":
                if int(value) != int(expected_value):
                    raise RuntimeError(f"{task}.{label}.support mismatch")
            else:
                _assert_close(
                    float(value),
                    float(expected_value),
                    f"{task}.{label}.{key}",
                )


def _terminal_spr_target(sample) -> dict[str, int]:
    payload = json.loads(Path(str(sample.metadata["annotation_path"])).read_text())
    raw = str(
        payload["event_annotation"][int(sample.metadata["event_index"])]["type"]
    )
    labels = TASK_SPECS["spr_seven"]["labels"]
    if raw not in labels:
        raise RuntimeError(f"unknown verifier SPR label {raw}")
    return {
        "spr_binary": int(raw != "Normal"),
        "spr_seven": labels.index(raw),
    }


def _verify_prediction_pair(
    label_free_path: Path,
    scored_path: Path,
    samples,
    recorded_metrics: dict[str, object],
) -> dict[str, int]:
    label_free = _read_predictions(label_free_path)
    scored = _read_predictions(scored_path)
    if not label_free or len(label_free) != len(scored):
        raise RuntimeError("label-free/scored prediction row mismatch")
    if "true_json" in label_free[0] or "true_json" not in scored[0]:
        raise RuntimeError("label-free column boundary failed")
    free_map = {
        (row["sample_id"], row["task"]): row for row in label_free
    }
    scored_map = {(row["sample_id"], row["task"]): row for row in scored}
    if len(free_map) != len(label_free) or free_map.keys() != scored_map.keys():
        raise RuntimeError("prediction key identity failed")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    task_counts = {}
    for task, spec in TASK_SPECS.items():
        rows = [
            scored_map[key] for key in scored_map if key[1] == task
        ]
        rows.sort(key=lambda row: row["sample_id"])
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate prediction IDs for {task}")
        probabilities = np.asarray(
            [json.loads(row["probabilities_json"]) for row in rows], dtype=float
        )
        predicted = np.asarray(
            [json.loads(row["pred_json"]) for row in rows], dtype=int
        )
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"non-finite probabilities for {task}")
        targets = []
        for row in rows:
            key = (row["sample_id"], task)
            if (
                free_map[key]["pred_json"] != row["pred_json"]
                or free_map[key]["probabilities_json"]
                != row["probabilities_json"]
            ):
                raise RuntimeError("terminal scoring changed model output")
            sample = sample_by_id[row["sample_id"]]
            if sample.dataset == "sprsound" and sample.partition == "test":
                if sample.targets or "raw_label" in sample.metadata:
                    raise RuntimeError("SPRSound test label leaked into sample")
                target = _terminal_spr_target(sample)[task]
            else:
                target = sample.targets[task]
            if json.loads(row["true_json"]) != target:
                raise RuntimeError(f"terminal target join mismatch for {row['sample_id']}")
            targets.append(target)
        target_array = np.asarray(targets, dtype=int)
        if spec["kind"] == "multiclass":
            if (
                probabilities.shape != (len(rows), len(spec["labels"]))
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                or not np.array_equal(predicted, probabilities.argmax(axis=1))
            ):
                raise RuntimeError(f"multiclass probability/argmax failure {task}")
            recomputed = _multiclass_metrics(
                target_array, predicted, spec["labels"], task
            )
            if int(np.asarray(recomputed["confusion"]).sum()) != len(rows):
                raise RuntimeError(f"confusion total failed {task}")
        else:
            if probabilities.shape != (len(rows), len(spec["labels"])):
                raise RuntimeError(f"multilabel probability shape failed {task}")
            if not np.array_equal(predicted, (probabilities >= 0.5).astype(int)):
                raise RuntimeError(f"multilabel threshold failure {task}")
            recomputed = _multilabel_metrics(
                target_array, probabilities, spec["labels"]
            )
        _compare_metrics(recomputed, recorded_metrics[task], task)
        task_counts[task] = len(rows)
    return task_counts


def verify_gate(
    result_root: Path, dataset_root: Path
) -> dict[str, object]:
    data = json.loads((result_root / "data_receipt.json").read_text())
    smoke = json.loads((result_root / "smoke_receipt.json").read_text())
    profile = json.loads((result_root / "profile_receipt.json").read_text())
    if (
        data["status"] != "four_dataset_data_audit_passed"
        or smoke["status"] != "four_dataset_real_data_smoke_passed"
        or profile["status"] != "profile_passed"
    ):
        raise RuntimeError("gate receipt failed")
    eligible_for_full = (
        profile["decision"] == "local full allowed"
        and profile["projected_end_to_end_with_2x_safety_minutes"]
        <= profile["gate"]["max_minutes"]
        and profile["peak_rss_gib"] < profile["gate"]["max_peak_rss_gib"]
    )
    if profile["decision"] not in {"local full allowed", "hold"}:
        raise RuntimeError("unknown profile decision")
    if data["folds"]["0"]["dataset_rows"] != EXPECTED_DATASET_ROWS:
        raise RuntimeError("dataset row gate failed")
    samples, _ = build_samples(dataset_root, 0)
    spr_test = [
        sample
        for sample in samples
        if sample.dataset == "sprsound" and sample.partition == "test"
    ]
    if (
        len(spr_test) != 1429
        or any(sample.targets or "raw_label" in sample.metadata for sample in spr_test)
    ):
        raise RuntimeError("SPRSound label-free manifest gate failed")
    if smoke["extraction"]["window_count_by_dataset"]["hf_lung"] != (
        smoke["dataset_rows"]["hf_lung"] * 3
    ):
        raise RuntimeError("HF three-window smoke gate failed")
    for condition in CONDITIONS:
        directory = result_root / "smoke" / condition
        metrics = json.loads((directory / "metrics.json").read_text())
        state = torch.load(directory / "best.pth", map_location="cpu")
        if not all(
            torch.isfinite(value).all()
            for value in state["model"].values()
            if torch.is_tensor(value)
        ):
            raise RuntimeError("non-finite smoke checkpoint")
        _verify_prediction_pair(
            directory / "predictions_label_free.csv.gz",
            directory / "predictions.csv.gz",
            samples,
            metrics["test_metrics"],
        )
    return {
        "status": (
            "four_dataset_gate_verified_full_allowed"
            if eligible_for_full
            else "four_dataset_gate_verified_hold"
        ),
        "rows": sum(EXPECTED_DATASET_ROWS.values()),
        "spr_label_free_test_rows": len(spr_test),
        "hf_smoke_windows_per_recording": 3,
        "projected_minutes_with_safety": profile[
            "projected_end_to_end_with_2x_safety_minutes"
        ],
        "peak_rss_gib": profile["peak_rss_gib"],
        "eligible_for_full": eligible_for_full,
        "profile_decision": profile["decision"],
        "hard_wall_clock_minutes": profile["gate"]["max_minutes"],
    }


def verify_full(
    result_root: Path, cache_root: Path, dataset_root: Path
) -> dict[str, object]:
    gate = verify_gate(result_root, dataset_root)
    if not gate["eligible_for_full"]:
        raise RuntimeError("full verification cannot run after HOLD gate")
    protocol = json.loads(
        (ROOT / "baseline/four_dataset_frozen_encoder/protocol.json").read_text()
    )
    embedding = json.loads((result_root / "embedding_receipt.json").read_text())
    manifest = json.loads((result_root / "run_manifest.json").read_text())
    encoder = embedding["encoder"]
    if (
        encoder["task_checkpoint_sha256"]
        != protocol["encoder"]["task_checkpoint_sha256"]
        or encoder["beats_checkpoint_sha256"]
        != protocol["encoder"]["beats_checkpoint_sha256"]
        or encoder["author_repo_commit"]
        != protocol["encoder"]["author_repo_commit"]
        or not encoder["encoder_frozen"]
        or encoder["discarded_source_states"] != ["classifier", "projector"]
        or embedding["cache_sha256"] != sha256_file(cache_root / "embeddings.npz")
        or manifest["status"] != "four_dataset_matrix_complete"
    ):
        raise RuntimeError("encoder/cache/manifest identity gate failed")
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    sample_ids = [sample.sample_id for sample in samples_by_fold[0]]
    archive = np.load(cache_root / "embeddings.npz", allow_pickle=False)
    if (
        archive["sample_ids"].astype(str).tolist() != sample_ids
        or archive["embeddings"].shape != (25084, 768)
        or not np.isfinite(archive["embeddings"]).all()
    ):
        raise RuntimeError("embedding content/order gate failed")
    final_by_id = dict(
        zip(
            archive["sample_ids"].astype(str).tolist(),
            archive["embeddings"].astype(np.float32),
        )
    )
    for dataset in ("icbhi", "sprsound", "hf_lung", "kauh"):
        shard_path = cache_root / "embedding_shards" / f"{dataset}.npz"
        shard_receipt_path = result_root / "embedding_shards" / f"{dataset}.json"
        shard_receipt = json.loads(shard_receipt_path.read_text())
        shard = np.load(shard_path, allow_pickle=False)
        shard_ids = shard["sample_ids"].astype(str).tolist()
        expected_ids = [
            sample.sample_id
            for sample in samples_by_fold[0]
            if sample.dataset == dataset
        ]
        shard_values = shard["embeddings"].astype(np.float32)
        if (
            shard_ids != expected_ids
            or shard_values.shape != (len(expected_ids), 768)
            or not np.isfinite(shard_values).all()
            or sha256_file(shard_path) != shard_receipt["cache_sha256"]
            or not all(
                np.array_equal(value, final_by_id[sample_id])
                for sample_id, value in zip(shard_ids, shard_values)
            )
        ):
            raise RuntimeError(f"embedding shard verification failed: {dataset}")

    kauh_test_sets = []
    non_kauh_metrics: dict[str, dict[str, list[dict[str, object]]]] = {
        condition: {
            task: [] for task in TASK_SPECS if task != "kauh_raw9"
        }
        for condition in CONDITIONS
    }
    prediction_pairs = 0
    for fold, samples in enumerate(samples_by_fold):
        kauh_test = {
            sample.sample_id
            for sample in samples
            if sample.dataset == "kauh" and sample.partition == "test"
        }
        kauh_test_sets.append(kauh_test)
        for condition in CONDITIONS:
            directory = result_root / f"fold_{fold}" / condition
            metrics = json.loads((directory / "metrics.json").read_text())
            state = torch.load(directory / "best.pth", map_location="cpu")
            if (
                metrics["condition"] != condition
                or not all(
                    torch.isfinite(value).all()
                    for value in state["model"].values()
                    if torch.is_tensor(value)
                )
                or any("test" in json.dumps(item).lower() for item in metrics["history"])
                or "validation" not in json.dumps(metrics["selection"]).lower()
            ):
                raise RuntimeError("checkpoint/selection provenance gate failed")
            _verify_prediction_pair(
                directory / "predictions_label_free.csv.gz",
                directory / "predictions.csv.gz",
                samples,
                metrics["test_metrics"],
            )
            prediction_pairs += 1
            for task in non_kauh_metrics[condition]:
                non_kauh_metrics[condition][task].append(
                    metrics["test_metrics"][task]
                )
    if (
        len(set().union(*kauh_test_sets)) != 336
        or sum(len(values) for values in kauh_test_sets) != 336
        or any(
            kauh_test_sets[left] & kauh_test_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("KAUH five-fold OOF partition identity failed")

    for condition in CONDITIONS:
        oof_rows = []
        for fold in range(5):
            rows = _read_predictions(
                result_root / f"fold_{fold}" / condition / "predictions.csv.gz"
            )
            oof_rows.extend(row for row in rows if row["task"] == "kauh_raw9")
        ids = [row["sample_id"] for row in oof_rows]
        if len(ids) != 336 or len(ids) != len(set(ids)):
            raise RuntimeError("KAUH OOF prediction identity failed")
        y_true = np.asarray([json.loads(row["true_json"]) for row in oof_rows])
        y_pred = np.asarray([json.loads(row["pred_json"]) for row in oof_rows])
        recomputed = _multiclass_metrics(
            y_true, y_pred, KAUH_LABELS, "kauh_raw9"
        )
        _compare_metrics(
            recomputed,
            manifest["kauh_oof"][condition],
            f"{condition}.kauh_oof",
        )
        for task, runs in non_kauh_metrics[condition].items():
            for metric in manifest["kauh_oof"][condition][
                "non_kauh_fold_conditioned"
            ][task]["metrics"]:
                values = [float(run[metric]) for run in runs]
                recorded = manifest["kauh_oof"][condition][
                    "non_kauh_fold_conditioned"
                ][task]["metrics"][metric]
                _assert_close(
                    float(np.mean(values)),
                    float(recorded["mean"]),
                    f"{condition}.{task}.{metric}.mean",
                )
                _assert_close(
                    float(np.std(values, ddof=1)),
                    float(recorded["sample_std"]),
                    f"{condition}.{task}.{metric}.std",
                )
    summary = list(csv.DictReader((result_root / "summary.csv").open()))
    per_class_summary = list(
        csv.DictReader((result_root / "per_class_summary.csv").open())
    )
    if len(summary) != len(CONDITIONS) * len(TASK_SPECS):
        raise RuntimeError("summary row count failed")
    if len(per_class_summary) != len(CONDITIONS) * sum(
        len(spec["labels"]) for spec in TASK_SPECS.values()
    ):
        raise RuntimeError("per-class summary row count failed")
    execution = json.loads((result_root / "execution_state.json").read_text())
    if (
        float(execution["active_seconds"])
        > float(execution["hard_wall_clock_seconds"])
        or float(execution["peak_rss_gib"])
        > float(execution["peak_rss_limit_gib"])
    ):
        raise RuntimeError("actual runtime/resource gate failed")
    return {
        **gate,
        "status": "four_dataset_matrix_verified",
        "conditions": len(CONDITIONS),
        "tasks": len(TASK_SPECS),
        "summary_rows": len(summary),
        "per_class_summary_rows": len(per_class_summary),
        "prediction_pairs": prediction_pairs,
        "embedding_rows": 25084,
        "embedding_dim": 768,
        "kauh_oof_rows_per_condition": 336,
        "active_wall_seconds": execution["active_seconds"],
        "peak_rss_gib": execution["peak_rss_gib"],
        "embedding_shards": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gate", "full"], default="full")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(f"result/{EXPERIMENT_ID}"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(f".cache/{EXPERIMENT_ID}"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    receipt = (
        verify_gate(result_root, args.dataset_root.resolve())
        if args.mode == "gate"
        else verify_full(
            result_root,
            args.cache_root.resolve(),
            args.dataset_root.resolve(),
        )
    )
    path = result_root / (
        "gate_verification.json"
        if args.mode == "gate"
        else "verification.json"
    )
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.mode == "full":
        state_path = result_root / "execution_state.json"
        state = json.loads(state_path.read_text())
        state["status"] = "completed_and_independently_verified"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
