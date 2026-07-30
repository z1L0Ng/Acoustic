"""Independent verifier for shared compatible-head harmonization."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from baseline.four_dataset_frozen_encoder.data import build_samples
from baseline.four_dataset_frozen_encoder.encoder import sha256_file
from baseline.four_dataset_frozen_encoder.verify import (
    _compare_metrics,
    _multiclass_metrics,
    _multilabel_metrics,
    _terminal_spr_target,
)
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS
from baseline.shared_compatible_head_harmonization.run import (
    COMPATIBLE_TASKS,
    CONDITIONS,
    EXPERIMENT_ID,
    NARROW_LABELS,
    PROTOCOL_PATH,
    SELECTED_CACHE_SHA256,
    SELECTION_RECEIPT,
    SPR_NARROW_MAP,
    HarmonizationModel,
    _ordered_sha,
    analyze,
)


def _read(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"{name}: {actual} != {expected}")


def _compatible_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, object]:
    indices = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
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
    return {
        "rows": len(y_true),
        "macro_f1": f1_score(
            y_true, y_pred, labels=indices, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=indices, average="weighted", zero_division=0
        ),
        "uar": float(np.mean(recall)),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "average_score": average,
        "harmonic_score": harmonic,
        "sprsound_score": (average + harmonic) / 2,
        "predicted_abnormal_prevalence": float(np.mean(y_pred != 0)),
        "per_class": {
            label: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion": matrix.astype(int).tolist(),
    }


def _compare_compatible(
    actual: dict[str, object], recorded: dict[str, object], task: str
) -> None:
    for key in (
        "rows",
        "macro_f1",
        "weighted_f1",
        "uar",
        "specificity",
        "sensitivity",
        "average_score",
        "harmonic_score",
        "sprsound_score",
        "predicted_abnormal_prevalence",
    ):
        if key == "rows":
            if int(actual[key]) != int(recorded[key]):
                raise RuntimeError(f"{task} rows mismatch")
        else:
            _assert_close(float(actual[key]), float(recorded[key]), f"{task}.{key}")
    if actual["confusion"] != recorded["confusion"]:
        raise RuntimeError(f"{task} confusion mismatch")
    for label, values in actual["per_class"].items():
        for key, value in values.items():
            recorded_value = recorded["per_class"][label][key]
            if key == "support":
                if int(value) != int(recorded_value):
                    raise RuntimeError(f"{task}.{label}.support mismatch")
            else:
                _assert_close(
                    float(value), float(recorded_value), f"{task}.{label}.{key}"
                )


def _verify_pair(
    directory: Path,
    samples,
    metrics: dict[str, object],
) -> dict[str, object]:
    label_free = _read(directory / "predictions_label_free.csv.gz")
    scored = _read(directory / "predictions.csv.gz")
    if not label_free or "true_json" in label_free[0] or "true_json" not in scored[0]:
        raise RuntimeError("label-free column boundary failed")
    free_map = {(row["sample_id"], row["task"]): row for row in label_free}
    scored_map = {(row["sample_id"], row["task"]): row for row in scored}
    if len(free_map) != len(label_free) or len(scored_map) != len(scored):
        raise RuntimeError("duplicate prediction keys")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    terminal_cache = {
        sample.sample_id: _terminal_spr_target(sample)
        for sample in samples
        if sample.dataset == "sprsound" and sample.partition == "test"
    }
    task_counts = {}
    for task, spec in TASK_SPECS.items():
        rows = sorted(
            [row for row in scored if row["task"] == task],
            key=lambda row: row["sample_id"],
        )
        probabilities = np.asarray(
            [json.loads(row["probabilities_json"]) for row in rows], dtype=float
        )
        predicted = np.asarray(
            [json.loads(row["pred_json"]) for row in rows], dtype=int
        )
        targets = []
        for row in rows:
            free = free_map[(row["sample_id"], task)]
            if (
                free["pred_json"] != row["pred_json"]
                or free["probabilities_json"] != row["probabilities_json"]
            ):
                raise RuntimeError("terminal join changed native prediction")
            sample = sample_by_id[row["sample_id"]]
            target = (
                terminal_cache[sample.sample_id][task]
                if sample.dataset == "sprsound" and sample.partition == "test"
                else sample.targets[task]
            )
            if json.loads(row["true_json"]) != target:
                raise RuntimeError("native target join failed")
            targets.append(target)
        target_array = np.asarray(targets, dtype=int)
        if spec["kind"] == "multiclass":
            if (
                not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                or not np.array_equal(predicted, probabilities.argmax(axis=1))
            ):
                raise RuntimeError(f"native probability gate failed: {task}")
            recomputed = _multiclass_metrics(
                target_array, predicted, spec["labels"], task
            )
            if int(np.asarray(recomputed["confusion"]).sum()) != len(rows):
                raise RuntimeError("native confusion total failed")
        else:
            if (
                not np.isfinite(probabilities).all()
                or not np.array_equal(predicted, (probabilities >= 0.5).astype(int))
            ):
                raise RuntimeError("native multilabel probability gate failed")
            recomputed = _multilabel_metrics(
                target_array, probabilities, spec["labels"]
            )
        _compare_metrics(recomputed, metrics["native_test_metrics"][task], task)
        task_counts[task] = len(rows)
    compatible_coverage = {}
    for task in COMPATIBLE_TASKS:
        ontology = "binary" if "binary" in task else "narrow4"
        dataset = "icbhi" if task.endswith("icbhi") else "sprsound"
        free_rows = [row for row in label_free if row["task"] == task]
        scored_rows = [row for row in scored if row["task"] == task]
        probabilities = np.asarray(
            [json.loads(row["probabilities_json"]) for row in scored_rows],
            dtype=float,
        )
        predicted = np.asarray(
            [json.loads(row["pred_json"]) for row in scored_rows], dtype=int
        )
        if (
            not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            or not np.array_equal(predicted, probabilities.argmax(axis=1))
        ):
            raise RuntimeError(f"compatible probability gate failed: {task}")
        targets = []
        expected_scored_ids = []
        excluded_ids = []
        for row in free_rows:
            sample = sample_by_id[row["sample_id"]]
            if dataset == "icbhi":
                native = int(sample.targets["icbhi_flat4"])
                target = int(native != 0) if ontology == "binary" else native
            else:
                source = terminal_cache[sample.sample_id]
                target = (
                    int(source["spr_binary"])
                    if ontology == "binary"
                    else SPR_NARROW_MAP.get(int(source["spr_seven"]))
                )
            if target is None:
                excluded_ids.append(sample.sample_id)
            else:
                expected_scored_ids.append(sample.sample_id)
        if [row["sample_id"] for row in scored_rows] != expected_scored_ids:
            raise RuntimeError(f"compatible eligibility set failed: {task}")
        for row in scored_rows:
            free = free_map[(row["sample_id"], task)]
            if (
                free["pred_json"] != row["pred_json"]
                or free["probabilities_json"] != row["probabilities_json"]
            ):
                raise RuntimeError("terminal join changed compatible prediction")
            sample = sample_by_id[row["sample_id"]]
            if dataset == "icbhi":
                native = int(sample.targets["icbhi_flat4"])
                target = int(native != 0) if ontology == "binary" else native
            else:
                source = terminal_cache[sample.sample_id]
                target = (
                    int(source["spr_binary"])
                    if ontology == "binary"
                    else SPR_NARROW_MAP[int(source["spr_seven"])]
                )
            if json.loads(row["true_json"]) != target:
                raise RuntimeError("compatible target join failed")
            targets.append(target)
        labels = ["normal", "abnormal"] if ontology == "binary" else NARROW_LABELS
        recomputed = _compatible_metrics(
            np.asarray(targets, dtype=int), predicted, labels
        )
        if int(np.asarray(recomputed["confusion"]).sum()) != len(scored_rows):
            raise RuntimeError("compatible confusion total failed")
        _compare_compatible(
            recomputed, metrics["compatible_test_metrics"][task], task
        )
        coverage = metrics["compatible_coverage"][task]
        if (
            coverage["label_free_prediction_rows"] != len(free_rows)
            or coverage["scored_rows"] != len(scored_rows)
            or coverage["excluded_rows"] != len(excluded_ids)
            or coverage["excluded_id_sha256"] != _ordered_sha(excluded_ids)
        ):
            raise RuntimeError(f"compatible coverage receipt failed: {task}")
        compatible_coverage[task] = {
            "label_free": len(free_rows),
            "scored": len(scored_rows),
            "excluded": len(excluded_ids),
        }
    return {
        "native_task_rows": task_counts,
        "compatible_coverage": compatible_coverage,
        "label_free_rows": len(label_free),
        "scored_rows": len(scored),
    }


def verify_gate(
    result_root: Path, dataset_root: Path, selected_cache: Path
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    selection = json.loads(SELECTION_RECEIPT.read_text())
    data = json.loads((result_root / "data_receipt.json").read_text())
    smoke = json.loads((result_root / "smoke_receipt.json").read_text())
    samples, _ = build_samples(dataset_root, 0)
    if (
        selection["selected_representation"]
        != protocol["encoder_selection_dependency"]["selected_representation"]
        or selection["selected_cache_sha256"] != SELECTED_CACHE_SHA256
        or sha256_file(selected_cache) != SELECTED_CACHE_SHA256
        or data["protocol_sha256"] != sha256_file(PROTOCOL_PATH)
        or data["rows"] != 25084
        or data["ordered_id_sha256"]
        != _ordered_sha([sample.sample_id for sample in samples])
    ):
        raise RuntimeError("selection/data/cache gate failed")
    checks = smoke["gradient_and_parameter_checks"]
    if (
        smoke["status"] != "harmonization_real_data_smoke_passed"
        or not smoke["finite"]
        or not smoke["spr_label_free_then_terminal_join"]
        or not checks["hf_kauh_missing_compatible_label_zero_gradient"]
        or checks["h1_compatible_parameters"] != 1548
        or checks["h2_compatible_parameters"] != 1548
        or checks["fixed_projection_trainable"]
    ):
        raise RuntimeError("smoke routing/parameter gate failed")
    pairs = {}
    for condition in CONDITIONS:
        directory = result_root / "smoke" / condition
        metrics = json.loads((directory / "metrics.json").read_text())
        pairs[condition] = _verify_pair(directory, samples, metrics)
    return {
        "status": "harmonization_gate_verified",
        "rows": 25084,
        "conditions": len(CONDITIONS),
        "smoke_prediction_pairs": len(pairs),
        "selected_cache_sha256": SELECTED_CACHE_SHA256,
    }


def verify_full(
    result_root: Path, dataset_root: Path, selected_cache: Path
) -> dict[str, object]:
    gate = verify_gate(result_root, dataset_root, selected_cache)
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    prediction_pairs = 0
    kauh_sets = []
    for fold, samples in enumerate(samples_by_fold):
        kauh_sets.append(
            {
                sample.sample_id
                for sample in samples
                if sample.dataset == "kauh" and sample.partition == "test"
            }
        )
        for condition in CONDITIONS:
            directory = result_root / f"fold_{fold}" / condition
            metrics = json.loads((directory / "metrics.json").read_text())
            state = torch.load(directory / "best.pth", map_location="cpu")
            if (
                metrics["condition"] != condition
                or any("test" in json.dumps(row).lower() for row in metrics["history"])
                or "validation" not in json.dumps(metrics["selection"]).lower()
                or not all(
                    torch.isfinite(value).all()
                    for value in state["model"].values()
                    if torch.is_tensor(value)
                )
            ):
                raise RuntimeError("full selection/checkpoint gate failed")
            model = HarmonizationModel(condition)
            if metrics["parameters"]["compatible"] != model.compatible_parameter_count():
                raise RuntimeError("compatible parameter identity failed")
            _verify_pair(directory, samples, metrics)
            prediction_pairs += 1
    if (
        len(set().union(*kauh_sets)) != 336
        or sum(len(values) for values in kauh_sets) != 336
        or any(
            kauh_sets[left] & kauh_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("KAUH patient OOF partition failed")
    before = json.loads((result_root / "harmonization_decision.json").read_text())
    recomputed = analyze(result_root)
    if before != recomputed:
        raise RuntimeError("harmonization decision is not reproducible")
    with (result_root / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    with (result_root / "per_class_summary.csv").open(newline="") as handle:
        per_class = list(csv.DictReader(handle))
    if len(summary) != len(CONDITIONS) * (len(TASK_SPECS) + len(COMPATIBLE_TASKS)):
        raise RuntimeError("summary row count failed")
    return {
        **gate,
        "status": "harmonization_full_verified",
        "full_prediction_pairs": prediction_pairs,
        "summary_rows": len(summary),
        "per_class_rows": len(per_class),
        "decision": recomputed["decision"],
        "gates": recomputed["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gate", "full"], default="full")
    parser.add_argument(
        "--result-root", type=Path, default=Path(f"result/{EXPERIMENT_ID}")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--selected-cache",
        type=Path,
        default=Path(
            ".cache/four_dataset_representation_attribution/"
            "r1_beats_as2m_audioset_only/embeddings.npz"
        ),
    )
    args = parser.parse_args()
    receipt = (
        verify_gate(
            args.result_root.resolve(),
            args.dataset_root.resolve(),
            args.selected_cache.resolve(),
        )
        if args.mode == "gate"
        else verify_full(
            args.result_root.resolve(),
            args.dataset_root.resolve(),
            args.selected_cache.resolve(),
        )
    )
    path = args.result_root / (
        "gate_verification.json" if args.mode == "gate" else "verification.json"
    )
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
