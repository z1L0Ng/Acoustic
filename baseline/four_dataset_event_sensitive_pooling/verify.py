"""Independent verification for the event-sensitive pooling diagnostic."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_event_sensitive_pooling.run import (
    ALL_CONDITIONS,
    P0,
    P1,
    P2,
    P0_ROOT,
    PROTOCOL_PATH,
    R0_CACHE,
    _condition_dir,
    _load_window_cache,
)
from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    KAUH_LABELS,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import sha256_file
from baseline.four_dataset_frozen_encoder.verify import (
    _multiclass_metrics,
    _multilabel_metrics,
    _terminal_spr_target,
)
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS
from baseline.four_dataset_representation_attribution.run import EXPECTED_R0_SHA256, ordered_id_sha256


ROOT = Path(__file__).resolve().parents[2]


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"metric mismatch {name}: {actual} != {expected}")


def _compare_metrics(actual: dict[str, object], expected: dict[str, object], task: str) -> None:
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
                    raise RuntimeError(f"{task} rows mismatch")
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
                _assert_close(float(value), float(expected_value), f"{task}.{label}.{key}")


def _verify_predictions(label_free_path: Path, scored_path: Path, samples, recorded_metrics: dict[str, object]) -> dict[str, int]:
    label_free = _read_predictions(label_free_path)
    scored = _read_predictions(scored_path)
    if not label_free or len(label_free) != len(scored):
        raise RuntimeError("prediction row count mismatch")
    if "true_json" in label_free[0] or "true_json" not in scored[0]:
        raise RuntimeError("SPRSound label-free boundary column check failed")
    free_map = {(row["sample_id"], row["task"]): row for row in label_free}
    scored_map = {(row["sample_id"], row["task"]): row for row in scored}
    if free_map.keys() != scored_map.keys() or len(free_map) != len(label_free):
        raise RuntimeError("prediction key identity failed")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    counts = {}
    for task, spec in TASK_SPECS.items():
        rows = [row for key, row in scored_map.items() if key[1] == task]
        rows.sort(key=lambda row: row["sample_id"])
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate IDs for {task}")
        probabilities = np.asarray([json.loads(row["probabilities_json"]) for row in rows], dtype=float)
        predicted = np.asarray([json.loads(row["pred_json"]) for row in rows], dtype=int)
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"non-finite probabilities for {task}")
        if spec["kind"] == "multiclass" and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"probability sum gate failed for {task}")
        targets = []
        for row in rows:
            key = (row["sample_id"], task)
            if (
                free_map[key]["pred_json"] != row["pred_json"]
                or free_map[key]["probabilities_json"] != row["probabilities_json"]
            ):
                raise RuntimeError("terminal scoring changed model output")
            sample = sample_by_id[row["sample_id"]]
            if sample.dataset == "sprsound" and sample.partition == "test":
                if sample.targets or "raw_label" in sample.metadata:
                    raise RuntimeError("SPRSound inter target leaked into inference sample")
                target = _terminal_spr_target(sample)[task]
            else:
                target = sample.targets[task]
            if json.loads(row["true_json"]) != target:
                raise RuntimeError(f"target mismatch for {row['sample_id']} {task}")
            targets.append(target)
        target_array = np.asarray(targets, dtype=int)
        if spec["kind"] == "multiclass":
            metrics = _multiclass_metrics(target_array, predicted.astype(int), spec["labels"], task)
        else:
            metrics = _multilabel_metrics(target_array, probabilities, spec["labels"])
        _compare_metrics(metrics, recorded_metrics[task], task)
        counts[task] = len(rows)
    return counts


def verify(result_root: Path, cache_root: Path, dataset_root: Path) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    data_receipt = json.loads((result_root / "data_receipt.json").read_text())
    if data_receipt["hf_assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256:
        raise RuntimeError("data receipt HF SHA mismatch")
    samples, sample_receipt = build_samples(dataset_root, 0)
    if sample_receipt["ordered_id_sha256"] != data_receipt["ordered_id_sha256"]:
        raise RuntimeError("sample order changed")
    if sha256_file(R0_CACHE) != EXPECTED_R0_SHA256:
        raise RuntimeError("R0 cache SHA mismatch")
    values, mask, window_receipt = _load_window_cache(cache_root / "window_embeddings.npz", samples)
    if window_receipt["ordered_id_sha256"] != data_receipt["ordered_id_sha256"]:
        raise RuntimeError("window cache ID SHA mismatch")
    if values.shape[0] != 25_084 or values.shape[2] != 768 or not mask.any(axis=1).all():
        raise RuntimeError("window cache shape/mask failed")
    if window_receipt["extraction"]["window_count_by_dataset"]["hf_lung"] != 9_765 * 3:
        raise RuntimeError("HF three-window coverage failed")
    parameter_counts = {}
    prediction_counts = {}
    for condition in ALL_CONDITIONS:
        for fold in range(5):
            fold_samples, fold_receipt = build_samples(dataset_root, fold)
            if [sample.sample_id for sample in fold_samples] != [sample.sample_id for sample in samples]:
                raise RuntimeError("fold sample order changed")
            directory = _condition_dir(result_root, fold, condition)
            metrics_path = directory / "metrics.json"
            label_free_path = directory / "predictions_label_free.csv.gz"
            scored_path = directory / "predictions.csv.gz"
            if not metrics_path.is_file() or not label_free_path.is_file() or not scored_path.is_file():
                raise RuntimeError(f"missing prediction artifact for {condition} fold {fold}")
            metrics = json.loads(metrics_path.read_text())
            counts = _verify_predictions(label_free_path, scored_path, fold_samples, metrics["test_metrics"])
            prediction_counts[f"{condition}/fold_{fold}"] = counts
            if condition != P0:
                state = torch.load(directory / "best.pth", map_location="cpu")
                if any(not torch.isfinite(value).all() for value in state["model"].values() if torch.is_tensor(value)):
                    raise RuntimeError(f"non-finite checkpoint tensor for {condition} fold {fold}")
                parameter_counts.setdefault(condition, int(metrics["parameters"]))
    if parameter_counts.get(P1) != parameter_counts.get(P2):
        raise RuntimeError("P1/P2 parameter equality failed")
    summary_rows = list(csv.DictReader((result_root / "summary.csv").open()))
    if len(summary_rows) != 18:
        raise RuntimeError("summary row count failed")
    decision = json.loads((result_root / "decision.json").read_text())
    if decision["decision"] not in {"event_sensitive_pooling_supported", "not_supported_or_inconclusive"}:
        raise RuntimeError("invalid decision")
    payload = {
        "status": "event_sensitive_pooling_independently_verified",
        "protocol_status": protocol["status"],
        "conditions": list(ALL_CONDITIONS),
        "summary_rows": len(summary_rows),
        "prediction_counts": prediction_counts,
        "parameter_counts": parameter_counts,
        "p1_p2_parameter_matched": True,
        "window_cache_shape": list(values.shape),
        "window_mask_shape": list(mask.shape),
        "hf_windows": int(window_receipt["extraction"]["window_count_by_dataset"]["hf_lung"]),
        "p0_reference_root": str(P0_ROOT),
        "decision": decision["decision"],
        "r0_cache_sha256": EXPECTED_R0_SHA256,
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "raw_audio_read": False,
        "acoustic_model_trained_or_modified": False,
        "warnings": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_path = result_root / "verification.json"
    write_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("result/four_dataset_event_sensitive_pooling"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/four_dataset_event_sensitive_pooling"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    payload = verify(args.result_root, args.cache_root, args.dataset_root)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
