"""Independent verification for the preregistered residual experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import build_samples
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS
from baseline.four_dataset_frozen_encoder.verify import _verify_prediction_pair
from model.four_dataset_general_specific_v1.models import (
    CONDITIONS,
    EXPECTED_PARAMETERS,
    GeneralSpecificModel,
    parameter_count,
)
from model.four_dataset_general_specific_v1.run import (
    DATASET_ROOT,
    DEFAULT_CACHE,
    DEFAULT_RESULT,
    ELIGIBLE_PRIOR_TASKS,
    EXPECTED_CACHE_SHA256,
    assert_sample_alignment,
    load_cache,
    loss_route,
)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_smoke(
    result_root: Path, cache_path: Path
) -> dict[str, object]:
    receipt = json.loads((result_root / "smoke_receipt.json").read_text())
    if (
        receipt["status"] != "four_dataset_general_specific_smoke_passed"
        or receipt["outer_test_metrics_evaluated"]
        or receipt["warning_count"] != 0
        or receipt["cache"]["cache_sha256"] != EXPECTED_CACHE_SHA256
    ):
        raise RuntimeError("smoke receipt boundary failed")
    counts = []
    for condition in CONDITIONS:
        model = GeneralSpecificModel(condition)
        count = parameter_count(model)
        if (
            count != EXPECTED_PARAMETERS[condition]
            or receipt["conditions"][condition]["parameters"] != count
            or not receipt["conditions"][condition]["finite_gradients"]
        ):
            raise RuntimeError(f"smoke model gate failed for {condition}")
        counts.append(count)
    if counts[1:] != [303900, 303900, 303900]:
        raise RuntimeError("parameter-matched condition gate failed")
    for task in TASK_SPECS:
        expected = (
            "logit_adjustment"
            if task == "kauh_raw9"
            else "pos_weight_bce"
            if task == "hf_adventitious_presence"
            else "cross_entropy"
            if TASK_SPECS[task]["kind"] == "multiclass"
            else "bce"
        )
        if loss_route("d2_task_residual_selective_prior", task) != expected:
            raise RuntimeError(f"selective route failed for {task}")
    sample_ids, _, _ = load_cache(cache_path)
    samples, _ = build_samples(DATASET_ROOT, 0)
    assert_sample_alignment(samples, sample_ids, 0)
    return {
        "status": "four_dataset_general_specific_smoke_verified",
        "conditions": len(CONDITIONS),
        "parameters": EXPECTED_PARAMETERS,
        "eligible_prior_tasks": sorted(ELIGIBLE_PRIOR_TASKS),
        "warnings": 0,
        "outer_test_metrics_evaluated": False,
    }


def verify_full(
    result_root: Path, cache_path: Path
) -> dict[str, object]:
    smoke = verify_smoke(result_root, cache_path)
    manifest = json.loads((result_root / "run_manifest.json").read_text())
    if (
        manifest["status"] != "four_dataset_general_specific_full_complete"
        or manifest["models"] != 20
        or manifest["conditions"] != list(CONDITIONS)
        or manifest["outer_test_used_for_selection"]
        or manifest["warning_count"] != 0
        or manifest["cache"]["cache_sha256"] != EXPECTED_CACHE_SHA256
    ):
        raise RuntimeError("full manifest gate failed")
    sample_ids, _, _ = load_cache(cache_path)
    condition_counts = Counter()
    task_prediction_counts: dict[str, dict[str, int]] = {}
    kauh_oof: dict[str, list[str]] = {condition: [] for condition in CONDITIONS}
    label_free_spr_rows = 0
    for fold in range(5):
        samples, data_receipt = build_samples(DATASET_ROOT, fold)
        assert_sample_alignment(samples, sample_ids, fold)
        if data_receipt["datasets"]["kauh"]["patient_overlap"] != 0:
            raise RuntimeError(f"KAUH patient overlap in fold {fold}")
        for condition in CONDITIONS:
            directory = result_root / f"fold_{fold}" / condition
            metrics = json.loads((directory / "metrics.json").read_text())
            checkpoint = torch.load(
                directory / "best.pth", map_location="cpu", weights_only=False
            )
            if (
                metrics["parameters"] != EXPECTED_PARAMETERS[condition]
                or metrics["warning_count"] != 0
                or checkpoint["cache_sha256"] != EXPECTED_CACHE_SHA256
                or checkpoint["loss_routes"] != metrics["loss_routes"]
                or not all(
                    torch.isfinite(value).all()
                    for value in checkpoint["model"].values()
                    if torch.is_tensor(value)
                )
            ):
                raise RuntimeError(
                    f"artifact/provenance gate failed fold={fold} condition={condition}"
                )
            counts = _verify_prediction_pair(
                directory / "predictions_label_free.csv.gz",
                directory / "predictions.csv.gz",
                samples,
                metrics["test_metrics"],
            )
            task_prediction_counts[f"{fold}:{condition}"] = counts
            free_rows = read_predictions(
                directory / "predictions_label_free.csv.gz"
            )
            label_free_spr_rows += sum(
                row["dataset"] == "sprsound" for row in free_rows
            )
            scored_rows = read_predictions(directory / "predictions.csv.gz")
            kauh_rows = [
                row for row in scored_rows if row["task"] == "kauh_raw9"
            ]
            kauh_oof[condition].extend(
                row["sample_id"] for row in kauh_rows
            )
            condition_counts[condition] += 1
    for condition, ids in kauh_oof.items():
        if len(ids) != 336 or len(set(ids)) != 336:
            raise RuntimeError(f"KAUH OOF coverage failed for {condition}")
    with (result_root / "summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with (result_root / "per_class_summary.csv").open(newline="") as handle:
        per_class_rows = list(csv.DictReader(handle))
    with (result_root / "task_fold_results.csv").open(newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    if (
        condition_counts != Counter({condition: 5 for condition in CONDITIONS})
        or len(summary_rows) != 24
        or len(per_class_rows) != 112
        or len(fold_rows) != 120
        or label_free_spr_rows != 4 * 5 * 1429 * 2
    ):
        raise RuntimeError("aggregate row-count gate failed")
    return {
        "status": "four_dataset_general_specific_full_verified",
        "smoke": smoke,
        "models": sum(condition_counts.values()),
        "condition_counts": dict(condition_counts),
        "cache_sha256_verified": True,
        "sample_rows": len(sample_ids),
        "kauh_oof_unique_rows_per_condition": 336,
        "spr_label_free_rows_across_two_tasks": label_free_spr_rows,
        "summary_rows": len(summary_rows),
        "per_class_summary_rows": len(per_class_rows),
        "task_fold_rows": len(fold_rows),
        "patient_overlap": 0,
        "warnings": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    payload = (
        verify_smoke(args.result_root, args.cache)
        if args.mode == "smoke"
        else verify_full(args.result_root, args.cache)
    )
    output = args.result_root / f"{args.mode}_verification.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
