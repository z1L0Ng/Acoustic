"""Independently verify the corrected support-aware cRT control."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    KAUH_LABELS,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import SharedNativeModel, TASK_SPECS
from baseline.four_dataset_frozen_encoder.verify import (
    _multiclass_metrics,
    _verify_prediction_pair,
)
from baseline.four_dataset_support_aware_crt.run import (
    CACHE_PATH,
    CACHE_SHA256,
    CONTRACT_PATH,
    D2_CONDITION,
    GATE_B_ROOT,
    PROTOCOL_PATH,
    R0,
    RESULT_ROOT,
    T0,
    T1,
    TASK_ID_MAP,
    _module_digest,
    _read_dependencies,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: object, expected: object, name: str) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise RuntimeError(f"{name} null mismatch")
        return
    if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-12):
        raise RuntimeError(f"{name} mismatch: {actual} != {expected}")


def _state_digest(state: dict[str, torch.Tensor], prefix: str) -> str:
    subset = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not subset:
        raise RuntimeError(f"missing checkpoint state prefix: {prefix}")
    digest = hashlib.sha256()
    for key in sorted(subset):
        value = subset[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _eligibility(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    for row in contract["label_assignments"]:
        task = TASK_ID_MAP.get(str(row["task_id"]), str(row["task_id"]))
        record = tasks.setdefault(
            task,
            {
                "primary": [],
                "diagnostic": [],
                "not_evaluable": [],
                "support": {},
            },
        )
        bucket = {
            "primary_evaluable": "primary",
            "diagnostic_only": "diagnostic",
            "not_evaluable": "not_evaluable",
        }[str(row["eligibility"])]
        label = str(row["label"])
        record[bucket].append(label)
        support = row["support"]
        record["support"][label] = int(
            support["subtrain"][0]
            if "subtrain" in support
            else support["full_release"][0]
        )
    if set(tasks) != set(TASK_SPECS):
        raise RuntimeError("tail contract task coverage mismatch")
    for task, record in tasks.items():
        if set(record["primary"] + record["diagnostic"] + record["not_evaluable"]) != set(
            TASK_SPECS[task]["labels"]
        ):
            raise RuntimeError(f"tail contract label coverage mismatch: {task}")
        maximum = max(record["support"][label] for label in record["primary"])
        tail = [
            label
            for label in record["primary"]
            if record["support"][label] < maximum
        ]
        record["tail"] = tail or list(record["primary"])
    return tasks


def _verify_sampling(
    samples: list,
    task: str,
    sampling: object,
) -> None:
    spec = TASK_SPECS[task]
    indices = [
        index
        for index, sample in enumerate(samples)
        if sample.partition == "subtrain" and task in sample.targets
    ]
    if spec["kind"] == "multiclass":
        actual: dict[int, int] = {}
        for index in indices:
            label = int(samples[index].targets[task])
            actual[label] = actual.get(label, 0) + 1
        for epoch in range(1, 6):
            row = sampling[str(epoch)]
            source = {int(key): int(value) for key, value in row["source_class_counts"].items()}
            drawn = {int(key): int(value) for key, value in row["draw_class_counts"].items()}
            if (
                int(row["eligible_rows"]) != len(indices)
                or source != actual
                or sum(drawn.values()) != len(indices)
                or set(drawn) != set(actual)
                or max(drawn.values()) - min(drawn.values()) > 1
                or not row["replacement"]
            ):
                raise RuntimeError(f"multiclass sampling mismatch: {task}/{epoch}")
        return

    if set(sampling) != set(spec["labels"]):
        raise RuntimeError(f"multilabel sampling label mismatch: {task}")
    for label_index, label in enumerate(spec["labels"]):
        target = np.asarray(
            [samples[index].targets[task][label_index] for index in indices],
            dtype=int,
        )
        negative = int(np.sum(target == 0))
        positive = int(np.sum(target == 1))
        if not negative or not positive:
            raise RuntimeError(f"missing observed polarity: {task}/{label}")
        for epoch in range(1, 6):
            row = sampling[label][str(epoch)]
            if (
                int(row["eligible_rows"]) != len(indices)
                or int(row["observed_negative_rows"]) != negative
                or int(row["observed_positive_rows"]) != positive
                or int(row["draw_negative_rows"]) + int(row["draw_positive_rows"])
                != len(indices)
                or abs(
                    int(row["draw_negative_rows"])
                    - int(row["draw_positive_rows"])
                )
                > 1
                or not row["replacement"]
                or not row["unknown_not_annotated_rows_omitted"]
            ):
                raise RuntimeError(f"multilabel sampling mismatch: {task}/{label}/{epoch}")


def _metric_from_oof(rows: list[dict[str, str]]) -> dict[str, object]:
    target = np.asarray([json.loads(row["true_json"]) for row in rows], dtype=int)
    predicted = np.asarray([json.loads(row["pred_json"]) for row in rows], dtype=int)
    return _multiclass_metrics(target, predicted, KAUH_LABELS, "kauh_raw9")


def _eligible_values(
    metrics: dict[str, object],
    eligibility: dict[str, object],
) -> tuple[float, float]:
    per_class = metrics["per_class"]
    primary = eligibility["primary"]
    tail = eligibility["tail"]
    return (
        float(np.mean([float(per_class[label]["f1"]) for label in primary])),
        float(np.mean([float(per_class[label]["recall"]) for label in tail])),
    )


def _verify_summary(
    recorded: list[dict[str, str]],
    per_class_recorded: list[dict[str, str]],
    metrics: dict[tuple[str, int, str], dict[str, object]],
    eligibility: dict[str, dict[str, object]],
) -> dict[str, dict[str, float]]:
    by_key = {(row["condition"], row["task"]): row for row in recorded}
    if len(recorded) != 12 or len(per_class_recorded) != 56:
        raise RuntimeError("summary row count mismatch")
    task_values: dict[str, dict[str, float]] = {}
    for condition in (T0, T1):
        for task, spec in TASK_SPECS.items():
            row = by_key[(condition, task)]
            if task == "kauh_raw9":
                runs = [metrics[(condition, -1, task)]]
            else:
                runs = [metrics[(condition, fold, task)] for fold in range(5)]
            eligible = [_eligible_values(run, eligibility[task]) for run in runs]
            _assert_close(
                row["eligible_macro_f1_mean"],
                np.mean([value[0] for value in eligible]),
                f"{condition}.{task}.eligible_macro_f1",
            )
            _assert_close(
                row["tail_recall_mean"],
                np.mean([value[1] for value in eligible]),
                f"{condition}.{task}.tail_recall",
            )
            for metric in (
                "macro_f1",
                "weighted_f1",
                "micro_f1",
                "uar",
                "native_score",
                "specificity",
            ):
                available = [float(run[metric]) for run in runs if metric in run]
                key = f"{metric}_mean"
                if available:
                    _assert_close(row[key], np.mean(available), f"{condition}.{task}.{metric}")
                elif row.get(key) not in {None, ""}:
                    raise RuntimeError(f"unexpected summary metric: {condition}/{task}/{metric}")
            task_values[f"{condition}:{task}"] = {
                "eligible_macro_f1": float(np.mean([value[0] for value in eligible])),
                "tail_recall": float(np.mean([value[1] for value in eligible])),
            }
    return task_values


def _verify_decision(
    decision: dict[str, object],
    summary: list[dict[str, str]],
) -> None:
    by_key = {(row["condition"], row["task"]): row for row in summary}
    material = []
    guardrails = []
    for task, spec in TASK_SPECS.items():
        t0 = by_key[(T0, task)]
        t1 = by_key[(T1, task)]
        eligible_gap = float(t1["eligible_macro_f1_mean"]) - float(
            t0["eligible_macro_f1_mean"]
        )
        tail_gap = float(t1["tail_recall_mean"]) - float(t0["tail_recall_mean"])
        constrained = (
            ["micro_f1"]
            if spec["kind"] == "multilabel"
            else ["weighted_f1", "native_score", "specificity"]
        )
        regressions = {}
        for metric in constrained:
            key = f"{metric}_mean"
            if t0.get(key) not in {None, ""} and t1.get(key) not in {None, ""}:
                regressions[metric] = float(t1[key]) - float(t0[key])
        material_improvement = max(eligible_gap, tail_gap) >= 0.03
        guardrail_pass = all(value >= -0.03 for value in regressions.values())
        if material_improvement:
            material.append(task)
        guardrails.append(guardrail_pass)
        recorded = decision["tasks"][task]
        _assert_close(
            recorded["eligible_macro_f1_delta"],
            eligible_gap,
            f"decision.{task}.eligible",
        )
        _assert_close(
            recorded["tail_recall_delta"],
            tail_gap,
            f"decision.{task}.tail",
        )
        if (
            bool(recorded["material_improvement"]) != material_improvement
            or bool(recorded["guardrail_pass"]) != guardrail_pass
        ):
            raise RuntimeError(f"decision task gate mismatch: {task}")
    expected = (
        "go_support_aware_classifier_retraining"
        if len(material) >= 2 and all(guardrails)
        else "hold_or_negative_support_aware_classifier_retraining"
    )
    if (
        decision["material_improvement_tasks"] != material
        or int(decision["material_improvement_count"]) != len(material)
        or bool(decision["all_regression_guardrails_pass"]) != all(guardrails)
        or decision["decision"] != expected
    ):
        raise RuntimeError("decision gate mismatch")


def verify(dataset_root: Path) -> dict[str, object]:
    protocol, contract = _read_dependencies()
    preregistration = json.loads((RESULT_ROOT / "preregistration.json").read_text())
    smoke = json.loads((RESULT_ROOT / "smoke_receipt.json").read_text())
    if (
        preregistration["status"] != "preregistered_before_t1_outer_test_scoring"
        or preregistration["outer_test_metrics_read_by_this_run"]
        or smoke["status"] != "support_aware_crt_smoke_passed"
        or not smoke["finite_loss_gradient"]
        or not smoke["missing_not_annotated_omitted"]
    ):
        raise RuntimeError("preregistration/smoke gate failed")
    if (
        sha256_file(PROTOCOL_PATH) != preregistration["protocol_sha256"]
        or sha256_file(CONTRACT_PATH) != preregistration["tail_contract_sha256"]
        or sha256_file(CACHE_PATH) != CACHE_SHA256
    ):
        raise RuntimeError("protocol/contract/cache identity failed")

    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    canonical = samples_by_fold[0]
    embeddings, _ = load_cache(CACHE_PATH, canonical)
    if embeddings.shape != (25_084, 768) or not np.isfinite(embeddings).all():
        raise RuntimeError("cache alignment/finite gate failed")
    if any(
        receipt["datasets"]["hf_lung"]["assignment_sha256"]
        != EXPECTED_HF_ASSIGNMENT_SHA256
        for _, receipt in [build_samples(dataset_root, fold) for fold in range(5)]
    ):
        raise RuntimeError("corrected HF assignment failed")
    kauh_sets = [
        {
            sample.sample_id
            for sample in samples
            if sample.dataset == "kauh" and sample.partition == "test"
        }
        for samples in samples_by_fold
    ]
    if (
        len(set().union(*kauh_sets)) != 336
        or sum(map(len, kauh_sets)) != 336
        or any(
            kauh_sets[left] & kauh_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("KAUH patient OOF partition failed")

    eligibility = _eligibility(contract)
    metrics: dict[tuple[str, int, str], dict[str, object]] = {}
    prediction_pairs = 0
    for fold, samples in enumerate(samples_by_fold):
        for condition in (T0, T1):
            directory = (
                GATE_B_ROOT
                / R0
                / f"fold_{fold}"
                / D2_CONDITION
                if condition == T0
                else RESULT_ROOT / f"fold_{fold}" / T1
            )
            recorded = json.loads((directory / "metrics.json").read_text())
            checkpoint_path = directory / "best.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not all(
                torch.isfinite(value).all()
                for value in checkpoint["model"].values()
                if torch.is_tensor(value)
            ):
                raise RuntimeError(f"non-finite checkpoint: {condition}/{fold}")
            counts = _verify_prediction_pair(
                directory / "predictions_label_free.csv.gz",
                directory / "predictions.csv.gz",
                samples,
                recorded["test_metrics"],
            )
            if set(counts) != set(TASK_SPECS):
                raise RuntimeError("prediction task coverage failed")
            prediction_pairs += 1
            for task, values in recorded["test_metrics"].items():
                metrics[(condition, fold, task)] = values

            if condition == T1:
                source_path = (
                    GATE_B_ROOT
                    / R0
                    / f"fold_{fold}"
                    / D2_CONDITION
                    / "best.pth"
                )
                source = torch.load(source_path, map_location="cpu")
                receipt = recorded["classifier_retraining"]
                checkpoint_receipt = checkpoint["classifier_retraining"]
                source_model = SharedNativeModel()
                source_model.load_state_dict(source["model"], strict=True)
                torch.manual_seed(20260728)
                expected_reinitialized = SharedNativeModel()
                if (
                    checkpoint["condition"] != T1
                    or receipt["source_checkpoint"]["sha256"] != sha256_file(source_path)
                    or receipt["source_checkpoint"]["sha256"]
                    != checkpoint_receipt["source_checkpoint"]["sha256"]
                    or receipt["adapter_trainable_parameters"] != 0
                    or receipt["source_adapter_digest"]
                    != receipt["initial_adapter_digest"]
                    or receipt["source_adapter_digest"]
                    != _module_digest(source_model.adapter)
                    or receipt["source_heads_digest"]
                    != _module_digest(source_model.heads)
                    or receipt["reinitialized_heads_digest"]
                    != _module_digest(expected_reinitialized.heads)
                    or receipt["source_adapter_digest"]
                    != receipt["final_adapter_digest"]
                    or receipt["source_heads_digest"]
                    == receipt["reinitialized_heads_digest"]
                    or receipt["reinitialized_heads_digest"]
                    == receipt["final_heads_digest"]
                    or _state_digest(source["model"], "adapter.")
                    != _state_digest(checkpoint["model"], "adapter.")
                    or "test" in json.dumps(recorded["history"]).lower()
                    or "validation" not in json.dumps(recorded["selection"]).lower()
                ):
                    raise RuntimeError(f"frozen adapter/reinitialized head gate failed: {fold}")
                for task in TASK_SPECS:
                    _verify_sampling(samples, task, receipt["sampling"][task])

    for condition in (T0, T1):
        rows = []
        for fold in range(5):
            directory = (
                GATE_B_ROOT / R0 / f"fold_{fold}" / D2_CONDITION
                if condition == T0
                else RESULT_ROOT / f"fold_{fold}" / T1
            )
            rows.extend(
                row
                for row in _read_predictions(directory / "predictions.csv.gz")
                if row["task"] == "kauh_raw9"
            )
        ids = [row["sample_id"] for row in rows]
        if len(ids) != 336 or len(ids) != len(set(ids)):
            raise RuntimeError(f"KAUH OOF prediction coverage failed: {condition}")
        metrics[(condition, -1, "kauh_raw9")] = _metric_from_oof(rows)

    summary = _read_csv(RESULT_ROOT / "summary.csv")
    per_class = _read_csv(RESULT_ROOT / "per_class_summary.csv")
    task_values = _verify_summary(summary, per_class, metrics, eligibility)
    decision = json.loads((RESULT_ROOT / "decision.json").read_text())
    _verify_decision(decision, summary)
    receipt = {
        "status": "support_aware_crt_full_verified",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "tail_contract_sha256": sha256_file(CONTRACT_PATH),
        "cache_sha256": CACHE_SHA256,
        "representation": R0,
        "selection_caveat": "ICBHI official-test-selected PAFA task encoder",
        "target_supervised": True,
        "rows": len(canonical),
        "folds": 5,
        "prediction_pairs": prediction_pairs,
        "kauh_oof_rows_per_condition": 336,
        "summary_rows": len(summary),
        "per_class_rows": len(per_class),
        "task_values": task_values,
        "decision": decision["decision"],
        "warnings": 0,
        "finite": True,
        "spr_label_free_terminal_join": True,
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
    }
    temporary = RESULT_ROOT / "verification.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULT_ROOT / "verification.json")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    receipt = verify(args.dataset_root.resolve())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
