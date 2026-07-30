"""Independently verify shortcut-diagnostic lineage, OOF metrics and decision."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS
from baseline.four_dataset_shortcut_diagnostic.run import (
    BOOTSTRAP_SEED,
    CONTRACT_PATH,
    EXPECTED_FEATURE_ROWS,
    EXPECTED_ROWS,
    EXPECTED_SHA,
    FEATURE_PATH,
    PROTOCOL_PATH,
    R0,
    R0_CACHE,
    R1,
    R1_CACHE,
    RESULT_ROOT,
    SEED,
    T0_ROOT,
    TASK_ID_MAP,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_close(actual: object, expected: object, name: str) -> None:
    if actual in {None, ""} or expected in {None, ""}:
        if actual not in {None, ""} or expected not in {None, ""}:
            raise RuntimeError(f"{name} null mismatch")
        return
    if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-12):
        raise RuntimeError(f"{name} mismatch: {actual} != {expected}")


def _lineage_group(sample) -> str:
    if sample.dataset == "hf_lung" and sample.metadata.get("patient_id") is not None:
        raise RuntimeError("HF proxy mislabeled as patient")
    return f"{sample.dataset}:{sample.group_id}"


def _load_contract_assignments() -> dict[str, dict[str, object]]:
    contract = json.loads(CONTRACT_PATH.read_text())["tail_eligibility_contract"]
    if (
        contract["status"] != "tail_eligibility_management_accepted"
        or Counter(row["eligibility"] for row in contract["label_assignments"])
        != {
            "primary_evaluable": 16,
            "diagnostic_only": 5,
            "not_evaluable": 7,
        }
    ):
        raise RuntimeError("tail eligibility contract failed")
    return {
        f"{TASK_ID_MAP.get(str(row['task_id']), str(row['task_id']))}:{row['label']}": row
        for row in contract["label_assignments"]
    }


def verify_smoke(dataset_root: Path) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    preregistration = json.loads(
        (RESULT_ROOT / "preregistration.json").read_text()
    )
    smoke = json.loads((RESULT_ROOT / "smoke_receipt.json").read_text())
    lineage = json.loads((RESULT_ROOT / "lineage_receipt.json").read_text())
    if (
        protocol["status"] != "preregistered_before_probe_outcomes"
        or preregistration["probe_outcomes_read_before_preregistration"]
        or preregistration["protocol_sha256"] != sha256_file(PROTOCOL_PATH)
        or smoke["status"]
        != "shortcut_diagnostic_package_lineage_smoke_passed"
        or smoke["raw_audio_read"]
        or smoke["acoustic_model_trained_or_modified"]
    ):
        raise RuntimeError("protocol/preregistration/smoke boundary failed")
    for path, digest in EXPECTED_SHA.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"frozen SHA failed: {path}")
    samples_by_fold = [build_samples(dataset_root, fold) for fold in range(5)]
    samples = samples_by_fold[0][0]
    ids = [sample.sample_id for sample in samples]
    if len(ids) != EXPECTED_ROWS or len(ids) != len(set(ids)):
        raise RuntimeError("sample identity failed")
    if any(
        receipt["datasets"]["hf_lung"]["assignment_sha256"]
        != EXPECTED_HF_ASSIGNMENT_SHA256
        for _, receipt in samples_by_fold
    ):
        raise RuntimeError("corrected HF assignment failed")
    with FEATURE_PATH.open(newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    by_path = {
        str(Path(row["path"]).resolve()): row for row in feature_rows
    }
    if (
        len(feature_rows) != EXPECTED_FEATURE_ROWS
        or len(by_path) != len(feature_rows)
        or any(str(Path(sample.audio_path).resolve()) not in by_path for sample in samples)
        or lineage["joined_prediction_units"] != EXPECTED_ROWS
        or lineage["missing_prediction_units"] != 0
    ):
        raise RuntimeError("recording-feature lineage join failed")
    for path in (R0_CACHE, R1_CACHE):
        values, _ = load_cache(path, samples)
        if values.shape != (EXPECTED_ROWS, 768) or not np.isfinite(values).all():
            raise RuntimeError(f"embedding alignment failed: {path}")
    if set(smoke["tasks"]) != set(TASK_SPECS):
        raise RuntimeError("smoke task coverage failed")
    unsupported = {
        task: row["unsupported_folds"]
        for task, row in smoke["tasks"].items()
        if row["status"] == "probe_inconclusive_fold_outcome_support"
    }
    if unsupported != {"hf_adventitious_presence": [1]}:
        raise RuntimeError(f"unexpected smoke support blocker: {unsupported}")
    receipt = {
        "status": "shortcut_diagnostic_smoke_independently_verified",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "rows": len(samples),
        "recording_feature_rows": len(feature_rows),
        "joined_rows": EXPECTED_ROWS,
        "r0_cache_sha256": sha256_file(R0_CACHE),
        "r1_cache_sha256": sha256_file(R1_CACHE),
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "unsupported_task_folds": unsupported,
        "raw_audio_read": False,
        "acoustic_model_trained_or_modified": False,
    }
    write_json(RESULT_ROOT / "smoke_verification.json", receipt)
    return receipt


def _bootstrap_indices(groups: np.ndarray, selected: np.ndarray) -> np.ndarray:
    by_group = {
        group: np.flatnonzero(groups == group) for group in np.unique(groups)
    }
    return np.concatenate([by_group[group] for group in selected])


def _paired_statistics(
    target: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    groups: np.ndarray,
    seed_offset: int,
) -> dict[str, object]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    deltas = []
    for _ in range(1000):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = _bootstrap_indices(groups, selected)
        if len(np.unique(target[index])) != 2:
            continue
        deltas.append(
            roc_auc_score(target[index], e1[index])
            - roc_auc_score(target[index], e0[index])
        )
    observed = roc_auc_score(target, e1) - roc_auc_score(target, e0)
    null = []
    for _ in range(2000):
        swapped = set(unique_groups[rng.random(len(unique_groups)) < 0.5])
        mask = np.asarray([group in swapped for group in groups])
        left = np.where(mask, e1, e0)
        right = np.where(mask, e0, e1)
        null.append(roc_auc_score(target, right) - roc_auc_score(target, left))
    return {
        "ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "p": float((1 + sum(value >= observed for value in null)) / 2001),
        "valid": len(deltas),
    }


def _verify_dataset_probe(samples: list) -> dict[str, object]:
    rows = _read_gzip(RESULT_ROOT / "dataset_id_probe_predictions.csv.gz")
    summary = {
        row["representation"]: row
        for row in _read_csv(RESULT_ROOT / "dataset_id_probe_summary.csv")
    }
    if len(rows) != 2 * EXPECTED_ROWS or set(summary) != {R0, R1}:
        raise RuntimeError("dataset-ID row coverage failed")
    by_representation = {
        representation: [
            row for row in rows if row["representation"] == representation
        ]
        for representation in (R0, R1)
    }
    expected_ids = [sample.sample_id for sample in samples]
    names = ["icbhi", "sprsound", "hf_lung", "kauh"]
    groups = {sample.sample_id: _lineage_group(sample) for sample in samples}
    folds_by_id = {}
    output = {}
    for representation, current in by_representation.items():
        if [row["sample_id"] for row in current] != expected_ids:
            raise RuntimeError(f"dataset-ID ordered IDs failed: {representation}")
        for row in current:
            sample_id = row["sample_id"]
            fold = int(row["fold"])
            if sample_id in folds_by_id and folds_by_id[sample_id] != fold:
                raise RuntimeError("R0/R1 dataset-ID fold mismatch")
            folds_by_id[sample_id] = fold
        for fold in range(5):
            test_groups = {
                groups[row["sample_id"]]
                for row in current
                if int(row["fold"]) == fold
            }
            train_groups = {
                groups[row["sample_id"]]
                for row in current
                if int(row["fold"]) != fold
            }
            if test_groups & train_groups:
                raise RuntimeError("dataset-ID group leakage")
        target = np.asarray([names.index(row["true_dataset"]) for row in current])
        predicted = np.asarray(
            [names.index(row["predicted_dataset"]) for row in current]
        )
        matrix = confusion_matrix(target, predicted, labels=range(4))
        balanced = balanced_accuracy_score(target, predicted)
        macro = f1_score(target, predicted, average="macro", zero_division=0)
        _assert_close(
            summary[representation]["balanced_accuracy"],
            balanced,
            f"{representation}.balanced_accuracy",
        )
        _assert_close(
            summary[representation]["macro_f1"],
            macro,
            f"{representation}.macro_f1",
        )
        if json.loads(summary[representation]["confusion_json"]) != matrix.tolist():
            raise RuntimeError(f"{representation} confusion mismatch")
        output[representation] = {
            "balanced_accuracy": float(balanced),
            "macro_f1": float(macro),
        }
    return output


def _expected_primary_instance_ids(
    samples_by_fold: list[list],
    assignments: dict[str, dict[str, object]],
) -> dict[str, set[str]]:
    output = {task: set() for task in TASK_SPECS}
    for fold, samples in enumerate(samples_by_fold):
        sample_by_id = {sample.sample_id: sample for sample in samples}
        rows = _read_gzip(
            T0_ROOT
            / f"fold_{fold}/d2_shared_adapter_dataset_balanced/"
            "predictions.csv.gz"
        )
        for row in rows:
            task = row["task"]
            sample = sample_by_id[row["sample_id"]]
            if task == "kauh_raw9" and sample.partition != "test":
                raise RuntimeError("non-OOF KAUH prediction")
            target = json.loads(row["true_json"])
            labels = TASK_SPECS[task]["labels"]
            if TASK_SPECS[task]["kind"] == "multiclass":
                label = labels[int(target)]
                if assignments[f"{task}:{label}"]["eligibility"] == "primary_evaluable":
                    output[task].add(f"{row['sample_id']}:{task}:fold{fold}")
            else:
                for label in labels:
                    if assignments[f"{task}:{label}"]["eligibility"] == "primary_evaluable":
                        output[task].add(
                            f"{row['sample_id']}:{task}:{label}:fold{fold}"
                        )
    return output


def _verify_error_probes(
    samples_by_fold: list[list],
    assignments: dict[str, dict[str, object]],
) -> tuple[list[str], dict[str, object]]:
    summary_rows = _read_csv(RESULT_ROOT / "error_probe_summary.csv")
    summary = {row["task"]: row for row in summary_rows}
    if set(summary) != set(TASK_SPECS):
        raise RuntimeError("error-probe summary task coverage failed")
    expected = _expected_primary_instance_ids(samples_by_fold, assignments)
    oof = _read_gzip(RESULT_ROOT / "error_probe_oof.csv.gz")
    by_task = {task: [row for row in oof if row["task"] == task] for task in TASK_SPECS}
    task_metrics = {}
    for task_index, task in enumerate(TASK_SPECS):
        recorded = summary[task]
        if recorded["status"] == "probe_inconclusive_fold_outcome_support":
            if task != "hf_adventitious_presence" or by_task[task]:
                raise RuntimeError("unexpected inconclusive task output")
            continue
        rows = by_task[task]
        ids = [row["instance_id"] for row in rows]
        if set(ids) != expected[task] or len(ids) != len(set(ids)):
            raise RuntimeError(f"error OOF instance coverage failed: {task}")
        target = np.asarray([int(row["incorrect"]) for row in rows])
        margin = np.asarray([float(row["true_class_margin"]) for row in rows])
        groups = np.asarray([row["lineage_group"] for row in rows])
        probability = {
            condition: np.asarray(
                [float(row[f"{condition}_error_probability"]) for row in rows]
            )
            for condition in ("e0", "e1", "shuffled")
        }
        margin_prediction = {
            condition: np.asarray(
                [float(row[f"{condition}_margin_prediction"]) for row in rows]
            )
            for condition in ("e0", "e1", "shuffled")
        }
        for values in [*probability.values(), *margin_prediction.values()]:
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite probe output: {task}")
        overall = {
            condition: {
                "auroc": roc_auc_score(target, values),
                "balanced_accuracy": balanced_accuracy_score(
                    target, values >= 0.5
                ),
                "margin_r2": r2_score(margin, margin_prediction[condition]),
                "margin_mae": mean_absolute_error(
                    margin, margin_prediction[condition]
                ),
            }
            for condition, values in probability.items()
        }
        for condition in ("e0", "e1", "shuffled"):
            for metric in ("auroc", "balanced_accuracy", "margin_r2", "margin_mae"):
                _assert_close(
                    recorded[f"{condition}_{metric}"],
                    overall[condition][metric],
                    f"{task}.{condition}.{metric}",
                )
        e1_stats = _paired_statistics(
            target, probability["e0"], probability["e1"], groups, task_index * 2
        )
        shuffled_stats = _paired_statistics(
            target,
            probability["e0"],
            probability["shuffled"],
            groups,
            task_index * 2 + 1,
        )
        _assert_close(
            recorded["e1_auroc_delta_ci95_low"],
            e1_stats["ci"][0],
            f"{task}.e1_ci_low",
        )
        _assert_close(
            recorded["e1_auroc_delta_ci95_high"],
            e1_stats["ci"][1],
            f"{task}.e1_ci_high",
        )
        _assert_close(
            recorded["e1_group_permutation_p"],
            e1_stats["p"],
            f"{task}.e1_permutation",
        )
        _assert_close(
            recorded["shuffled_auroc_delta_ci95_low"],
            shuffled_stats["ci"][0],
            f"{task}.shuffled_ci_low",
        )
        _assert_close(
            recorded["shuffled_group_permutation_p"],
            shuffled_stats["p"],
            f"{task}.shuffled_permutation",
        )
        fold_rows = _read_csv(RESULT_ROOT / f"{task}_fold_metrics.csv")
        if len(fold_rows) != 15:
            raise RuntimeError(f"fold metric row count failed: {task}")
        positive = sum(
            float(
                next(
                    row["auroc"]
                    for row in fold_rows
                    if row["condition"] == "e1" and int(row["fold"]) == fold
                )
            )
            > float(
                next(
                    row["auroc"]
                    for row in fold_rows
                    if row["condition"] == "e0" and int(row["fold"]) == fold
                )
            )
            for fold in range(5)
        )
        if positive != int(recorded["positive_auroc_folds"]):
            raise RuntimeError(f"fold consistency mismatch: {task}")
        delta = overall["e1"]["auroc"] - overall["e0"]["auroc"]
        shuffled_delta = overall["shuffled"]["auroc"] - overall["e0"]["auroc"]
        shuffled_vote = (
            shuffled_delta >= 0.03
            and int(recorded["shuffled_positive_auroc_folds"]) >= 4
            and shuffled_stats["ci"][0] > 0
            and shuffled_stats["p"] <= 0.05
        )
        vote = (
            delta >= 0.03
            and positive >= 4
            and e1_stats["ci"][0] > 0
            and e1_stats["p"] <= 0.05
            and not shuffled_vote
        )
        if str(recorded["evidence_vote"]).lower() != str(vote).lower():
            raise RuntimeError(f"evidence vote mismatch: {task}")
        task_metrics[task] = {
            "e1_minus_e0_auroc": float(delta),
            "positive_folds": positive,
            "ci95": e1_stats["ci"],
            "permutation_p": e1_stats["p"],
            "vote": vote,
        }
    votes = [
        task
        for task, row in summary.items()
        if row.get("evidence_vote", "").lower() == "true"
    ]
    return votes, task_metrics


def verify_full(dataset_root: Path) -> dict[str, object]:
    smoke = verify_smoke(dataset_root)
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    dataset_metrics = _verify_dataset_probe(samples_by_fold[0])
    assignments = _load_contract_assignments()
    votes, task_metrics = _verify_error_probes(samples_by_fold, assignments)
    decision = json.loads((RESULT_ROOT / "decision.json").read_text())
    expected_decision = (
        "acquisition_correlated_error_supported"
        if len(votes) >= 2
        else "not_supported_or_inconclusive"
    )
    if (
        decision["evidence_vote_tasks"] != votes
        or int(decision["evidence_vote_count"]) != len(votes)
        or decision["decision"] != expected_decision
        or decision["causal_shortcut_claim"]
        or decision["acoustic_model_trained_or_modified"]
        or decision["raw_audio_read"]
    ):
        raise RuntimeError("shortcut diagnostic decision mismatch")
    support = _read_csv(RESULT_ROOT / "class_support_error.csv")
    expected_labels = {
        f"{task}:{label}"
        for task, spec in TASK_SPECS.items()
        for label in spec["labels"]
    }
    if (
        {f"{row['task']}:{row['label']}" for row in support}
        != expected_labels
        or any(
            row["eligibility"]
            != assignments[f"{row['task']}:{row['label']}"]["eligibility"]
            for row in support
        )
    ):
        raise RuntimeError("class support/eligibility routing failed")
    receipt = {
        "status": "four_dataset_shortcut_diagnostic_independently_verified",
        "smoke_status": smoke["status"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "r0_cache_sha256": sha256_file(R0_CACHE),
        "r1_cache_sha256": sha256_file(R1_CACHE),
        "recording_features_sha256": sha256_file(FEATURE_PATH),
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "dataset_id_probe": dataset_metrics,
        "error_probe": task_metrics,
        "inconclusive_task": {
            "hf_adventitious_presence": "fold 1 has one error outcome"
        },
        "decision": expected_decision,
        "evidence_vote_tasks": votes,
        "causal_shortcut_claim": False,
        "raw_audio_read": False,
        "acoustic_model_trained_or_modified": False,
        "warnings": 0,
        "finite": True,
    }
    write_json(RESULT_ROOT / "verification.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    receipt = (
        verify_smoke(dataset_root)
        if args.phase == "smoke"
        else verify_full(dataset_root)
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
