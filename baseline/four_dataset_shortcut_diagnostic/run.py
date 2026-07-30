"""Run frozen-embedding dataset-ID and acquisition-correlated error probes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    Sample,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result/four_dataset_shortcut_diagnostic"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
CONTRACT_PATH = ROOT / "docs/datasets/four_dataset_task_contract_draft_2026-07-28.json"
FEATURE_PATH = ROOT / "result/acoustic_distribution/recording_features.csv"
R0_CACHE = ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
R1_CACHE = (
    ROOT
    / ".cache/four_dataset_representation_attribution/"
    "r1_beats_as2m_audioset_only/embeddings.npz"
)
T0_ROOT = (
    ROOT
    / "result/four_dataset_representation_attribution/hf_proxy_fixed_v2/"
    "r0_pafa_icbhi_task_encoder"
)
GATE_D_VERIFICATION = ROOT / "result/four_dataset_support_aware_crt/verification.json"
R0 = "r0_pafa_icbhi_task_encoder"
R1 = "r1_beats_as2m_audioset_only"
SEED = 20260729
BOOTSTRAP_SEED = 20260731
EXPECTED_ROWS = 25_084
EXPECTED_FEATURE_ROWS = 13_704
EXPECTED_SHA = {
    R0_CACHE: "f40ae7fe581457bc86d76b93b1ee811e7ea01bc5e098a6daa73db451f96d1b31",
    R1_CACHE: "3b3798cc9d01dbdfa8168a1cd641d658eb2fd4553799e59b84b7aae7ad0f5a69",
    FEATURE_PATH: "1a25a6a9171b25a7cab2807b3119d105d553c138b858b251524287b860b11ebb",
    CONTRACT_PATH: "48ed66a28674b41354f259c13283054de59ca9c1f07e91ce5ed28a7d80f2bc09",
}
TASK_ID_MAP = {
    "spr_event_binary": "spr_binary",
    "spr_event_seven": "spr_seven",
}
BASE_CATEGORICAL = ["true_label", "true_state", "model_fold"]
BASE_NUMERIC = [
    "log1p_accepted_support",
    "unit_duration_s",
    "units_per_recording",
]
COVARIATE_CATEGORICAL = ["device", "site", "native_sr_category"]
COVARIATE_NUMERIC = [
    "recording_duration_s",
    "dbfs",
    "crest_db",
    "clip_frac",
    "snr_proxy_db",
    "centroid_hz",
    "bandwidth_hz",
    "rolloff85_hz",
    "flatness",
    "zcr",
    "band_0_100",
    "band_100_250",
    "band_250_500",
    "band_500_1000",
    "band_1000_2000",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_contract() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    payload = json.loads(CONTRACT_PATH.read_text())
    contract = payload["tail_eligibility_contract"]
    counts = Counter(row["eligibility"] for row in contract["label_assignments"])
    if (
        contract["status"] != "tail_eligibility_management_accepted"
        or dict(counts)
        != {
            "primary_evaluable": 16,
            "diagnostic_only": 5,
            "not_evaluable": 7,
        }
    ):
        raise RuntimeError("accepted tail contract failed")
    assignments: dict[str, dict[str, object]] = {}
    for row in contract["label_assignments"]:
        task = TASK_ID_MAP.get(str(row["task_id"]), str(row["task_id"]))
        assignments[f"{task}:{row['label']}"] = row
    expected = {
        f"{task}:{label}"
        for task, spec in TASK_SPECS.items()
        for label in spec["labels"]
    }
    if set(assignments) != expected:
        raise RuntimeError("tail contract task/label coverage failed")
    return contract, assignments


def _load_dependencies(dataset_root: Path) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["status"] != "preregistered_before_probe_outcomes":
        raise RuntimeError("protocol status failed")
    for path, digest in EXPECTED_SHA.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen input SHA failed: {path}")
    gate_d = json.loads(GATE_D_VERIFICATION.read_text())
    if (
        gate_d["status"] != "support_aware_crt_full_verified"
        or gate_d["cache_sha256"] != EXPECTED_SHA[R0_CACHE]
        or gate_d["hf_assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256
    ):
        raise RuntimeError("Gate D verification dependency failed")
    samples_by_fold = [build_samples(dataset_root, fold) for fold in range(5)]
    samples = samples_by_fold[0][0]
    ids = [sample.sample_id for sample in samples]
    if (
        len(samples) != EXPECTED_ROWS
        or len(ids) != len(set(ids))
        or any(
            receipt["datasets"]["hf_lung"]["assignment_sha256"]
            != EXPECTED_HF_ASSIGNMENT_SHA256
            for _, receipt in samples_by_fold
        )
    ):
        raise RuntimeError("canonical sample/HF assignment failed")
    return {
        "protocol": protocol,
        "contract": _load_contract()[0],
        "assignments": _load_contract()[1],
        "samples_by_fold": [item[0] for item in samples_by_fold],
        "sample_receipts": [item[1] for item in samples_by_fold],
    }


def _recording_features() -> dict[str, dict[str, str]]:
    with FEATURE_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_path = {str(Path(row["path"]).resolve()): row for row in rows}
    if len(rows) != EXPECTED_FEATURE_ROWS or len(by_path) != len(rows):
        raise RuntimeError("recording feature path uniqueness failed")
    for row in rows:
        for key in COVARIATE_NUMERIC:
            source_key = "duration_s" if key == "recording_duration_s" else key
            if not math.isfinite(float(row[source_key])):
                raise RuntimeError(f"non-finite recording covariate: {key}")
    return by_path


def _lineage_key(sample: Sample) -> str:
    if sample.dataset == "hf_lung" and sample.metadata.get("patient_id") is not None:
        raise RuntimeError("HF date proxy was mislabeled as patient")
    return f"{sample.dataset}:{sample.group_id}"


def _recording_key(sample: Sample) -> str:
    return f"{sample.dataset}:{Path(sample.audio_path).resolve()}"


def _unit_duration(sample: Sample, recording: dict[str, str]) -> float:
    if sample.crop_start_s is not None and sample.crop_end_s is not None:
        return float(sample.crop_end_s - sample.crop_start_s)
    return float(recording["duration_s"])


def lineage_audit(
    samples_by_fold: list[list[Sample]],
    features: dict[str, dict[str, str]],
) -> dict[str, object]:
    canonical = samples_by_fold[0]
    joined = [
        str(Path(sample.audio_path).resolve()) in features for sample in canonical
    ]
    if not all(joined):
        missing = Counter(
            sample.dataset
            for sample, present in zip(canonical, joined)
            if not present
        )
        raise RuntimeError(f"recording feature join failed: {missing}")
    canonical_ids = [sample.sample_id for sample in canonical]
    if any(
        [sample.sample_id for sample in fold_samples] != canonical_ids
        for fold_samples in samples_by_fold[1:]
    ):
        raise RuntimeError("fold sample order changed")
    per_dataset = {}
    for dataset in ("icbhi", "sprsound", "hf_lung", "kauh"):
        current = [sample for sample in canonical if sample.dataset == dataset]
        per_dataset[dataset] = {
            "prediction_units": len(current),
            "source_recordings": len({_recording_key(sample) for sample in current}),
            "lineage_groups": len({_lineage_key(sample) for sample in current}),
            "partitions": dict(sorted(Counter(sample.partition for sample in current).items())),
        }
    kauh_test_sets = [
        {
            sample.sample_id
            for sample in fold_samples
            if sample.dataset == "kauh" and sample.partition == "test"
        }
        for fold_samples in samples_by_fold
    ]
    if (
        len(set().union(*kauh_test_sets)) != 336
        or sum(map(len, kauh_test_sets)) != 336
        or any(
            kauh_test_sets[left] & kauh_test_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("KAUH OOF lineage failed")
    receipt = {
        "status": "shortcut_diagnostic_lineage_verified",
        "prediction_units": len(canonical),
        "unique_prediction_unit_ids": len(set(canonical_ids)),
        "ordered_id_sha256": ordered_sha(canonical_ids),
        "recording_feature_rows": len(features),
        "recording_feature_sha256": sha256_file(FEATURE_PATH),
        "joined_prediction_units": sum(joined),
        "missing_prediction_units": 0,
        "per_dataset": per_dataset,
        "lineage": {
            "icbhi": "cycle -> recording -> patient",
            "sprsound": "event -> recording -> patient",
            "hf_lung": "recording -> canonical date proxy; not patient",
            "kauh": "recording -> patient; B/D/E siblings grouped",
        },
        "kauh_oof_unique_rows": 336,
        "icbhi_official_test_patient_overlap_caveat": [156, 218],
    }
    write_json(RESULT_ROOT / "lineage_receipt.json", receipt)
    return receipt


def _accepted_support(
    assignment: dict[str, object],
    true_state: int,
) -> int:
    support = assignment["support"]
    values = (
        support["subtrain"]
        if "subtrain" in support
        else support["full_release"]
    )
    return int(values[0] if true_state == 1 else values[2])


def _instance_covariates(
    sample: Sample,
    recording: dict[str, str],
    units_per_recording: int,
) -> dict[str, object]:
    output = {
        "recording_key": _recording_key(sample),
        "lineage_group": _lineage_key(sample),
        "unit_duration_s": _unit_duration(sample, recording),
        "units_per_recording": units_per_recording,
        "device": recording["device"] or "NA",
        "site": recording["site"] or "NA",
        "native_sr_category": recording["native_sr"],
    }
    for key in COVARIATE_NUMERIC:
        source_key = "duration_s" if key == "recording_duration_s" else key
        output[key] = float(recording[source_key])
    return output


def build_prediction_instances(
    samples_by_fold: list[list[Sample]],
    features: dict[str, dict[str, str]],
    assignments: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    canonical = samples_by_fold[0]
    canonical_by_id = {sample.sample_id: sample for sample in canonical}
    recording_counts = Counter(_recording_key(sample) for sample in canonical)
    instances = []
    for fold, samples in enumerate(samples_by_fold):
        sample_by_id = {sample.sample_id: sample for sample in samples}
        path = (
            T0_ROOT
            / f"fold_{fold}/d2_shared_adapter_dataset_balanced/"
            "predictions.csv.gz"
        )
        for row in _read_predictions(path):
            task = row["task"]
            sample = sample_by_id[row["sample_id"]]
            canonical_sample = canonical_by_id[row["sample_id"]]
            if task == "kauh_raw9" and sample.partition != "test":
                raise RuntimeError("KAUH non-OOF prediction entered diagnostic")
            probabilities = np.asarray(json.loads(row["probabilities_json"]), dtype=float)
            predicted = np.asarray(json.loads(row["pred_json"]), dtype=int)
            target = np.asarray(json.loads(row["true_json"]), dtype=int)
            if not np.isfinite(probabilities).all():
                raise RuntimeError("non-finite corrected T0 probability")
            base = {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "task": task,
                "model_fold": str(fold),
                **_instance_covariates(
                    canonical_sample,
                    features[str(Path(canonical_sample.audio_path).resolve())],
                    recording_counts[_recording_key(canonical_sample)],
                ),
            }
            labels = list(TASK_SPECS[task]["labels"])
            if TASK_SPECS[task]["kind"] == "multiclass":
                true_index = int(target)
                true_label = labels[true_index]
                assignment = assignments[f"{task}:{true_label}"]
                other = np.delete(probabilities, true_index)
                instances.append(
                    {
                        **base,
                        "instance_id": f"{row['sample_id']}:{task}:fold{fold}",
                        "unit_label_id": f"{row['sample_id']}:{task}",
                        "true_label": true_label,
                        "true_state": "class",
                        "eligibility": assignment["eligibility"],
                        "log1p_accepted_support": math.log1p(
                            _accepted_support(assignment, 1)
                        ),
                        "incorrect": int(int(predicted) != true_index),
                        "true_class_margin": float(
                            probabilities[true_index] - np.max(other)
                        ),
                    }
                )
            else:
                for label_index, label in enumerate(labels):
                    truth = int(target[label_index])
                    prediction = int(predicted[label_index])
                    probability = float(probabilities[label_index])
                    assignment = assignments[f"{task}:{label}"]
                    true_probability = probability if truth else 1.0 - probability
                    instances.append(
                        {
                            **base,
                            "instance_id": (
                                f"{row['sample_id']}:{task}:{label}:fold{fold}"
                            ),
                            "unit_label_id": f"{row['sample_id']}:{task}:{label}",
                            "true_label": label,
                            "true_state": str(truth),
                            "eligibility": assignment["eligibility"],
                            "log1p_accepted_support": math.log1p(
                                _accepted_support(assignment, truth)
                            ),
                            "incorrect": int(prediction != truth),
                            "true_class_margin": float(2 * true_probability - 1),
                        }
                    )
    ids = [str(row["instance_id"]) for row in instances]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate prediction-instance ID")
    return instances


def _processor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )


def _error_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _processor(categorical, numeric)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _margin_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _processor(categorical, numeric)),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _shuffle_covariates(
    frame: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    output = frame.copy()
    recording_keys = sorted(output["recording_key"].unique())
    rng = np.random.default_rng(seed)
    source_keys = rng.permutation(recording_keys)
    first = output.drop_duplicates("recording_key").set_index("recording_key")
    covariates = [*COVARIATE_CATEGORICAL, *COVARIATE_NUMERIC]
    mapping = dict(zip(recording_keys, source_keys))
    for key in covariates:
        output[key] = [
            first.loc[mapping[recording], key]
            for recording in output["recording_key"]
        ]
    return output


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_f1": float(
            f1_score(target, predicted, average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


def _margin_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(target, predicted)),
        "mae": float(mean_absolute_error(target, predicted)),
    }


def _group_resample_indices(
    groups: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
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
    bootstrap_auroc = []
    bootstrap_balanced = []
    for _ in range(1000):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = _group_resample_indices(groups, selected)
        if len(np.unique(target[index])) != 2:
            continue
        bootstrap_auroc.append(
            roc_auc_score(target[index], e1[index])
            - roc_auc_score(target[index], e0[index])
        )
        bootstrap_balanced.append(
            balanced_accuracy_score(target[index], e1[index] >= 0.5)
            - balanced_accuracy_score(target[index], e0[index] >= 0.5)
        )
    if len(bootstrap_auroc) < 950:
        raise RuntimeError("insufficient valid grouped bootstrap replicates")
    observed = roc_auc_score(target, e1) - roc_auc_score(target, e0)
    null = []
    for _ in range(2000):
        swap_groups = set(unique_groups[rng.random(len(unique_groups)) < 0.5])
        mask = np.asarray([group in swap_groups for group in groups])
        left = np.where(mask, e1, e0)
        right = np.where(mask, e0, e1)
        null.append(roc_auc_score(target, right) - roc_auc_score(target, left))
    p_value = (1 + sum(value >= observed for value in null)) / (len(null) + 1)
    return {
        "bootstrap_repetitions": 1000,
        "bootstrap_valid": len(bootstrap_auroc),
        "auroc_delta_ci95": [
            float(np.quantile(bootstrap_auroc, 0.025)),
            float(np.quantile(bootstrap_auroc, 0.975)),
        ],
        "balanced_accuracy_delta_ci95": [
            float(np.quantile(bootstrap_balanced, 0.025)),
            float(np.quantile(bootstrap_balanced, 0.975)),
        ],
        "paired_group_permutation_repetitions": 2000,
        "paired_group_permutation_p_one_sided": float(p_value),
        "groups": len(unique_groups),
    }


def _task_folds(
    frame: pd.DataFrame, task_index: int, task: str
) -> tuple[np.ndarray, list[int]]:
    target = frame["incorrect"].to_numpy(dtype=int)
    groups = frame["lineage_group"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED + task_index
    )
    assignment = np.full(len(frame), -1, dtype=int)
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros(len(frame)), target, groups)
    ):
        assignment[test_index] = fold
    if np.any(assignment < 0):
        raise RuntimeError("probe fold assignment incomplete")
    unsupported = []
    for fold in range(5):
        train_groups = set(groups[assignment != fold])
        test_groups = set(groups[assignment == fold])
        if train_groups & test_groups:
            raise RuntimeError(f"probe lineage-group overlap: {task}/{fold}")
        if len(np.unique(target[assignment == fold])) != 2:
            unsupported.append(fold)
    return assignment, unsupported


def run_error_probes(
    instances: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_frame = pd.DataFrame(instances)
    summaries = []
    oof_rows = []
    for task_index, task in enumerate(TASK_SPECS):
        frame = all_frame[
            (all_frame["task"] == task)
            & (all_frame["eligibility"] == "primary_evaluable")
        ].copy()
        frame.reset_index(drop=True, inplace=True)
        if (
            len(frame) == 0
            or frame["incorrect"].nunique() != 2
            or frame["lineage_group"].nunique() < 10
        ):
            raise RuntimeError(f"primary error probe support failed: {task}")
        folds, unsupported_folds = _task_folds(frame, task_index, task)
        if unsupported_folds:
            summaries.append(
                {
                    "task": task,
                    "dataset": TASK_SPECS[task]["dataset"],
                    "status": "probe_inconclusive_fold_outcome_support",
                    "primary_prediction_instances": len(frame),
                    "unique_unit_label_rows": frame["unit_label_id"].nunique(),
                    "lineage_groups": frame["lineage_group"].nunique(),
                    "error_prevalence": float(frame["incorrect"].mean()),
                    "unsupported_folds_json": json.dumps(unsupported_folds),
                    "evidence_vote": False,
                }
            )
            continue
        target = frame["incorrect"].to_numpy(dtype=int)
        margin = frame["true_class_margin"].to_numpy(dtype=float)
        predictions = {
            condition: {
                "error": np.full(len(frame), np.nan),
                "margin": np.full(len(frame), np.nan),
            }
            for condition in ("e0", "e1", "shuffled")
        }
        fold_rows = []
        for fold in range(5):
            train_index = np.flatnonzero(folds != fold)
            test_index = np.flatnonzero(folds == fold)
            train = frame.iloc[train_index].copy()
            test = frame.iloc[test_index].copy()
            shuffled_train = _shuffle_covariates(
                train, SEED + 10_000 + task_index * 100 + fold
            )
            shuffled_test = _shuffle_covariates(
                test, SEED + 20_000 + task_index * 100 + fold
            )
            conditions = {
                "e0": (
                    train,
                    test,
                    BASE_CATEGORICAL,
                    BASE_NUMERIC,
                ),
                "e1": (
                    train,
                    test,
                    [*BASE_CATEGORICAL, *COVARIATE_CATEGORICAL],
                    [*BASE_NUMERIC, *COVARIATE_NUMERIC],
                ),
                "shuffled": (
                    shuffled_train,
                    shuffled_test,
                    [*BASE_CATEGORICAL, *COVARIATE_CATEGORICAL],
                    [*BASE_NUMERIC, *COVARIATE_NUMERIC],
                ),
            }
            for condition, (
                train_frame,
                test_frame,
                categorical,
                numeric,
            ) in conditions.items():
                classifier = _error_model(categorical, numeric)
                regressor = _margin_model(categorical, numeric)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    classifier.fit(train_frame, target[train_index])
                    regressor.fit(train_frame, margin[train_index])
                if caught:
                    raise RuntimeError(
                        f"probe fit warning: {task}/{fold}/{condition}: "
                        f"{[str(item.message) for item in caught]}"
                    )
                error_probability = classifier.predict_proba(test_frame)[:, 1]
                margin_prediction = regressor.predict(test_frame)
                if (
                    not np.isfinite(error_probability).all()
                    or not np.isfinite(margin_prediction).all()
                ):
                    raise RuntimeError("non-finite OOF probe prediction")
                predictions[condition]["error"][test_index] = error_probability
                predictions[condition]["margin"][test_index] = margin_prediction
                error_metrics = _binary_metrics(
                    target[test_index], error_probability
                )
                margin_metrics = _margin_metrics(
                    margin[test_index], margin_prediction
                )
                fold_rows.append(
                    {
                        "task": task,
                        "fold": fold,
                        "condition": condition,
                        **error_metrics,
                        "margin_r2": margin_metrics["r2"],
                        "margin_mae": margin_metrics["mae"],
                        "rows": len(test_index),
                        "groups": frame.iloc[test_index][
                            "lineage_group"
                        ].nunique(),
                    }
                )
        if any(
            not np.isfinite(values[kind]).all()
            for values in predictions.values()
            for kind in ("error", "margin")
        ):
            raise RuntimeError("incomplete OOF predictions")
        overall = {
            condition: {
                **_binary_metrics(target, values["error"]),
                **{
                    f"margin_{key}": value
                    for key, value in _margin_metrics(
                        margin, values["margin"]
                    ).items()
                },
            }
            for condition, values in predictions.items()
        }
        e1_stats = _paired_statistics(
            target,
            predictions["e0"]["error"],
            predictions["e1"]["error"],
            frame["lineage_group"].to_numpy(),
            task_index * 2,
        )
        shuffled_stats = _paired_statistics(
            target,
            predictions["e0"]["error"],
            predictions["shuffled"]["error"],
            frame["lineage_group"].to_numpy(),
            task_index * 2 + 1,
        )
        fold_by = {
            (row["condition"], int(row["fold"])): row for row in fold_rows
        }
        positive_folds = sum(
            fold_by[("e1", fold)]["auroc"]
            > fold_by[("e0", fold)]["auroc"]
            for fold in range(5)
        )
        shuffled_positive_folds = sum(
            fold_by[("shuffled", fold)]["auroc"]
            > fold_by[("e0", fold)]["auroc"]
            for fold in range(5)
        )
        delta = overall["e1"]["auroc"] - overall["e0"]["auroc"]
        shuffled_delta = (
            overall["shuffled"]["auroc"] - overall["e0"]["auroc"]
        )
        task_vote = (
            delta >= 0.03
            and positive_folds >= 4
            and e1_stats["auroc_delta_ci95"][0] > 0
            and e1_stats["paired_group_permutation_p_one_sided"] <= 0.05
        )
        shuffled_vote = (
            shuffled_delta >= 0.03
            and shuffled_positive_folds >= 4
            and shuffled_stats["auroc_delta_ci95"][0] > 0
            and shuffled_stats["paired_group_permutation_p_one_sided"] <= 0.05
        )
        task_vote = bool(task_vote and not shuffled_vote)
        summaries.append(
            {
                "task": task,
                "dataset": TASK_SPECS[task]["dataset"],
                "status": "probe_complete",
                "primary_prediction_instances": len(frame),
                "unique_unit_label_rows": frame["unit_label_id"].nunique(),
                "lineage_groups": frame["lineage_group"].nunique(),
                "error_prevalence": float(target.mean()),
                "e0_auroc": overall["e0"]["auroc"],
                "e1_auroc": overall["e1"]["auroc"],
                "shuffled_auroc": overall["shuffled"]["auroc"],
                "e1_minus_e0_auroc": delta,
                "shuffled_minus_e0_auroc": shuffled_delta,
                "e0_balanced_accuracy": overall["e0"]["balanced_accuracy"],
                "e1_balanced_accuracy": overall["e1"]["balanced_accuracy"],
                "shuffled_balanced_accuracy": overall["shuffled"][
                    "balanced_accuracy"
                ],
                "e1_minus_e0_balanced_accuracy": (
                    overall["e1"]["balanced_accuracy"]
                    - overall["e0"]["balanced_accuracy"]
                ),
                "e0_margin_r2": overall["e0"]["margin_r2"],
                "e1_margin_r2": overall["e1"]["margin_r2"],
                "shuffled_margin_r2": overall["shuffled"]["margin_r2"],
                "e1_minus_e0_margin_r2": (
                    overall["e1"]["margin_r2"]
                    - overall["e0"]["margin_r2"]
                ),
                "e0_margin_mae": overall["e0"]["margin_mae"],
                "e1_margin_mae": overall["e1"]["margin_mae"],
                "shuffled_margin_mae": overall["shuffled"]["margin_mae"],
                "e1_minus_e0_margin_mae": (
                    overall["e1"]["margin_mae"]
                    - overall["e0"]["margin_mae"]
                ),
                "positive_auroc_folds": positive_folds,
                "shuffled_positive_auroc_folds": shuffled_positive_folds,
                "e1_auroc_delta_ci95_low": e1_stats["auroc_delta_ci95"][0],
                "e1_auroc_delta_ci95_high": e1_stats["auroc_delta_ci95"][1],
                "e1_group_permutation_p": e1_stats[
                    "paired_group_permutation_p_one_sided"
                ],
                "shuffled_auroc_delta_ci95_low": shuffled_stats[
                    "auroc_delta_ci95"
                ][0],
                "shuffled_auroc_delta_ci95_high": shuffled_stats[
                    "auroc_delta_ci95"
                ][1],
                "shuffled_group_permutation_p": shuffled_stats[
                    "paired_group_permutation_p_one_sided"
                ],
                "shuffled_control_full_rule": shuffled_vote,
                "evidence_vote": task_vote,
            }
        )
        for index, row in frame.iterrows():
            oof_rows.append(
                {
                    "instance_id": row["instance_id"],
                    "unit_label_id": row["unit_label_id"],
                    "sample_id": row["sample_id"],
                    "recording_key": row["recording_key"],
                    "lineage_group": row["lineage_group"],
                    "dataset": row["dataset"],
                    "task": task,
                    "true_label": row["true_label"],
                    "true_state": row["true_state"],
                    "eligibility": row["eligibility"],
                    "probe_fold": int(folds[index]),
                    "incorrect": int(target[index]),
                    "true_class_margin": float(margin[index]),
                    "e0_error_probability": float(
                        predictions["e0"]["error"][index]
                    ),
                    "e1_error_probability": float(
                        predictions["e1"]["error"][index]
                    ),
                    "shuffled_error_probability": float(
                        predictions["shuffled"]["error"][index]
                    ),
                    "e0_margin_prediction": float(
                        predictions["e0"]["margin"][index]
                    ),
                    "e1_margin_prediction": float(
                        predictions["e1"]["margin"][index]
                    ),
                    "shuffled_margin_prediction": float(
                        predictions["shuffled"]["margin"][index]
                    ),
                }
            )
        write_csv(RESULT_ROOT / f"{task}_fold_metrics.csv", fold_rows)
    return summaries, oof_rows


def class_support_rows(
    instances: list[dict[str, object]],
    assignments: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(instances)
    rows = []
    for (task, label, eligibility), current in frame.groupby(
        ["task", "true_label", "eligibility"], sort=True
    ):
        rows.append(
            {
                "task": task,
                "dataset": TASK_SPECS[task]["dataset"],
                "label": label,
                "eligibility": eligibility,
                "unique_unit_label_rows": current["unit_label_id"].nunique(),
                "prediction_instances": len(current),
                "source_recordings": current["recording_key"].nunique(),
                "lineage_groups": current["lineage_group"].nunique(),
                "error_rate": float(current["incorrect"].mean()),
                "mean_true_class_margin": float(
                    current["true_class_margin"].mean()
                ),
            }
        )
    present = {f"{row['task']}:{row['label']}" for row in rows}
    for key, assignment in assignments.items():
        if key in present:
            continue
        task, label = key.split(":", 1)
        rows.append(
            {
                "task": task,
                "dataset": TASK_SPECS[task]["dataset"],
                "label": label,
                "eligibility": assignment["eligibility"],
                "unique_unit_label_rows": 0,
                "prediction_instances": 0,
                "source_recordings": 0,
                "lineage_groups": 0,
                "error_rate": None,
                "mean_true_class_margin": None,
            }
        )
    rows.sort(key=lambda row: (str(row["task"]), str(row["label"])))
    return rows


def _dataset_folds(samples: list[Sample]) -> np.ndarray:
    labels = np.asarray(
        [["icbhi", "sprsound", "hf_lung", "kauh"].index(sample.dataset) for sample in samples]
    )
    groups = np.asarray([_lineage_key(sample) for sample in samples])
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    folds = np.full(len(samples), -1, dtype=int)
    for fold, (_, test_index) in enumerate(
        splitter.split(np.zeros(len(samples)), labels, groups)
    ):
        folds[test_index] = fold
    if np.any(folds < 0):
        raise RuntimeError("dataset-ID fold assignment incomplete")
    for fold in range(5):
        if set(groups[folds == fold]) & set(groups[folds != fold]):
            raise RuntimeError("dataset-ID group overlap")
    return folds


def run_dataset_id_probes(
    samples: list[Sample],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = np.asarray(
        [["icbhi", "sprsound", "hf_lung", "kauh"].index(sample.dataset) for sample in samples],
        dtype=int,
    )
    groups = np.asarray([_lineage_key(sample) for sample in samples])
    folds = _dataset_folds(samples)
    names = ["icbhi", "sprsound", "hf_lung", "kauh"]
    summary = []
    predictions = []
    for representation, values in embeddings.items():
        oof = np.full(len(samples), -1, dtype=int)
        for fold in range(5):
            train_index = np.flatnonzero(folds != fold)
            test_index = np.flatnonzero(folds == fold)
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=500,
                            random_state=SEED,
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(values[train_index], labels[train_index])
            if caught:
                raise RuntimeError(
                    f"dataset-ID probe warning: {representation}/{fold}: "
                    f"{[str(item.message) for item in caught]}"
                )
            oof[test_index] = model.predict(values[test_index])
        if np.any(oof < 0):
            raise RuntimeError("dataset-ID OOF prediction incomplete")
        matrix = confusion_matrix(labels, oof, labels=range(4))
        summary.append(
            {
                "representation": representation,
                "rows": len(samples),
                "groups": len(set(groups)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, oof)
                ),
                "macro_f1": float(
                    f1_score(labels, oof, average="macro", zero_division=0)
                ),
                "confusion_json": json.dumps(matrix.astype(int).tolist()),
            }
        )
        for index, sample in enumerate(samples):
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "lineage_group": groups[index],
                    "fold": int(folds[index]),
                    "representation": representation,
                    "true_dataset": names[labels[index]],
                    "predicted_dataset": names[oof[index]],
                }
            )
    return summary, predictions


def smoke(
    dependencies: dict[str, object],
    features: dict[str, dict[str, str]],
) -> dict[str, object]:
    samples_by_fold = dependencies["samples_by_fold"]
    lineage = lineage_audit(samples_by_fold, features)
    instances = build_prediction_instances(
        samples_by_fold, features, dependencies["assignments"]
    )
    primary = pd.DataFrame(instances)
    checks = {}
    for task_index, task in enumerate(TASK_SPECS):
        frame = primary[
            (primary["task"] == task)
            & (primary["eligibility"] == "primary_evaluable")
        ].copy()
        frame.reset_index(drop=True, inplace=True)
        folds, unsupported_folds = _task_folds(frame, task_index, task)
        if unsupported_folds:
            checks[task] = {
                "status": "probe_inconclusive_fold_outcome_support",
                "primary_prediction_instances": len(frame),
                "groups": frame["lineage_group"].nunique(),
                "unsupported_folds": unsupported_folds,
                "finite": True,
            }
            continue
        test = np.flatnonzero(folds == 0)
        train = np.flatnonzero(folds != 0)
        classifier = _error_model(BASE_CATEGORICAL, BASE_NUMERIC)
        regressor = _margin_model(BASE_CATEGORICAL, BASE_NUMERIC)
        classifier.fit(frame.iloc[train], frame.iloc[train]["incorrect"])
        regressor.fit(frame.iloc[train], frame.iloc[train]["true_class_margin"])
        error_probability = classifier.predict_proba(frame.iloc[test])[:, 1]
        margin_prediction = regressor.predict(frame.iloc[test])
        if (
            not np.isfinite(error_probability).all()
            or not np.isfinite(margin_prediction).all()
        ):
            raise RuntimeError(f"non-finite smoke probe: {task}")
        checks[task] = {
            "status": "probe_smoke_fit_passed",
            "primary_prediction_instances": len(frame),
            "groups": frame["lineage_group"].nunique(),
            "held_out_rows": len(test),
            "finite": True,
        }
    canonical = samples_by_fold[0]
    r0, _ = load_cache(R0_CACHE, canonical)
    r1, _ = load_cache(R1_CACHE, canonical)
    if (
        r0.shape != (EXPECTED_ROWS, 768)
        or r1.shape != (EXPECTED_ROWS, 768)
        or not np.isfinite(r0).all()
        or not np.isfinite(r1).all()
    ):
        raise RuntimeError("embedding smoke failed")
    folds = _dataset_folds(canonical)
    receipt = {
        "status": "shortcut_diagnostic_package_lineage_smoke_passed",
        "lineage_status": lineage["status"],
        "prediction_instances": len(instances),
        "unique_instance_ids": len({row["instance_id"] for row in instances}),
        "tasks": checks,
        "embedding_shapes": {R0: list(r0.shape), R1: list(r1.shape)},
        "dataset_id_fold_rows": {
            str(int(key)): int(value)
            for key, value in sorted(Counter(folds).items())
        },
        "probe_outcomes_read_after_preregistration": True,
        "raw_audio_read": False,
        "acoustic_model_trained_or_modified": False,
    }
    write_json(RESULT_ROOT / "smoke_receipt.json", receipt)
    return receipt


def run_full(
    dependencies: dict[str, object],
    features: dict[str, dict[str, str]],
) -> dict[str, object]:
    smoke_receipt = json.loads((RESULT_ROOT / "smoke_receipt.json").read_text())
    preregistration = json.loads(
        (RESULT_ROOT / "preregistration.json").read_text()
    )
    if (
        smoke_receipt["status"]
        != "shortcut_diagnostic_package_lineage_smoke_passed"
        or preregistration["probe_outcomes_read_before_preregistration"]
    ):
        raise RuntimeError("smoke/preregistration dependency failed")
    samples = dependencies["samples_by_fold"][0]
    r0, _ = load_cache(R0_CACHE, samples)
    r1, _ = load_cache(R1_CACHE, samples)
    dataset_summary_path = RESULT_ROOT / "dataset_id_probe_summary.csv"
    dataset_predictions_path = RESULT_ROOT / "dataset_id_probe_predictions.csv.gz"
    if dataset_summary_path.is_file() and dataset_predictions_path.is_file():
        dataset_summary = _read_csv(dataset_summary_path)
    elif dataset_summary_path.exists() or dataset_predictions_path.exists():
        raise RuntimeError("partial dataset-ID probe artifact set")
    else:
        dataset_summary, dataset_predictions = run_dataset_id_probes(
            samples, {R0: r0, R1: r1}
        )
        write_csv(dataset_summary_path, dataset_summary)
        write_gzip_csv(dataset_predictions_path, dataset_predictions)
    instances = build_prediction_instances(
        dependencies["samples_by_fold"],
        features,
        dependencies["assignments"],
    )
    support = class_support_rows(instances, dependencies["assignments"])
    error_summary_path = RESULT_ROOT / "error_probe_summary.csv"
    error_oof_path = RESULT_ROOT / "error_probe_oof.csv.gz"
    if error_summary_path.is_file() and error_oof_path.is_file():
        error_summary = _read_csv(error_summary_path)
    elif error_summary_path.exists() or error_oof_path.exists():
        raise RuntimeError("partial error-probe artifact set")
    else:
        error_summary, error_oof = run_error_probes(instances)
        write_csv(error_summary_path, error_summary)
        write_gzip_csv(error_oof_path, error_oof)
    write_csv(RESULT_ROOT / "class_support_error.csv", support)
    votes = [
        row["task"]
        for row in error_summary
        if str(row["evidence_vote"]).lower() == "true"
    ]
    decision_name = (
        "acquisition_correlated_error_supported"
        if len(votes) >= 2
        else "not_supported_or_inconclusive"
    )
    decision = {
        "status": "four_dataset_shortcut_diagnostic_complete",
        "decision": decision_name,
        "evidence_vote_tasks": votes,
        "evidence_vote_count": len(votes),
        "required_votes": 2,
        "dataset_id_probe_interpretation": (
            "accessible domain information only; not sufficient for a shortcut claim"
        ),
        "claim": (
            "acquisition-correlated error supported under the preregistered probe"
            if decision_name == "acquisition_correlated_error_supported"
            else "acquisition-correlated error not supported or inconclusive under "
            "the preregistered probe"
        ),
        "causal_shortcut_claim": False,
        "recommended_next": (
            "normalization_or_domain_residual_frozen_control"
            if decision_name == "acquisition_correlated_error_supported"
            else "window_or_token_level_event_sensitive_pooling"
        ),
        "next_method_implemented": False,
        "r0_caveat": "ICBHI official-test-selected PAFA task encoder",
        "raw_audio_read": False,
        "acoustic_model_trained_or_modified": False,
    }
    write_json(RESULT_ROOT / "decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["smoke", "full", "all"], default="all"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    dependencies = _load_dependencies(dataset_root)
    features = _recording_features()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    preregistration_path = RESULT_ROOT / "preregistration.json"
    expected_preregistration = {
        "status": "preregistered_before_probe_outcomes",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "recording_features_sha256": sha256_file(FEATURE_PATH),
        "r0_cache_sha256": sha256_file(R0_CACHE),
        "r1_cache_sha256": sha256_file(R1_CACHE),
        "tail_contract_sha256": sha256_file(CONTRACT_PATH),
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "probe_outcomes_read_before_preregistration": False,
    }
    if preregistration_path.exists():
        if json.loads(preregistration_path.read_text()) != expected_preregistration:
            raise RuntimeError("existing preregistration differs from protocol")
    else:
        write_json(preregistration_path, expected_preregistration)
    if args.phase in {"smoke", "all"}:
        receipt = smoke(dependencies, features)
        if args.phase == "smoke":
            print(json.dumps(receipt, sort_keys=True))
            return
    decision = run_full(dependencies, features)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
