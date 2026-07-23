from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .config import CORE_FEATURES, LABELS, MANIFEST_PATH, Protocol, REPO_ROOT
from .data import load_aligned_features, load_manifest, provenance_row, sha256_file
from .metrics import evaluate_predictions, flatten_metrics
from .run import environment_receipt, write_json
from .strict_patient_v3_policies import POLICIES, run_policy


PROTOCOL_VERSION = "formal-strict-patient-v3"
RESULT_ROOT = REPO_ROOT / "result/icbhi_strict_patient_long_tail"
BEATS_ROOT = RESULT_ROOT / "beats"
COMPARISON_ROOT = RESULT_ROOT / "comparison"
BEATS_SHA256 = "7040a40ddb8fa30a57d5b4a0722a5d42116e383653312a1bacea13fdc1b74ac3"
OUTER_FOLDS = 5
OUTER_SEED = 20260712
INNER_SEED_BASE = 20260712
SMOKE_OUTER_FOLD = 2
ECE_BINS = 15
METRICS = [
    "macro_f1",
    "weighted_f1",
    "uar",
    "abnormal_sensitivity",
    "normal_specificity",
    "icbhi_score",
    "both_recall",
    "ece",
    "brier",
]
PARETO_METRICS = ["macro_f1", "uar", "both_recall", "normal_specificity", "icbhi_score"]


def source_audit_receipt() -> dict:
    return {
        "scope": (
            "Local controlled strict-patient benchmark on frozen BEATs features; "
            "not an official reproduction of the long-tail method papers."
        ),
        "logit_adjusted_ce": {
            "paper": "https://openreview.net/forum?id=37nvvqkCo5",
            "official_code": (
                "https://github.com/google-research/google-research/tree/master/logit_adjustment"
            ),
            "prior": "inner-train empirical class count divided by inner-train size",
            "tau": 1.0,
            "training_sign": "plus",
            "training_formula": "CrossEntropy(logits + tau * log(class_prior), target)",
            "inference": "raw logits; no post-hoc adjustment and no threshold search",
            "audit_note": (
                "The paper defines train-time logit adjustment with + tau log pi. "
                "Tau=1 is fixed before evaluation; no sweep is performed."
            ),
        },
        "ldam_drw": {
            "paper": (
                "https://papers.neurips.cc/paper_files/paper/2019/hash/"
                "621461af90cadfdaf0e8d4cc25129f91-Paper.pdf"
            ),
            "official_code": "https://github.com/kaidic/LDAM-DRW",
            "official_loss_code": "https://github.com/kaidic/LDAM-DRW/blob/master/losses.py",
            "margin_formula": "Delta_c proportional to 1 / n_c^(1/4)",
            "max_margin": 0.5,
            "scale": 30.0,
            "official_drw": (
                "Official CIFAR code switches beta from 0 to 0.9999 at epoch 160/200."
            ),
            "local_schedule": (
                "Epochs 1-40 use LDAM without weights; epochs 41-50 use LDAM with "
                "effective-number beta=0.9999 weights. The run cannot early-stop before "
                "the switch; checkpoint selection is restricted to epochs 41-50."
            ),
            "adaptation_note": (
                "The 80% deferred-switch ratio is preserved in the formal-v1 50-epoch budget; "
                "the original SGD/LR schedule is not copied into the AdamW MLP control."
            ),
        },
        "crt": {
            "paper": "https://openreview.net/forum?id=r1gRTCVFvB",
            "official_code": "https://github.com/facebookresearch/classifier-balancing",
            "official_config": (
                "https://github.com/facebookresearch/classifier-balancing/blob/"
                "master/config/ImageNet_LT/cls_crt.yaml"
            ),
            "stage1": (
                "Train MLP hidden representation and classifier with natural sampling and "
                "unweighted CE; select only by inner validation macro F1."
            ),
            "stage2": (
                "Reload stage-1 hidden representation, freeze hidden layers, reinitialize the "
                "final classifier, and retrain only that classifier with class-balanced replacement "
                "sampling and unweighted CE."
            ),
            "local_schedule": (
                "Both stages use formal-v1 AdamW lr/wd, batch size, max 50 epochs, patience 8; "
                "each stage is selected only by the same inner validation split."
            ),
        },
    }


def protocol_receipt() -> dict:
    base = Protocol()
    return {
        **asdict(base),
        "seeds": list(base.seeds),
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "ICBHI 2017 cycle-level flat4",
        "test_split": (
            "Five fixed outer StratifiedGroupKFold folds over all 6,898 cycles, grouped by "
            "patient_id; the official challenge split is not used for v3 evaluation"
        ),
        "validation_method": (
            "Within each outer development partition, fixed patient-grouped "
            "StratifiedGroupKFold fold 0 is used for validation"
        ),
        "validation_group": "patient_id",
        "split_caveat": (
            "This is a local strict patient-grouped protocol, not the official ICBHI challenge "
            "split; outer fold 0 contains only four both-class cycles"
        ),
        "representation": "BEATs frozen cycle embedding",
        "feature_key": CORE_FEATURES["beats"]["key"],
        "feature_sha256": BEATS_SHA256,
        "outer_split": (
            "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260712) "
            "over all 6,898 cycles, grouped by patient_id"
        ),
        "inner_split": (
            "For each outer fold, StratifiedGroupKFold(n_splits=5, shuffle=True, "
            "random_state=20260712+outer_fold), fold 0 as validation"
        ),
        "outer_assignment_changes_with_model_seed": False,
        "inner_assignment_changes_with_model_seed": False,
        "normalization": "StandardScaler fit separately on each outer fold inner-train only",
        "class_balance": "Policy-specific; unweighted CE is the natural-sampling control",
        "task": "flat4 normal/crackle/wheeze/both only",
        "architecture": "MLP2 768 -> 256 -> 128 -> 4, ReLU, dropout 0.3",
        "policies": list(POLICIES),
        "calibration": {
            "ece": f"top-label ECE with {ECE_BINS} equal-width confidence bins",
            "brier": "mean per-example sum of squared multiclass probability errors",
        },
        "selection": "inner validation macro F1 only; outer test never selects epoch/hyperparameter",
        "decision": "fixed argmax; no threshold search",
        "guardrail_interpretation": (
            "Minority-sensitive track reports both recall only for policy-level OOF operating "
            "points with mean normal specificity >= 0.55; this is post-run interpretation, "
            "not model or threshold selection."
        ),
        "primary_source_audit": source_audit_receipt(),
    }


def build_fold_assignments(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = manifest.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["outer_fold"] = -1
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    for fold, (_, test_position) in enumerate(
        outer.split(frame, frame["native_four_class_label"], frame["patient_id"])
    ):
        frame.loc[test_position, "outer_fold"] = fold
    if frame.groupby("patient_id")["outer_fold"].nunique().max() != 1:
        raise AssertionError("A patient crosses outer folds")

    long_rows = []
    statistics = []
    labels = LABELS["flat4"]
    for fold in range(OUTER_FOLDS):
        test_mask = frame["outer_fold"].eq(fold)
        development = frame.loc[~test_mask].copy()
        inner = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=INNER_SEED_BASE + fold,
        )
        train_position, validation_position = next(
            inner.split(
                development,
                development["native_four_class_label"],
                development["patient_id"],
            )
        )
        validation_ids = set(development.iloc[validation_position]["cycle_id"])
        role = np.full(len(frame), "inner_train", dtype=object)
        role[test_mask.to_numpy()] = "outer_test"
        role[frame["cycle_id"].isin(validation_ids).to_numpy()] = "inner_validation"
        fold_assignment = frame[[
            "cycle_id", "recording_id", "patient_id", "native_four_class_label", "outer_fold"
        ]].copy()
        fold_assignment["evaluation_outer_fold"] = fold
        fold_assignment["role"] = role
        long_rows.append(fold_assignment)

        patient_sets = {
            name: set(fold_assignment.loc[fold_assignment["role"].eq(name), "patient_id"])
            for name in ("inner_train", "inner_validation", "outer_test")
        }
        if patient_sets["inner_train"] & patient_sets["inner_validation"]:
            raise AssertionError(f"Inner patient overlap in fold {fold}")
        if (patient_sets["inner_train"] | patient_sets["inner_validation"]) & patient_sets["outer_test"]:
            raise AssertionError(f"Outer patient overlap in fold {fold}")
        for role_name in ("inner_train", "inner_validation", "outer_test"):
            subset = fold_assignment[fold_assignment["role"].eq(role_name)]
            counts = subset["native_four_class_label"].value_counts()
            if any(int(counts.get(label, 0)) <= 0 for label in labels):
                raise AssertionError(f"Missing class in fold {fold} {role_name}")
            statistics.append({
                "outer_fold": fold,
                "role": role_name,
                "cycles": len(subset),
                "patients": subset["patient_id"].nunique(),
                **{f"{label}_cycles": int(counts.get(label, 0)) for label in labels},
            })
    outer_assignment = frame[["cycle_id", "recording_id", "patient_id", "outer_fold"]].copy()
    long_assignment = pd.concat(long_rows, ignore_index=True)
    return outer_assignment, long_assignment, pd.DataFrame(statistics)


def _feature_audit(manifest: pd.DataFrame) -> tuple[np.ndarray, dict]:
    config = CORE_FEATURES["beats"]
    features, metadata = load_aligned_features(
        manifest,
        Path(config["path"]),
        config["key"],
        config["expected_dim"],
        expected_sha256=BEATS_SHA256,
    )
    provenance = provenance_row("beats", config, metadata)
    provenance.update({
        "sha256_matches_strict_registry": provenance["sha256"] == BEATS_SHA256,
        "cycle_id_join": "explicit unique cycle_id join against 6,898-row manifest",
        "features_finite": bool(np.isfinite(features).all()),
        "official_reproduction_claim": False,
    })
    if not all((provenance["sha256_matches_strict_registry"], provenance["features_finite"])):
        raise AssertionError("BEATs provenance audit failed")
    return features, provenance


def _fold_partitions(
    features: np.ndarray,
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    outer_fold: int,
) -> tuple[dict, StandardScaler]:
    fold = assignments[assignments["evaluation_outer_fold"].eq(outer_fold)]
    role_by_id = fold.set_index("cycle_id")["role"]
    roles = manifest["cycle_id"].map(role_by_id).to_numpy()
    labels = manifest["native_four_class_label"].astype(str).to_numpy()
    ids = manifest["cycle_id"].astype(str).to_numpy()
    raw = {}
    for role in ("inner_train", "inner_validation", "outer_test"):
        mask = roles == role
        raw[role] = (features[mask], labels[mask], ids[mask])
    scaler = StandardScaler().fit(raw["inner_train"][0].astype(np.float64, copy=False))
    scaled = {
        role: (
            scaler.transform(values[0].astype(np.float64, copy=False)).astype(np.float32),
            values[1],
            values[2],
        )
        for role, values in raw.items()
    }
    return scaled, scaler


def _smoke_subset(values: tuple[np.ndarray, np.ndarray, np.ndarray], limit: int = 96):
    x, y, ids = values
    positions = []
    per_class = max(4, limit // len(LABELS["flat4"]))
    for label in LABELS["flat4"]:
        positions.extend(np.flatnonzero(y == label)[:per_class].tolist())
    positions = np.asarray(sorted(positions[:limit]), dtype=np.int64)
    return x[positions], y[positions], ids[positions]


def calibration_metrics(y_true: np.ndarray, probability: np.ndarray, labels: list[str]) -> dict:
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_index = np.asarray([label_to_index[value] for value in y_true], dtype=np.int64)
    one_hot = np.eye(len(labels), dtype=np.float64)[true_index]
    brier = float(np.mean(np.sum((probability.astype(np.float64) - one_hot) ** 2, axis=1)))
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == true_index
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    ece = 0.0
    for index in range(ECE_BINS):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < ECE_BINS - 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"ece": ece, "brier": brier}


def _write_prediction_artifacts(
    run_dir: Path,
    ids: np.ndarray,
    y_true: np.ndarray,
    trained,
    metrics: dict,
    labels: list[str],
    outer_fold: int,
) -> None:
    prediction = pd.DataFrame({
        "cycle_id": ids,
        "outer_fold": outer_fold,
        "y_true": y_true,
        "y_pred": trained.prediction,
    })
    for index, label in enumerate(labels):
        prediction[f"prob_{label}"] = trained.probability[:, index]
    prediction.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
        run_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(trained.curve).to_csv(run_dir / "training_curve.csv", index=False)


def _existing_complete_run(run_dir: Path) -> dict | None:
    required = [
        run_dir / "metrics.json",
        run_dir / "predictions.csv",
        run_dir / "confusion_matrix.csv",
        run_dir / "training_curve.csv",
        run_dir / "checkpoint.pt",
        run_dir / "run_log.json",
    ]
    if not all(path.exists() for path in required):
        return None
    payload = json.loads((run_dir / "metrics.json").read_text())
    row = payload.get("row")
    return row if row and row.get("protocol_version") == PROTOCOL_VERSION else None


def run_strict_patient_v3(mode: str = "smoke", resume: bool = True) -> list[dict]:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    labels = LABELS["flat4"]
    protocol = Protocol()
    manifest = load_manifest()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    features, provenance = _feature_audit(manifest)
    outer_assignment, assignments, fold_statistics = build_fold_assignments(manifest)
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    outer_assignment.to_csv(COMPARISON_ROOT / "outer_fold_assignment.csv", index=False)
    assignments.to_csv(COMPARISON_ROOT / "nested_fold_assignment.csv", index=False)
    fold_statistics.to_csv(COMPARISON_ROOT / "fold_class_patient_counts.csv", index=False)
    pd.DataFrame([provenance]).to_csv(COMPARISON_ROOT / "feature_provenance.csv", index=False)
    write_json(COMPARISON_ROOT / "protocol.json", protocol_receipt())
    write_json(COMPARISON_ROOT / "primary_source_audit.json", source_audit_receipt())

    mode_root = BEATS_ROOT / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    write_json(mode_root / "environment_receipt.json", environment_receipt())
    write_json(mode_root / "config.json", {
        "protocol": protocol_receipt(),
        "feature": provenance,
        "mode": mode,
    })
    folds = [SMOKE_OUTER_FOLD] if mode == "smoke" else list(range(OUTER_FOLDS))
    seeds = [protocol.seeds[0]] if mode == "smoke" else list(protocol.seeds)
    rows = []
    for outer_fold in folds:
        scaled_full, scaler = _fold_partitions(features, manifest, assignments, outer_fold)
        reference_y = scaled_full["inner_train"][1]
        scaled = (
            {name: _smoke_subset(values) for name, values in scaled_full.items()}
            if mode == "smoke"
            else scaled_full
        )
        if mode == "full":
            scaler_root = COMPARISON_ROOT / "scalers"
            scaler_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_root / f"outer_fold_{outer_fold}_standard_scaler.joblib")
        x_train, y_train, _ = scaled["inner_train"]
        x_val, y_val, _ = scaled["inner_validation"]
        x_test, y_test, test_ids = scaled["outer_test"]
        for policy in POLICIES:
            for seed in seeds:
                run_name = f"beats_{policy}_outer{outer_fold}_seed{seed}"
                run_dir = mode_root / "runs" / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                if resume:
                    existing = _existing_complete_run(run_dir)
                    if existing is not None:
                        rows.append(existing)
                        print(f"resumed {run_name}", flush=True)
                        continue
                started = time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    trained = run_policy(
                        policy,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        x_test,
                        labels,
                        protocol,
                        seed,
                        reference_y,
                        mode == "smoke",
                    )
                warning_text = [f"{type(item.message).__name__}: {item.message}" for item in caught]
                warning_text.extend(trained.warnings)
                metrics = evaluate_predictions(y_test, trained.prediction, labels)
                calibration = calibration_metrics(y_test, trained.probability, labels)
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "representation": "beats",
                    "encoder": CORE_FEATURES["beats"]["encoder"],
                    "feature_key": CORE_FEATURES["beats"]["key"],
                    "feature_sha256": BEATS_SHA256,
                    "task": "flat4",
                    "head": "mlp2",
                    "policy": policy,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "mode": mode,
                    "inner_train_cycles": len(y_train),
                    "inner_validation_cycles": len(y_val),
                    "outer_test_cycles": len(y_test),
                    "parameter_count": trained.parameter_count,
                    "trainable_parameter_count": trained.trainable_parameter_count,
                    "runtime_seconds": trained.runtime_seconds,
                    "best_epoch": trained.best_epoch,
                    "stage1_best_epoch": trained.stage1_best_epoch,
                    "best_validation_macro_f1": trained.best_validation_macro_f1,
                    "warning_count": len(warning_text),
                    "warnings": " | ".join(warning_text),
                    "converged": True,
                    "loss_finite": trained.loss_finite,
                    "gradient_finite": trained.gradient_finite,
                    "policy_details_json": json.dumps(trained.details, sort_keys=True),
                    **flatten_metrics(metrics),
                    **calibration,
                }
                rows.append(row)
                _write_prediction_artifacts(
                    run_dir, test_ids, y_test, trained, metrics, labels, outer_fold
                )
                torch.save({
                    "state_dict": trained.state_dict,
                    "policy_details": trained.details,
                    "best_epoch": trained.best_epoch,
                    "best_validation_macro_f1": trained.best_validation_macro_f1,
                }, run_dir / "checkpoint.pt")
                write_json(run_dir / "metrics.json", {
                    "row": row,
                    "class_metrics": metrics["class_metrics"],
                    "calibration": calibration,
                })
                write_json(run_dir / "run_log.json", {
                    "run_name": run_name,
                    "started_unix": started,
                    "completed_unix": time.time(),
                    "partition_sizes": {name: len(values[1]) for name, values in scaled.items()},
                    "full_partition_sizes": {
                        name: len(values[1]) for name, values in scaled_full.items()
                    },
                    "policy_details": trained.details,
                    "loss_finite": trained.loss_finite,
                    "gradient_finite": trained.gradient_finite,
                })
                print(
                    f"completed {run_name}: macro={metrics['macro_f1']:.4f} "
                    f"uar={metrics['uar']:.4f} both={metrics['both_recall']:.4f} "
                    f"spec={metrics['normal_specificity']:.4f}",
                    flush=True,
                )
    output = mode_root / f"beats_strict_patient_v3_{mode}_results.csv"
    pd.DataFrame(rows).sort_values(["policy", "outer_fold", "seed"]).to_csv(output, index=False)
    if mode == "full":
        consolidate_strict_patient_v3()
    return rows


def _pareto_flags(frame: pd.DataFrame, columns: list[str]) -> list[bool]:
    values = frame[columns].to_numpy(dtype=float)
    flags = []
    for index, candidate in enumerate(values):
        dominated = any(
            other != index
            and np.all(values[other] >= candidate)
            and np.any(values[other] > candidate)
            for other in range(len(values))
        )
        flags.append(not dominated)
    return flags


def consolidate_strict_patient_v3() -> None:
    labels = LABELS["flat4"]
    results = pd.read_csv(BEATS_ROOT / "full/beats_strict_patient_v3_full_results.csv")
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(COMPARISON_ROOT / "strict_patient_v3_results.csv", index=False)

    fold_group = results.groupby(["policy", "outer_fold"], dropna=False)
    fold_summary = fold_group[METRICS + ["runtime_seconds", "parameter_count"]].agg(
        ["mean", "std"]
    ).reset_index()
    fold_summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in fold_summary.columns
    ]
    fold_summary["n_runs"] = fold_group.size().to_numpy()
    fold_summary.to_csv(COMPARISON_ROOT / "fold_level_summary.csv", index=False)

    seed_rows = []
    for policy in POLICIES:
        for seed in Protocol().seeds:
            paths = [
                BEATS_ROOT / "full/runs" / f"beats_{policy}_outer{fold}_seed{seed}" / "predictions.csv"
                for fold in range(OUTER_FOLDS)
            ]
            prediction = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            prediction = prediction.sort_values("cycle_id").reset_index(drop=True)
            probability = prediction[[f"prob_{label}" for label in labels]].to_numpy()
            metrics = evaluate_predictions(
                prediction["y_true"].to_numpy(), prediction["y_pred"].to_numpy(), labels
            )
            calibration = calibration_metrics(prediction["y_true"].to_numpy(), probability, labels)
            seed_row = {
                "protocol_version": PROTOCOL_VERSION,
                "representation": "beats",
                "policy": policy,
                "seed": seed,
                "cycles": len(prediction),
                **flatten_metrics(metrics),
                **calibration,
            }
            seed_rows.append(seed_row)
            prediction.to_csv(
                COMPARISON_ROOT / f"oof_predictions_{policy}_seed{seed}.csv", index=False
            )
            pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
                COMPARISON_ROOT / f"oof_confusion_{policy}_seed{seed}.csv"
            )
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(COMPARISON_ROOT / "seed_oof_metrics.csv", index=False)

    policy_group = seed_metrics.groupby("policy", dropna=False)
    policy_summary = policy_group[METRICS].agg(["mean", "std"]).reset_index()
    policy_summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in policy_summary.columns
    ]
    policy_summary["n_seeds"] = policy_group.size().to_numpy()
    policy_summary.to_csv(COMPARISON_ROOT / "policy_oof_summary.csv", index=False)

    paired_rows = []
    indexed = results.set_index(["policy", "outer_fold", "seed"])
    for policy in POLICIES:
        for reference in ("unweighted_ce", "class_weighted_ce"):
            for outer_fold in range(OUTER_FOLDS):
                for seed in Protocol().seeds:
                    current = indexed.loc[(policy, outer_fold, seed)]
                    baseline = indexed.loc[(reference, outer_fold, seed)]
                    receipt = {
                        "policy": policy,
                        "reference": reference,
                        "outer_fold": outer_fold,
                        "seed": seed,
                    }
                    for metric in METRICS:
                        delta = float(current[metric] - baseline[metric])
                        receipt[f"delta_{metric}"] = delta
                        receipt[f"win_{metric}"] = delta < 0 if metric in {"ece", "brier"} else delta > 0
                    paired_rows.append(receipt)
    pd.DataFrame(paired_rows).to_csv(COMPARISON_ROOT / "paired_fold_deltas.csv", index=False)

    seed_delta_rows = []
    seed_indexed = seed_metrics.set_index(["policy", "seed"])
    for policy in POLICIES:
        for reference in ("unweighted_ce", "class_weighted_ce"):
            for seed in Protocol().seeds:
                current = seed_indexed.loc[(policy, seed)]
                baseline = seed_indexed.loc[(reference, seed)]
                receipt = {"policy": policy, "reference": reference, "seed": seed}
                for metric in METRICS:
                    delta = float(current[metric] - baseline[metric])
                    receipt[f"delta_{metric}"] = delta
                    receipt[f"win_{metric}"] = (
                        delta < 0 if metric in {"ece", "brier"} else delta > 0
                    )
                seed_delta_rows.append(receipt)
    pd.DataFrame(seed_delta_rows).to_csv(COMPARISON_ROOT / "seed_oof_deltas.csv", index=False)

    pareto = policy_summary[[
        "policy",
        *[column for metric in PARETO_METRICS for column in (f"{metric}_mean", f"{metric}_std")],
    ]].copy()
    pareto["pareto_nondominated"] = _pareto_flags(
        pareto.rename(columns={f"{metric}_mean": metric for metric in PARETO_METRICS}),
        PARETO_METRICS,
    )
    pareto["normal_specificity_guardrail_ge_0_55"] = pareto["normal_specificity_mean"] >= 0.55
    pareto.to_csv(COMPARISON_ROOT / "pareto_tradeoff.csv", index=False)

    targets = policy_summary.copy()
    for metric in ("macro_f1", "uar", "icbhi_score"):
        targets[f"rank_{metric}"] = targets[f"{metric}_mean"].rank(ascending=False, method="min")
    targets["overall_balanced_mean_rank"] = targets[[
        "rank_macro_f1", "rank_uar", "rank_icbhi_score"
    ]].mean(axis=1)
    targets["minority_guardrail_pass"] = targets["normal_specificity_mean"] >= 0.55
    targets["minority_both_recall_rank_within_guardrail"] = np.nan
    mask = targets["minority_guardrail_pass"]
    targets.loc[mask, "minority_both_recall_rank_within_guardrail"] = targets.loc[
        mask, "both_recall_mean"
    ].rank(ascending=False, method="min")
    targets.sort_values("overall_balanced_mean_rank").to_csv(
        COMPARISON_ROOT / "target_track_summary.csv", index=False
    )
