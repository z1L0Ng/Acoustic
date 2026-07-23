from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import (
    CORE_FEATURES, FEATURES, IMBALANCE_LOSSES, IMBALANCE_RESULT_ROOT, LABELS, MANIFEST_PATH,
    Protocol, imbalance_protocol,
)
from .data import (
    fit_scaler, load_aligned_features, load_manifest, provenance_row, split_arrays,
    validation_assignment,
)
from .losses import build_loss
from .metrics import evaluate_predictions, flatten_metrics
from .models import run_mlp
from .run import _subset_smoke, environment_receipt, write_json


METRICS = [
    "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
    "icbhi_score", "both_recall", "parameter_count", "runtime_seconds",
]
TRADEOFF_METRICS = ["both_recall", "normal_specificity", "uar", "icbhi_score"]


def _loss_receipts(reference_y: np.ndarray, labels: list[str]) -> dict:
    receipts = {}
    for loss_name in IMBALANCE_LOSSES:
        _, spec = build_loss(loss_name, reference_y, labels, torch.device("cpu"))
        receipts[loss_name] = spec.__dict__
    return receipts


def run_imbalance_backbone(
    backbone: str,
    mode: str = "full",
    *,
    result_base: Path = IMBALANCE_RESULT_ROOT,
    protocol_version: str = "formal-imbalance-loss-v1",
    scope: str | None = None,
) -> list[dict]:
    protocol = Protocol()
    labels = LABELS["flat4"]
    feature_config = FEATURES[backbone]
    result_name = feature_config.get("result_name", backbone)
    result_root = result_base / result_name / ("smoke" if mode == "smoke" else "")
    comparison_root = result_base / "comparison"
    result_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assignment = validation_assignment(manifest, protocol)
    features, feature_metadata = load_aligned_features(
        manifest, Path(feature_config["path"]), feature_config["key"], feature_config["expected_dim"],
        allow_pickle_metadata=feature_config.get("allow_pickle_metadata", False),
        expected_sha256=feature_config.get("expected_sha256"),
    )
    partitions = split_arrays(features, manifest, assignment, "flat4")
    scaler, scaled_full = fit_scaler(partitions)
    reference_y = scaled_full["subtrain"][1]
    receipts = _loss_receipts(reference_y, labels)
    scaled = _subset_smoke(scaled_full) if mode == "smoke" else scaled_full

    protocol_receipt = {
        **imbalance_protocol(protocol),
        "protocol_version": protocol_version,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_rows": len(manifest),
        "official_split_counts": manifest["official_split"].value_counts().to_dict(),
        "partition_counts": assignment["partition"].value_counts().to_dict(),
        "cross_official_split_patients": [156, 218],
        "label_order": labels,
        "official_subtrain_class_counts": receipts["unweighted_ce"]["class_counts"],
        "resolved_loss_specs": receipts,
    }
    if scope is not None:
        protocol_receipt["scope"] = scope
    write_json(
        comparison_root / ("smoke_protocol.json" if mode == "smoke" else "protocol.json"),
        protocol_receipt,
    )
    if mode == "full":
        assignment.to_csv(comparison_root / "validation_assignment.csv", index=False)
        joblib.dump(scaler, result_root / "flat4_standard_scaler.joblib")
        write_json(result_root / "environment_receipt.json", environment_receipt())
        provenance = provenance_row(backbone, feature_config, feature_metadata)
        write_json(result_root / "config.json", {
            "backbone": backbone,
            "feature": provenance,
            "protocol": protocol_receipt,
        })
        provenance_path = comparison_root / "feature_provenance.csv"
        old = pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["backbone"] != backbone]
        pd.concat([old, pd.DataFrame([provenance])], ignore_index=True).sort_values("backbone").to_csv(
            provenance_path, index=False
        )

    x_train, y_train, _ = scaled["subtrain"]
    x_val, y_val, _ = scaled["validation"]
    x_test, y_test, test_ids = scaled["test"]
    seeds = (protocol.seeds[0],) if mode == "smoke" else protocol.seeds
    rows = []
    for loss_name in IMBALANCE_LOSSES:
        for seed in seeds:
            run_name = f"{backbone}_flat4_mlp2_{loss_name}_seed{seed}"
            run_dir = result_root / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            trained = run_mlp(
                x_train, y_train, x_val, y_val, x_test, labels, protocol, seed, "mlp2",
                smoke=mode == "smoke", loss_name=loss_name, loss_reference_y=reference_y,
            )
            metrics = evaluate_predictions(y_test, trained.prediction, labels)
            row = {
                "protocol_version": protocol_version,
                "backbone": backbone,
                "encoder": feature_config["encoder"],
                "feature_key": feature_config["key"],
                "task": "flat4",
                "head": "mlp2",
                "loss": loss_name,
                "seed": seed,
                "mode": mode,
                "parameter_count": trained.parameter_count,
                "runtime_seconds": trained.runtime_seconds,
                "best_epoch": trained.best_epoch,
                "best_validation_macro_f1": trained.best_validation_macro_f1,
                "warning_count": len(trained.warnings),
                "warnings": " | ".join(trained.warnings),
                "converged": True,
                "loss_finite": trained.loss_finite,
                "gradient_finite": trained.gradient_finite,
                "class_counts_json": json.dumps(trained.loss_spec["class_counts"]),
                "class_weights_json": json.dumps(trained.loss_spec["class_weights"]),
                "loss_spec_json": json.dumps(trained.loss_spec, sort_keys=True),
                **flatten_metrics(metrics),
            }
            rows.append(row)
            pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
                run_dir / "confusion_matrix.csv"
            )
            pd.DataFrame({
                "cycle_id": test_ids,
                "y_true": y_test,
                "y_pred": trained.prediction,
            }).to_csv(run_dir / "predictions.csv", index=False)
            pd.DataFrame(trained.curve).to_csv(run_dir / "training_curve.csv", index=False)
            if mode == "full":
                torch.save({
                    "state_dict": trained.state_dict,
                    "loss_spec": trained.loss_spec,
                    "best_epoch": trained.best_epoch,
                }, run_dir / "checkpoint.pt")
            write_json(run_dir / "metrics.json", {**row, "class_metrics": metrics["class_metrics"]})
            write_json(run_dir / "run_log.json", {
                "run_name": run_name,
                "started_unix": started,
                "completed_unix": time.time(),
                "partition_sizes": {name: len(values[1]) for name, values in scaled.items()},
                "loss_spec": trained.loss_spec,
                "loss_finite": trained.loss_finite,
                "gradient_finite": trained.gradient_finite,
            })
            print(
                f"completed {run_name}: macro_f1={metrics['macro_f1']:.4f} "
                f"uar={metrics['uar']:.4f} both={metrics['both_recall']:.4f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(result_root / f"{result_name}_{mode}_results.csv", index=False)
    return rows


def _pareto_flags(frame: pd.DataFrame) -> list[bool]:
    values = frame[TRADEOFF_METRICS].to_numpy()
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


def consolidate_imbalance_results(
    backbones: tuple[str, ...] | list[str] = tuple(CORE_FEATURES),
    *,
    result_base: Path = IMBALANCE_RESULT_ROOT,
    result_filename: str = "formal_imbalance_results.csv",
    summary_filename: str = "formal_imbalance_summary.csv",
) -> tuple[Path, Path]:
    comparison_root = result_base / "comparison"
    frames = []
    for backbone in backbones:
        result_name = FEATURES[backbone].get("result_name", backbone)
        path = result_base / result_name / f"{result_name}_full_results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No full imbalance result files found")
    results = pd.concat(frames, ignore_index=True)
    result_path = comparison_root / result_filename
    results.to_csv(result_path, index=False)

    grouped = results.groupby(["backbone", "encoder", "task", "head", "loss"], dropna=False)
    summary = grouped[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary["n_runs"] = grouped.size().to_numpy()
    summary_path = comparison_root / summary_filename
    summary.to_csv(summary_path, index=False)

    delta_rows = []
    for backbone, frame in summary.groupby("backbone"):
        indexed = frame.set_index("loss")
        for loss_name, row in indexed.iterrows():
            delta = {"backbone": backbone, "loss": loss_name}
            for baseline_loss in ("unweighted_ce", "class_weighted_ce"):
                for metric in TRADEOFF_METRICS + ["macro_f1", "weighted_f1"]:
                    delta[f"delta_{metric}_vs_{baseline_loss}"] = (
                        row[f"{metric}_mean"] - indexed.loc[baseline_loss, f"{metric}_mean"]
                    )
            delta_rows.append(delta)
    pd.DataFrame(delta_rows).to_csv(comparison_root / "loss_deltas.csv", index=False)

    seed_rows = []
    for (backbone, seed), frame in results.groupby(["backbone", "seed"]):
        indexed = frame.set_index("loss")
        for loss_name, row in indexed.iterrows():
            receipt = {"backbone": backbone, "seed": seed, "loss": loss_name}
            for baseline_loss in ("unweighted_ce", "class_weighted_ce"):
                for metric in TRADEOFF_METRICS + ["macro_f1", "weighted_f1"]:
                    delta = row[metric] - indexed.loc[baseline_loss, metric]
                    receipt[f"delta_{metric}_vs_{baseline_loss}"] = delta
                    receipt[f"win_{metric}_vs_{baseline_loss}"] = bool(delta > 0)
            seed_rows.append(receipt)
    pd.DataFrame(seed_rows).to_csv(comparison_root / "seed_level_deltas.csv", index=False)

    pareto = summary[[
        "backbone", "loss", "both_recall_mean", "both_recall_std",
        "normal_specificity_mean", "normal_specificity_std", "uar_mean", "uar_std",
        "icbhi_score_mean", "icbhi_score_std",
    ]].copy()
    pareto["pareto_nondominated"] = False
    for _, indexes in pareto.groupby("backbone").groups.items():
        subset = pareto.loc[indexes].rename(columns={f"{name}_mean": name for name in TRADEOFF_METRICS})
        pareto.loc[indexes, "pareto_nondominated"] = _pareto_flags(subset)
    pareto.to_csv(comparison_root / "pareto_tradeoff.csv", index=False)
    return result_path, summary_path
