from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from .config import CORE_FEATURES, FEATURES, LABELS, MANIFEST_PATH, RESULT_ROOT, Protocol
from .data import (
    fit_scaler, load_aligned_features, load_manifest, provenance_row, split_arrays,
    validation_assignment,
)
from .metrics import evaluate_predictions, flatten_metrics
from .models import run_lr, run_mlp


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")


def environment_receipt() -> dict:
    try:
        conda_prefix = str(Path(sys.prefix).resolve())
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except Exception as error:
        conda_prefix, pip_freeze = str(sys.prefix), [f"receipt_error={error}"]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": conda_prefix,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_device": "cpu",
        "pip_freeze": pip_freeze,
    }


def _subset_smoke(partitions: dict, limit: int = 96) -> dict:
    output = {}
    for name, (x, y, ids) in partitions.items():
        positions = []
        for label in np.unique(y):
            positions.extend(np.flatnonzero(y == label)[: max(4, limit // len(np.unique(y)))].tolist())
        positions = np.asarray(sorted(positions[:limit]), dtype=int)
        output[name] = (x[positions], y[positions], ids[positions])
    return output


def run_backbone(
    backbone: str,
    mode: str = "full",
    tasks: tuple[str, ...] = ("flat4", "binary"),
    *,
    result_base: Path = RESULT_ROOT,
    protocol_version: str | None = None,
    scope: str | None = None,
) -> list[dict]:
    protocol = Protocol()
    config = FEATURES[backbone]
    result_name = config.get("result_name", backbone)
    result_root = result_base / result_name / ("smoke" if mode == "smoke" else "")
    comparison_root = result_base / "comparison"
    result_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assignment = validation_assignment(manifest, protocol)
    features, feature_metadata = load_aligned_features(
        manifest, Path(config["path"]), config["key"], config["expected_dim"],
        allow_pickle_metadata=config.get("allow_pickle_metadata", False),
        expected_sha256=config.get("expected_sha256"),
    )
    provenance = provenance_row(backbone, config, feature_metadata)

    if mode == "full":
        assignment.to_csv(comparison_root / "validation_assignment.csv", index=False)
        write_json(comparison_root / "protocol.json", {
            **protocol.to_dict(),
            "manifest_path": str(MANIFEST_PATH.resolve()),
            "manifest_rows": len(manifest),
            "official_split_counts": manifest["official_split"].value_counts().to_dict(),
            "partition_counts": assignment["partition"].value_counts().to_dict(),
            "cross_official_split_patients": [156, 218],
            "protocol_version": protocol_version or protocol.protocol_version,
            "scope": scope or "Formal local controlled frozen-feature downstream comparison; not encoder-paper reproduction.",
        })
        write_json(result_root / "environment_receipt.json", environment_receipt())
        write_json(result_root / "config.json", {"backbone": backbone, "feature": provenance, "protocol": protocol.to_dict()})
        provenance_path = comparison_root / "feature_provenance.csv"
        old = pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
        combined = pd.concat([old[old.get("backbone", pd.Series(dtype=str)) != backbone], pd.DataFrame([provenance])], ignore_index=True)
        combined.sort_values("backbone").to_csv(provenance_path, index=False)

    rows = []
    for task in tasks:
        labels = LABELS[task]
        partitions = split_arrays(features, manifest, assignment, task)
        scaler, scaled = fit_scaler(partitions)
        if mode == "smoke":
            scaled = _subset_smoke(scaled)
        else:
            joblib.dump(scaler, result_root / f"{task}_standard_scaler.joblib")
        x_train, y_train, _ = scaled["subtrain"]
        x_val, y_val, _ = scaled["validation"]
        x_test, y_test, test_ids = scaled["test"]

        heads = ("lr", "mlp1", "mlp2")
        for head in heads:
            seeds = (protocol.seeds[0],) if head == "lr" or mode == "smoke" else protocol.seeds
            for seed in seeds:
                run_name = f"{backbone}_{task}_{head}_seed{seed}"
                run_dir = result_root / "runs" / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                if head == "lr":
                    trained = run_lr(x_train, y_train, x_test, labels, protocol, seed)
                    prediction = trained["prediction"]
                    details = {
                        "best_epoch": None, "best_validation_macro_f1": None,
                        "warnings": trained["warnings"], "converged": trained["converged"],
                    }
                    if mode == "full":
                        joblib.dump(trained["model"], run_dir / "model.joblib")
                else:
                    trained = run_mlp(
                        x_train, y_train, x_val, y_val, x_test, labels, protocol, seed, head,
                        smoke=mode == "smoke",
                    )
                    prediction = trained.prediction
                    details = {
                        "best_epoch": trained.best_epoch,
                        "best_validation_macro_f1": trained.best_validation_macro_f1,
                        "warnings": trained.warnings,
                        "converged": True,
                        "class_weights": trained.class_weights,
                    }
                    pd.DataFrame(trained.curve).to_csv(run_dir / "training_curve.csv", index=False)
                    if mode == "full":
                        torch.save({"state_dict": trained.state_dict, "details": details}, run_dir / "checkpoint.pt")
                metrics = evaluate_predictions(y_test, prediction, labels)
                runtime = trained["runtime_seconds"] if head == "lr" else trained.runtime_seconds
                parameters = trained["parameter_count"] if head == "lr" else trained.parameter_count
                row = {
                    "protocol_version": protocol_version or protocol.protocol_version,
                    "backbone": backbone,
                    "encoder": config["encoder"],
                    "feature_key": config["key"],
                    "task": task,
                    "head": head,
                    "seed": seed,
                    "mode": mode,
                    "parameter_count": parameters,
                    "runtime_seconds": runtime,
                    "best_epoch": details["best_epoch"],
                    "best_validation_macro_f1": details["best_validation_macro_f1"],
                    "warning_count": len(details["warnings"]),
                    "warnings": " | ".join(details["warnings"]),
                    "converged": bool(details["converged"]),
                    **flatten_metrics(metrics),
                }
                rows.append(row)
                pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
                    run_dir / "confusion_matrix.csv"
                )
                pd.DataFrame({"cycle_id": test_ids, "y_true": y_test, "y_pred": prediction}).to_csv(
                    run_dir / "predictions.csv", index=False
                )
                write_json(run_dir / "metrics.json", {**row, "class_metrics": metrics["class_metrics"]})
                write_json(run_dir / "run_log.json", {
                    "run_name": run_name, "started_unix": started, "completed_unix": time.time(),
                    "partition_sizes": {name: len(values[1]) for name, values in scaled.items()},
                    "details": details,
                })
                print(f"completed {run_name}: macro_f1={metrics['macro_f1']:.4f} uar={metrics['uar']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(result_root / f"{result_name}_{mode}_results.csv", index=False)
    return rows


def consolidate_full_results(
    backbones: tuple[str, ...] | list[str] = tuple(CORE_FEATURES),
    *,
    result_base: Path = RESULT_ROOT,
    result_filename: str = "formal_downstream_results.csv",
    summary_filename: str = "formal_downstream_summary.csv",
) -> tuple[Path, Path]:
    comparison_root = result_base / "comparison"
    frames = []
    for backbone in backbones:
        result_name = FEATURES[backbone].get("result_name", backbone)
        path = result_base / result_name / f"{result_name}_full_results.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No full result files found")
    results = pd.concat(frames, ignore_index=True)
    result_path = comparison_root / result_filename
    results.to_csv(result_path, index=False)
    metric_columns = [
        "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
        "icbhi_score", "both_recall", "parameter_count", "runtime_seconds",
    ]
    group = results.groupby(["backbone", "encoder", "task", "head"], dropna=False)
    summary = group[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary["n_runs"] = group.size().to_numpy()
    summary_path = comparison_root / summary_filename
    summary.to_csv(summary_path, index=False)
    return result_path, summary_path
