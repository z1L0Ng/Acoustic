from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .config import CORE_FEATURES, IMBALANCE_LOSSES, IMBALANCE_RESULT_ROOT, LABELS, MANIFEST_PATH, RESULT_ROOT
from .losses import balanced_weights, effective_number_weights


METRICS = [
    "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
    "icbhi_score", "both_recall", "parameter_count", "runtime_seconds",
]


def _check_weights(results: pd.DataFrame) -> None:
    expected_counts = np.asarray([1578, 805, 408, 264])
    expected = {
        "unweighted_ce": None,
        "class_weighted_ce": balanced_weights(expected_counts),
        "focal_loss": None,
        "class_balanced_ce": effective_number_weights(expected_counts, beta=0.9999),
    }
    for _, row in results.iterrows():
        assert np.array_equal(np.asarray(json.loads(row["class_counts_json"])), expected_counts)
        weights = None if pd.isna(row["class_weights_json"]) else json.loads(row["class_weights_json"])
        if expected[row["loss"]] is None:
            assert weights is None
        else:
            assert np.allclose(np.asarray(weights), expected[row["loss"]], rtol=1e-7, atol=1e-7)


def _check_notebooks() -> None:
    baseline_root = RESULT_ROOT.parents[2] / "baseline"
    notebook_paths = []
    for backbone in CORE_FEATURES:
        candidates = list((baseline_root / backbone).glob("*.ipynb"))
        assert len(candidates) == 1
        notebook_paths.extend(candidates)
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            assert cell.get("id")
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []


def verify_smoke() -> None:
    frames = [
        pd.read_csv(IMBALANCE_RESULT_ROOT / backbone / "smoke" / f"{backbone}_smoke_results.csv")
        for backbone in ("ast", "clap", "beats")
    ]
    results = pd.concat(frames, ignore_index=True)
    assert len(results) == 12
    assert results.groupby(["backbone", "loss"]).size().eq(1).all()
    assert set(results["loss"]) == set(IMBALANCE_LOSSES)
    assert results["warning_count"].eq(0).all()
    assert results["loss_finite"].eq(True).all()
    assert results["gradient_finite"].eq(True).all()
    assert np.isfinite(results[METRICS + ["best_epoch", "best_validation_macro_f1"]].to_numpy()).all()
    class_metrics = [
        f"{label}_{metric}"
        for label in LABELS["flat4"]
        for metric in ("precision", "recall", "f1", "support")
    ]
    assert np.isfinite(results[class_metrics].to_numpy()).all()
    _check_weights(results)
    paths = list(IMBALANCE_RESULT_ROOT.glob("*/smoke/runs/*/predictions.csv"))
    assert len(paths) == 12
    for path in paths:
        predictions = pd.read_csv(path)
        assert len(predictions) == 96 and predictions["cycle_id"].is_unique
        confusion = pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0)
        assert int(confusion.to_numpy().sum()) == 96
        for filename in ("metrics.json", "run_log.json", "training_curve.csv"):
            assert path.with_name(filename).exists()
    _check_notebooks()
    print("imbalance_smoke_verification_ok rows=12 predictions=12")


def _check_class_weighted_identity(results: pd.DataFrame) -> None:
    formal = pd.read_csv(RESULT_ROOT / "comparison/formal_downstream_results.csv")
    metrics = ["macro_f1", "weighted_f1", "uar", "normal_specificity", "icbhi_score", "both_recall"]
    current = results[results["loss"].eq("class_weighted_ce")]
    for _, row in current.iterrows():
        original = formal[
            formal["backbone"].eq(row["backbone"])
            & formal["task"].eq("flat4")
            & formal["head"].eq("mlp2")
            & formal["seed"].eq(row["seed"])
        ].iloc[0]
        assert np.allclose(row[metrics].astype(float), original[metrics].astype(float), rtol=0, atol=0)
        run_name = f"{row['backbone']}_flat4_mlp2_seed{int(row['seed'])}"
        current_name = f"{row['backbone']}_flat4_mlp2_class_weighted_ce_seed{int(row['seed'])}"
        old_pred = pd.read_csv(RESULT_ROOT / row["backbone"] / "runs" / run_name / "predictions.csv")
        new_pred = pd.read_csv(IMBALANCE_RESULT_ROOT / row["backbone"] / "runs" / current_name / "predictions.csv")
        assert old_pred.equals(new_pred)


def verify_full() -> None:
    results = pd.read_csv(IMBALANCE_RESULT_ROOT / "comparison/formal_imbalance_results.csv")
    summary = pd.read_csv(IMBALANCE_RESULT_ROOT / "comparison/formal_imbalance_summary.csv")
    assert len(results) == 36
    assert results.groupby(["backbone", "loss"]).size().eq(3).all()
    assert set(results["loss"]) == set(IMBALANCE_LOSSES)
    assert results["warning_count"].eq(0).all()
    assert results["converged"].eq(True).all()
    assert results["loss_finite"].eq(True).all()
    assert results["gradient_finite"].eq(True).all()
    assert np.isfinite(results[METRICS + ["best_epoch", "best_validation_macro_f1"]].to_numpy()).all()
    class_metrics = [
        f"{label}_{metric}"
        for label in LABELS["flat4"]
        for metric in ("precision", "recall", "f1", "support")
    ]
    assert np.isfinite(results[class_metrics].to_numpy()).all()
    _check_weights(results)

    manifest = pd.read_csv(MANIFEST_PATH)
    test_ids = set(manifest.loc[manifest["official_split"].eq("test"), "cycle_id"])
    paths = list(IMBALANCE_RESULT_ROOT.glob("*/runs/*/predictions.csv"))
    assert len(paths) == 36
    for path in paths:
        predictions = pd.read_csv(path)
        assert len(predictions) == 2756
        assert predictions["cycle_id"].is_unique
        assert set(predictions["cycle_id"]) == test_ids
        confusion = pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0)
        assert int(confusion.to_numpy().sum()) == 2756
        for filename in ("metrics.json", "run_log.json", "training_curve.csv", "checkpoint.pt"):
            assert path.with_name(filename).exists()
    assert summary["n_runs"].eq(3).all()
    assert len(pd.read_csv(IMBALANCE_RESULT_ROOT / "comparison/loss_deltas.csv")) == 12
    assert len(pd.read_csv(IMBALANCE_RESULT_ROOT / "comparison/seed_level_deltas.csv")) == 36
    assert len(pd.read_csv(IMBALANCE_RESULT_ROOT / "comparison/pareto_tradeoff.csv")) == 12
    _check_class_weighted_identity(results)
    _check_notebooks()
    print("imbalance_full_verification_ok rows=36 predictions=36 class_weighted_identity=true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    verify_smoke() if args.mode == "smoke" else verify_full()


if __name__ == "__main__":
    main()
