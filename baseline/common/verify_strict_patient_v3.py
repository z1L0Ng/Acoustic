from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .config import LABELS, REPO_ROOT
from .strict_patient_v3 import (
    BEATS_ROOT,
    BEATS_SHA256,
    COMPARISON_ROOT,
    METRICS,
    OUTER_FOLDS,
    POLICIES,
    PROTOCOL_VERSION,
    RESULT_ROOT,
)


CLASS_METRICS = [
    f"{label}_{metric}"
    for label in LABELS["flat4"]
    for metric in ("precision", "recall", "f1", "support")
]


def _check_notebook() -> None:
    path = REPO_ROOT / "baseline/beats/beats_frozen_downstream.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert any("strict_patient_v3" in "".join(cell.source) for cell in notebook.cells)
    for cell in notebook.cells:
        assert cell.get("id")
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []


def _check_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = pd.read_csv(COMPARISON_ROOT / "outer_fold_assignment.csv", dtype={"patient_id": str})
    nested = pd.read_csv(COMPARISON_ROOT / "nested_fold_assignment.csv", dtype={"patient_id": str})
    counts = pd.read_csv(COMPARISON_ROOT / "fold_class_patient_counts.csv")
    assert len(outer) == 6898 and outer["cycle_id"].is_unique
    assert outer.groupby("patient_id")["outer_fold"].nunique().max() == 1
    assert set(outer["outer_fold"]) == set(range(OUTER_FOLDS))
    assert len(nested) == 6898 * OUTER_FOLDS
    assert len(counts) == OUTER_FOLDS * 3
    for fold in range(OUTER_FOLDS):
        frame = nested[nested["evaluation_outer_fold"].eq(fold)]
        assert len(frame) == 6898 and frame["cycle_id"].is_unique
        patients = {
            role: set(frame.loc[frame["role"].eq(role), "patient_id"])
            for role in ("inner_train", "inner_validation", "outer_test")
        }
        assert not patients["inner_train"] & patients["inner_validation"]
        assert not (patients["inner_train"] | patients["inner_validation"]) & patients["outer_test"]
        test_ids = set(frame.loc[frame["role"].eq("outer_test"), "cycle_id"])
        assert test_ids == set(outer.loc[outer["outer_fold"].eq(fold), "cycle_id"])
    return outer, nested


def _check_receipts() -> None:
    provenance = pd.read_csv(COMPARISON_ROOT / "feature_provenance.csv")
    assert len(provenance) == 1
    assert provenance.iloc[0]["sha256"] == BEATS_SHA256
    assert bool(provenance.iloc[0]["sha256_matches_strict_registry"])
    assert bool(provenance.iloc[0]["features_finite"])
    protocol = json.loads((COMPARISON_ROOT / "protocol.json").read_text())
    sources = json.loads((COMPARISON_ROOT / "primary_source_audit.json").read_text())
    assert protocol["protocol_version"] == PROTOCOL_VERSION
    assert protocol["policies"] == list(POLICIES)
    assert "official challenge split is not used" in protocol["test_split"]
    assert protocol["validation_group"] == "patient_id"
    assert "outer fold 0 contains only four both-class cycles" in protocol["split_caveat"]
    assert sources["logit_adjusted_ce"]["training_sign"] == "plus"
    assert sources["logit_adjusted_ce"]["tau"] == 1.0
    assert sources["ldam_drw"]["max_margin"] == 0.5
    assert sources["ldam_drw"]["scale"] == 30.0
    assert "reinitialize" in sources["crt"]["stage2"]


def _check_probabilities(prediction: pd.DataFrame) -> None:
    columns = [f"prob_{label}" for label in LABELS["flat4"]]
    probability = prediction[columns].to_numpy(dtype=float)
    assert np.isfinite(probability).all()
    assert (probability >= 0).all() and (probability <= 1).all()
    assert np.allclose(probability.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
    expected = np.asarray(LABELS["flat4"])[probability.argmax(axis=1)]
    assert np.array_equal(expected, prediction["y_pred"].to_numpy())


def _check_policy_invariants(results: pd.DataFrame, mode_root: Path) -> None:
    for _, row in results.iterrows():
        run_name = f"beats_{row['policy']}_outer{int(row['outer_fold'])}_seed{int(row['seed'])}"
        run_dir = mode_root / "runs" / run_name
        details = json.loads(row["policy_details_json"])
        curve = pd.read_csv(run_dir / "training_curve.csv")
        assert not curve.empty
        assert np.isfinite(curve[["epoch", "train_loss", "validation_macro_f1"]].to_numpy()).all()
        if row["policy"] == "logit_adjusted_ce":
            spec = details["loss_spec"]
            assert spec["tau"] == 1.0
            assert spec["training_adjustment"].startswith("logits +")
        elif row["policy"] == "ldam_drw":
            assert set(curve["stage"]) == {"ldam_pre_drw", "ldam_drw"}
            assert int(row["best_epoch"]) > int(details["switch_after_epoch"])
            assert details["scale"] == 30.0
            assert np.isclose(max(details["margins"]), 0.5)
        elif row["policy"] == "crt":
            stage2 = details["stage2"]
            assert stage2["hidden_frozen"]
            assert stage2["classifier_reinitialized"]
            assert stage2["hidden_unchanged_after_training"]
            assert set(curve["stage"]) == {
                "crt_stage1_natural", "crt_stage2_classifier_balanced"
            }


def _check_result_rows(results: pd.DataFrame) -> None:
    assert results["protocol_version"].eq(PROTOCOL_VERSION).all()
    assert results["warning_count"].eq(0).all()
    assert results["converged"].eq(True).all()
    assert results["loss_finite"].eq(True).all()
    assert results["gradient_finite"].eq(True).all()
    finite = [
        *METRICS,
        *CLASS_METRICS,
        "parameter_count",
        "trainable_parameter_count",
        "runtime_seconds",
        "best_epoch",
        "best_validation_macro_f1",
    ]
    assert np.isfinite(results[finite].to_numpy(dtype=float)).all()
    assert results["ece"].between(0, 1).all()
    assert results["brier"].between(0, 2).all()


def verify_smoke() -> None:
    _check_assignments()
    _check_receipts()
    path = BEATS_ROOT / "smoke/beats_strict_patient_v3_smoke_results.csv"
    results = pd.read_csv(path)
    assert len(results) == 6
    assert set(results["policy"]) == set(POLICIES)
    assert results.groupby("policy").size().eq(1).all()
    _check_result_rows(results)
    mode_root = BEATS_ROOT / "smoke"
    prediction_paths = list(mode_root.glob("runs/*/predictions.csv"))
    assert len(prediction_paths) == 6
    for path in prediction_paths:
        prediction = pd.read_csv(path)
        assert len(prediction) > 0 and len(prediction) <= 96 and prediction["cycle_id"].is_unique
        _check_probabilities(prediction)
        confusion = pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0)
        assert int(confusion.to_numpy().sum()) == len(prediction)
        assert path.with_name("checkpoint.pt").exists()
    _check_policy_invariants(results, mode_root)
    _check_notebook()
    print("strict_patient_v3_smoke_verification_ok policies=6 assignments=5 notebook=clean")


def verify_full() -> None:
    outer, nested = _check_assignments()
    _check_receipts()
    results = pd.read_csv(COMPARISON_ROOT / "strict_patient_v3_results.csv")
    assert len(results) == 90
    assert results.groupby("policy").size().eq(15).all()
    assert results.groupby(["policy", "outer_fold"]).size().eq(3).all()
    _check_result_rows(results)
    mode_root = BEATS_ROOT / "full"
    prediction_paths = list(mode_root.glob("runs/*/predictions.csv"))
    assert len(prediction_paths) == 90
    for path in prediction_paths:
        prediction = pd.read_csv(path)
        assert prediction["cycle_id"].is_unique
        _check_probabilities(prediction)
        fold = int(prediction["outer_fold"].iloc[0])
        expected_ids = set(outer.loc[outer["outer_fold"].eq(fold), "cycle_id"])
        assert set(prediction["cycle_id"]) == expected_ids
        confusion = pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0)
        assert int(confusion.to_numpy().sum()) == len(prediction)
        for filename in ("checkpoint.pt", "training_curve.csv", "metrics.json", "run_log.json"):
            assert path.with_name(filename).exists()

    for policy in POLICIES:
        for seed in (20260712, 20260713, 20260714):
            paths = list(mode_root.glob(f"runs/beats_{policy}_outer*_seed{seed}/predictions.csv"))
            merged = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            assert len(paths) == OUTER_FOLDS
            assert len(merged) == 6898 and merged["cycle_id"].is_unique
            assert set(merged["cycle_id"]) == set(outer["cycle_id"])

    seed_metrics = pd.read_csv(COMPARISON_ROOT / "seed_oof_metrics.csv")
    summary = pd.read_csv(COMPARISON_ROOT / "policy_oof_summary.csv")
    fold_summary = pd.read_csv(COMPARISON_ROOT / "fold_level_summary.csv")
    paired = pd.read_csv(COMPARISON_ROOT / "paired_fold_deltas.csv")
    seed_deltas = pd.read_csv(COMPARISON_ROOT / "seed_oof_deltas.csv")
    pareto = pd.read_csv(COMPARISON_ROOT / "pareto_tradeoff.csv")
    targets = pd.read_csv(COMPARISON_ROOT / "target_track_summary.csv")
    assert len(seed_metrics) == 18 and seed_metrics.groupby("policy").size().eq(3).all()
    assert len(summary) == 6 and summary["n_seeds"].eq(3).all()
    assert len(fold_summary) == 30 and fold_summary["n_runs"].eq(3).all()
    assert len(paired) == 180
    assert len(seed_deltas) == 36
    assert len(pareto) == 6 and len(targets) == 6
    assert np.isfinite(seed_metrics[METRICS + CLASS_METRICS].to_numpy(dtype=float)).all()
    assert len(list((COMPARISON_ROOT / "scalers").glob("*.joblib"))) == 5
    _check_policy_invariants(results, mode_root)
    _check_notebook()
    ignored = subprocess.run(
        ["git", "check-ignore", str(RESULT_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0
    assert nested.groupby(["evaluation_outer_fold", "cycle_id"]).size().eq(1).all()
    print(
        "strict_patient_v3_full_verification_ok rows=90 predictions=90 "
        "oof_policy_seeds=18 patient_overlap=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    verify_smoke() if args.mode == "smoke" else verify_full()


if __name__ == "__main__":
    main()
