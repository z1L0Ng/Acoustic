from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .config import (
    EXTENSION_ARCHITECTURE_ROOT, EXTENSION_FEATURES, EXTENSION_IMBALANCE_ROOT,
    EXTENSION_RESULT_ROOT, IMBALANCE_LOSSES, LABELS, MANIFEST_PATH,
)
from .losses import balanced_weights, effective_number_weights


BACKBONES = tuple(EXTENSION_FEATURES)
EXPECTED_NOTEBOOKS = (
    "baseline/ast/ast_frozen_downstream.ipynb",
    "baseline/beats/beats_frozen_downstream.ipynb",
    "baseline/clap/clap_frozen_downstream.ipynb",
    "baseline/hear/hear_frozen_downstream.ipynb",
    "baseline/opera/opera_ct_official_like_downstream.ipynb",
    "baseline/simple_acoustic/simple_acoustic_downstream.ipynb",
)
RESULT_NAMES = {name: config.get("result_name", name) for name, config in EXTENSION_FEATURES.items()}
BASE_METRICS = [
    "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
    "icbhi_score", "parameter_count", "runtime_seconds",
]
FLAT4_CLASS_METRICS = [
    f"{label}_{metric}"
    for label in LABELS["flat4"]
    for metric in ("precision", "recall", "f1", "support")
    if not (label == "both" and metric == "recall")
]
BINARY_CLASS_METRICS = [
    f"{label}_{metric}"
    for label in LABELS["binary"]
    for metric in ("precision", "recall", "f1", "support")
]


def check_metric_finiteness(results: pd.DataFrame) -> None:
    assert np.isfinite(results[BASE_METRICS].to_numpy()).all()
    flat4 = results[results["task"].eq("flat4")]
    binary = results[results["task"].eq("binary")]
    if not flat4.empty:
        assert np.isfinite(flat4[["both_recall", *FLAT4_CLASS_METRICS]].to_numpy()).all()
    if not binary.empty:
        assert np.isfinite(binary[BINARY_CLASS_METRICS].to_numpy()).all()


def check_training_curves(paths: list[Path]) -> None:
    for path in paths:
        curve = pd.read_csv(path)
        assert not curve.empty
        assert np.isfinite(curve[["epoch", "train_loss", "validation_macro_f1"]].to_numpy()).all()


def check_notebooks() -> None:
    project_root = EXTENSION_RESULT_ROOT.parents[2]
    paths = [project_root / relative for relative in EXPECTED_NOTEBOOKS]
    assert all(path.is_file() for path in paths)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            assert cell.get("id")
            if cell.cell_type == "code":
                assert cell.execution_count is None and cell.outputs == []


def check_audit() -> None:
    audit = pd.read_csv(EXTENSION_RESULT_ROOT / "comparison/input_provenance_audit.csv")
    assert len(audit) == 3
    flags = [
        "cycle_id_unique", "cycle_id_set_matches_manifest", "labels_match_manifest_after_join",
        "split_matches_manifest_after_join", "features_finite", "usable_all",
        "object_metadata_values_are_strings", "sha256_matches_registry",
    ]
    assert audit[flags].eq(True).all().all()
    assert audit["official_reproduction_claim"].eq(False).all()
    assert set(audit["shape"]) == {"6898x114", "6898x512", "6898x768"}


def check_loss_weights(results: pd.DataFrame) -> None:
    counts = np.asarray([1578, 805, 408, 264])
    expected = {
        "unweighted_ce": None,
        "class_weighted_ce": balanced_weights(counts),
        "focal_loss": None,
        "class_balanced_ce": effective_number_weights(counts, beta=0.9999),
    }
    for _, row in results.iterrows():
        assert np.array_equal(np.asarray(json.loads(row["class_counts_json"])), counts)
        actual = None if pd.isna(row["class_weights_json"]) else np.asarray(json.loads(row["class_weights_json"]))
        if expected[row["loss"]] is None:
            assert actual is None
        else:
            assert np.allclose(actual, expected[row["loss"]], rtol=1e-7, atol=1e-7)


def verify_smoke() -> None:
    check_audit()
    architecture = pd.concat([
        pd.read_csv(
            EXTENSION_ARCHITECTURE_ROOT / result_name / "smoke" / f"{result_name}_smoke_results.csv"
        )
        for result_name in RESULT_NAMES.values()
    ], ignore_index=True)
    assert len(architecture) == 18
    assert architecture.groupby(["backbone", "task", "head"]).size().eq(1).all()
    assert architecture["warning_count"].eq(0).all()
    check_metric_finiteness(architecture)
    arch_paths = list(EXTENSION_ARCHITECTURE_ROOT.glob("*/smoke/runs/*/predictions.csv"))
    assert len(arch_paths) == 18
    for path in arch_paths:
        prediction = pd.read_csv(path)
        assert len(prediction) == 96 and prediction["cycle_id"].is_unique
        assert int(pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0).to_numpy().sum()) == 96

    imbalance = pd.concat([
        pd.read_csv(
            EXTENSION_IMBALANCE_ROOT / result_name / "smoke" / f"{result_name}_smoke_results.csv"
        )
        for result_name in RESULT_NAMES.values()
    ], ignore_index=True)
    assert len(imbalance) == 12
    assert imbalance.groupby(["backbone", "loss"]).size().eq(1).all()
    assert set(imbalance["loss"]) == set(IMBALANCE_LOSSES)
    assert imbalance["warning_count"].eq(0).all()
    assert imbalance["loss_finite"].eq(True).all() and imbalance["gradient_finite"].eq(True).all()
    check_metric_finiteness(imbalance)
    check_loss_weights(imbalance)
    loss_paths = list(EXTENSION_IMBALANCE_ROOT.glob("*/smoke/runs/*/predictions.csv"))
    assert len(loss_paths) == 12
    for path in loss_paths:
        prediction = pd.read_csv(path)
        assert len(prediction) == 96 and prediction["cycle_id"].is_unique
        assert int(pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0).to_numpy().sum()) == 96
    check_notebooks()
    print("extension_smoke_verification_ok architecture=18 imbalance=12 audit=3 notebooks=6")


def check_class_weighted_identity(architecture: pd.DataFrame, imbalance: pd.DataFrame) -> None:
    metrics = [
        "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
        "icbhi_score", "both_recall", "parameter_count",
    ]
    weighted = imbalance[imbalance["loss"].eq("class_weighted_ce")]
    for _, row in weighted.iterrows():
        match = architecture[
            architecture["backbone"].eq(row["backbone"])
            & architecture["task"].eq("flat4")
            & architecture["head"].eq("mlp2")
            & architecture["seed"].eq(row["seed"])
        ].iloc[0]
        assert np.allclose(row[metrics].astype(float), match[metrics].astype(float), rtol=0, atol=0)
        result_name = RESULT_NAMES[row["backbone"]]
        old_name = f"{row['backbone']}_flat4_mlp2_seed{int(row['seed'])}"
        new_name = f"{row['backbone']}_flat4_mlp2_class_weighted_ce_seed{int(row['seed'])}"
        arch_pred = pd.read_csv(EXTENSION_ARCHITECTURE_ROOT / result_name / "runs" / old_name / "predictions.csv")
        loss_pred = pd.read_csv(EXTENSION_IMBALANCE_ROOT / result_name / "runs" / new_name / "predictions.csv")
        assert arch_pred.equals(loss_pred)


def verify_full() -> None:
    check_audit()
    architecture = pd.read_csv(
        EXTENSION_ARCHITECTURE_ROOT / "comparison/formal_extension_architecture_results.csv"
    )
    arch_summary = pd.read_csv(
        EXTENSION_ARCHITECTURE_ROOT / "comparison/formal_extension_architecture_summary.csv"
    )
    assert len(architecture) == 42
    expected = {("lr", 1), ("mlp1", 3), ("mlp2", 3)}
    for _, group in architecture.groupby(["backbone", "task"]):
        assert {(head, len(rows)) for head, rows in group.groupby("head")} == expected
    assert architecture["warning_count"].eq(0).all() and architecture["converged"].eq(True).all()
    check_metric_finiteness(architecture)
    assert arch_summary.loc[arch_summary["head"].eq("lr"), "n_runs"].eq(1).all()
    assert arch_summary.loc[~arch_summary["head"].eq("lr"), "n_runs"].eq(3).all()

    imbalance = pd.read_csv(
        EXTENSION_IMBALANCE_ROOT / "comparison/formal_extension_imbalance_results.csv"
    )
    loss_summary = pd.read_csv(
        EXTENSION_IMBALANCE_ROOT / "comparison/formal_extension_imbalance_summary.csv"
    )
    assert len(imbalance) == 36 and imbalance.groupby(["backbone", "loss"]).size().eq(3).all()
    assert imbalance["warning_count"].eq(0).all() and imbalance["converged"].eq(True).all()
    assert imbalance["loss_finite"].eq(True).all() and imbalance["gradient_finite"].eq(True).all()
    check_metric_finiteness(imbalance)
    assert loss_summary["n_runs"].eq(3).all()
    check_loss_weights(imbalance)

    manifest = pd.read_csv(MANIFEST_PATH)
    test_ids = set(manifest.loc[manifest["official_split"].eq("test"), "cycle_id"])
    paths = list(EXTENSION_ARCHITECTURE_ROOT.glob("*/runs/*/predictions.csv")) + list(
        EXTENSION_IMBALANCE_ROOT.glob("*/runs/*/predictions.csv")
    )
    assert len(paths) == 78
    for path in paths:
        prediction = pd.read_csv(path)
        assert len(prediction) == 2756 and prediction["cycle_id"].is_unique
        assert set(prediction["cycle_id"]) == test_ids
        assert int(pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0).to_numpy().sum()) == 2756
        for filename in ("metrics.json", "run_log.json"):
            assert path.with_name(filename).exists()
    assert len(list(EXTENSION_ARCHITECTURE_ROOT.glob("*/runs/*/checkpoint.pt"))) == 36
    assert len(list(EXTENSION_ARCHITECTURE_ROOT.glob("*/runs/*/model.joblib"))) == 6
    architecture_curves = list(EXTENSION_ARCHITECTURE_ROOT.glob("*/runs/*/training_curve.csv"))
    assert len(architecture_curves) == 36
    check_training_curves(architecture_curves)
    assert len(list(EXTENSION_IMBALANCE_ROOT.glob("*/runs/*/checkpoint.pt"))) == 36
    imbalance_curves = list(EXTENSION_IMBALANCE_ROOT.glob("*/runs/*/training_curve.csv"))
    assert len(imbalance_curves) == 36
    check_training_curves(imbalance_curves)
    assert len(pd.read_csv(EXTENSION_ARCHITECTURE_ROOT / "comparison/seed_level_deltas.csv")) == 36
    assert len(pd.read_csv(EXTENSION_IMBALANCE_ROOT / "comparison/loss_deltas.csv")) == 12
    assert len(pd.read_csv(EXTENSION_IMBALANCE_ROOT / "comparison/seed_level_deltas.csv")) == 36
    assert len(pd.read_csv(EXTENSION_IMBALANCE_ROOT / "comparison/pareto_tradeoff.csv")) == 12
    check_class_weighted_identity(architecture, imbalance)
    combined = pd.read_csv(EXTENSION_RESULT_ROOT / "comparison/six_representation_summary.csv")
    assert len(combined) == 6 and combined["representation"].is_unique
    check_notebooks()
    print("extension_full_verification_ok architecture=42 imbalance=36 predictions=78 identity=true combined=6")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    verify_smoke() if args.mode == "smoke" else verify_full()


if __name__ == "__main__":
    main()
