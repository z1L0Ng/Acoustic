from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .config import CORE_FEATURES, MANIFEST_PATH, RESULT_ROOT


def main() -> None:
    results = pd.read_csv(RESULT_ROOT / "comparison/formal_downstream_results.csv")
    summary = pd.read_csv(RESULT_ROOT / "comparison/formal_downstream_summary.csv")
    assert len(results) == 42
    expected = {("lr", 1), ("mlp1", 3), ("mlp2", 3)}
    for _, group in results.groupby(["backbone", "task"]):
        assert {(head, len(rows)) for head, rows in group.groupby("head")} == expected
    assert results["warning_count"].eq(0).all()
    assert results["converged"].eq(True).all()
    metrics = [
        "macro_f1", "weighted_f1", "uar", "abnormal_sensitivity", "normal_specificity",
        "icbhi_score", "parameter_count", "runtime_seconds",
    ]
    assert np.isfinite(results[metrics].to_numpy()).all()
    assert np.isfinite(results.loc[results["task"].eq("flat4"), "both_recall"]).all()
    assert results.loc[results["task"].eq("binary"), "both_recall"].isna().all()

    manifest = pd.read_csv(MANIFEST_PATH)
    test_ids = set(manifest.loc[manifest["official_split"].eq("test"), "cycle_id"])
    prediction_paths = list(RESULT_ROOT.glob("*/runs/*/predictions.csv"))
    assert len(prediction_paths) == 42
    for path in prediction_paths:
        predictions = pd.read_csv(path)
        assert len(predictions) == 2756
        assert predictions["cycle_id"].is_unique
        assert set(predictions["cycle_id"]) == test_ids
        confusion = pd.read_csv(path.with_name("confusion_matrix.csv"), index_col=0)
        assert int(confusion.to_numpy().sum()) == 2756
    for _, row in summary.iterrows():
        assert row["n_runs"] == (1 if row["head"] == "lr" else 3)

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
    print("formal_verification_ok rows=42 predictions=42 notebooks=3")


if __name__ == "__main__":
    main()
