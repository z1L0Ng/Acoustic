"""Independent verifier for Shared Respiratory Attributes Phase 1A."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.shared_respiratory_attributes_phase1a.models import CONDITIONS, condition_parameter_receipt
from model.shared_respiratory_attributes_phase1a.run import (
    CACHE_SHA256,
    DRAW_SEEDS,
    OPTIMIZATION_SEEDS,
    RESULT_ROOT,
)


EXPECTED_PARAMETER_TOTALS = {
    "N": 231_179,
    "I16": 235_375,
    "S16": 233_277,
    "S32": 235_375,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_audit(result_root: Path) -> dict[str, object]:
    protocol = read_json(result_root / "protocol.json")
    data = read_json(result_root / "data_receipt.json")
    strict = read_json(result_root / "strict_icbhi_split_receipt.json")
    shot = read_json(result_root / "shot_support_receipt.json")
    parameter = read_json(result_root / "parameter_receipt.json")
    audit = read_json(result_root / "audit_receipt.json")
    outer = pd.read_csv(result_root / "strict_icbhi_outer_fold_assignment.csv", dtype={"patient_id": str})
    nested = pd.read_csv(result_root / "strict_icbhi_nested_assignment.csv", dtype={"patient_id": str})
    counts = pd.read_csv(result_root / "strict_icbhi_fold_class_patient_counts.csv")

    if protocol["encoder"]["cache_sha256"] != CACHE_SHA256 or data["cache_sha256"] != CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if audit["retained_shots"] != ["4", "8", "16", "32", "64", "full"]:
        raise RuntimeError("unexpected retained shot levels")
    if audit["conditions"] != list(CONDITIONS) or audit["draws"] != len(DRAW_SEEDS):
        raise RuntimeError("condition/draw receipt mismatch")
    if audit["seeds"] != OPTIMIZATION_SEEDS:
        raise RuntimeError("optimization seed mismatch")
    if data["icbhi_official_train_rows"] != 4142 or data["spr_official_train_rows"] != 6656 or data["spr_inter_rows"] != 1429:
        raise RuntimeError("source/target pool size mismatch")
    if len(outer) != 6898 or len(nested) != 6898 * 5 or len(counts) != 15:
        raise RuntimeError("strict split row count mismatch")
    if outer["cycle_id"].nunique() != 6898 or nested.groupby(["evaluation_outer_fold", "cycle_id"]).size().max() != 1:
        raise RuntimeError("strict split identity mismatch")
    for fold in range(5):
        frame = nested[nested["evaluation_outer_fold"].eq(fold)]
        patients = {
            role: set(frame.loc[frame["role"].eq(role), "patient_id"])
            for role in ("inner_train", "inner_validation", "outer_test")
        }
        if patients["inner_train"] & patients["inner_validation"]:
            raise RuntimeError(f"inner patient overlap fold={fold}")
        if (patients["inner_train"] | patients["inner_validation"]) & patients["outer_test"]:
            raise RuntimeError(f"outer patient overlap fold={fold}")
    for condition, values in parameter.items():
        if values["trainable_total"] != EXPECTED_PARAMETER_TOTALS[condition]:
            raise RuntimeError(f"parameter total mismatch for {condition}")
    if parameter != condition_parameter_receipt():
        raise RuntimeError("parameter receipt is not reproducible")
    for draw_index in range(len(DRAW_SEEDS)):
        r1 = shot["r1_icbhi_to_spr"][f"draw_{draw_index}"]
        if r1["shot_support"]["4"]["groups"] != 4 or r1["shot_support"]["8"]["groups"] != 8:
            raise RuntimeError("R1 draw support mismatch")
    return {
        "status": "phase1a_audit_verified",
        "cache_sha256": CACHE_SHA256,
        "retained_shots": audit["retained_shots"],
        "conditions": len(CONDITIONS),
        "draws": len(DRAW_SEEDS),
        "seeds": len(OPTIMIZATION_SEEDS),
        "strict_nested_rows": len(nested),
        "warnings": 0,
    }


def verify_smoke(result_root: Path) -> dict[str, object]:
    audit = verify_audit(result_root)
    smoke = read_json(result_root / "smoke_receipt.json")
    if smoke["status"] != "phase1a_real_data_smoke_passed" or smoke["outer_test_metrics_evaluated"] or smoke["warnings"] != 0:
        raise RuntimeError("smoke boundary failed")
    for condition in CONDITIONS:
        directory_r1 = result_root / "smoke" / f"r1_cond{condition}"
        directory_r2 = result_root / "smoke" / f"r2_cond{condition}"
        for directory in (directory_r1, directory_r2):
            metrics = read_json(directory / "metrics.json")
            if metrics["metrics"]["outer_test_scored"] or not metrics["loss_finite"] or not metrics["gradient_finite"] or metrics["warnings"]:
                raise RuntimeError(f"smoke metrics failed for {directory}")
            if metrics["parameter_receipt"]["trainable_total"] != EXPECTED_PARAMETER_TOTALS[condition]:
                raise RuntimeError(f"smoke parameter mismatch for {condition}")
            label_free = read_gzip_csv(directory / "predictions_label_free.csv.gz")
            if not label_free or (directory / "predictions_scored.csv.gz").exists():
                raise RuntimeError("smoke prediction boundary failed")
    return {
        "status": "phase1a_smoke_verified",
        "audit": audit,
        "conditions": len(CONDITIONS),
        "warnings": 0,
        "outer_test_metrics_evaluated": False,
    }


def verify_full(result_root: Path) -> dict[str, object]:
    smoke = verify_smoke(result_root)
    manifest = read_json(result_root / "run_manifest.json")
    decision = read_json(result_root / "decision_receipt.json")
    merged = list(csv.DictReader((result_root / "merged_oof_metrics.csv").open()))
    aulc = list(csv.DictReader((result_root / "aulc_by_draw_seed.csv").open()))
    paired = list(csv.DictReader((result_root / "paired_aulc_deltas.csv").open()))
    summary = list(csv.DictReader((result_root / "summary.csv").open()))
    guard = list(csv.DictReader((result_root / "guardrail_deltas.csv").open()))
    intervals = list(csv.DictReader((result_root / "patient_cluster_interval.csv").open()))

    if manifest["status"] != "phase1a_full_complete" or manifest["warnings"] != 0 or manifest["decision"] != decision["decision"]:
        raise RuntimeError("full manifest/decision mismatch")
    if len(merged) != len(CONDITIONS) * 2 * 6 * len(DRAW_SEEDS) * len(OPTIMIZATION_SEEDS):
        raise RuntimeError("merged metric row count mismatch")
    if len(aulc) != len(CONDITIONS) * 2 * len(DRAW_SEEDS) * len(OPTIMIZATION_SEEDS):
        raise RuntimeError("AULC row count mismatch")
    if len(paired) != 4 * 2 * len(DRAW_SEEDS) * len(OPTIMIZATION_SEEDS):
        raise RuntimeError("paired delta row count mismatch")
    if len(summary) != 10 or len(intervals) != 2:
        raise RuntimeError("summary / interval row count mismatch")
    if len(guard) != 2 * 6 * 4:
        raise RuntimeError("guardrail row count mismatch")

    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        for condition in CONDITIONS:
            for shot in ("4", "8", "16", "32", "64", "full"):
                for draw_index in range(len(DRAW_SEEDS)):
                    for seed in OPTIMIZATION_SEEDS:
                        merged_path = result_root / "merged_predictions" / f"{direction}_shot{shot}_draw{draw_index}_seed{seed}_cond{condition}.csv.gz"
                        rows = read_gzip_csv(merged_path)
                        ids = [row["sample_id"] for row in rows]
                        if len(ids) != len(set(ids)):
                            raise RuntimeError(f"duplicate merged IDs: {merged_path.name}")
                        if direction == "r1_icbhi_to_spr" and len(ids) != 1429:
                            raise RuntimeError(f"SPR inter coverage mismatch: {merged_path.name}")
                        if direction == "r2_spr_to_icbhi" and len(ids) != 6898:
                            raise RuntimeError(f"ICBHI OOF coverage mismatch: {merged_path.name}")
    return {
        "status": "phase1a_full_verified",
        "smoke": smoke,
        "merged_rows": len(merged),
        "aulc_rows": len(aulc),
        "paired_delta_rows": len(paired),
        "summary_rows": len(summary),
        "interval_rows": len(intervals),
        "decision": decision["decision"],
        "warnings": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "smoke", "full"], required=True)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    payload = (
        verify_audit(args.result_root.resolve())
        if args.mode == "audit"
        else verify_smoke(args.result_root.resolve())
        if args.mode == "smoke"
        else verify_full(args.result_root.resolve())
    )
    output = args.result_root / f"{args.mode}_verification.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
