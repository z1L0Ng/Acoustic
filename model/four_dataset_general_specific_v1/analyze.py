"""Create preregistered comparisons and decision receipts after full verification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result/model/2026-07-28/four_dataset_general_specific_v1"
FROZEN = ROOT / "result/four_dataset_pafa_frozen_encoder"

C0 = "d2_local_reference"
C1 = "d2_shared_residual_param_matched"
M1 = "d2_task_residual"
M2 = "d2_task_residual_selective_prior"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def external_sanity(
    summary: pd.DataFrame, frozen: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    local = summary[summary["condition"] == C0].set_index("task")
    reference = frozen[
        frozen["condition"] == "d2_shared_adapter_dataset_balanced"
    ].set_index("task")
    rows = []
    for task in local.index:
        for metric in ("macro_f1_mean", "uar_mean", "native_score_mean"):
            if pd.isna(local.loc[task, metric]) or pd.isna(
                reference.loc[task, metric]
            ):
                continue
            delta = float(local.loc[task, metric] - reference.loc[task, metric])
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "local_c0": float(local.loc[task, metric]),
                    "frozen_d2": float(reference.loc[task, metric]),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "within_0_02": abs(delta) <= 0.02,
                }
            )
    output = pd.DataFrame(rows)
    return output, float(output["abs_delta"].max())


def summary_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index(["condition", "task"])
    comparisons = [
        ("c1_vs_c0", C1, C0),
        ("m1_vs_c1_parameter_matched", M1, C1),
        ("m1_vs_c0", M1, C0),
        ("m2_vs_m1_tail_objective", M2, M1),
        ("m2_vs_c0", M2, C0),
    ]
    rows = []
    metrics = [
        "macro_f1_mean",
        "weighted_or_micro_f1_mean",
        "uar_mean",
        "native_score_mean",
    ]
    for comparison, candidate, reference in comparisons:
        tasks = sorted(
            set(indexed.loc[candidate].index)
            & set(indexed.loc[reference].index)
        )
        for task in tasks:
            row = {
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
                "task": task,
            }
            for metric in metrics:
                candidate_value = indexed.loc[(candidate, task), metric]
                reference_value = indexed.loc[(reference, task), metric]
                row[f"candidate_{metric}"] = candidate_value
                row[f"reference_{metric}"] = reference_value
                row[f"delta_{metric}"] = (
                    candidate_value - reference_value
                    if pd.notna(candidate_value) and pd.notna(reference_value)
                    else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def fold_deltas(folds: pd.DataFrame) -> pd.DataFrame:
    indexed = folds.set_index(["condition", "fold", "task"])
    rows = []
    comparisons = [
        ("m1_vs_c1_parameter_matched", M1, C1),
        ("m1_vs_c0", M1, C0),
        ("m2_vs_m1_tail_objective", M2, M1),
    ]
    for comparison, candidate, reference in comparisons:
        for fold in range(5):
            for task in sorted(folds["task"].unique()):
                candidate_row = indexed.loc[(candidate, fold, task)]
                reference_row = indexed.loc[(reference, fold, task)]
                rows.append(
                    {
                        "comparison": comparison,
                        "fold": fold,
                        "task": task,
                        "delta_macro_f1": (
                            float(candidate_row["macro_f1"])
                            - float(reference_row["macro_f1"])
                        ),
                        "delta_uar": (
                            float(candidate_row["uar"])
                            - float(reference_row["uar"])
                            if pd.notna(candidate_row["uar"])
                            else None
                        ),
                        "delta_native_score": (
                            float(candidate_row["native_score"])
                            - float(reference_row["native_score"])
                            if pd.notna(candidate_row["native_score"])
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def class_deltas(per_class: pd.DataFrame) -> pd.DataFrame:
    indexed = per_class.set_index(["condition", "task", "label"])
    rows = []
    for comparison, candidate, reference in (
        ("m1_vs_c1_parameter_matched", M1, C1),
        ("m1_vs_c0", M1, C0),
        ("m2_vs_m1_tail_objective", M2, M1),
    ):
        keys = sorted(
            set(indexed.loc[candidate].index)
            & set(indexed.loc[reference].index)
        )
        for task, label in keys:
            candidate_row = indexed.loc[(candidate, task, label)]
            reference_row = indexed.loc[(reference, task, label)]
            rows.append(
                {
                    "comparison": comparison,
                    "task": task,
                    "label": label,
                    "support": int(candidate_row["support"]),
                    "candidate_recall": float(candidate_row["recall_mean"]),
                    "reference_recall": float(reference_row["recall_mean"]),
                    "delta_recall": float(
                        candidate_row["recall_mean"]
                        - reference_row["recall_mean"]
                    ),
                    "candidate_f1": float(candidate_row["f1_mean"]),
                    "reference_f1": float(reference_row["f1_mean"]),
                    "delta_f1": float(
                        candidate_row["f1_mean"]
                        - reference_row["f1_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    verification = json.loads((RESULT / "full_verification.json").read_text())
    if verification["status"] != "four_dataset_general_specific_full_verified":
        raise RuntimeError("full verification must pass before analysis")
    summary = pd.read_csv(RESULT / "summary.csv")
    frozen = pd.read_csv(FROZEN / "summary.csv")
    folds = pd.read_csv(RESULT / "task_fold_results.csv")
    per_class = pd.read_csv(RESULT / "per_class_summary.csv")

    sanity, max_delta = external_sanity(summary, frozen)
    sanity.to_csv(RESULT / "external_d2_sanity.csv", index=False)
    deltas = summary_deltas(summary)
    deltas.to_csv(RESULT / "task_summary_deltas.csv", index=False)
    fold_delta = fold_deltas(folds)
    fold_delta.to_csv(RESULT / "paired_fold_deltas.csv", index=False)
    class_delta = class_deltas(per_class)
    class_delta.to_csv(RESULT / "per_class_deltas.csv", index=False)

    m1 = deltas[deltas["comparison"] == "m1_vs_c1_parameter_matched"]
    improved_tasks = sorted(
        m1.loc[
            (m1["delta_macro_f1_mean"] >= 0.01)
            | (m1["delta_uar_mean"] >= 0.01),
            "task",
        ].tolist()
    )
    severe_regressions = sorted(
        m1.loc[
            (m1["delta_macro_f1_mean"] < -0.02)
            | (m1["delta_native_score_mean"] < -0.02),
            "task",
        ].tolist()
    )
    spr_binary_folds = fold_delta[
        (fold_delta["comparison"] == "m1_vs_c1_parameter_matched")
        & (fold_delta["task"] == "spr_binary")
    ]
    m2 = deltas[deltas["comparison"] == "m2_vs_m1_tail_objective"].set_index(
        "task"
    )
    hf_micro_drop = float(
        m2.loc[
            "hf_adventitious_presence",
            "delta_weighted_or_micro_f1_mean",
        ]
    )
    kauh_weighted_drop = float(
        m2.loc["kauh_raw9", "delta_weighted_or_micro_f1_mean"]
    )
    receipt = {
        "status": "four_dataset_general_specific_decision_complete",
        "external_d2_sanity": {
            "threshold": 0.02,
            "max_abs_delta": max_delta,
            "passed": max_delta <= 0.02,
        },
        "m1_structure_gate": {
            "comparison": "M1 task-specific rank16 x6 vs C1 shared rank96; exact parameter match",
            "tasks_with_macro_f1_or_uar_gain_ge_0_01": improved_tasks,
            "tasks_with_macro_f1_or_native_regression_gt_0_02": severe_regressions,
            "spr_binary_macro_f1_fold_wins": int(
                (spr_binary_folds["delta_macro_f1"] > 0).sum()
            ),
            "formal_gate_passed": (
                len(improved_tasks) >= 2
                and not severe_regressions
                and int((spr_binary_folds["delta_macro_f1"] > 0).sum()) > 1
            ),
            "claim": (
                "partial task-specialization signal, not universal dominance; "
                "ICBHI both and HF phase E recall regress"
            ),
        },
        "m2_tail_gate": {
            "hf_adventitious_macro_f1_delta": float(
                m2.loc["hf_adventitious_presence", "delta_macro_f1_mean"]
            ),
            "hf_adventitious_micro_f1_delta": hf_micro_drop,
            "hf_micro_drop_within_0_03": hf_micro_drop >= -0.03,
            "kauh_macro_f1_delta": float(
                m2.loc["kauh_raw9", "delta_macro_f1_mean"]
            ),
            "kauh_uar_delta": float(
                m2.loc["kauh_raw9", "delta_uar_mean"]
            ),
            "kauh_weighted_f1_delta": kauh_weighted_drop,
            "kauh_weighted_drop_within_0_03": kauh_weighted_drop >= -0.03,
            "formal_gate_passed": (
                hf_micro_drop >= -0.03 and kauh_weighted_drop >= -0.03
            ),
            "claim": (
                "tail coverage tradeoff; both eligible heads exceed the "
                "preregistered weighted/micro regression allowance"
            ),
        },
        "router": {
            "evaluated": False,
            "reason": "preregistered out of the bounded first matrix",
            "shortcut_claim_allowed": False,
        },
        "full_encoder_upgrade": False,
        "next_action": (
            "stop this branch after reporting; do not add router, seeds, "
            "loss weights, or full-encoder training without a new preregistration"
        ),
    }
    write_json(RESULT / "decision_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
