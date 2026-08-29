"""Read-only checkpoint-selection sensitivity for repaired BEATs Core-2 logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


RUN_DIRECTORIES = {
    "R0": "R0_server_4s2s_truebatch8_repaired_seed42",
    "N1": "N1_server_4s2s_truebatch8_repaired_seed42",
    "A1": "A1_server_4s2s_truebatch8_repaired_seed42",
    "L1": "L1_server_4s2s_truebatch8_repaired_seed42",
    "L2": "L2_server_4s2s_truebatch8_repaired_seed42",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _minimum(rows: Sequence[Mapping[str, object]], field: str) -> Mapping[str, object]:
    return min(rows, key=lambda row: (float(row[field]), int(row["epoch"])))


def analyze_run(run_id: str, run_dir: Path) -> dict[str, object]:
    curve = _read_jsonl(run_dir / "train_log.jsonl")
    epoch_one = next(row for row in curve if int(row["epoch"]) == 1)
    base_losses = epoch_one["validation"]["dataset_losses"]
    derived = []
    for row in curve:
        dataset_losses = row["validation"]["dataset_losses"]
        icbhi_ratio = float(dataset_losses["icbhi"]) / float(base_losses["icbhi"])
        spr_ratio = float(dataset_losses["sprsound"]) / float(base_losses["sprsound"])
        derived.append(
            {
                "epoch": int(row["epoch"]),
                "current_equal_raw_loss": float(row["validation"]["selection_loss"]),
                "e1_normalized_mean": (icbhi_ratio + spr_ratio) / 2.0,
                "e1_normalized_worst": max(icbhi_ratio, spr_ratio),
                "icbhi_loss": float(dataset_losses["icbhi"]),
                "sprsound_loss": float(dataset_losses["sprsound"]),
            }
        )
    current = _minimum(derived, "current_equal_raw_loss")
    normalized_mean = _minimum(derived, "e1_normalized_mean")
    normalized_worst = _minimum(derived, "e1_normalized_worst")
    alternative_epochs = sorted(
        {
            int(normalized_mean["epoch"]),
            int(normalized_worst["epoch"]),
        }
        - {int(current["epoch"])}
    )
    available_epoch_checkpoints = {
        int(path.stem.split("_")[-1])
        for path in run_dir.rglob("epoch_*.pt")
        if path.stem.split("_")[-1].isdigit()
    }
    missing_alternatives = [
        epoch for epoch in alternative_epochs if epoch not in available_epoch_checkpoints
    ]
    if not alternative_epochs:
        alternative_status = "not_applicable_same_epoch"
    elif missing_alternatives:
        alternative_status = "HOLD_missing_epoch_checkpoint"
    else:
        alternative_status = "available_for_future_approved_audit"
    return {
        "run_id": run_id,
        "epochs_logged": len(curve),
        "current_equal_raw_loss": current,
        "e1_normalized_mean": normalized_mean,
        "e1_normalized_worst": normalized_worst,
        "existing_result_policy": "retain current selected epoch; no posthoc reselection",
        "alternative_checkpoint_status": alternative_status,
        "missing_alternative_epochs": missing_alternatives,
    }


def native_composite_design() -> dict[str, object]:
    return {
        "status": "prospective_preregistration_only",
        "test_role": "never used for checkpoint or threshold selection",
        "validation_split": {
            "calibration": "group-safe calibration groups within validation",
            "selection": "disjoint group-safe selection groups within validation",
        },
        "calibration": {
            "level1": "softmax argmax; no threshold",
            "crackle": "one core-shared threshold maximizing equal ICBHI/SPRSound calibration F1",
            "wheeze": "one core-shared threshold maximizing equal ICBHI/SPRSound calibration F1",
            "tie": "higher threshold",
        },
        "selection": {
            "icbhi": "official flat4 Sp/Se/Score from calibrated hierarchical decoder",
            "sprsound": "official inter Task1-1 Normal/Adventitious Score from Level1 argmax",
            "composite": "0.5 * ICBHI_validation_Score + 0.5 * SPRSound_validation_Task1_1_Score",
            "tie": "earliest epoch",
        },
        "after_selection": (
            "freeze epoch; refit the same shared thresholds on full validation, then access test once"
        ),
        "current_five_runs": "not eligible for posthoc native-composite reselection",
    }


def analyze_root(logs_root: Path) -> dict[str, object]:
    runs = [
        analyze_run(run_id, logs_root / directory)
        for run_id, directory in RUN_DIRECTORIES.items()
    ]
    return {
        "status": "validation_log_only_design_sensitivity",
        "test_accessed": False,
        "current_criterion": (
            "equal mean of ICBHI and SPRSound validation dataset losses; each dataset loss is "
            "the equal mean of eligible Level1, Crackle, and Wheeze plain CE/BCE"
        ),
        "normalization_note": (
            "epoch-1 normalization is a sensitivity view, not an approved replacement criterion"
        ),
        "runs": runs,
        "prospective_native_composite": native_composite_design(),
    }


def render_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# BEATs Core-2 checkpoint-selection sensitivity",
        "",
        "Status: validation-log-only design sensitivity. No test data are read, and the existing five runs are not reselected.",
        "",
        "| Run | Current raw-loss epoch | e1-normalized mean epoch | normalized-worst epoch | Alternative checkpoint |",
        "|---|---:|---:|---:|---|",
    ]
    for run in analysis["runs"]:
        lines.append(
            "| {run_id} | {current} | {mean} | {worst} | {status} |".format(
                run_id=run["run_id"],
                current=run["current_equal_raw_loss"]["epoch"],
                mean=run["e1_normalized_mean"]["epoch"],
                worst=run["e1_normalized_worst"]["epoch"],
                status=run["alternative_checkpoint_status"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `current raw loss` is the implemented equal-dataset mean of plain eligible-node validation CE/BCE.",
            "- `e1-normalized mean` divides each dataset curve by its own epoch-1 loss before averaging.",
            "- `normalized worst` minimizes the worse of the two epoch-1-relative dataset losses.",
            "- The normalized views are descriptive sensitivity analyses. They do not replace the selected checkpoint of R0/N1/A1/L1/L2.",
            "- When an alternative epoch checkpoint is absent, deployment or terminal scoring is `HOLD`; the train log alone is not a model artifact.",
            "",
            "## Prospective native composite",
            "",
            "Use disjoint patient/group-safe calibration and selection subsets inside validation. Fit one shared Crackle and one shared Wheeze threshold on calibration only. On selection, compute ICBHI flat4 Score and SPRSound inter Task1-1 Score, then select the epoch by their equal mean; ties choose the earliest epoch. Test remains terminal-only.",
            "",
            "This composite must be preregistered before reading per-epoch native metrics. It is not applied retrospectively to the five repaired runs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("result/reproduce/beats_nal_ablation"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    analysis = analyze_root(args.logs_root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(analysis))
    if not args.json_output and not args.markdown_output:
        print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
