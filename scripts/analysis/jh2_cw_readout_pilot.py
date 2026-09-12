"""CPU post-hoc C/W-only readout pilot for the selected JH2 ICBHI runs.

The pilot reuses each run's saved ICBHI probabilities and validation-fitted
thresholds.  It rebuilds the saved Hard Hierarchy readout first, requires that
it matches the saved selected metric artifact, and then compares the
thresholded Crackle/Wheeze-only flat4 readout.  It does not load a model or
access audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from baseline.multidataset_pipeline.beats_nal_protocol import (
    decode_icbhi_flat4,
    decode_icbhi_hierarchical_flat4,
)
from baseline.multidataset_pipeline.posthoc_native_readout import (
    ICBHI_LABELS,
    native_metrics,
)


SEEDS = (0, 1, 42)
MAIN_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed")
OUTPUT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_CW_READOUT_PILOT"
)
EVIDENCE_LABEL = "posthoc_test_selected_cw_readout_pilot"
PILOT_PURPOSE = "C/W-only ICBHI readout pilot; excluded from formal Table 2"


def _label(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _metric_payload(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    metrics = native_metrics(target, prediction, ICBHI_LABELS)
    metrics.update(
        {
            "task": "ICBHI official-test flat4 readout pilot",
            "official_score": metrics["average_score"],
            "both_recall": metrics["per_class"]["both"]["recall"],
            "predicted_class_counts": np.asarray(
                metrics["confusion"], dtype=np.int64
            ).sum(axis=0).tolist(),
        }
    )
    return metrics


def _selected_inputs(repo_root: Path, seed: int) -> dict[str, object]:
    run_dir = repo_root / MAIN_RELATIVE / f"seed_{seed}"
    summary_path = run_dir / "run_summary.json"
    selection_path = run_dir / "validation_selection.json"
    prediction_path = run_dir / "terminal/selected_icbhi_test_predictions_scored.npz"
    saved_metric_path = run_dir / "terminal/selected_icbhi_native_metrics.json"
    summary = json.loads(summary_path.read_text())
    selection = json.loads(selection_path.read_text())
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    selected_epoch = int(summary["selected_epoch"])
    if selected_epoch != int(selection["selected_epoch"]):
        raise RuntimeError(f"selected epoch mismatch for seed {seed}")
    return {
        "seed": seed,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "selection_path": selection_path,
        "prediction_path": prediction_path,
        "saved_metric_path": saved_metric_path,
        "selected_epoch": selected_epoch,
        "thresholds": thresholds,
    }


def _saved_full_match(
    rebuilt: Mapping[str, object],
    saved: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "confusion",
        "predicted_class_counts",
        "specificity",
        "sensitivity",
        "average_score",
        "macro_f1",
        "uar",
    )
    differences: dict[str, object] = {}
    for field in fields:
        actual = rebuilt[field]
        expected = saved[field]
        if isinstance(actual, list):
            if actual != expected:
                differences[field] = {"rebuilt": actual, "saved": expected}
        elif not np.isclose(float(actual), float(expected), rtol=1e-7, atol=1e-9):
            differences[field] = {"rebuilt": float(actual), "saved": float(expected)}
    return {
        "match": not differences,
        "checked_fields": list(fields),
        "differences": differences,
    }


def _readout_for_seed(repo_root: Path, seed: int) -> dict[str, object]:
    selected = _selected_inputs(repo_root, seed)
    with np.load(selected["prediction_path"], allow_pickle=False) as predictions:
        target = np.asarray(
            [ICBHI_LABELS.index(_label(value)) for value in predictions["raw_ground_truth"]],
            dtype=np.int64,
        )
        probabilities = np.asarray(predictions["attribute_probabilities"], dtype=np.float64)
        level1 = np.asarray(predictions["level1_predictions"], dtype=np.int64)
        full_prediction = decode_icbhi_hierarchical_flat4(
            level1,
            probabilities,
            selected["thresholds"],
        )
        cw_prediction = decode_icbhi_flat4(probabilities, selected["thresholds"])

    saved_payload = json.loads(selected["saved_metric_path"].read_text())
    saved_full = saved_payload["icbhi_flat4"]
    full_metrics = _metric_payload(target, full_prediction)
    full_match = _saved_full_match(full_metrics, saved_full)
    if not full_match["match"]:
        raise RuntimeError(
            f"rebuilt Full readout differs from saved seed {seed} metric: "
            f"{json.dumps(full_match['differences'], sort_keys=True)}"
        )
    cw_metrics = _metric_payload(target, cw_prediction)
    delta_fields = (
        "specificity",
        "sensitivity",
        "official_score",
        "macro_f1",
        "uar",
        "both_recall",
    )
    paired_delta = {
        field: float(cw_metrics[field]) - float(full_metrics[field])
        for field in delta_fields
    }
    return {
        "seed": seed,
        "source_run": str(selected["run_dir"]),
        "source_prediction_npz": str(selected["prediction_path"]),
        "source_validation_selection": str(selected["selection_path"]),
        "source_saved_full_metrics": str(selected["saved_metric_path"]),
        "selected_epoch": selected["selected_epoch"],
        "shared_attribute_thresholds": selected["thresholds"],
        "full_rebuild_match": full_match,
        "full_hard_hierarchy": full_metrics,
        "cw_only": cw_metrics,
        "paired_delta_cw_minus_full": paired_delta,
    }


def _existing_pilot_for_seed(repo_root: Path, seed: int) -> dict[str, object]:
    run_dir = repo_root / MAIN_RELATIVE / f"seed_{seed}"
    metric_path = run_dir / "terminal/selected_icbhi_native_metrics.json"
    selection_path = run_dir / "validation_selection.json"
    summary_path = run_dir / "run_summary.json"
    saved = json.loads(metric_path.read_text())
    selection = json.loads(selection_path.read_text())
    summary = json.loads(summary_path.read_text())
    full = saved["icbhi_flat4"]
    cw_only = saved["icbhi_flat4_bits_only_ablation"]
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    if full["rows"] != cw_only["rows"] or full["thresholds"] != cw_only["thresholds"]:
        raise RuntimeError(f"saved Full/CW-only metric contract mismatch for seed {seed}")
    delta_fields = (
        "specificity",
        "sensitivity",
        "official_score",
        "macro_f1",
        "uar",
        "both_recall",
    )
    full_with_both = {
        **full,
        "both_recall": full["per_class"]["both"]["recall"],
    }
    cw_with_both = {
        **cw_only,
        "both_recall": cw_only["per_class"]["both"]["recall"],
    }
    return {
        "seed": seed,
        "source_run": str(run_dir),
        "source_prediction_npz": str(
            run_dir / "terminal/selected_icbhi_test_predictions_scored.npz"
        ),
        "source_validation_selection": str(selection_path),
        "source_saved_full_metrics": str(metric_path),
        "selected_epoch": int(summary["selected_epoch"]),
        "shared_attribute_thresholds": thresholds,
        "full_rebuild_match": {
            "status": "existing_saved_metric_aggregate; no probability rebuild",
            "match": None,
            "saved_metric_semantics": "selected_icbhi_native_metrics.json.icbhi_flat4",
        },
        "full_hard_hierarchy": full_with_both,
        "cw_only": cw_with_both,
        "paired_delta_cw_minus_full": {
            field: float(cw_with_both[field]) - float(full_with_both[field])
            for field in delta_fields
        },
    }


def _aggregate_readout_stats(
    results: list[Mapping[str, object]],
    readout_key: str,
) -> dict[str, object]:
    fields = (
        "specificity",
        "sensitivity",
        "official_score",
        "macro_f1",
        "uar",
        "both_recall",
    )
    return {
        field: _stat([float(result[readout_key][field]) for result in results])
        for field in fields
    }


def _stat(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "ddof": 1,
        "values": [float(value) for value in array],
    }


def _format(value: Mapping[str, object]) -> str:
    return f"{float(value['mean']):.6f} ± {float(value['sample_std']):.6f}"


def aggregate_existing(repo_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite pilot output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [_existing_pilot_for_seed(repo_root, seed) for seed in SEEDS]
    summary = {
        "status": "existing_saved_cw_pilot_aggregate",
        "evidence_label": EVIDENCE_LABEL,
        "purpose": PILOT_PURPOSE,
        "formal_table2_included": False,
        "new_model_inference": False,
        "new_audio_access": False,
        "new_probability_rebuild": False,
        "threshold_refit": False,
        "checkpoint_reselection": False,
        "seeds": list(SEEDS),
        "per_seed": results,
        "aggregate": {
            "full_hard_hierarchy": _aggregate_readout_stats(
                results, "full_hard_hierarchy"
            ),
            "cw_only": _aggregate_readout_stats(results, "cw_only"),
            "paired_delta_cw_minus_full": _aggregate_readout_stats(
                results, "paired_delta_cw_minus_full"
            ),
        },
        "source_metric_artifacts": [result["source_saved_full_metrics"] for result in results],
        "claim_boundary": (
            "post-hoc ICBHI test-selected readout diagnostic aggregated from existing "
            "saved metrics; not a clean estimate and not a formal Table 2 result"
        ),
        "outputs": {
            "summary_json": str(output_dir / "cw_readout_existing_pilot.json"),
            "summary_markdown": str(output_dir / "cw_readout_existing_pilot.md"),
        },
    }
    (output_dir / "cw_readout_existing_pilot.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# JH2 existing C/W-only readout pilot aggregate",
        "",
        f"Evidence: `{EVIDENCE_LABEL}`.",
        f"{PILOT_PURPOSE}. This aggregate reads existing selected metric JSON only; it does not rebuild probabilities.",
        "",
        "| Readout | Sp | Se | Score | Macro-F1 | UAR | Both recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("full_hard_hierarchy", "Full Hard Hierarchy"),
        ("cw_only", "C/W-only"),
        ("paired_delta_cw_minus_full", "C/W-only minus Full"),
    ):
        row = summary["aggregate"][key]
        lines.append(
            f"| {label} | {_format(row['specificity'])} | {_format(row['sensitivity'])} | "
            f"{_format(row['official_score'])} | {_format(row['macro_f1'])} | "
            f"{_format(row['uar'])} | {_format(row['both_recall'])} |"
        )
    lines.extend(
        [
            "",
            "Per-seed selected epochs: seed0=15, seed1=11, seed42=17.",
            "Full/CW-only rows were checked for equal row counts and thresholds before aggregation.",
            "The pilot is excluded from formal Table 2 and primary claims.",
        ]
    )
    (output_dir / "cw_readout_existing_pilot.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def _markdown(results: list[Mapping[str, object]]) -> str:
    lines = [
        "# JH2 ICBHI C/W-only readout pilot",
        "",
        f"Evidence: `{EVIDENCE_LABEL}`.",
        f"{PILOT_PURPOSE}. No model inference is used; probabilities and thresholds are copied from the selected runs.",
        "",
        "| Seed | Selected epoch | Full Score | C/W-only Score | Delta | Full Sp | C/W-only Sp | Full Se | C/W-only Se | Full Both recall | C/W-only Both recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        full = result["full_hard_hierarchy"]
        cw = result["cw_only"]
        delta = result["paired_delta_cw_minus_full"]
        lines.append(
            f"| {result['seed']} | {result['selected_epoch']} | "
            f"{float(full['official_score']):.6f} | {float(cw['official_score']):.6f} | "
            f"{float(delta['official_score']):+.6f} | "
            f"{float(full['specificity']):.6f} | {float(cw['specificity']):.6f} | "
            f"{float(full['sensitivity']):.6f} | {float(cw['sensitivity']):.6f} | "
            f"{float(full['both_recall']):.6f} | {float(cw['both_recall']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Full is rebuilt with the canonical Level1-gated Hard Hierarchy decoder. C/W-only ignores Level1 and maps thresholded Crackle/Wheeze bits to Normal/Crackle/Wheeze/Both.",
            "",
            "The pilot is test-selected diagnostic evidence and is excluded from formal Table 2 and primary claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(repo_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite pilot output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        result = _readout_for_seed(repo_root, seed)
        seed_path = output_dir / f"seed_{seed}.json"
        seed_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    summary = {
        "status": "cw_readout_pilot_complete",
        "evidence_label": EVIDENCE_LABEL,
        "purpose": PILOT_PURPOSE,
        "formal_table2_included": False,
        "new_model_inference": False,
        "new_audio_access": False,
        "threshold_refit": False,
        "checkpoint_reselection": False,
        "seeds": list(SEEDS),
        "per_seed": results,
        "outputs": {
            "summary_json": str(output_dir / "cw_readout_pilot.json"),
            "summary_markdown": str(output_dir / "cw_readout_pilot.md"),
        },
        "claim_boundary": (
            "post-hoc ICBHI test-selected readout diagnostic; not a clean estimate "
            "and not a formal Table 2 result"
        ),
    }
    (output_dir / "cw_readout_pilot.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cw_readout_pilot.md").write_text(
        _markdown(results), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / OUTPUT_RELATIVE
    if args.aggregate_existing:
        print(
            json.dumps(aggregate_existing(args.repo_root, output_dir), indent=2, sort_keys=True)
        )
        return
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "evidence_label": EVIDENCE_LABEL,
                    "purpose": PILOT_PURPOSE,
                    "output_dir": str(output_dir),
                    "seeds": list(SEEDS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(args.repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
