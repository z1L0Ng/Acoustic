"""Close a completed HF-on run from saved terminal artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _hf_off_metrics_path(repo_root: Path, seed: int) -> Path:
    if seed == 42:
        return repo_root / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2/hf_metrics.json"
    return repo_root / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH" / f"seed_{seed}" / "hf_metrics.json"


def _comparison(hf_off: dict[str, object], hf_on: dict[str, object]) -> dict[str, object]:
    off = hf_off["recording_presence_pool"]["metrics"]
    on = hf_on["recording_presence_pool"]["metrics"]
    result = {}
    for node, hf_key, interval_key in (
        ("crackle", "D", "D"),
        ("wheeze", "Wheeze", "Wheeze"),
    ):
        result[node] = {
            "hf_off_auroc": off[hf_key]["curve"]["auroc"],
            "hf_on_auroc": on[hf_key]["curve"]["auroc"],
            "delta_auroc": on[hf_key]["curve"]["auroc"] - off[hf_key]["curve"]["auroc"],
            "hf_off_auprc": off[hf_key]["curve"]["auprc"],
            "hf_on_auprc": on[hf_key]["curve"]["auprc"],
            "delta_auprc": on[hf_key]["curve"]["auprc"] - off[hf_key]["curve"]["auprc"],
            "hf_off_interval_recall": hf_off["positive_interval_coverage"][interval_key]["positive_recall"],
            "hf_on_interval_recall": hf_on["positive_interval_coverage"][interval_key]["positive_recall"],
        }
    return result


def finalize(run_dir: Path, repo_root: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    validation_path = run_dir / "validation_selection.json"
    terminal_path = run_dir / "terminal/native_metrics.json"
    hf_path = run_dir / "terminal/hf_metrics.json"
    kauh_path = run_dir / "terminal/kauh_metrics.json"
    required = (config_path, validation_path, terminal_path, hf_path, kauh_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing completed terminal artifacts: " + ", ".join(missing))
    existing_summary = run_dir / "run_summary.json"
    if existing_summary.is_file():
        existing = json.loads(existing_summary.read_text())
        if existing.get("status") in {
            "early_stopped_epochwise_test_selected",
            "complete_epochwise_test_selected",
        }:
            return {"status": "already_complete", "run_summary": str(existing_summary)}

    config = json.loads(config_path.read_text())
    selection = json.loads(validation_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    hf_metrics = json.loads(hf_path.read_text())
    kauh_metrics = json.loads(kauh_path.read_text())
    seed = int(config["seed"])
    condition = str(config["condition"])
    selection["condition"] = condition
    selection["evidence_label"] = config["evidence_label"]
    selection["base_condition"] = config["base_condition"]
    validation_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    terminal["condition"] = condition
    terminal["evidence_label"] = config["evidence_label"]
    terminal["base_condition"] = config["base_condition"]
    terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    hf_off_path = _hf_off_metrics_path(repo_root, seed)
    hf_off_metrics = json.loads(hf_off_path.read_text())
    hf_comparison = _comparison(hf_off_metrics, hf_metrics)
    selected_epoch = int(selection["selected_epoch"])
    completed_epochs = int(terminal["completed_training_epochs"])
    selected_icbhi = terminal["icbhi_flat4"]
    spr_metrics = terminal["sprsound_inter_task1_1"]
    summary = {
        "status": "early_stopped_epochwise_test_selected" if terminal.get("early_stopped") else "complete_epochwise_test_selected",
        "evidence_label": config["evidence_label"],
        "condition": condition,
        "seed": seed,
        "base_condition": config["base_condition"],
        "training_config_reused_from": config.get("training_config_reused_from"),
        "completed_training_epochs": completed_epochs,
        "selected_epoch": selected_epoch,
        "selection_loss": float(selection["selection_loss"]),
        "icbhi_official_score": float(selected_icbhi["official_score"]),
        "sprsound_task1_1_official_score": float(spr_metrics["official_score"]),
        "outer_test_accessed": True,
        "test_access_counts": {
            "icbhi_official_test": int(terminal.get("icbhi_official_test_accesses", completed_epochs)),
            "sprsound_official_inter_test": 1,
            "hf_source_test": 1,
            "kauh_external_test": 1,
        },
        "training": True,
        "finalization_only": True,
        "new_model_inference": False,
        "new_test_target_access": False,
        "source_artifacts": {
            "config": str(config_path),
            "validation_selection": str(validation_path),
            "terminal_metrics": str(terminal_path),
            "hf_metrics": str(hf_path),
            "kauh_metrics": str(kauh_path),
            "hf_off_reference": str(hf_off_path),
        },
        "selected_metrics": {
            "icbhi_flat4": selected_icbhi,
            "sprsound_inter_task1_1": spr_metrics,
            "hf_source_test": hf_metrics,
            "kauh_external": kauh_metrics,
        },
        "hf_on_off_comparison": hf_comparison,
        "checkpoint_selection": config["checkpoint_selection"],
        "threshold_source": selection["threshold_source"],
        "claim_boundary": config["evidence_label"],
    }
    (run_dir / "hf_on_off_comparison.json").write_text(
        json.dumps(
            {
                "status": "matched_fixed_HF_off_reference_comparison",
                "evidence_label": config["evidence_label"],
                "hf_off_reference": str(hf_off_path),
                "hf_on_run": str(run_dir),
                "selected_epoch": selected_epoch,
                "seed": seed,
                "metrics": hf_comparison,
                "finalization_only": True,
                "new_model_inference": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    existing_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"status": "finalized", "run_summary": str(existing_summary), "seed": seed, "selected_epoch": selected_epoch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir, args.repo_root), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
