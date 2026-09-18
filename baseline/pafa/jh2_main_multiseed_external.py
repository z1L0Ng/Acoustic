"""Post-hoc HF Lung and KAUH evaluation for the selected JH2 main seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from baseline.pafa import jh2_hf_kauh_external as external
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _build_components,
    _prepare_waveforms,
    _save_predictions,
    _write_json,
)


SEEDS = (0, 1, 42)
MAIN_RELATIVE = Path("result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed")
OUTPUT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH"
)
EVIDENCE_LABEL = "posthoc_fixed_JH2_main_selected_checkpoints_HF_KAUH_external_diagnostic"


def _load_selected(
    repo_root: Path, seed: int, main_relative: Path = MAIN_RELATIVE
) -> dict[str, object]:
    run_dir = repo_root / main_relative / f"seed_{seed}"
    config = json.loads((run_dir / "config.json").read_text())
    selection = json.loads((run_dir / "validation_selection.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    if int(summary["selected_epoch"]) != int(selection["selected_epoch"]):
        raise RuntimeError(f"selected epoch mismatch for seed {seed}")
    return {
        "seed": seed,
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "backbone_checkpoint": Path(str(config["checkpoint"])),
        "selected_epoch": int(summary["selected_epoch"]),
        "thresholds": thresholds,
    }


def _config(
    repo_root: Path,
    selected: Mapping[str, object],
    output_dir: Path,
    device: str = "mps",
) -> PAFAJointHierarchyConfig:
    return PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=repo_root
        / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        checkpoint=selected["backbone_checkpoint"],
        icbhi_audio_dir=repo_root
        / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
        output_dir=output_dir,
        device=device,
        cpu_threads=4,
    )


def _config_payload(
    repo_root: Path,
    selected: Mapping[str, object],
    output_dir: Path,
    device: str = "mps",
) -> dict[str, object]:
    return {
        "status": "fixed_selected_checkpoint_external_transfer_inference",
        "evidence_label": EVIDENCE_LABEL,
        "repo_root": str(repo_root),
        "source_run": str(selected["run_dir"]),
        "output_dir": str(output_dir),
        "checkpoint": str(selected["checkpoint"]),
        "backbone_checkpoint_for_model_construction": str(
            selected["backbone_checkpoint"]
        ),
        "seed": int(selected["seed"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "frozen_shared_thresholds": dict(selected["thresholds"]),
        "threshold_source": "same selected run validation_selection.json; no HF or KAUH tuning",
        "model": "BEATs iter3+ AS2M full fine-tuned PAFA-JH2 hierarchy",
        "device": device,
        "precision": "FP32",
        "input": "mono 16 kHz; 5 s JH2 waveform geometry",
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "hf_policy": {
            "source_split": "test only",
            "windows": "exactly [0,5], [5,10], [10,15] seconds",
            "compatible_attributes": {"D": "Crackle", "Wheeze": "Wheeze"},
            "gap_unknown_excluded": True,
            "unsupported": ["Rhonchi", "Stridor", "I", "E", "CAS", "DAS"],
        },
        "kauh_policy": {
            "recordings": 336,
            "patients": 112,
            "filters": ["B", "D", "E"],
            "window_to_recording_to_patient": (
                "recording readout, then B/D/E probability mean at patient level"
            ),
            "compatible_overlay": {
                "N": "Normal",
                "E W": "Wheeze",
                "I E W": "Wheeze",
                "C": "Crackle",
                "I C": "Crackle",
                "I C E W": "Both",
            },
            "unresolved": ["Crep", "Bronchial", "I C B"],
        },
        "claim_boundary": (
            "post-hoc external transfer diagnostic; not HF/KAUH native reproduction, "
            "not checkpoint reselection, and not a paper claim"
        ),
    }


def _augment_metrics(
    metrics: Mapping[str, object],
    selected: Mapping[str, object],
    output_dir: Path,
    kind: str,
) -> dict[str, object]:
    return {
        **metrics,
        "evidence_label": EVIDENCE_LABEL,
        "source_run": str(selected["run_dir"]),
        "checkpoint_source": str(selected["checkpoint"]),
        "seed": int(selected["seed"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "shared_attribute_thresholds": dict(selected["thresholds"]),
        "threshold_source": "selected run validation thresholds; no external tuning",
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "external_evaluation_kind": kind,
        "posthoc_output_dir": str(output_dir),
    }


def run_seed(
    repo_root: Path,
    seed: int,
    output_dir: Path,
    *,
    main_relative: Path = MAIN_RELATIVE,
    device_name: str = "mps",
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite external output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_selected(repo_root, seed, main_relative)
    config = _config(repo_root, selected, output_dir, device_name)
    _write_json(
        output_dir / "config.json",
        _config_payload(repo_root, selected, output_dir, device_name),
    )

    kauh_samples = external._load_kauh_samples(repo_root)
    hf_samples = external._load_hf_test_samples(repo_root)
    device = torch.device(device_name)
    model, _ = _build_components(config, device)
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    thresholds = selected["thresholds"]

    kauh_waveforms = _prepare_waveforms(kauh_samples, config)
    kauh_label_free = external._predict_kauh(
        model, kauh_samples, kauh_waveforms, config, device, thresholds
    )
    kauh_label_free_path = output_dir / "kauh_predictions_label_free.npz"
    _save_predictions(kauh_label_free_path, kauh_label_free)

    hf_label_free = external._predict_hf(model, hf_samples, config, device)
    hf_label_free_path = output_dir / "hf_predictions_label_free.npz"
    _save_predictions(hf_label_free_path, hf_label_free)

    hf_annotations = external._parse_hf_annotations(hf_samples)
    hf_scored, hf_metrics = external._hf_scored_predictions(
        hf_label_free, hf_annotations, thresholds
    )
    hf_scored_path = output_dir / "hf_predictions_scored.npz"
    _save_predictions(hf_scored_path, hf_scored)

    kauh_scored, _ = external._kauh_scored_predictions(
        kauh_label_free, kauh_samples, thresholds
    )
    kauh_metrics, patient_scored = external._kauh_metrics(kauh_scored, thresholds)
    kauh_scored_path = output_dir / "kauh_predictions_scored.npz"
    kauh_patient_path = output_dir / "kauh_patient_predictions_scored.npz"
    _save_predictions(kauh_scored_path, kauh_scored)
    _save_predictions(kauh_patient_path, patient_scored)

    hf_metrics = _augment_metrics(hf_metrics, selected, output_dir, "HF Lung source-test")
    kauh_metrics = _augment_metrics(kauh_metrics, selected, output_dir, "KAUH all-filtered external test")
    hf_metrics_path = output_dir / "hf_metrics.json"
    kauh_metrics_path = output_dir / "kauh_metrics.json"
    _write_json(hf_metrics_path, {
        **hf_metrics,
        "label_free_predictions": str(hf_label_free_path),
        "scored_predictions": str(hf_scored_path),
    })
    _write_json(kauh_metrics_path, {
        **kauh_metrics,
        "label_free_predictions": str(kauh_label_free_path),
        "scored_predictions": str(kauh_scored_path),
        "patient_predictions": str(kauh_patient_path),
    })
    summary = {
        "status": "posthoc_external_evaluation_complete",
        "evidence_label": EVIDENCE_LABEL,
        "seed": seed,
        "source_run": str(selected["run_dir"]),
        "checkpoint_source": str(selected["checkpoint"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "frozen_shared_thresholds": dict(thresholds),
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "test_accessed_datasets": ["hf_lung_source_test", "kauh_v3_all_filtered"],
        "hf_metrics_path": str(hf_metrics_path),
        "kauh_metrics_path": str(kauh_metrics_path),
        "outputs": {
            "hf_label_free": str(hf_label_free_path),
            "hf_scored": str(hf_scored_path),
            "kauh_label_free": str(kauh_label_free_path),
            "kauh_scored": str(kauh_scored_path),
            "kauh_patient_scored": str(kauh_patient_path),
        },
        "claim_boundary": (
            "post-hoc external transfer diagnostic; not HF/KAUH native reproduction, "
            "not JH2 reselection, and not a paper claim"
        ),
    }
    _write_json(output_dir / "run_summary.json", summary)
    del model
    return {**summary, "hf_metrics": hf_metrics, "kauh_metrics": kauh_metrics}


def _stat(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "ddof": 1,
        "values": [float(value) for value in array],
    }


def _hf_scalars(metrics: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for token in ("D", "Wheeze"):
        node = metrics["recording_presence_pool"]["metrics"][token]
        curve = node["curve"]
        thresholded = node["thresholded"]
        prefix = token.lower()
        for key in ("auroc", "auprc"):
            values[f"{prefix}_recording_{key}"] = float(curve[key])
        for key in ("specificity", "sensitivity", "average_score", "macro_f1", "uar"):
            values[f"{prefix}_recording_{key}"] = float(thresholded[key])
        values[f"{prefix}_positive_interval_recall"] = float(
            metrics["positive_interval_coverage"][token]["positive_recall"]
        )
    return values


def _kauh_scalars(metrics: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for level_key, level_label in (
        ("recording_level", "recording"),
        ("patient_level_after_BDE_probability_mean", "patient"),
    ):
        for task in ("level1_binary", "flat4"):
            metric = metrics[level_key][task]
            for key in ("specificity", "sensitivity", "average_score", "macro_f1", "uar"):
                values[f"{level_label}_{task}_{key}"] = float(metric[key])
    return values


def aggregate(repo_root: Path, seed_results: list[Mapping[str, object]], output_dir: Path) -> dict[str, object]:
    hf_by_seed = {int(row["seed"]): _hf_scalars(row["hf_metrics"]) for row in seed_results}
    kauh_by_seed = {int(row["seed"]): _kauh_scalars(row["kauh_metrics"]) for row in seed_results}
    hf_keys = sorted(next(iter(hf_by_seed.values())))
    kauh_keys = sorted(next(iter(kauh_by_seed.values())))
    summary = {
        "status": "complete_JH2_main_multiseed_HF_KAUH_posthoc_diagnostic",
        "evidence_label": EVIDENCE_LABEL,
        "source_main_aggregate": str(repo_root / MAIN_RELATIVE / "multiseed_summary.json"),
        "seeds": list(SEEDS),
        "n": 3,
        "sample_std_ddof": 1,
        "selection_unchanged": True,
        "training": False,
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "selected_epoch": int(row["selected_epoch"]),
                "source_run": row["source_run"],
                "hf_metrics": row["hf_metrics"],
                "kauh_metrics": row["kauh_metrics"],
            }
            for row in seed_results
        ],
        "hf_mean_sample_std": {
            key: _stat([hf_by_seed[seed][key] for seed in SEEDS]) for key in hf_keys
        },
        "kauh_mean_sample_std": {
            key: _stat([kauh_by_seed[seed][key] for seed in SEEDS]) for key in kauh_keys
        },
        "kauh_support": {
            "recordings": 336,
            "patients": 112,
            "filters": {"B": 112, "D": 112, "E": 112},
            "compatible_overlay_only": True,
            "unresolved_raw_sounds": ["Crep", "Bronchial", "I C B"],
        },
        "claim_boundary": (
            "post-hoc HF/KAUH external transfer diagnostic on fixed JH2 selected "
            "checkpoints; not native HF/KAUH reproduction or a paper result"
        ),
        "actions_not_performed": [
            "No retraining",
            "No epoch reselection",
            "No HF or KAUH threshold tuning",
            "No JH2 main aggregate modification",
            "No server, other baseline, Git commit/push, or Notion change",
        ],
    }
    _write_json(output_dir / "external_multiseed_summary.json", summary)
    (output_dir / "external_multiseed_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def _format(value: Mapping[str, object]) -> str:
    return f"{float(value['mean']):.6f} ± {float(value['sample_std']):.6f}"


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# JH2 selected-checkpoint HF Lung and KAUH post-hoc diagnostic",
        "",
        f"Evidence: `{summary['evidence_label']}`.",
        "This artifact evaluates fixed selected JH2 checkpoints. It does not change the main selection or aggregate and is not a paper result.",
        "",
        "## Per-seed selected checkpoints",
        "",
        "| Seed | Selected epoch | Source run |",
        "|---:|---:|---|",
    ]
    for row in summary["per_seed"]:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | `{row['source_run']}` |")
    lines.extend(["", "## HF Lung source-test mean ± sample std", "", "| Metric | Mean ± sample std |", "|---|---:|"])
    for key, value in summary["hf_mean_sample_std"].items():
        lines.append(f"| {key} | {_format(value)} |")
    lines.extend([
        "",
        "HF uses three continuous 5 s windows over each 15 s source-test recording; gap-unknown, unsupported labels, and Level1 normal detection are not scored.",
        "",
        "## KAUH compatible-overlay mean ± sample std",
        "",
        "| Metric | Mean ± sample std |",
        "|---|---:|",
    ])
    for key, value in summary["kauh_mean_sample_std"].items():
        lines.append(f"| {key} | {_format(value)} |")
    lines.extend([
        "",
        "KAUH support is 336 recordings / 112 patients / 112 each for B, D, and E. Compatible overlay metrics exclude unresolved Crep, Bronchial, and I C B; patient readout uses B/D/E probability means.",
        "",
        "## Boundary",
        "",
        "Thresholds are copied from each selected run's validation_selection.json. No HF/KAUH tuning, retraining, epoch reselection, main aggregate modification, server task, or other experiment was performed.",
        "",
        f"Per-seed artifacts are under `{Path(summary['per_seed'][0]['source_run']).parent.parent / OUTPUT_RELATIVE.name}`.",
    ])
    return "\n".join(lines) + "\n"


def run(repo_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite external root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in SEEDS:
        seed_results.append(
            run_seed(repo_root, seed, output_root / f"seed_{seed}")
        )
    return aggregate(repo_root, seed_results, output_root)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / OUTPUT_RELATIVE
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "evidence_label": EVIDENCE_LABEL, "output_dir": str(output_dir)}, indent=2))
        return
    print(json.dumps(run(args.repo_root, output_dir), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
