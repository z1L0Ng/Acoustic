"""Post-hoc HF Lung and KAUH evaluation for the selected JH3.3 checkpoint.

This runner performs external inference only.  It reuses the selected JH3.3
checkpoint and validation-frozen thresholds, and does not retrain, reselection
the checkpoint, or tune thresholds on HF Lung or KAUH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from baseline.pafa import jh2_hf_kauh_external as external
from baseline.pafa import joint_hierarchy_soft_bridge_mvn as jh3_3
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _prepare_waveforms,
    _save_predictions,
    _write_json,
)


SOURCE_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_soft_bridge_mvn_seed42"
)
OUTPUT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_HF_KAUH_external_seed42"
)
EVIDENCE_LABEL = "posthoc_fixed_JH3_3_selected_checkpoint_HF_KAUH_external_diagnostic"


def _load_selected(repo_root: Path) -> dict[str, object]:
    run_dir = repo_root / SOURCE_RELATIVE
    config = json.loads((run_dir / "config.json").read_text())
    selection = json.loads((run_dir / "validation_selection.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    selected_epoch = int(summary["selected_epoch"])
    if selected_epoch != int(selection["selected_epoch"]):
        raise RuntimeError("JH3.3 selected epoch mismatch")
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "backbone_checkpoint": Path(str(config["checkpoint"])),
        "selected_epoch": selected_epoch,
        "thresholds": thresholds,
        "selection": selection["checkpoint_selection"],
        "source_evidence_label": selection["evidence_label"],
    }


def _config(
    repo_root: Path,
    selected: Mapping[str, object],
    output_dir: Path,
    device: str,
    cpu_threads: int,
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
        cpu_threads=cpu_threads,
    )


def _config_payload(
    repo_root: Path,
    selected: Mapping[str, object],
    output_dir: Path,
    device: str,
    cpu_threads: int,
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
        "seed": 42,
        "selected_epoch": int(selected["selected_epoch"]),
        "source_selection_policy": selected["selection"],
        "source_evidence_label": selected["source_evidence_label"],
        "frozen_shared_thresholds": dict(selected["thresholds"]),
        "threshold_source": "JH3.3 validation_selection.json; no HF or KAUH tuning",
        "model": "BEATs iter3+ AS2M full fine-tuned JH3.3 soft-bridge model with unit-scalar log-fbank MVN frontend",
        "device": device,
        "precision": "FP32",
        "input": "mono 16 kHz; 5 s repeat-pad/front-truncate for KAUH; three continuous 5 s windows over each 15 s HF test recording",
        "cpu_threads": cpu_threads,
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "test_accessed_datasets": ["hf_lung_source_test", "kauh_v3_all_filtered"],
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


def _annotate(
    metrics: Mapping[str, object],
    selected: Mapping[str, object],
    output_dir: Path,
    kind: str,
) -> dict[str, object]:
    annotated = dict(metrics)
    annotated.update(
        {
            "evidence_label": EVIDENCE_LABEL,
            "source_run": str(selected["run_dir"]),
            "checkpoint_source": str(selected["checkpoint"]),
            "seed": 42,
            "selected_epoch": int(selected["selected_epoch"]),
            "shared_attribute_thresholds": dict(selected["thresholds"]),
            "threshold_source": "JH3.3 selected validation thresholds; no external tuning",
            "training": False,
            "checkpoint_selection": False,
            "threshold_tuning": False,
            "external_evaluation_kind": kind,
            "posthoc_output_dir": str(output_dir),
        }
    )
    return annotated


def _markdown(summary: Mapping[str, object]) -> str:
    hf = summary["hf_metrics"]
    kauh = summary["kauh_metrics"]
    lines = [
        "# JH3.3 HF Lung and KAUH external test diagnostic",
        "",
        f"Evidence: `{EVIDENCE_LABEL}`.",
        "",
        f"Source run: `{summary['source_run']}`; selected epoch `{summary['selected_epoch']}`.",
        "Thresholds are copied from the selected run's validation artifact. No external threshold tuning or checkpoint reselection was performed.",
        "",
        "## HF Lung",
        "",
        "HF uses three continuous non-overlapping 5 s windows per 15 s source-test recording. Gap-unknown and unsupported labels are excluded from scored compatible attributes.",
        "",
        f"- Source-test recordings: {hf['source_test_recordings']}",
        f"- Annotation status counts: `{json.dumps(hf['annotation_status_counts'], sort_keys=True)}`",
        "- Crackle/D presence: see `hf_metrics.json` under `recording_presence_pool.metrics.D`.",
        "- Wheeze presence: see `hf_metrics.json` under `recording_presence_pool.metrics.Wheeze`.",
        "",
        "## KAUH",
        "",
        "KAUH is reported on the compatible shared overlay at recording level and after B/D/E probability mean at patient level. Crep, Bronchial, and I C B remain unresolved and outside the scored overlay.",
        "",
        f"- Recordings: {kauh['recording_support']}; patients: {kauh['patient_support']}; compatible recordings: {kauh['compatible_recording_support']}",
        f"- Recording Level1 Score: {kauh['recording_level']['level1_binary']['average_score']:.6f}",
        f"- Recording flat4 Score: {kauh['recording_level']['flat4']['average_score']:.6f}",
        f"- Patient Level1 Score: {kauh['patient_level_after_BDE_probability_mean']['level1_binary']['average_score']:.6f}",
        f"- Patient flat4 Score: {kauh['patient_level_after_BDE_probability_mean']['flat4']['average_score']:.6f}",
        "",
        "## Boundary",
        "",
        "This is a posthoc/test-exposed external-transfer diagnostic for one fixed JH3.3 seed42 checkpoint. It is not a clean estimate, native HF/KAUH reproduction, or paper primary result.",
        "",
        f"Artifacts: `{summary['output_dir']}`.",
    ]
    return "\n".join(lines) + "\n"


def run(
    repo_root: Path,
    output_dir: Path,
    device_name: str,
    cpu_threads: int,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite external output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_selected(repo_root)
    config = _config(repo_root, selected, output_dir, device_name, cpu_threads)
    _write_json(
        output_dir / "config.json",
        _config_payload(repo_root, selected, output_dir, device_name, cpu_threads),
    )

    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    hf_samples = external._load_hf_test_samples(repo_root)
    kauh_samples = external._load_kauh_samples(repo_root)
    model, _ = jh3_3._build_components(config, device)
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    kauh_waveforms = _prepare_waveforms(kauh_samples, config)
    kauh_label_free = external._predict_kauh(
        model,
        kauh_samples,
        kauh_waveforms,
        config,
        device,
        selected["thresholds"],
    )
    kauh_label_free_path = output_dir / "kauh_predictions_label_free.npz"
    _save_predictions(kauh_label_free_path, kauh_label_free)

    hf_label_free = external._predict_hf(model, hf_samples, config, device)
    hf_label_free_path = output_dir / "hf_predictions_label_free.npz"
    _save_predictions(hf_label_free_path, hf_label_free)

    hf_annotations = external._parse_hf_annotations(hf_samples)
    hf_scored, hf_metrics = external._hf_scored_predictions(
        hf_label_free,
        hf_annotations,
        selected["thresholds"],
    )
    hf_scored_path = output_dir / "hf_predictions_scored.npz"
    _save_predictions(hf_scored_path, hf_scored)

    kauh_scored, _ = external._kauh_scored_predictions(
        kauh_label_free,
        kauh_samples,
        selected["thresholds"],
    )
    kauh_metrics, patient_scored = external._kauh_metrics(
        kauh_scored,
        selected["thresholds"],
    )
    kauh_scored_path = output_dir / "kauh_predictions_scored.npz"
    kauh_patient_path = output_dir / "kauh_patient_predictions_scored.npz"
    _save_predictions(kauh_scored_path, kauh_scored)
    _save_predictions(kauh_patient_path, patient_scored)

    hf_metrics = _annotate(
        hf_metrics,
        selected,
        output_dir,
        "HF Lung source-test",
    )
    kauh_metrics = _annotate(
        kauh_metrics,
        selected,
        output_dir,
        "KAUH all-filtered external test",
    )
    hf_metrics_path = output_dir / "hf_metrics.json"
    kauh_metrics_path = output_dir / "kauh_metrics.json"
    _write_json(
        hf_metrics_path,
        {
            **hf_metrics,
            "label_free_predictions": str(hf_label_free_path),
            "scored_predictions": str(hf_scored_path),
        },
    )
    _write_json(
        kauh_metrics_path,
        {
            **kauh_metrics,
            "label_free_predictions": str(kauh_label_free_path),
            "scored_predictions": str(kauh_scored_path),
            "patient_predictions": str(kauh_patient_path),
        },
    )
    summary = {
        "status": "posthoc_external_evaluation_complete",
        "evidence_label": EVIDENCE_LABEL,
        "seed": 42,
        "source_run": str(selected["run_dir"]),
        "checkpoint_source": str(selected["checkpoint"]),
        "output_dir": str(output_dir),
        "selected_epoch": int(selected["selected_epoch"]),
        "frozen_shared_thresholds": dict(selected["thresholds"]),
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
        "hf_metrics": hf_metrics,
        "kauh_metrics": kauh_metrics,
        "claim_boundary": (
            "post-hoc external transfer diagnostic; not HF/KAUH native reproduction, "
            "not checkpoint reselection, and not a paper claim"
        ),
    }
    _write_json(output_dir / "run_summary.json", summary)
    (output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / OUTPUT_RELATIVE
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "evidence_label": EVIDENCE_LABEL,
                    "source_run": str(args.repo_root / SOURCE_RELATIVE),
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            run(args.repo_root, output_dir, args.device, args.cpu_threads),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
