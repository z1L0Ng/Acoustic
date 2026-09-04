"""Fixed-checkpoint JH4 KAUH external evaluation and JH2 comparison.

This module performs one KAUH external inference pass for the selected JH4
checkpoint and then assembles a read-only comparison from the saved JH2/JH4
core, SPRSound, HF, and KAUH artifacts.  It is permanently test-exposed and is
not a native KAUH raw9 reproduction or a paper result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from baseline.multidataset_pipeline.posthoc_native_readout import ICBHI_LABELS
from baseline.pafa.jh2_hf_kauh_external import (
    _kauh_metrics,
    _kauh_scored_predictions,
    _load_kauh_samples,
    _predict_kauh,
)
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _build_components,
    _prepare_waveforms,
    _save_predictions,
    _write_json,
)


EVIDENCE_LABEL = "fixed_test_selected_JH4_checkpoint_KAUH_external_and_JH2_comparison"
JH4_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH4_JH2_HFaux_seed42_attempt2"
)
JH2_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2"
)
JH2_KAUH_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2"
)
KAUH_OUTPUT_RELATIVE = Path("kauh_external_jh4")
JH4_THRESHOLDS = {
    "crackle": 0.23046129941940308,
    "wheeze": 0.47414448857307434,
}


def _config(repo_root: Path, output_dir: Path) -> PAFAJointHierarchyConfig:
    return PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=repo_root
        / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        checkpoint=repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
        icbhi_audio_dir=repo_root
        / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
        output_dir=output_dir,
        device="mps",
        cpu_threads=4,
    )


def _metric_scalars(metric: Mapping[str, object]) -> dict[str, float]:
    keys = (
        "specificity",
        "sensitivity",
        "official_score",
        "macro_f1",
        "uar",
        "average_score",
        "harmonic_score",
    )
    return {
        key: float(metric[key])
        for key in keys
        if key in metric and metric[key] is not None
    }


def _metric_comparison(
    jh2_metric: Mapping[str, object],
    jh4_metric: Mapping[str, object],
) -> dict[str, object]:
    jh2_values = _metric_scalars(jh2_metric)
    jh4_values = _metric_scalars(jh4_metric)
    keys = sorted(set(jh2_values) | set(jh4_values))
    return {
        key: {
            "jh2": jh2_values.get(key),
            "jh4": jh4_values.get(key),
            "delta": (
                jh4_values[key] - jh2_values[key]
                if key in jh2_values and key in jh4_values
                else None
            ),
        }
        for key in keys
    }


def _per_class_recall(metric: Mapping[str, object]) -> dict[str, float]:
    return {
        str(label): float(row["recall"])
        for label, row in metric.get("per_class", {}).items()
        if row.get("recall") is not None
    }


def _recall_comparison(
    jh2_metric: Mapping[str, object],
    jh4_metric: Mapping[str, object],
) -> dict[str, object]:
    jh2_values = _per_class_recall(jh2_metric)
    jh4_values = _per_class_recall(jh4_metric)
    labels = sorted(set(jh2_values) | set(jh4_values))
    return {
        label: {
            "jh2": jh2_values.get(label),
            "jh4": jh4_values.get(label),
            "delta": (
                jh4_values[label] - jh2_values[label]
                if label in jh2_values and label in jh4_values
                else None
            ),
        }
        for label in labels
    }


def _hf_node_comparison(
    jh2_metrics: Mapping[str, object],
    jh4_metrics: Mapping[str, object],
    *,
    node_key: str,
    interval_key: str,
) -> dict[str, object]:
    jh2_node = jh2_metrics["recording_presence_pool"]["metrics"][node_key]
    jh4_node = jh4_metrics["recording_presence_pool"]["metrics"][node_key]
    jh2_curve = jh2_node["curve"]
    jh4_curve = jh4_node["curve"]
    jh2_thresholded = jh2_node["thresholded"]
    jh4_thresholded = jh4_node["thresholded"]
    jh2_interval = jh2_metrics["positive_interval_coverage"][interval_key]
    jh4_interval = jh4_metrics["positive_interval_coverage"][interval_key]
    scalar_values = {
        "recording_auroc": (jh2_curve["auroc"], jh4_curve["auroc"]),
        "recording_auprc": (jh2_curve["auprc"], jh4_curve["auprc"]),
        "recording_sensitivity": (
            jh2_thresholded["sensitivity"],
            jh4_thresholded["sensitivity"],
        ),
        "recording_specificity": (
            jh2_thresholded["specificity"],
            jh4_thresholded["specificity"],
        ),
        "recording_macro_f1": (
            jh2_thresholded["macro_f1"],
            jh4_thresholded["macro_f1"],
        ),
        "recording_positive_f1": (
            jh2_thresholded["per_class"]["positive"]["f1"],
            jh4_thresholded["per_class"]["positive"]["f1"],
        ),
        "positive_interval_recall": (
            jh2_interval["positive_recall"],
            jh4_interval["positive_recall"],
        ),
    }
    return {
        key: {"jh2": float(values[0]), "jh4": float(values[1]), "delta": float(values[1] - values[0])}
        for key, values in scalar_values.items()
    }


def _kauh_task_comparison(
    jh2_metric: Mapping[str, object],
    jh4_metric: Mapping[str, object],
) -> dict[str, object]:
    return {
        "scalars": _metric_comparison(jh2_metric, jh4_metric),
        "per_class_recall": _recall_comparison(jh2_metric, jh4_metric),
        "jh2_confusion": jh2_metric["confusion"],
        "jh4_confusion": jh4_metric["confusion"],
    }


def _kauh_comparison(
    jh2_metrics: Mapping[str, object],
    jh4_metrics: Mapping[str, object],
) -> dict[str, object]:
    support_fields = (
        "recording_support",
        "patient_support",
        "compatible_recording_support",
    )
    support = {
        field: {
            "jh2": int(jh2_metrics[field]),
            "jh4": int(jh4_metrics[field]),
            "delta": int(jh4_metrics[field]) - int(jh2_metrics[field]),
        }
        for field in support_fields
    }
    unresolved = {}
    for raw_label in ("Crep", "Bronchial", "I C B"):
        jh2_row = jh2_metrics["unresolved_raw_sounds"][raw_label]
        jh4_row = jh4_metrics["unresolved_raw_sounds"][raw_label]
        unresolved[raw_label] = {
            "status": "unresolved_not_scored",
            "jh2": jh2_row,
            "jh4": jh4_row,
            "recording_support_delta": int(jh4_row["recording_support"])
            - int(jh2_row["recording_support"]),
            "level1_prediction_distribution_delta": {
                label: int(jh4_row["prediction_distribution_level1"][label])
                - int(jh2_row["prediction_distribution_level1"][label])
                for label in ("normal", "abnormal")
            },
        }
    per_filter = {}
    for filter_mode in ("B", "D", "E"):
        jh2_filter = jh2_metrics["per_filter_BDE_audit"][filter_mode]
        jh4_filter = jh4_metrics["per_filter_BDE_audit"][filter_mode]
        per_filter[filter_mode] = {
            "recordings": {
                "jh2": int(jh2_filter["recordings"]),
                "jh4": int(jh4_filter["recordings"]),
                "delta": int(jh4_filter["recordings"])
                - int(jh2_filter["recordings"]),
            },
            "compatible_recordings": {
                "jh2": int(jh2_filter["compatible_recordings"]),
                "jh4": int(jh4_filter["compatible_recordings"]),
                "delta": int(jh4_filter["compatible_recordings"])
                - int(jh2_filter["compatible_recordings"]),
            },
            "level1_binary": _kauh_task_comparison(
                jh2_filter["level1_binary"], jh4_filter["level1_binary"]
            ),
            "flat4": _kauh_task_comparison(
                jh2_filter["flat4"], jh4_filter["flat4"]
            ),
            "raw_sound_support_jh2": jh2_filter["raw_sound_support"],
            "raw_sound_support_jh4": jh4_filter["raw_sound_support"],
        }
    return {
        "support": support,
        "compatible_overlay": {
            "jh2": jh2_metrics["compatible_raw_sounds"],
            "jh4": jh4_metrics["compatible_raw_sounds"],
        },
        "unresolved_raw_sounds": unresolved,
        "recording_level": {
            task: _kauh_task_comparison(
                jh2_metrics["recording_level"][task],
                jh4_metrics["recording_level"][task],
            )
            for task in ("level1_binary", "flat4")
        },
        "patient_level_after_BDE_probability_mean": {
            task: _kauh_task_comparison(
                jh2_metrics["patient_level_after_BDE_probability_mean"][task],
                jh4_metrics["patient_level_after_BDE_probability_mean"][task],
            )
            for task in ("level1_binary", "flat4")
        },
        "per_filter_BDE_audit": per_filter,
        "jh2_full_metrics": jh2_metrics,
        "jh4_full_metrics": jh4_metrics,
    }


def _run_kauh(
    repo_root: Path,
    jh4_run: Path,
    kauh_output: Path,
) -> dict[str, object]:
    if kauh_output.exists() and any(kauh_output.iterdir()):
        raise FileExistsError(f"refusing to overwrite KAUH output: {kauh_output}")
    kauh_output.mkdir(parents=True, exist_ok=True)
    config = _config(repo_root, kauh_output)
    config.validate()
    selection = json.loads((jh4_run / "validation_selection.json").read_text())
    thresholds = {
        key: float(value)
        for key, value in selection["shared_attribute_thresholds"].items()
    }
    if int(selection["selected_epoch"]) != 9 or thresholds != JH4_THRESHOLDS:
        raise RuntimeError("JH4 selected epoch or frozen thresholds changed")
    checkpoint_path = jh4_run / "best_checkpoint.pt"
    model, _ = _build_components(config, torch.device("mps"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device("mps"))

    kauh_samples = _load_kauh_samples(repo_root)
    kauh_waveforms = _prepare_waveforms(kauh_samples, config)
    label_free = _predict_kauh(
        model,
        kauh_samples,
        kauh_waveforms,
        config,
        torch.device("mps"),
        thresholds,
    )
    label_free_path = kauh_output / "kauh_predictions_label_free.npz"
    _save_predictions(label_free_path, label_free)
    scored, _ = _kauh_scored_predictions(label_free, kauh_samples, thresholds)
    metrics, patient_scored = _kauh_metrics(scored, thresholds)
    scored_path = kauh_output / "kauh_predictions_scored.npz"
    patient_path = kauh_output / "kauh_patient_predictions_scored.npz"
    _save_predictions(scored_path, scored)
    _save_predictions(patient_path, patient_scored)
    metrics = {
        **metrics,
        "status": "fixed_test_selected_JH4_KAUH_external_complete",
        "evidence_label": EVIDENCE_LABEL,
        "checkpoint_source": str(checkpoint_path),
        "selected_epoch": 9,
        "shared_attribute_thresholds": thresholds,
        "threshold_source": "JH4 validation_selection.json; no KAUH tuning",
        "training": False,
        "checkpoint_selection": False,
        "label_free_written_before_raw_targets": True,
        "kauh_native_raw9_reproduction": False,
        "compatible_overlay_only": True,
        "unresolved_labels_not_scored": ["Crep", "Bronchial", "I C B"],
        "label_free_predictions": str(label_free_path),
        "scored_predictions": str(scored_path),
        "patient_predictions": str(patient_path),
    }
    _write_json(kauh_output / "config.json", {
        "status": "fixed_checkpoint_external_transfer_inference",
        "evidence_label": EVIDENCE_LABEL,
        "checkpoint": str(checkpoint_path),
        "backbone_checkpoint_for_model_construction": str(config.checkpoint),
        "selected_epoch": 9,
        "frozen_shared_thresholds": thresholds,
        "model": "BEATs iter3+ AS2M full fine-tuned PAFA-JH4 hierarchy",
        "device": "mps",
        "precision": "FP32",
        "input": "mono 16 kHz; 5 s front-truncate; B/D/E sibling recordings retained",
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "kauh_protocol": {
            "recordings": 336,
            "patients": 112,
            "filters": ["B", "D", "E"],
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
    })
    _write_json(kauh_output / "kauh_metrics.json", metrics)
    return metrics


def build_comparison(repo_root: Path, jh4_run: Path, kauh_metrics: Mapping[str, object]) -> dict[str, object]:
    jh4_summary = json.loads((jh4_run / "run_summary.json").read_text())
    jh4_selected = jh4_summary["selected_metrics"]
    jh2_terminal = json.loads(
        (repo_root / JH2_RELATIVE / "terminal/native_metrics.json").read_text()
    )
    jh2_hf = json.loads(
        (repo_root / JH2_KAUH_RELATIVE / "hf_metrics.json").read_text()
    )
    jh2_kauh = json.loads(
        (repo_root / JH2_KAUH_RELATIVE / "kauh_metrics.json").read_text()
    )
    icbhi_comparison = _metric_comparison(
        jh2_terminal["icbhi_flat4"], jh4_selected["icbhi_flat4"]
    )
    icbhi_comparison["per_class_recall"] = _recall_comparison(
        jh2_terminal["icbhi_flat4"], jh4_selected["icbhi_flat4"]
    )
    spr_comparison = _metric_comparison(
        jh2_terminal["sprsound_inter_task1_1"],
        jh4_selected["sprsound_inter_task1_1"],
    )
    hf_comparison = {
        "crackle_D": _hf_node_comparison(
            jh2_hf,
            jh4_selected["hf_source_test"],
            node_key="D",
            interval_key="D",
        ),
        "wheeze": _hf_node_comparison(
            jh2_hf,
            jh4_selected["hf_source_test"],
            node_key="Wheeze",
            interval_key="Wheeze",
        ),
        "support": {
            "jh2_source_test_recordings": jh2_hf["source_test_recordings"],
            "jh4_source_test_recordings": jh4_selected["hf_source_test"][
                "source_test_recordings"
            ],
        },
    }
    comparison = {
        "status": "complete_fixed_test_selected_JH4_KAUH_external_and_JH2_comparison",
        "evidence_label": EVIDENCE_LABEL,
        "claim_boundary": (
            "JH2/JH4 are test-selected diagnostics; KAUH is compatible-overlay external "
            "diagnostic, not raw9 native reproduction or a paper result"
        ),
        "method_difference": {
            "jh2": "JH2 Hard Hierarchy, no HF auxiliary",
            "jh4": "JH2 base plus HF shared Crackle/Wheeze auxiliary, lambda=0.25",
            "hf_selection_role": "HF excluded from threshold fitting, checkpoint selection, and early stopping",
            "core_contract": "same ICBHI+SPRSound joint training/data/model/input/batching/optimizer contract",
            "epoch_selection": "ICBHI official Hard Hierarchy Score only; strict improvement; tie-earliest",
        },
        "selected_epochs": {"jh2": 19, "jh4": 9},
        "icbhi": {
            "jh2": jh2_terminal["icbhi_flat4"],
            "jh4": jh4_selected["icbhi_flat4"],
            "comparison": icbhi_comparison,
        },
        "sprsound": {
            "jh2": jh2_terminal["sprsound_inter_task1_1"],
            "jh4": jh4_selected["sprsound_inter_task1_1"],
            "comparison": spr_comparison,
        },
        "hf": {
            "jh2_hf_off": jh2_hf,
            "jh4_hf_on": jh4_selected["hf_source_test"],
            "comparison": hf_comparison,
        },
        "kauh": {
            "jh2": jh2_kauh,
            "jh4": kauh_metrics,
            "comparison": _kauh_comparison(jh2_kauh, kauh_metrics),
        },
        "tradeoffs": [
            "JH4 ICBHI Score is within 0.01 of JH2 and specificity is higher, but sensitivity, Macro-F1, UAR, and several ICBHI class recalls are lower.",
            "JH4 improves SPRSound Task1-1 official Score relative to JH2.",
            "HF-on improves D/Crackle recording curves and interval recall; Wheeze AUROC/AUPRC improve while positive-interval recall decreases.",
            "KAUH compatible-overlay results must be read at both recording and patient levels; unresolved Crep/Bronchial/I C B remain outside shared metrics.",
        ],
        "actions_not_performed": [
            "No JH2 re-inference",
            "No JH4 retraining or epoch reselection",
            "No KAUH threshold tuning",
            "No raw9 KAUH native reproduction",
            "No server, KAUH follow-up selection, N1/A1/L1/L2, Git commit/push, or Notion change",
        ],
    }
    return comparison


def _markdown(comparison: Mapping[str, object], kauh_output: Path) -> str:
    icbhi = comparison["icbhi"]["comparison"]
    spr = comparison["sprsound"]["comparison"]
    hf = comparison["hf"]["comparison"]
    kauh = comparison["kauh"]["comparison"]
    lines = [
        "# JH2 vs JH4: fixed-checkpoint KAUH external comparison",
        "",
        f"Evidence: `{comparison['evidence_label']}`.",
        "Both runs are test-selected diagnostics. KAUH uses the compatible shared overlay and is not a raw9 native reproduction or paper result.",
        "",
        "## Method",
        "",
        "JH4 keeps the JH2 core contract and adds only the HF shared Crackle/Wheeze auxiliary (`lambda=0.25`). HF is excluded from threshold fitting, checkpoint selection, and early stopping. Epoch selection uses ICBHI official Hard Hierarchy Score only.",
        "",
        "## Core comparison",
        "",
        "| Metric | JH2 | JH4 | Delta |",
        "|---|---:|---:|---:|",
        f"| ICBHI Score | {icbhi['official_score']['jh2']:.6f} | {icbhi['official_score']['jh4']:.6f} | {icbhi['official_score']['delta']:+.6f} |",
        f"| ICBHI Sp | {icbhi['specificity']['jh2']:.6f} | {icbhi['specificity']['jh4']:.6f} | {icbhi['specificity']['delta']:+.6f} |",
        f"| ICBHI Se | {icbhi['sensitivity']['jh2']:.6f} | {icbhi['sensitivity']['jh4']:.6f} | {icbhi['sensitivity']['delta']:+.6f} |",
        f"| ICBHI Macro-F1 | {icbhi['macro_f1']['jh2']:.6f} | {icbhi['macro_f1']['jh4']:.6f} | {icbhi['macro_f1']['delta']:+.6f} |",
        f"| ICBHI UAR | {icbhi['uar']['jh2']:.6f} | {icbhi['uar']['jh4']:.6f} | {icbhi['uar']['delta']:+.6f} |",
        f"| SPRSound Se | {spr['sensitivity']['jh2']:.6f} | {spr['sensitivity']['jh4']:.6f} | {spr['sensitivity']['delta']:+.6f} |",
        f"| SPRSound Sp | {spr['specificity']['jh2']:.6f} | {spr['specificity']['jh4']:.6f} | {spr['specificity']['delta']:+.6f} |",
        f"| SPRSound AS | {spr['average_score']['jh2']:.6f} | {spr['average_score']['jh4']:.6f} | {spr['average_score']['delta']:+.6f} |",
        f"| SPRSound HS | {spr['harmonic_score']['jh2']:.6f} | {spr['harmonic_score']['jh4']:.6f} | {spr['harmonic_score']['delta']:+.6f} |",
        f"| SPRSound official Score | {spr['official_score']['jh2']:.6f} | {spr['official_score']['jh4']:.6f} | {spr['official_score']['delta']:+.6f} |",
        f"| SPRSound Macro-F1 | {spr['macro_f1']['jh2']:.6f} | {spr['macro_f1']['jh4']:.6f} | {spr['macro_f1']['delta']:+.6f} |",
        f"| SPRSound UAR | {spr['uar']['jh2']:.6f} | {spr['uar']['jh4']:.6f} | {spr['uar']['delta']:+.6f} |",
        "",
        "ICBHI per-class recall deltas:",
    ]
    for label, values in icbhi["per_class_recall"].items():
        lines.append(
            f"- {label}: {values['jh2']:.6f} -> {values['jh4']:.6f} ({values['delta']:+.6f})"
        )
    lines.extend(["", "## HF source-test: JH2 off vs JH4 on", ""])
    lines.append("| Node | Measure | JH2 off | JH4 on | Delta |\n|---|---|---:|---:|---:|")
    for node, values in (("D/Crackle", hf["crackle_D"]), ("Wheeze", hf["wheeze"])):
        for key, label in (
            ("recording_auroc", "recording AUROC"),
            ("recording_auprc", "recording AUPRC"),
            ("recording_sensitivity", "recording Se"),
            ("recording_specificity", "recording Sp"),
            ("recording_macro_f1", "recording Macro-F1"),
            ("recording_positive_f1", "recording positive F1"),
            ("positive_interval_recall", "positive-interval recall"),
        ):
            row = values[key]
            lines.append(
                f"| {node} | {label} | {row['jh2']:.6f} | {row['jh4']:.6f} | {row['delta']:+.6f} |"
            )
    lines.extend(["", "## KAUH compatible-overlay comparison", ""])
    lines.append("Support: 336 recordings / 112 patients / 258 compatible recordings in both runs; Crep, Bronchial, and I C B remain unresolved and unscored.")
    lines.append("")
    lines.append("| Level/task | Metric | JH2 | JH4 | Delta |\n|---|---|---:|---:|---:|")
    for level, key in (("Recording Level1", "recording_level"), ("Recording flat4", "recording_level"), ("Patient Level1", "patient_level_after_BDE_probability_mean"), ("Patient flat4", "patient_level_after_BDE_probability_mean")):
        task = "level1_binary" if "Level1" in level else "flat4"
        values = kauh[key][task]["scalars"]
        for metric_key, label in (("specificity", "Sp"), ("sensitivity", "Se"), ("official_score", "Score"), ("macro_f1", "Macro-F1"), ("uar", "UAR")):
            if metric_key in values:
                row = values[metric_key]
                lines.append(f"| {level} | {label} | {row['jh2']:.6f} | {row['jh4']:.6f} | {row['delta']:+.6f} |")
    lines.extend(["", "### KAUH per-filter audit", ""])
    lines.append("| Filter | Task | Metric | JH2 | JH4 | Delta |\n|---|---|---|---:|---:|---:|")
    for filter_mode in ("B", "D", "E"):
        for task, label in (("level1_binary", "Level1"), ("flat4", "flat4")):
            values = kauh["per_filter_BDE_audit"][filter_mode][task]["scalars"]
            for metric_key, metric_label in (("specificity", "Sp"), ("sensitivity", "Se"), ("official_score", "Score"), ("macro_f1", "Macro-F1"), ("uar", "UAR")):
                row = values[metric_key]
                lines.append(f"| {filter_mode} | {label} | {metric_label} | {row['jh2']:.6f} | {row['jh4']:.6f} | {row['delta']:+.6f} |")
    lines.extend(["", "Unresolved raw-label support and prediction distributions are retained in the JSON under `unresolved_raw_sounds` and are not scored.", ""])
    for raw_label, values in kauh["unresolved_raw_sounds"].items():
        jh2_row = values["jh2"]
        jh4_row = values["jh4"]
        lines.append(
            f"- {raw_label}: support {jh2_row['recording_support']} -> {jh4_row['recording_support']}; "
            f"Level1 normal/abnormal {jh2_row['prediction_distribution_level1']['normal']}/"
            f"{jh2_row['prediction_distribution_level1']['abnormal']} -> "
            f"{jh4_row['prediction_distribution_level1']['normal']}/"
            f"{jh4_row['prediction_distribution_level1']['abnormal']}"
        )
    lines.append("## Trade-offs and boundary")
    lines.append("")
    for item in comparison["tradeoffs"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "KAUH was evaluated once after the fixed JH4 checkpoint and never used for selection or threshold fitting. No subsequent experiment was launched.",
        "",
        f"KAUH artifacts: `{kauh_output}`.",
    ])
    return "\n".join(lines) + "\n"


def run(repo_root: Path) -> dict[str, object]:
    jh4_run = repo_root / JH4_RELATIVE
    kauh_output = jh4_run / KAUH_OUTPUT_RELATIVE
    kauh_metrics = _run_kauh(repo_root, jh4_run, kauh_output)
    comparison = build_comparison(repo_root, jh4_run, kauh_metrics)
    comparison_json = jh4_run / "JH2_vs_JH4_KAUH_external_comparison.json"
    comparison_md = jh4_run / "JH2_vs_JH4_KAUH_external_comparison.md"
    _write_json(comparison_json, comparison)
    comparison_md.write_text(_markdown(comparison, kauh_output), encoding="utf-8")
    return {
        "status": comparison["status"],
        "evidence_label": EVIDENCE_LABEL,
        "kauh_output": str(kauh_output),
        "comparison_json": str(comparison_json),
        "comparison_markdown": str(comparison_md),
        "selected_epoch": 9,
        "kauh_recordings": 336,
        "kauh_patients": 112,
    }


def write_comparison_only(repo_root: Path) -> dict[str, object]:
    """Rewrite comparison artifacts from the already saved JH4 KAUH metrics."""

    jh4_run = repo_root / JH4_RELATIVE
    kauh_output = jh4_run / KAUH_OUTPUT_RELATIVE
    kauh_metrics = json.loads((kauh_output / "kauh_metrics.json").read_text())
    comparison = build_comparison(repo_root, jh4_run, kauh_metrics)
    comparison_json = jh4_run / "JH2_vs_JH4_KAUH_external_comparison.json"
    comparison_md = jh4_run / "JH2_vs_JH4_KAUH_external_comparison.md"
    _write_json(comparison_json, comparison)
    comparison_md.write_text(_markdown(comparison, kauh_output), encoding="utf-8")
    return {
        "status": comparison["status"],
        "evidence_label": EVIDENCE_LABEL,
        "comparison_json": str(comparison_json),
        "comparison_markdown": str(comparison_md),
        "kauh_inference_reused": True,
        "new_model_inference": False,
        "new_test_target_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args()
    if args.comparison_only:
        print(
            json.dumps(
                write_comparison_only(args.repo_root),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "evidence_label": EVIDENCE_LABEL,
                    "output_dir": str(args.repo_root / JH4_RELATIVE / KAUH_OUTPUT_RELATIVE),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(args.repo_root), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
