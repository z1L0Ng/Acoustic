"""Rebuild the missing SPR C/W readout for the exact historical JH2 HF-off pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline.four_dataset_frozen_encoder.data import (
    _load_spr,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_terminal import _attach_targets
from baseline.multidataset_pipeline.posthoc_native_readout import native_metrics
from baseline.pafa.joint_hierarchy import (
    PAFAJointHierarchyConfig,
    _build_components,
    _prepare_waveforms,
    _save_predictions,
    _write_json,
    infer,
)


BASE_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2"
)
JH4_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH4_JH2_HFaux_seed42_attempt2"
)
OUTPUT_RELATIVE = JH4_RELATIVE / "historical_jh2_spr_cw_posthoc"
EVIDENCE_LABEL = "historical_fixed_JH2_HF_pair_spr_cw_posthoc"


def _load_contract(repo_root: Path) -> dict[str, object]:
    base_dir = repo_root / BASE_RELATIVE
    jh4_dir = repo_root / JH4_RELATIVE
    base_config = json.loads((base_dir / "config.json").read_text())
    jh4_config = json.loads((jh4_dir / "config.json").read_text())
    external_config = json.loads(
        (
            repo_root
            / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2/config.json"
        ).read_text()
    )
    validation = json.loads((base_dir / "validation_selection.json").read_text())
    checkpoint_path = base_dir / "best_checkpoint.pt"
    if Path(jh4_config["base_reference"]) != base_dir:
        raise RuntimeError("JH4 base_reference does not identify the exact JH2 pair")
    if Path(external_config["checkpoint"]) != checkpoint_path:
        raise RuntimeError("external JH2 checkpoint source does not match base checkpoint")
    if int(validation["selected_epoch"]) != 19:
        raise RuntimeError("historical JH2 selected epoch changed")
    thresholds = {
        key: float(value)
        for key, value in validation["shared_attribute_thresholds"].items()
    }
    expected_thresholds = {
        "crackle": 0.4441142678260803,
        "wheeze": 0.10498092323541641,
    }
    if thresholds != expected_thresholds:
        raise RuntimeError("historical JH2 thresholds changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint["epoch"]) != 19 or int(checkpoint["update"]) != 6194:
        raise RuntimeError("historical JH2 checkpoint metadata changed")
    if float(checkpoint["icbhi_official_score"]) != 0.600502129962986:
        raise RuntimeError("historical JH2 checkpoint selection score changed")
    return {
        "base_dir": base_dir,
        "jh4_dir": jh4_dir,
        "base_config": base_config,
        "jh4_config": jh4_config,
        "external_config": external_config,
        "validation": validation,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "thresholds": thresholds,
    }


def _make_config(
    repo_root: Path,
    contract: Mapping[str, object],
    output_dir: Path,
    device_name: str,
) -> PAFAJointHierarchyConfig:
    raw = contract["base_config"]
    return PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=Path(raw["author_repo"]),
        checkpoint=Path(raw["checkpoint"]),
        icbhi_audio_dir=Path(raw["icbhi_audio_dir"]),
        output_dir=output_dir,
        device=device_name,
        seed=int(raw["seed"]),
        sample_rate=int(raw["sample_rate"]),
        desired_seconds=float(raw["desired_seconds"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        cpu_threads=int(raw["cpu_threads"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        cosine_eta_min_ratio=float(raw["cosine_eta_min_ratio"]),
        ema_beta=float(raw["ema_beta"]),
        classification_weight=float(raw["classification_weight"]),
        pafa_weight=float(raw["pafa_weight"]),
        lambda_pcsl=float(raw["lambda_pcsl"]),
        lambda_gpal=float(raw["lambda_gpal"]),
        projection_output_dim=int(raw["projection_output_dim"]),
        projection_norm=str(raw["projection_norm"]),
        projection_attention=bool(raw["projection_attention"]),
    )


def _attribute_metrics(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    aurocs = []
    auprcs = []
    for probability_index, target_index, name in (
        (0, 1, "crackle"),
        (1, 2, "wheeze"),
    ):
        mask = predictions["eligible"][:, target_index].astype(bool)
        target = predictions["targets"][mask, target_index].astype(np.int64)
        probability = predictions["attribute_probabilities"][mask, probability_index]
        auroc = float(roc_auc_score(target, probability))
        auprc = float(average_precision_score(target, probability))
        aurocs.append(auroc)
        auprcs.append(auprc)
        rows[name] = {
            "rows": int(len(target)),
            "eligible": int(mask.sum()),
            "positive": int(target.sum()),
            "negative": int((target == 0).sum()),
            "auroc": auroc,
            "auprc": auprc,
        }
    rows["macro"] = {
        "definition": "mean of Crackle and Wheeze metrics within this seed",
        "auroc": float(np.mean(aurocs)),
        "auprc": float(np.mean(auprcs)),
    }
    return rows


def _native_spr_metrics(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, object]:
    target = predictions["targets"][:, 0].astype(np.int64)
    prediction = predictions["level1_predictions"].astype(np.int64)
    metrics = native_metrics(target, prediction, ("normal", "abnormal"))
    metrics.update(
        {
            "task": "SPRSound BioCAS2022 inter Task1-1 Normal/Adventitious",
            "protocol": "official inter-subject test; 1429 respiratory events",
            "average_score_as": metrics["average_score"],
            "harmonic_score_hs": metrics["harmonic_score"],
            "official_score": (
                metrics["average_score"] + metrics["harmonic_score"]
            )
            / 2,
            "task1_2_raw7": "not produced by the shared three-node head",
        }
    )
    return metrics


def _summary_markdown(summary: Mapping[str, object]) -> str:
    attrs = summary["attribute_metrics"]
    native = summary["native_metrics"]
    lines = [
        "# Historical JH2 HF-off SPRSound C/W posthoc",
        "",
        f"Evidence: `{EVIDENCE_LABEL}`.",
        "",
        f"Source checkpoint: `{summary['checkpoint_source']}`; selected epoch {summary['selected_epoch']}; no checkpoint reselection and no threshold tuning.",
        "",
        "| Readout | AUROC | AUPRC | Positive / negative |",
        "|---|---:|---:|---:|",
        f"| Crackle | {attrs['crackle']['auroc']:.6f} | {attrs['crackle']['auprc']:.6f} | {attrs['crackle']['positive']} / {attrs['crackle']['negative']} |",
        f"| Wheeze | {attrs['wheeze']['auroc']:.6f} | {attrs['wheeze']['auprc']:.6f} | {attrs['wheeze']['positive']} / {attrs['wheeze']['negative']} |",
        f"| C/W macro | {attrs['macro']['auroc']:.6f} | {attrs['macro']['auprc']:.6f} | within-seed mean |",
        "",
        f"Native SPRSound inter official Score: {native['official_score']:.6f}; BA/AS: {native['average_score']:.6f}; HS: {native['harmonic_score']:.6f}; Macro-F1: {native['macro_f1']:.6f}; UAR: {native['uar']:.6f}.",
        f"Existing saved native Score: {summary['existing_native_score']:.6f}; reconciliation delta: {summary['native_score_reconciliation']['delta']:.12f}; status: `{summary['native_score_reconciliation']['status']}`.",
        "",
        "This is the exact historical JH2 HF-off side of the JH4 HF-on/off single-seed pair. It is not the current JH2 main fresh42 run and is not a clean estimate or primary paper result.",
        "",
        f"Artifacts: `{summary['output_dir']}`.",
    ]
    return "\n".join(lines) + "\n"


def run(repo_root: Path, output_dir: Path, device_name: str) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract(repo_root)
    config = _make_config(repo_root, contract, output_dir, device_name)
    config.validate()
    _write_json(
        output_dir / "config.json",
        {
            "status": "fixed_checkpoint_spr_cw_posthoc_inference",
            "evidence_label": EVIDENCE_LABEL,
            "source_run": str(contract["base_dir"]),
            "source_pair": str(contract["jh4_dir"]),
            "checkpoint": str(contract["checkpoint_path"]),
            "checkpoint_epoch": int(contract["checkpoint"]["epoch"]),
            "checkpoint_update": int(contract["checkpoint"]["update"]),
            "selected_epoch": 19,
            "device": device_name,
            "precision": "FP32",
            "model": "BEATs iter3+ AS2M full fine-tuned PAFA-JH2 Hard Hierarchy; no MVN",
            "input": "mono 16 kHz; 5 s repeat-pad/front-truncate",
            "frozen_shared_thresholds": dict(contract["thresholds"]),
            "threshold_source": "historical JH2 validation_selection.json and external config; no test tuning",
            "datasets": ["sprsound_inter_test_only"],
            "training": False,
            "optimizer_update": False,
            "checkpoint_selection": False,
            "prediction_support": 1429,
            "claim_boundary": "historical single-seed HF-off side of the JH4 HF-on/off pair; not clean or primary",
        },
    )

    device = torch.device(device_name)
    torch.set_num_threads(config.cpu_threads)
    model, _ = _build_components(config, device)
    model.load_state_dict(contract["checkpoint"]["model"], strict=True)
    model.to(device)

    spr_rows, _ = _load_spr(repo_root / "dataset/raw", include_checksums=False)
    spr_samples = tuple(
        sorted(
            (row for row in spr_rows if row.partition == "test"),
            key=lambda row: row.sample_id,
        )
    )
    if len(spr_samples) != 1429:
        raise RuntimeError(f"SPRSound inter support changed: {len(spr_samples)}")
    waveform_store = _prepare_waveforms(spr_samples, config)
    label_free = infer(
        model,
        {"icbhi": tuple(), "sprsound": spr_samples},
        waveform_store,
        config,
        device,
        include_targets=False,
    )
    terminal_dir = output_dir / "terminal"
    label_free_path = terminal_dir / "selected_sprsound_predictions_label_free.npz"
    _save_predictions(label_free_path, label_free)

    spr_targets = load_terminal_spr_test_targets(
        list(spr_samples),
        include_checksums=False,
    )
    scored = _attach_targets(label_free, spr_samples, spr_targets)
    scored_path = terminal_dir / "selected_sprsound_predictions_scored.npz"
    _save_predictions(scored_path, scored)

    attribute_metrics = _attribute_metrics(scored)
    native_metrics_payload = _native_spr_metrics(scored)
    existing_native = json.loads(
        (
            contract["base_dir"] / "terminal/selected_native_metrics.json"
        ).read_text()
    )
    existing_native_score = float(
        existing_native["sprsound_inter_task1_1"]["official_score"]
    )
    delta = float(native_metrics_payload["official_score"] - existing_native_score)
    reconciliation = {
        "status": "match_within_float_tolerance" if np.isclose(delta, 0.0) else "HOLD_native_score_mismatch",
        "existing_native_score": existing_native_score,
        "recomputed_native_score": float(native_metrics_payload["official_score"]),
        "delta": delta,
        "existing_source": str(
            contract["base_dir"] / "terminal/selected_native_metrics.json"
        ),
    }
    summary = {
        "status": "complete_fixed_checkpoint_spr_cw_posthoc",
        "evidence_label": EVIDENCE_LABEL,
        "source_run": str(contract["base_dir"]),
        "source_pair": str(contract["jh4_dir"]),
        "checkpoint_source": str(contract["checkpoint_path"]),
        "selected_epoch": 19,
        "output_dir": str(output_dir),
        "frozen_shared_thresholds": dict(contract["thresholds"]),
        "training": False,
        "optimizer_update": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "test_accessed_datasets": ["sprsound_inter_test"],
        "prediction_support": 1429,
        "label_free_predictions": str(label_free_path),
        "scored_predictions": str(scored_path),
        "attribute_metrics": attribute_metrics,
        "native_metrics": native_metrics_payload,
        "existing_native_score": existing_native_score,
        "native_score_reconciliation": reconciliation,
        "claim_boundary": "historical single-seed HF-off side of the JH4 HF-on/off pair; not clean or primary",
    }
    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "run_summary.json", summary)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo_root / OUTPUT_RELATIVE
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "evidence_label": EVIDENCE_LABEL,
                    "source_run": str(args.repo_root / BASE_RELATIVE),
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            run(args.repo_root, output_dir, args.device),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
