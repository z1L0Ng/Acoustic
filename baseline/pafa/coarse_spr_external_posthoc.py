"""Fixed-checkpoint HF/KAUH post-hoc for the coarse-SPR controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.pafa.native_attributes_external_posthoc import _install_numpy_metric_compat

_install_numpy_metric_compat()

from sklearn.metrics import roc_auc_score

from baseline.pafa import jh2_hf_kauh_external as external
from baseline.pafa.joint_hierarchy import _prepare_waveforms, _save_predictions, _write_json
from baseline.pafa.table2_benchmark_controls import BenchmarkConfig
from baseline.pafa.table2_clean_controls import build_components


REPO = Path("/Users/zilongzeng/Research/Acoustic")
SOURCE_ROOT = REPO / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/coarse_spr"
SOURCE_SEED42 = REPO / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/coarse_spr/seed_42_attempt2"
OUTPUT_ROOT = REPO / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/coarse_spr_external_posthoc"
SEEDS = (0, 1, 42)
EVIDENCE_LABEL = "complete_coarse_spr_fixed_checkpoint_external_posthoc"
CAS_TOKENS = {"Wheeze", "Rhonchi", "Stridor"}
CAS_ELIGIBLE_TOKENS = {"D", "Wheeze", "Rhonchi", "Stridor"}


def _source_dir(seed: int) -> Path:
    return SOURCE_SEED42 if seed == 42 else SOURCE_ROOT / f"seed_{seed}"


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    target = target.astype(np.int64)
    prediction = prediction.astype(np.int64)
    tn = int(((target == 0) & (prediction == 0)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    tp = int(((target == 1) & (prediction == 1)).sum())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rows": int(len(target)),
        "confusion": [[tn, fp], [fn, tp]],
        "specificity": specificity,
        "sensitivity": sensitivity,
        "balanced_accuracy": (specificity + sensitivity) / 2.0,
    }


def _hf_readout(predictions: dict[str, np.ndarray], annotations: dict[str, dict[str, object]]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    recording_ids = predictions["recording_ids"].astype(str)
    rows = []
    for recording_id in sorted(set(recording_ids.tolist())):
        annotation = annotations[recording_id]
        tokens = set(annotation["tokens"])
        if not tokens & CAS_ELIGIBLE_TOKENS:
            continue
        indices = np.flatnonzero(recording_ids == recording_id)
        rows.append((recording_id, int(bool(tokens & CAS_TOKENS)), float(predictions["attribute_probabilities"][indices, 1].max())))
    target = np.asarray([row[1] for row in rows], dtype=np.int64)
    score = np.asarray([row[2] for row in rows], dtype=np.float64)
    scored = {
        "recording_ids": np.asarray([row[0] for row in rows]),
        "cas_union_targets": target,
        "wheeze_attribute_max_scores": score.astype(np.float32),
    }
    metrics = {
        "hf_cas_auroc": float(roc_auc_score(target, score)),
        "support": int(len(target)),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
        "score_source": "hierarchy attribute Wheeze probability; maximum over three windows",
        "semantic_limit": "fixed ranking proxy for CAS union; not a trained CAS/Rhonchi/Stridor head",
    }
    return scored, metrics


def _kauh_readout(predictions: dict[str, np.ndarray], samples: list[object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    targets = [external._parse_kauh_target(sample) for sample in samples]
    patient_ids = predictions["patient_ids"].astype(str)
    abnormal_scores = predictions["level1_probabilities"][:, 1].astype(np.float64)
    raw_sounds = np.asarray([row["raw_sound"] for row in targets])
    compatible = np.asarray([row["compatible"] for row in targets], dtype=bool)
    level1_targets = np.asarray([row["level1_target"] for row in targets], dtype=np.int64)
    scored_rows = []
    label_free_rows = []
    for patient_id in sorted(set(patient_ids.tolist()), key=lambda value: int(value[1:])):
        indices = np.flatnonzero(patient_ids == patient_id)
        score = float(abnormal_scores[indices].mean())
        prediction = int(score > 0.5)
        label_free_rows.append((patient_id, score, prediction, int(len(indices))))
        if bool(compatible[indices].all()):
            scored_rows.append((patient_id, score, prediction, int(level1_targets[indices[0]]), str(raw_sounds[indices[0]])))
    patient_targets = np.asarray([row[3] for row in scored_rows], dtype=np.int64)
    patient_predictions = np.asarray([row[2] for row in scored_rows], dtype=np.int64)
    report = _binary_metrics(patient_targets, patient_predictions)
    report.update({
        "kauh_patient_ba": report["balanced_accuracy"],
        "threshold": 0.5,
        "tie_policy": "score == 0.5 predicts Normal",
        "score_source": "hierarchy Level1 abnormal probability; mean over B/D/E per patient",
        "recordings": 336,
        "patients": 112,
        "compatible_recordings": int(compatible.sum()),
        "compatible_patients": int(len(scored_rows)),
        "unresolved": ["Crep", "Bronchial", "I C B"],
    })
    label_free = {
        "patient_ids": np.asarray([row[0] for row in label_free_rows]),
        "abnormal_scores": np.asarray([row[1] for row in label_free_rows], dtype=np.float32),
        "predictions": np.asarray([row[2] for row in label_free_rows], dtype=np.int64),
        "view_count": np.asarray([row[3] for row in label_free_rows], dtype=np.int64),
    }
    scored = {
        "patient_ids": np.asarray([row[0] for row in scored_rows]),
        "abnormal_scores": np.asarray([row[1] for row in scored_rows], dtype=np.float32),
        "predictions": patient_predictions,
        "targets": patient_targets,
        "raw_ground_truth": np.asarray([row[4] for row in scored_rows]),
    }
    return {"label_free": label_free, "scored": scored}, report


def run_seed(seed: int, device_name: str, output_root: Path) -> dict[str, object]:
    source_dir = _source_dir(seed)
    output_dir = output_root / f"seed_{seed}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite coarse-SPR post-hoc: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((source_dir / "config.json").read_text())
    selection = json.loads((source_dir / "selection.json").read_text())
    summary = json.loads((source_dir / "run_summary.json").read_text())
    selected_epoch = int(selection["selected_epoch"])
    if selected_epoch != int(summary["selected_epoch"]):
        raise RuntimeError(f"selected epoch mismatch for coarse-SPR seed {seed}")
    config = BenchmarkConfig(repo_root=REPO, variant="coarse_spr", output_dir=output_dir, device=device_name, cpu_threads=4, seed=seed)
    config.validate()
    base_config = config.core.base_config()
    device = torch.device(device_name)
    model, _ = build_components(config.core, device)
    checkpoint_path = source_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    hf_samples = external._load_hf_test_samples(REPO)
    kauh_samples = external._load_kauh_samples(REPO)
    kauh_waveforms = _prepare_waveforms(kauh_samples, base_config)
    hf_label_free = external._predict_hf(model, hf_samples, base_config, device)
    kauh_label_free = external._predict_kauh(model, kauh_samples, kauh_waveforms, base_config, device, selection.get("thresholds", {}))
    _save_predictions(output_dir / "hf_window_predictions_label_free.npz", hf_label_free)
    _save_predictions(output_dir / "kauh_view_predictions_label_free.npz", kauh_label_free)
    annotations = external._parse_hf_annotations(hf_samples)
    hf_scored, hf_metrics = _hf_readout(hf_label_free, annotations)
    kauh_arrays, kauh_metrics = _kauh_readout(kauh_label_free, kauh_samples)
    _save_predictions(output_dir / "hf_recording_predictions_scored.npz", hf_scored)
    _save_predictions(output_dir / "kauh_patient_predictions_label_free.npz", kauh_arrays["label_free"])
    _save_predictions(output_dir / "kauh_patient_predictions_scored.npz", kauh_arrays["scored"])
    metadata = {
        "condition": "coarse_spr",
        "seed": seed,
        "selected_epoch": selected_epoch,
        "source_run": str(source_dir),
        "checkpoint": str(checkpoint_path),
        "evidence_label": EVIDENCE_LABEL,
        "training": False,
        "checkpoint_selection": False,
        "threshold_tuning": False,
        "hf_cas_score": "max hierarchy Wheeze probability over three 5-s windows",
        "kauh_score": "hierarchy Level1 abnormal probability; B/D/E patient mean; >0.5 abnormal; tie Normal",
    }
    _write_json(output_dir / "config.json", {**metadata, "source_config": source_config, "support": {"hf_recordings": 1956, "hf_eligible": 957, "kauh_recordings": 336, "kauh_patients": 112, "kauh_compatible_patients": 86}})
    _write_json(output_dir / "hf_cas_metrics.json", {**metadata, **hf_metrics})
    _write_json(output_dir / "kauh_patient_metrics.json", {**metadata, **kauh_metrics})
    run_summary = {"status": EVIDENCE_LABEL, **metadata, "hf_cas": hf_metrics, "kauh_patient": kauh_metrics}
    _write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps({"seed": seed, "selected_epoch": selected_epoch, "output_dir": str(output_dir), "status": EVIDENCE_LABEL}, sort_keys=True), flush=True)
    return run_summary


def aggregate(rows: list[dict[str, object]], output_root: Path) -> dict[str, object]:
    hf = np.asarray([row["hf_cas"]["hf_cas_auroc"] for row in rows], dtype=np.float64)
    kauh = np.asarray([row["kauh_patient"]["kauh_patient_ba"] for row in rows], dtype=np.float64)
    summary = {
        "status": EVIDENCE_LABEL + "_three_seed",
        "condition": "coarse_spr",
        "seeds": list(SEEDS),
        "selected_epochs": {str(row["seed"]): int(row["selected_epoch"]) for row in rows},
        "hf_cas_auroc": {"mean": float(hf.mean()), "sample_sd": float(hf.std(ddof=1)), "values": [float(v) for v in hf]},
        "kauh_patient_ba": {"mean": float(kauh.mean()), "sample_sd": float(kauh.std(ddof=1)), "values": [float(v) for v in kauh]},
        "support": {"hf_recordings": 1956, "hf_eligible_recordings": 957, "hf_positive": 661, "hf_negative": 296, "kauh_recordings": 336, "kauh_patients": 112, "kauh_compatible_patients": 86},
        "readout": {"hf": "hierarchy attribute Wheeze probability, max over three 5-s windows", "kauh": "hierarchy Level1 abnormal probability, B/D/E patient mean, >0.5 abnormal, tie Normal"},
        "claim_boundary": "Fixed coarse-SPR checkpoint external post-hoc diagnostic; original native control results unchanged.",
        "per_seed": rows,
    }
    _write_json(output_root / "multiseed_summary.json", summary)
    lines = ["# Coarse SPR fixed-checkpoint HF/KAUH post-hoc", "", "| Seed | Selected epoch | HF CAS AUROC (%) | KAUH patient BA (%) |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | {100*row['hf_cas']['hf_cas_auroc']:.2f} | {100*row['kauh_patient']['kauh_patient_ba']:.2f} |")
    lines.extend(["", f"Mean ± sample SD: HF CAS AUROC **{100*hf.mean():.2f}±{100*hf.std(ddof=1):.2f}%**; KAUH patient BA **{100*kauh.mean():.2f}±{100*kauh.std(ddof=1):.2f}%**.", "", "HF: 957 eligible recordings (661/296 positive/negative), three windows each. KAUH: 336 recordings/112 patients, 86 compatible patients. Original coarse-SPR native results and summary were not modified."])
    (output_root / "multiseed_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "output_root": str(args.output_root)}, indent=2))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite coarse-SPR post-hoc output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(seed, args.device, args.output_root) for seed in SEEDS]
    print(json.dumps(aggregate(rows, args.output_root), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
