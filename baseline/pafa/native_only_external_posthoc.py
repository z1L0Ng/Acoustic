"""Fixed zero-target HF/KAUH post-hoc for selected Native-only checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from baseline.multidataset_pipeline.posthoc_native_readout import native_metrics
from baseline.pafa import jh2_hf_kauh_external as external
from baseline.pafa.joint_hierarchy import _prepare_waveforms, _save_predictions, _write_json
from baseline.pafa.table2_benchmark_controls import BenchmarkConfig
from baseline.pafa.table2_clean_controls import build_components


SEEDS = (0, 1, 42)
CLASS_ORDER = ("Normal", "Crackle", "Wheeze", "Both")
CAS_TOKENS = {"Wheeze", "Rhonchi", "Stridor"}
CAS_ELIGIBLE_TOKENS = {"D", "Wheeze", "Rhonchi", "Stridor"}
SOURCE_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only"
)
OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only_external_posthoc"
)


def native_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("non-finite ICBHI native softmax probability")
    return probabilities


def native_wheeze_marginal(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[-1] != 4:
        raise ValueError("ICBHI native probabilities must use four classes")
    score = probabilities[..., 2] + probabilities[..., 3]
    if not np.isfinite(score).all():
        raise FloatingPointError("non-finite native Wheeze+Both score")
    return score


def hf_recording_readout(
    window_predictions: Mapping[str, np.ndarray],
    annotations: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    recording_ids = window_predictions["recording_ids"].astype(str)
    window_indices = window_predictions["window_indices"].astype(np.int64)
    window_scores = window_predictions["wheeze_marginal_scores"].astype(np.float64)
    label_free_rows = []
    scored_rows = []
    for recording_id in sorted(set(recording_ids.tolist())):
        indices = np.flatnonzero(recording_ids == recording_id)
        label_free_rows.append(
            (
                recording_id,
                float(window_scores[indices].max()),
                int(len(indices)),
                tuple(int(value) for value in window_indices[indices]),
            )
        )
        if annotations is not None:
            tokens = set(annotations[recording_id]["tokens"])
            if tokens & CAS_ELIGIBLE_TOKENS:
                scored_rows.append(
                    (
                        recording_id,
                        int(bool(tokens & CAS_TOKENS)),
                        float(window_scores[indices].max()),
                    )
                )
    label_free = {
        "recording_ids": np.asarray([row[0] for row in label_free_rows]),
        "cas_ranking_scores": np.asarray(
            [row[1] for row in label_free_rows], dtype=np.float32
        ),
        "window_count": np.asarray([row[2] for row in label_free_rows], dtype=np.int64),
        "window_indices": np.asarray([row[3] for row in label_free_rows], dtype=np.int64),
    }
    if annotations is None:
        return label_free, None
    scored = {
        "recording_ids": np.asarray([row[0] for row in scored_rows]),
        "cas_union_targets": np.asarray([row[1] for row in scored_rows], dtype=np.int64),
        "cas_ranking_scores": np.asarray([row[2] for row in scored_rows], dtype=np.float32),
    }
    return label_free, scored


def hf_cas_metrics(scored: Mapping[str, np.ndarray]) -> dict[str, object]:
    target = scored["cas_union_targets"].astype(np.int64)
    score = scored["cas_ranking_scores"].astype(np.float64)
    if not np.isfinite(score).all():
        raise FloatingPointError("non-finite Native-only HF CAS score")
    return {
        "hf_cas_auroc": float(roc_auc_score(target, score)),
        "support": int(len(target)),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
        "score_source": "ICBHI native softmax P(Wheeze)+P(Both), maximum over three windows",
        "semantic_limit": "fixed ranking proxy for CAS union; not a trained CAS/Rhonchi/Stridor head",
    }


def kauh_patient_readout(
    view_predictions: Mapping[str, np.ndarray],
    view_targets: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    patient_ids = view_predictions["patient_ids"].astype(str)
    scores = view_predictions["abnormal_scores"].astype(np.float64)
    sample_ids = view_predictions["sample_ids"].astype(str)
    rows = []
    scored_rows = []
    for patient_id in sorted(set(patient_ids.tolist()), key=lambda value: int(value[1:])):
        indices = np.flatnonzero(patient_ids == patient_id)
        patient_score = float(scores[indices].mean())
        prediction = int(patient_score > 0.5)
        rows.append((patient_id, patient_score, prediction, int(len(indices))))
        if view_targets is not None:
            targets = [view_targets[str(sample_ids[index])] for index in indices]
            compatible = all(bool(value["compatible"]) for value in targets)
            if compatible:
                level1 = {int(value["level1_target"]) for value in targets}
                if len(level1) != 1:
                    raise RuntimeError("KAUH sibling target mismatch")
                raw_sound = {str(value["raw_sound"]) for value in targets}
                if len(raw_sound) != 1:
                    raise RuntimeError("KAUH sibling raw label mismatch")
                scored_rows.append(
                    (
                        patient_id,
                        patient_score,
                        prediction,
                        level1.pop(),
                        raw_sound.pop(),
                    )
                )
    label_free = {
        "patient_ids": np.asarray([row[0] for row in rows]),
        "abnormal_scores": np.asarray([row[1] for row in rows], dtype=np.float32),
        "predictions": np.asarray([row[2] for row in rows], dtype=np.int64),
        "view_count": np.asarray([row[3] for row in rows], dtype=np.int64),
    }
    if view_targets is None:
        return label_free, None
    scored = {
        "patient_ids": np.asarray([row[0] for row in scored_rows]),
        "abnormal_scores": np.asarray([row[1] for row in scored_rows], dtype=np.float32),
        "predictions": np.asarray([row[2] for row in scored_rows], dtype=np.int64),
        "targets": np.asarray([row[3] for row in scored_rows], dtype=np.int64),
        "raw_ground_truth": np.asarray([row[4] for row in scored_rows]),
    }
    return label_free, scored


def kauh_metrics(scored: Mapping[str, np.ndarray]) -> dict[str, object]:
    report = native_metrics(
        scored["targets"].astype(np.int64),
        scored["predictions"].astype(np.int64),
        ("normal", "abnormal"),
    )
    return {
        **report,
        "kauh_patient_ba": float(report["average_score"]),
        "threshold": 0.5,
        "tie_policy": "score == 0.5 predicts Normal",
        "score_source": "mean over B/D/E of 1-P(Normal) from ICBHI native softmax",
    }


def _selected_source(repo_root: Path, seed: int) -> dict[str, object]:
    run_dir = repo_root / SOURCE_ROOT_RELATIVE / f"seed_{seed}"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    if summary.get("status") != "complete_test_selected_benchmark_control":
        raise RuntimeError(f"Native-only source seed {seed} is not complete")
    if int(summary["selected_epoch"]) != int(selection["selected_epoch"]):
        raise RuntimeError(f"Native-only selected epoch mismatch for seed {seed}")
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "selected_epoch": int(summary["selected_epoch"]),
        "source_status": summary["status"],
    }


def _predict_kauh_native(
    model: torch.nn.Module,
    samples: Sequence[object],
    waveform_store: Mapping[str, torch.Tensor],
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "patient_ids": [],
        "filter_modes": [],
        "file_names": [],
        "icbhi_native_logits": [],
        "icbhi_native_probabilities": [],
        "abnormal_scores": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), config.core.batch_size):
            current = list(samples[start : start + config.core.batch_size])
            waveform = torch.stack(
                [waveform_store[row.sample_id] for row in current]
            ).to(device)
            output, _ = model(waveform, training=False)
            logits = output["icbhi_native"].float().cpu()
            probability = native_probabilities(logits).numpy()
            fields["sample_ids"].append(np.asarray([row.sample_id for row in current]))
            fields["patient_ids"].append(np.asarray([row.group_id for row in current]))
            fields["filter_modes"].append(
                np.asarray([str(row.metadata["filter_mode"]) for row in current])
            )
            fields["file_names"].append(
                np.asarray([Path(row.audio_path).name for row in current])
            )
            fields["icbhi_native_logits"].append(logits.numpy())
            fields["icbhi_native_probabilities"].append(probability)
            fields["abnormal_scores"].append(1.0 - probability[:, 0])
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def _predict_hf_native(
    model: torch.nn.Module,
    samples: Sequence[object],
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "recording_ids": [],
        "window_indices": [],
        "source_start_s": [],
        "source_end_s": [],
        "file_names": [],
        "icbhi_native_logits": [],
        "icbhi_native_probabilities": [],
        "wheeze_marginal_scores": [],
    }
    base = config.core.base_config()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), config.core.batch_size):
            current = list(samples[start : start + config.core.batch_size])
            windows = []
            recording_ids = []
            window_indices = []
            starts = []
            ends = []
            file_names = []
            for sample in current:
                current_windows, time_map = external._load_hf_windows(
                    Path(sample.audio_path), base.sample_rate
                )
                for index, (window, (source_start, source_end)) in enumerate(
                    zip(current_windows, time_map)
                ):
                    windows.append(window)
                    recording_ids.append(str(sample.metadata["recording_id"]))
                    window_indices.append(index)
                    starts.append(source_start)
                    ends.append(source_end)
                    file_names.append(Path(sample.audio_path).name)
            output, _ = model(torch.stack(windows).to(device), training=False)
            logits = output["icbhi_native"].float().cpu()
            probability = native_probabilities(logits).numpy()
            ids = np.asarray(
                [
                    f"hf:test:{recording}::window_{index:02d}"
                    for recording, index in zip(recording_ids, window_indices)
                ]
            )
            fields["prediction_ids"].append(ids)
            fields["sample_ids"].append(
                np.asarray([f"hf:test:{value}" for value in recording_ids])
            )
            fields["recording_ids"].append(np.asarray(recording_ids))
            fields["window_indices"].append(
                np.asarray(window_indices, dtype=np.int64)
            )
            fields["source_start_s"].append(np.asarray(starts, dtype=np.float32))
            fields["source_end_s"].append(np.asarray(ends, dtype=np.float32))
            fields["file_names"].append(np.asarray(file_names))
            fields["icbhi_native_logits"].append(logits.numpy())
            fields["icbhi_native_probabilities"].append(probability)
            fields["wheeze_marginal_scores"].append(
                native_wheeze_marginal(probability).astype(np.float32)
            )
    return {key: np.concatenate(value, axis=0) for key, value in fields.items()}


def run_seed(repo_root: Path, seed: int, output_dir: Path, device_name: str) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Native-only post-hoc: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_source(repo_root, seed)
    config = BenchmarkConfig(
        repo_root=repo_root,
        variant="native_only",
        output_dir=selected["run_dir"],
        device=device_name,
        cpu_threads=4,
        seed=seed,
    )
    config.validate()
    _write_json(
        output_dir / "config.json",
        {
            "status": "approved_fixed_native_head_external_posthoc",
            "seed": seed,
            "source_run": str(selected["run_dir"]),
            "checkpoint": str(selected["checkpoint"]),
            "selected_epoch": selected["selected_epoch"],
            "source_head": "icbhi_native",
            "class_order": list(CLASS_ORDER),
            "hf_score": "per-window P(Wheeze)+P(Both), then recording max over three windows",
            "kauh_score": "per-view 1-P(Normal), then patient B/D/E mean; >0.5 abnormal",
            "target_training": False,
            "target_selection": False,
            "target_threshold_fit": False,
            "source_evidence_boundary": "ICBHI-official-test-selected Native-only source checkpoint",
            "device": device_name,
            "precision": "FP32",
        },
    )

    device = torch.device(device_name)
    model, _ = build_components(config.core, device)
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    hf_samples = external._load_hf_test_samples(repo_root)
    kauh_samples = external._load_kauh_samples(repo_root)
    kauh_waveforms = _prepare_waveforms(kauh_samples, config.core.base_config())

    hf_windows = _predict_hf_native(model, hf_samples, config, device)
    hf_windows_path = output_dir / "hf_window_predictions_label_free.npz"
    _save_predictions(hf_windows_path, hf_windows)
    hf_recording_label_free, _ = hf_recording_readout(hf_windows)
    hf_recording_label_free_path = output_dir / "hf_recording_predictions_label_free.npz"
    _save_predictions(hf_recording_label_free_path, hf_recording_label_free)

    kauh_views = _predict_kauh_native(
        model, kauh_samples, kauh_waveforms, config, device
    )
    kauh_views_path = output_dir / "kauh_view_predictions_label_free.npz"
    _save_predictions(kauh_views_path, kauh_views)
    kauh_patient_label_free, _ = kauh_patient_readout(kauh_views)
    kauh_patient_label_free_path = output_dir / "kauh_patient_predictions_label_free.npz"
    _save_predictions(kauh_patient_label_free_path, kauh_patient_label_free)

    hf_annotations = external._parse_hf_annotations(hf_samples)
    _, hf_scored = hf_recording_readout(hf_windows, hf_annotations)
    if hf_scored is None:
        raise RuntimeError("HF CAS scoring did not produce predictions")
    hf_metrics = hf_cas_metrics(hf_scored)
    if (hf_metrics["support"], hf_metrics["positive"], hf_metrics["negative"]) != (
        957,
        661,
        296,
    ):
        raise RuntimeError("Native-only HF CAS support changed")
    hf_scored_path = output_dir / "hf_recording_predictions_scored.npz"
    _save_predictions(hf_scored_path, hf_scored)
    _write_json(output_dir / "hf_cas_metrics.json", hf_metrics)

    view_targets = {
        sample.sample_id: external._parse_kauh_target(sample) for sample in kauh_samples
    }
    kauh_view_scored = {
        **kauh_views,
        "raw_ground_truth": np.asarray(
            [view_targets[str(value)]["raw_sound"] for value in kauh_views["sample_ids"]]
        ),
        "compatible": np.asarray(
            [view_targets[str(value)]["compatible"] for value in kauh_views["sample_ids"]],
            dtype=bool,
        ),
        "targets": np.asarray(
            [view_targets[str(value)]["level1_target"] for value in kauh_views["sample_ids"]],
            dtype=np.int64,
        ),
        "predictions": (kauh_views["abnormal_scores"] > 0.5).astype(np.int64),
    }
    kauh_view_scored_path = output_dir / "kauh_view_predictions_scored.npz"
    _save_predictions(kauh_view_scored_path, kauh_view_scored)
    _, kauh_patient_scored = kauh_patient_readout(kauh_views, view_targets)
    if kauh_patient_scored is None:
        raise RuntimeError("KAUH scoring did not produce patient predictions")
    if len(kauh_patient_scored["patient_ids"]) != 86:
        raise RuntimeError("Native-only KAUH compatible patient support changed")
    kauh_patient_scored_path = output_dir / "kauh_patient_predictions_scored.npz"
    _save_predictions(kauh_patient_scored_path, kauh_patient_scored)
    kauh_report = kauh_metrics(kauh_patient_scored)
    _write_json(output_dir / "kauh_patient_metrics.json", kauh_report)

    summary = {
        "status": "complete_native_only_fixed_head_external_posthoc",
        "seed": seed,
        "selected_epoch": selected["selected_epoch"],
        "source_run": str(selected["run_dir"]),
        "checkpoint": str(selected["checkpoint"]),
        "hf_cas": hf_metrics,
        "kauh_patient": kauh_report,
        "outputs": {
            "hf_windows_label_free": str(hf_windows_path),
            "hf_recording_label_free": str(hf_recording_label_free_path),
            "hf_recording_scored": str(hf_scored_path),
            "kauh_views_label_free": str(kauh_views_path),
            "kauh_views_scored": str(kauh_view_scored_path),
            "kauh_patients_label_free": str(kauh_patient_label_free_path),
            "kauh_patients_scored": str(kauh_patient_scored_path),
        },
        "zero_target_boundary": {
            "target_training": False,
            "target_selection": False,
            "target_threshold_fit": False,
            "labels_loaded_after_label_free_prediction_write": True,
        },
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _stat(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite Native-only external metric")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def aggregate(repo_root: Path, output_root: Path) -> dict[str, object]:
    rows = []
    for seed in SEEDS:
        path = output_root / f"seed_{seed}" / "run_summary.json"
        if not path.is_file():
            raise RuntimeError(f"missing Native-only external seed {seed}")
        row = json.loads(path.read_text())
        if row.get("status") != "complete_native_only_fixed_head_external_posthoc":
            raise RuntimeError(f"Native-only external seed {seed} is not complete")
        rows.append(row)
    summary = {
        "status": "complete_native_only_fixed_head_external_posthoc_three_seed",
        "seeds": list(SEEDS),
        "n": 3,
        "sample_sd_ddof": 1,
        "hf_cas_auroc": _stat([row["hf_cas"]["hf_cas_auroc"] for row in rows]),
        "kauh_patient_ba": _stat(
            [row["kauh_patient"]["kauh_patient_ba"] for row in rows]
        ),
        "support": {
            "hf": [
                {
                    key: int(row["hf_cas"][key])
                    for key in ("support", "positive", "negative")
                }
                for row in rows
            ],
            "kauh_compatible_patients": [
                int(row["kauh_patient"]["rows"]) for row in rows
            ],
        },
        "per_seed": rows,
        "boundary": (
            "separate fixed-head zero-target post-hoc; does not modify Native-only "
            "training, selection, native predictions, thresholds, or primary summary"
        ),
    }
    _write_json(output_root / "multiseed_summary.json", summary)
    return summary


def run_queue(repo_root: Path, output_root: Path, device_name: str) -> dict[str, object]:
    for seed in SEEDS:
        run_seed(repo_root, seed, output_root / f"seed_{seed}", device_name)
    return aggregate(repo_root, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root or repo_root / OUTPUT_ROOT_RELATIVE
    if args.aggregate:
        print(json.dumps(aggregate(repo_root, output_root), indent=2))
        return
    if args.queue:
        if not args.run:
            print(json.dumps({
                "status": "CODE_READY_NOT_RUN",
                "seeds": list(SEEDS),
                "source_root": str(repo_root / SOURCE_ROOT_RELATIVE),
                "output_root": str(output_root),
            }, indent=2))
            return
        print(json.dumps(run_queue(repo_root, output_root, args.device), indent=2))
        return
    if args.seed is None:
        parser.error("--seed is required unless --queue or --aggregate is used")
    if not args.run:
        print(json.dumps({
            "status": "CODE_READY_NOT_RUN",
            "seed": args.seed,
            "source_run": str(repo_root / SOURCE_ROOT_RELATIVE / f"seed_{args.seed}"),
            "output_dir": str(output_root / f"seed_{args.seed}"),
            "device": args.device,
        }, indent=2))
        return
    print(json.dumps(run_seed(
        repo_root, args.seed, output_root / f"seed_{args.seed}", args.device
    ), indent=2))


if __name__ == "__main__":
    main()
