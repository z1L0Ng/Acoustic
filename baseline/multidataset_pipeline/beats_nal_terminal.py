"""Terminal ICBHI and SPRSound readouts for selected BEATs NAL runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    Sample,
    load_terminal_spr_test_targets,
)
from baseline.multidataset_pipeline.beats_nal_protocol import (
    BEATsCore2Model,
    BEATsNALConfig,
    HierarchicalLossConfig,
    WaveformAugmentationConfig,
    WaveformNormalizationConfig,
    decode_icbhi_flat4,
    decode_icbhi_hierarchical_flat4,
    load_core_samples,
    normalize_waveform,
)
from baseline.multidataset_pipeline.beats_window_encoder import load_local_beats_model
from baseline.multidataset_pipeline.m_unified import map_native_sample, set_determinism
from baseline.multidataset_pipeline.posthoc_native_readout import (
    ICBHI_LABELS,
    native_metrics,
)
from baseline.multidataset_pipeline.real_subtrain_provider import (
    LANE_BY_CANONICAL_DATASET,
    load_sample_waveform,
)
from baseline.multidataset_pipeline.sliding_window import collate_sliding_windows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_config(
    repo_root: Path,
    run_dir: Path,
    device: str,
    source_repo: Path | None = None,
    checkpoint: Path | None = None,
) -> BEATsNALConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    return BEATsNALConfig(
        repo_root=repo_root,
        source_repo=source_repo or Path(raw["source_repo"]),
        checkpoint=checkpoint or Path(raw["checkpoint"]),
        output_dir=run_dir,
        device=device,
        seed=int(raw["seed"]),
        sample_rate=int(raw["sample_rate"]),
        window_seconds=float(raw["window_seconds"]),
        stride_seconds=float(raw["stride_seconds"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        cpu_threads=int(raw["cpu_threads"]),
        encoder_scope=str(raw["encoder_scope"]),
        backbone_learning_rate=float(raw["backbone_learning_rate"]),
        head_learning_rate=float(raw["head_learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        normalization=WaveformNormalizationConfig(**raw["normalization"]),
        augmentation=WaveformAugmentationConfig(**raw["augmentation"]),
        loss=HierarchicalLossConfig(**raw["loss"]),
    )


def _test_samples(samples: Sequence[Sample]) -> dict[str, tuple[Sample, ...]]:
    output = {
        dataset: tuple(
            sorted(
                (
                    sample
                    for sample in samples
                    if sample.dataset == dataset and sample.partition == "test"
                ),
                key=lambda sample: sample.sample_id,
            )
        )
        for dataset in ("icbhi", "sprsound")
    }
    counts = {dataset: len(rows) for dataset, rows in output.items()}
    if counts != {"icbhi": 2756, "sprsound": 1429}:
        raise RuntimeError(f"terminal support changed: {counts}")
    return output


def _infer_label_free(
    model: BEATsCore2Model,
    samples_by_dataset: Mapping[str, Sequence[Sample]],
    config: BEATsNALConfig,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "prediction_ids": [],
        "sample_ids": [],
        "dataset_ids": [],
        "group_ids": [],
        "file_names": [],
        "level1_logits": [],
        "level1_probabilities": [],
        "level1_predictions": [],
        "attribute_logits": [],
        "attribute_probabilities": [],
    }
    device = torch.device(config.device)
    model.eval()
    with torch.no_grad():
        for dataset in ("icbhi", "sprsound"):
            samples = samples_by_dataset[dataset]
            for start in range(0, len(samples), config.batch_size):
                current = samples[start : start + config.batch_size]
                waveforms = []
                for sample in current:
                    lane = LANE_BY_CANONICAL_DATASET[sample.dataset]
                    decoded = load_sample_waveform(
                        sample,
                        lane,
                        outer_test_accessed=True,
                    )[0]
                    waveforms.append(
                        decoded
                        if config.normalization.mode == "none"
                        else replace(
                            decoded,
                            waveform=normalize_waveform(
                                decoded.waveform,
                                config.normalization,
                            ).contiguous(),
                        )
                    )
                windows = collate_sliding_windows(
                    waveforms,
                    window_samples=config.window_samples,
                    stride_samples=config.stride_samples,
                ).to(device)
                output = model(windows)
                level1_logits = output["level1"].detach().cpu()
                attributes = torch.stack(
                    (output["crackle"], output["wheeze"]), dim=-1
                ).detach().cpu()
                ids = np.asarray([sample.sample_id for sample in current])
                fields["prediction_ids"].append(ids)
                fields["sample_ids"].append(ids)
                fields["dataset_ids"].append(np.asarray([dataset] * len(current)))
                fields["group_ids"].append(
                    np.asarray([sample.group_id for sample in current])
                )
                fields["file_names"].append(
                    np.asarray([Path(sample.audio_path).name for sample in current])
                )
                fields["level1_logits"].append(level1_logits.numpy())
                fields["level1_probabilities"].append(
                    torch.softmax(level1_logits, dim=-1).numpy()
                )
                fields["level1_predictions"].append(
                    level1_logits.argmax(dim=-1).numpy()
                )
                fields["attribute_logits"].append(attributes.numpy())
                fields["attribute_probabilities"].append(
                    torch.sigmoid(attributes).numpy()
                )
    return {key: np.concatenate(values, axis=0) for key, values in fields.items()}


def _attach_targets(
    predictions: Mapping[str, np.ndarray],
    ordered_samples: Sequence[Sample],
    spr_targets: Mapping[str, Mapping[str, int]],
) -> dict[str, np.ndarray]:
    if list(predictions["sample_ids"]) != [sample.sample_id for sample in ordered_samples]:
        raise RuntimeError("terminal prediction/sample order changed")
    targets = np.zeros((len(ordered_samples), 3), dtype=np.float32)
    eligible = np.zeros((len(ordered_samples), 3), dtype=bool)
    raw_labels = []
    for row, sample in enumerate(ordered_samples):
        mapped = map_native_sample(sample, spr_terminal_targets=spr_targets)
        raw_labels.append(str(mapped["raw_label"]))
        for column, node in enumerate(("level1", "crackle", "wheeze")):
            if bool(mapped[f"{node}_eligible"]):
                eligible[row, column] = True
                targets[row, column] = float(mapped[f"{node}_target"])
    return {
        **predictions,
        "raw_ground_truth": np.asarray(raw_labels),
        "targets": targets,
        "eligible": eligible,
    }


def _score(
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    icbhi = predictions["dataset_ids"] == "icbhi"
    icbhi_target = np.asarray(
        [ICBHI_LABELS.index(str(value)) for value in predictions["raw_ground_truth"][icbhi]],
        dtype=np.int64,
    )
    icbhi_prediction = decode_icbhi_hierarchical_flat4(
        predictions["level1_predictions"][icbhi],
        predictions["attribute_probabilities"][icbhi],
        thresholds,
    )
    icbhi_metrics = native_metrics(icbhi_target, icbhi_prediction, ICBHI_LABELS)
    icbhi_metrics.update(
        {
            "task": "ICBHI official-test flat4",
            "protocol": "official recording split 60/40; 2756 respiratory cycles",
            "decoder": (
                "Level1 Normal->Normal; Level1 Abnormal->Crackle/Wheeze/Both "
                "from shared validation thresholds; neither attribute over "
                "threshold->larger probability-minus-threshold margin; "
                "Crackle wins ties"
            ),
            "thresholds": dict(thresholds),
            "official_score": icbhi_metrics["icbhi_score"],
        }
    )
    bits_only_prediction = decode_icbhi_flat4(
        predictions["attribute_probabilities"][icbhi],
        thresholds,
    )
    bits_only_metrics = native_metrics(
        icbhi_target,
        bits_only_prediction,
        ICBHI_LABELS,
    )
    bits_only_metrics.update(
        {
            "task": "ICBHI official-test flat4 diagnostic ablation",
            "decoder": (
                "Crackle/Wheeze bits only: 00 Normal, 10 Crackle, "
                "01 Wheeze, 11 Both"
            ),
            "thresholds": dict(thresholds),
            "official_score": bits_only_metrics["icbhi_score"],
        }
    )

    spr = predictions["dataset_ids"] == "sprsound"
    spr_target = predictions["targets"][spr, 0].astype(np.int64)
    spr_prediction = predictions["level1_predictions"][spr].astype(np.int64)
    spr_metrics = native_metrics(
        spr_target,
        spr_prediction,
        ("normal", "abnormal"),
    )
    spr_metrics.update(
        {
            "task": "SPRSound BioCAS2022 Task1-1 Normal/Adventitious",
            "protocol": "official inter-subject test; 1429 respiratory events",
            "average_score_as": spr_metrics["average_score"],
            "harmonic_score_hs": spr_metrics["harmonic_score"],
            "official_score": (
                spr_metrics["average_score"] + spr_metrics["harmonic_score"]
            )
            / 2,
            "task1_2_raw7": "not produced by the shared three-node head",
        }
    )
    return {
        "icbhi_flat4": icbhi_metrics,
        "icbhi_flat4_bits_only_ablation": bits_only_metrics,
        "sprsound_inter_task1_1": spr_metrics,
    }


def evaluate_run(
    repo_root: Path,
    run_dir: Path,
    device: str,
    source_repo: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    terminal_dir = run_dir / "terminal"
    if terminal_dir.exists():
        raise RuntimeError(f"terminal output already exists: {terminal_dir}")
    config = _load_config(
        repo_root,
        run_dir,
        device,
        source_repo=source_repo,
        checkpoint=checkpoint,
    )
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    set_determinism(config.seed)
    samples = load_core_samples(repo_root)
    test_by_dataset = _test_samples(samples)
    ordered_samples = [
        sample
        for dataset in ("icbhi", "sprsound")
        for sample in test_by_dataset[dataset]
    ]

    beats = load_local_beats_model(
        config.source_repo,
        config.checkpoint,
        device=torch.device(device),
        trainable=False,
    )
    model = BEATsCore2Model(beats, encoder_trainable=False).to(device)
    selected = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
    model.load_state_dict(selected["model"])

    label_free = _infer_label_free(model, test_by_dataset, config)
    terminal_dir.mkdir(parents=True)
    label_free_path = terminal_dir / "selected_test_predictions_label_free.npz"
    np.savez(label_free_path, **label_free)

    spr_targets = load_terminal_spr_test_targets(samples, include_checksums=False)
    scored = _attach_targets(label_free, ordered_samples, spr_targets)
    scored_path = terminal_dir / "selected_test_predictions_scored.npz"
    np.savez(scored_path, **scored)

    selection = json.loads((run_dir / "validation_selection.json").read_text())
    thresholds = {
        node: float(value)
        for node, value in selection["shared_attribute_thresholds"].items()
    }
    metrics = _score(scored, thresholds)
    payload = {
        "status": "terminal_native_scores_complete",
        "condition": json.loads((run_dir / "run_summary.json").read_text())["condition"],
        "selected_epoch": int(selection["selected_epoch"]),
        "threshold_source": "saved validation_selection.json; no test tuning",
        "outer_test_accessed": True,
        "terminal_targets_loaded_after_label_free_prediction_write": True,
        "prediction_support": {"icbhi": 2756, "sprsound": 1429},
        **metrics,
    }
    _write_json(terminal_dir / "native_metrics.json", payload)
    _write_json(
        terminal_dir / "run_summary.json",
        {
            "status": payload["status"],
            "condition": payload["condition"],
            "selected_epoch": payload["selected_epoch"],
            "outer_test_accessed": True,
            "icbhi_official_score": payload["icbhi_flat4"]["official_score"],
            "sprsound_task1_1_official_score": payload["sprsound_inter_task1_1"]["official_score"],
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = evaluate_run(
        args.repo_root,
        args.run_dir,
        args.device,
        source_repo=args.source_repo,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
