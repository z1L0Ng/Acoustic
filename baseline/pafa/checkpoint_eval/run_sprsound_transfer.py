"""Frozen PAFA B1 inference on the exact SPRSound B0 inter target."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms

from baseline.checkpoint_eval_common.sprsound_inter import (
    FOUR_LABELS,
    build_label_free_inter_manifest,
    finalize_full_result,
    group_by_recording,
    sha256_file,
    write_csv,
    write_json,
)


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def preprocess(rows, author_repo: Path):
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio

    args = SimpleNamespace(sample_rate=16000, desired_length=5, pad_types="repeat")
    waveform, sample_rate = torchaudio.load(str(rows[0]["audio_path"]))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != args.sample_rate:
        waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
    waveform = audio_transforms.Fade(
        fade_in_len=args.sample_rate // 16,
        fade_out_len=args.sample_rate // 16,
        fade_shape="linear",
    )(waveform)
    duration_ms = waveform.shape[1] / args.sample_rate * 1000
    output = []
    for row in rows:
        if float(row["end_ms"]) > duration_ms + 1:
            raise RuntimeError(f"event exceeds recording: {row['event_id']}")
        start = int(float(row["start_ms"]) / 1000 * args.sample_rate)
        end = int(float(row["end_ms"]) / 1000 * args.sample_rate)
        event = cut_pad_sample_torchaudio(waveform[:, start:end], args).squeeze(0)
        if event.shape != (80000,) or not torch.isfinite(event).all():
            raise RuntimeError(f"invalid PAFA waveform: {row['event_id']} {tuple(event.shape)}")
        output.append(event)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=8)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if not root.name.startswith("pafa_sprsound_transfer_"):
        raise ValueError("result root must be result/pafa_sprsound_transfer_<timestamp>")
    if args.mode == "smoke" and args.max_events != 8:
        raise ValueError("the immutable label-free smoke contract requires exactly 8 events")
    source_repo = (args.source_repo or root / "source" / "repo").resolve()
    checkpoint = args.checkpoint.resolve()
    backbone = args.backbone_checkpoint.resolve()
    if sha256_file(checkpoint) != args.checkpoint_sha256.lower() or sha256_file(backbone) != args.backbone_sha256.lower():
        raise RuntimeError("checkpoint SHA gate failed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = build_label_free_inter_manifest(args.dataset_root)
    rows = manifest if args.mode == "full" else manifest[: args.max_events]
    write_csv(root / "inference_manifest_label_free.csv", rows)

    sys.path.insert(0, str(source_repo))
    from models.beats import BEATsTransferLearningModel

    state = torch.load(checkpoint, map_location="cpu")
    if int(state.get("epoch", -1)) != 27 or not {"model", "classifier"} <= set(state):
        raise RuntimeError("expected verified PAFA epoch-27 checkpoint")
    model = BEATsTransferLearningModel(
        num_target_classes=4, model_path=str(backbone), ft_entire_network=True, spec_transform=None
    )
    classifier = torch.nn.Linear(model.final_feat_dim, 4)
    model_result = model.load_state_dict(normalize_state(state["model"]), strict=True)
    classifier_result = classifier.load_state_dict(normalize_state(state["classifier"]), strict=True)
    if model_result.missing_keys or model_result.unexpected_keys or classifier_result.missing_keys or classifier_result.unexpected_keys:
        raise RuntimeError("PAFA task checkpoint state mismatch")
    model, classifier = model.to(device).eval(), classifier.to(device).eval()
    prediction_rows = []
    pending_images, pending_rows = [], []

    def flush():
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        with torch.inference_mode():
            frame_features = model(batch, training=False)
            logits = classifier(frame_features).mean(dim=1)
            probabilities = torch.softmax(logits, dim=1)
        for row, logit, probability in zip(pending_rows, logits.cpu().numpy(), probabilities.cpu().numpy()):
            pred = int(np.argmax(probability))
            record = {
                **row,
                "pred_narrow_four_label": FOUR_LABELS[pred],
                "pred_narrow_four_index": pred,
                "pred_binary_broad_label": "normal" if pred == 0 else "abnormal",
                "pred_binary_broad_index": 0 if pred == 0 else 1,
                "binary_decision_rule": "collapse_source_four_class_argmax",
            }
            for index, label in enumerate(FOUR_LABELS):
                record[f"logit_{label}"] = float(logit[index])
                record[f"prob_{label}"] = float(probability[index])
            record["prob_binary_normal"] = float(probability[0])
            record["prob_binary_abnormal"] = float(probability[1:].sum())
            prediction_rows.append(record)
        pending_images.clear()
        pending_rows.clear()

    for recording_rows in group_by_recording(rows).values():
        for row, image in zip(recording_rows, preprocess(recording_rows, source_repo)):
            pending_rows.append(row)
            pending_images.append(image)
            if len(pending_images) >= args.batch_size:
                flush()
    flush()
    prediction_rows.sort(key=lambda row: str(row["event_id"]))
    if not all(
        math.isfinite(float(value))
        for row in prediction_rows
        for key, value in row.items()
        if key.startswith(("logit_", "prob_"))
    ):
        raise RuntimeError("non-finite PAFA output")
    write_csv(root / "predictions_label_free.csv", prediction_rows)
    run_receipt = {
        "method_id": "pafa",
        "claim": "verified one-seed ICBHI-test-selected PAFA checkpoint; exploratory zero-target-tuning transfer",
        "mode": args.mode,
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "checkpoint_epoch": 27,
        "source_preprocessing": "16 kHz mono; event crop; 5 s repeat/truncate+fade; raw waveform; --nospec",
        "augmentation": "off",
        "events": len(prediction_rows),
        "target_label_access": "none" if args.mode == "smoke" else "after all logits",
        "target_device_metadata": "not used",
        "binary_probability_rule": "P(normal)=P(class0); P(abnormal)=sum(P(class1:4))",
        "source_numerical_status": "Sp76.884 Se51.402 Score64.143; Score reasonably aligned one seed, componentwise different",
    }
    if args.mode == "smoke":
        write_json(root / "smoke_receipt.json", run_receipt)
        print(f"pafa_sprsound_smoke_ok events={len(prediction_rows)} labels=not_accessed")
        return
    finalize_full_result(root, manifest, prediction_rows, run_receipt)
    print("pafa_sprsound_full_complete_pending_verification events=1429")


if __name__ == "__main__":
    main()
