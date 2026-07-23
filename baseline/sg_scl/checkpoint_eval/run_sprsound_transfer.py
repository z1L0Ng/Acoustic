"""Frozen SG-SCL B2 inference on the exact SPRSound B0 inter target."""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from acoustic.evaluation.sprsound_inter import (
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


def preprocess(rows: list[dict[str, object]], author_repo: Path) -> list[torch.Tensor]:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    args = SimpleNamespace(
        sample_rate=16000,
        desired_length=8,
        pad_types="repeat",
        model="ast",
    )
    resize = transforms.Resize(size=(798, 128))
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
    output: list[torch.Tensor] = []
    for row in rows:
        if float(row["end_ms"]) > duration_ms + 1:
            raise RuntimeError(f"event exceeds recording: {row['event_id']}")
        start = int(float(row["start_ms"]) / 1000 * args.sample_rate)
        end = int(float(row["end_ms"]) / 1000 * args.sample_rate)
        event = cut_pad_sample_torchaudio(waveform[:, start:end], args)
        fbank = generate_fbank(args, event, args.sample_rate, n_mels=128)
        image = resize(transforms.ToTensor()(fbank))
        if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
            raise RuntimeError(f"invalid SG-SCL fbank: {row['event_id']} {tuple(image.shape)}")
        output.append(image)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=8)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if not root.name.startswith("sg_scl_sprsound_transfer_"):
        raise ValueError("result root must be result/sg_scl_sprsound_transfer_<timestamp>")
    if args.mode == "smoke" and args.max_events != 8:
        raise ValueError("the immutable label-free smoke contract requires exactly 8 events")
    source_repo = (args.source_repo or root / "source" / "repo").resolve()
    checkpoint = args.checkpoint.resolve()
    if sha256_file(checkpoint) != args.checkpoint_sha256.lower():
        raise RuntimeError("SG-SCL task checkpoint SHA gate failed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = build_label_free_inter_manifest(args.dataset_root)
    rows = manifest if args.mode == "full" else manifest[: args.max_events]
    write_csv(root / "inference_manifest_label_free.csv", rows)

    sys.path.insert(0, str(source_repo))
    from models.ast import ASTModel

    state = torch.load(checkpoint, map_location="cpu")
    if int(state.get("epoch", -1)) != 27 or not {"model", "classifier"} <= set(state):
        raise RuntimeError("expected verified SG-SCL epoch-27 checkpoint")
    # False/False prevents any initialization download. The full predictive
    # state is loaded strictly from the task checkpoint immediately below.
    model = ASTModel(
        label_dim=4,
        input_fdim=798,
        input_tdim=128,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
    )
    classifier = deepcopy(model.mlp_head)
    model_result = model.load_state_dict(normalize_state(state["model"]), strict=True)
    classifier_result = classifier.load_state_dict(normalize_state(state["classifier"]), strict=True)
    if model_result.missing_keys or model_result.unexpected_keys or classifier_result.missing_keys or classifier_result.unexpected_keys:
        raise RuntimeError("SG-SCL predictive checkpoint state mismatch")
    model, classifier = model.to(device).eval(), classifier.to(device).eval()
    inference_args = SimpleNamespace(domain_adaptation=False, domain_adaptation2=True)
    prediction_rows: list[dict[str, object]] = []
    pending_images: list[torch.Tensor] = []
    pending_rows: list[dict[str, object]] = []

    def flush() -> None:
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        with torch.inference_mode():
            # This is the author's validate() path. training=False returns the
            # audio embedding only; no target device/domain label is consumed.
            features = model(batch, args=inference_args, training=False)
            logits = classifier(features)
            probabilities = torch.softmax(logits, dim=1)
        for row, logit, probability in zip(pending_rows, logits.cpu().numpy(), probabilities.cpu().numpy()):
            pred = int(np.argmax(probability))
            record: dict[str, object] = {
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
        raise RuntimeError("non-finite SG-SCL output")
    write_csv(root / "predictions_label_free.csv", prediction_rows)
    run_receipt = {
        "method_id": "sg_scl",
        "claim": "verified one-seed ICBHI-test-selected SG-SCL checkpoint; exploratory zero-target-tuning transfer",
        "mode": args.mode,
        "checkpoint_sha256": args.checkpoint_sha256.lower(),
        "checkpoint_epoch": 27,
        "source_preprocessing": "16 kHz mono; event crop; 8 s repeat/truncate+fade; 128-bin fbank; resize 798x128",
        "augmentation": "off",
        "events": len(prediction_rows),
        "target_label_access": "none" if args.mode == "smoke" else "after all logits",
        "target_device_metadata": "not used or invented; author validation audio-classifier path",
        "source_training_boundary": "metadata/device-aware SG-SCL source training",
        "binary_probability_rule": "P(normal)=P(class0); P(abnormal)=sum(P(class1:4))",
        "source_numerical_status": "Sp74.984 Se46.984 Score60.984; all within about 0.6 SD for one seed",
    }
    if args.mode == "smoke":
        write_json(root / "smoke_receipt.json", run_receipt)
        print(f"sg_scl_sprsound_smoke_ok events={len(prediction_rows)} labels=not_accessed")
        return
    finalize_full_result(root, manifest, prediction_rows, run_receipt)
    print("sg_scl_sprsound_full_complete_pending_verification events=1429")


if __name__ == "__main__":
    main()
