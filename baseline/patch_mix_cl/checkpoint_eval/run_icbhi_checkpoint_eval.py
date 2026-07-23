"""Run checkpoint-only Patch-Mix inference on the official ICBHI test cycles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from .restricted_checkpoint import restricted_torch_load


LABELS = ["normal", "crackle", "wheeze", "both"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
AUTHOR_POSTED = {"specificity": 86.19, "sensitivity": 38.15, "icbhi_score": 62.17}
PAPER_MEAN = {"specificity": 81.66, "sensitivity": 43.07, "icbhi_score": 62.37}
PAPER_SD = {"specificity": 3.83, "sensitivity": 2.80, "icbhi_score": 0.61}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["official_split"] == "test"]
    if len(rows) != 2756 or len({row["cycle_id"] for row in rows}) != 2756:
        raise RuntimeError("official test manifest must contain 2,756 unique cycle IDs")
    for row in rows:
        expected = LABELS[(int(row["crackle_flag"]) > 0) + 2 * (int(row["wheeze_flag"]) > 0)]
        if expected != row["native_four_class_label"]:
            raise RuntimeError(f"label mismatch for {row['cycle_id']}")
    return rows


def build_model(author_repo: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(author_repo))
    from models.ast import ASTModel

    checkpoint = restricted_torch_load(checkpoint_path)
    model = ASTModel(
        label_dim=4,
        fstride=10,
        tstride=10,
        input_fdim=798,
        input_tdim=128,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
        mix_beta=1.0,
    )
    classifier = torch.nn.Sequential(
        torch.nn.LayerNorm(model.final_feat_dim),
        torch.nn.Linear(model.final_feat_dim, 4),
    )
    model_result = model.load_state_dict(checkpoint["model"], strict=True)
    classifier_result = classifier.load_state_dict(checkpoint["classifier"], strict=True)
    if model_result.missing_keys or model_result.unexpected_keys:
        raise RuntimeError(f"model state mismatch: {model_result}")
    if classifier_result.missing_keys or classifier_result.unexpected_keys:
        raise RuntimeError(f"classifier state mismatch: {classifier_result}")
    return model.to(device).eval(), classifier.to(device).eval(), int(checkpoint["epoch"])


def preprocess_recording(rows: list[dict[str, str]], generate_fbank, cut_pad, args):
    audio_path = Path(rows[0]["audio_path"])
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != args.sample_rate:
        waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
    fade_samples = int(args.sample_rate / 16)
    waveform = audio_transforms.Fade(
        fade_in_len=fade_samples,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )(waveform)
    resize = transforms.Resize((798, 128))
    output = []
    for row in rows:
        start = min(int(float(row["cycle_start_s"]) * args.sample_rate), waveform.shape[1])
        end = min(int(float(row["cycle_end_s"]) * args.sample_rate), waveform.shape[1])
        cycle = cut_pad(waveform[:, start:end], args)
        fbank = generate_fbank(cycle, args.sample_rate, n_mels=args.n_mels)
        image = resize(transforms.ToTensor()(fbank))
        if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
            raise RuntimeError(f"invalid fbank for {row['cycle_id']}: {tuple(image.shape)}")
        output.append(image)
    return output


def metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(4))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(4), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    score = (specificity + sensitivity) / 2
    metrics = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": score,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "both_recall": float(recall[3]),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }
    return metrics, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args_cli = parser.parse_args()

    torch.set_num_threads(args_cli.threads)
    device = torch.device(args_cli.device)
    manifest = args_cli.manifest.resolve()
    author_repo = args_cli.author_repo.resolve()
    checkpoint = args_cli.checkpoint.resolve()
    result_root = args_cli.result_root.resolve()
    output_dir = result_root / "icbhi"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_test_rows(manifest)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)

    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank
    protocol_args = argparse.Namespace(sample_rate=16000, desired_length=8, pad_types="repeat", n_mels=128)
    model, classifier, checkpoint_epoch = build_model(author_repo, checkpoint, device)

    prediction_rows = []
    pending_images = []
    pending_rows = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        with torch.inference_mode():
            logits = classifier(model(batch))
            probabilities = torch.softmax(logits, dim=1)
        logits_np = logits.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        for row, logit, probability in zip(pending_rows, logits_np, probabilities_np):
            prediction_index = int(probability.argmax())
            record = {
                "cycle_id": row["cycle_id"],
                "recording_id": row["recording_id"],
                "official_split": row["official_split"],
                "true_label": row["native_four_class_label"],
                "true_index": LABEL_TO_INDEX[row["native_four_class_label"]],
                "pred_label": LABELS[prediction_index],
                "pred_index": prediction_index,
            }
            for index, label in enumerate(LABELS):
                record[f"logit_{label}"] = float(logit[index])
                record[f"prob_{label}"] = float(probability[index])
            prediction_rows.append(record)
        pending_images.clear()
        pending_rows.clear()

    for recording_index, recording_id in enumerate(sorted(grouped), start=1):
        recording_rows = grouped[recording_id]
        images = preprocess_recording(
            recording_rows, generate_fbank, cut_pad_sample_torchaudio, protocol_args
        )
        for row, image in zip(recording_rows, images):
            pending_rows.append(row)
            pending_images.append(image)
            if len(pending_images) == args_cli.batch_size:
                flush()
        if recording_index % 25 == 0:
            print(f"recordings={recording_index}/{len(grouped)} cycles={len(prediction_rows)}", flush=True)
    flush()
    runtime_seconds = time.perf_counter() - started

    prediction_rows.sort(key=lambda row: row["cycle_id"])
    expected_ids = {row["cycle_id"] for row in rows}
    observed_ids = {row["cycle_id"] for row in prediction_rows}
    if len(prediction_rows) != 2756 or len(observed_ids) != 2756 or observed_ids != expected_ids:
        raise RuntimeError("prediction coverage mismatch")
    if not all(
        math.isfinite(value)
        for row in prediction_rows
        for key, value in row.items()
        if key.startswith(("logit_", "prob_"))
    ):
        raise RuntimeError("non-finite prediction output")

    y_true = np.asarray([row["true_index"] for row in prediction_rows], dtype=np.int64)
    y_pred = np.asarray([row["pred_index"] for row in prediction_rows], dtype=np.int64)
    metrics, matrix = metric_payload(y_true, y_pred)
    repo_hits = np.diag(matrix).astype(float)
    repo_counts = matrix.sum(axis=1).astype(float)
    repo_sp = repo_hits[0] / (repo_counts[0] + 1e-10) * 100
    repo_se = repo_hits[1:].sum() / (repo_counts[1:].sum() + 1e-10) * 100
    repo_score = (repo_sp + repo_se) / 2
    repo_derived = {
        "specificity": float(repo_sp),
        "sensitivity": float(repo_se),
        "icbhi_score": float(repo_score),
    }
    max_metric_difference = max(abs(metrics[key] - repo_derived[key]) for key in repo_derived)
    if max_metric_difference > 1e-10:
        raise RuntimeError(f"repo metric mismatch: {max_metric_difference}")

    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *LABELS])
        for label, values in zip(LABELS, matrix):
            writer.writerow([label, *values.tolist()])

    receipt = {
        "status": "completed_author_checkpoint_only_icbhi_test_selected",
        "claim_boundary": "exploratory author checkpoint inference; not server full reproduction",
        "author_repo": "https://github.com/raymin0223/patch-mix_contrastive_learning",
        "author_repo_commit": "836b09fea1b70eb29fe0b25afa481286b56f5104",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "selection": "author checkpoint selected on official test Score",
        "manifest": str(manifest),
        "test_cycles": len(prediction_rows),
        "unique_cycle_ids": len(observed_ids),
        "recordings": len(grouped),
        "label_order": LABELS,
        "preprocessing": "author eval path: mono, 16 kHz, full-recording fade, cycle crop, 8 s truncate/repeat with fade-out, 128-bin Kaldi fbank, AudioSet normalization, resize 798x128, no SpecAugment",
        "device": str(device),
        "threads": args_cli.threads,
        "batch_size": args_cli.batch_size,
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
        "repo_prediction_derived_metrics": repo_derived,
        "repo_metric_max_abs_difference": max_metric_difference,
        "author_posted_metrics": AUTHOR_POSTED,
        "absolute_gap_vs_author_posted": {
            key: metrics[key] - value for key, value in AUTHOR_POSTED.items()
        },
        "paper_five_run_mean": PAPER_MEAN,
        "paper_five_run_sd": PAPER_SD,
        "z_like_vs_paper_mean": {
            key: (metrics[key] - PAPER_MEAN[key]) / PAPER_SD[key] for key in PAPER_MEAN
        },
        "confusion_total": int(matrix.sum()),
        "predictions_path": str(prediction_path),
        "confusion_path": str(confusion_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
