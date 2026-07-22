"""Eight-cycle forward and finite-gradient smoke for one ADD-RSC track."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms
from torchvision import transforms


LABELS = ["normal", "crackle", "wheeze", "both"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cycle_index(cycle_id: str) -> int:
    return int(cycle_id.rsplit("_", 1)[-1])


def select_rows(manifest: Path, track: str) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if track == "paper_declared_reconstruction":
        candidates = [row for row in rows if row["official_split"] == "train"]
    else:
        recordings = sorted({row["recording_id"] for row in rows})
        indices = list(range(len(recordings)))
        random.Random(1).shuffle(indices)
        train_ids = {recordings[index] for index in indices[: int(len(indices) * 0.6)]}
        candidates = [row for row in rows if row["recording_id"] in train_ids]
    candidates = sorted(candidates, key=lambda row: (row["recording_id"], cycle_index(row["cycle_id"])))
    selected = []
    for label in LABELS:
        selected.extend([row for row in candidates if row["native_four_class_label"] == label][:2])
    selected.sort(key=lambda row: (row["recording_id"], cycle_index(row["cycle_id"])))
    if len(selected) != 8 or len({row["cycle_id"] for row in selected}) != 8:
        raise ValueError("failed to select two unique cycles per ADD-RSC class")
    return selected


def preprocess(rows: list[dict[str, str]], util, n_mels: int) -> torch.Tensor:
    crop_args = SimpleNamespace(sample_rate=16000, desired_length=8, pad_types="repeat")
    resize = transforms.Resize((1024, 256), antialias=None)
    output = []
    for row in rows:
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = audio_transforms.Resample(sample_rate, 16000)(waveform)
        waveform = audio_transforms.Fade(1000, 1000, fade_shape="linear")(waveform)
        start = min(int(float(row["cycle_start_s"]) * 16000), waveform.shape[1])
        end = min(int(float(row["cycle_end_s"]) * 16000), waveform.shape[1])
        cycle = util.cut_pad_sample_torchaudio(waveform[:, start:end], crop_args)
        fbank = util.generate_fbank(cycle, 16000, n_mels=n_mels)
        output.append(resize(transforms.ToTensor()(fbank)))
    batch = torch.stack(output)
    if batch.shape != (8, 1, 1024, 256) or not torch.isfinite(batch).all():
        raise ValueError(f"unexpected ADD-RSC input batch: {tuple(batch.shape)}")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["paper_declared_reconstruction", "author_repo_default_official_like"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.train_batch_size <= 8:
        raise ValueError("steps must be positive and train batch size must be 1..8")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    source = args.author_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    sys.path.insert(0, str(source))
    util = importlib.import_module("util.icbhi_util")
    ASTModel = importlib.import_module("models.ast").ASTModel
    DiffTransformerLayer = importlib.import_module("models.adapt_diff_denoise").DiffTransformerLayer
    LabelSmoothingLoss = importlib.import_module("models.bias_denoise_loss").LabelSmoothingLoss

    config = (
        {"n_mels": 64, "weight_decay": 0.1, "split_protocol": "paper_declared_official"}
        if args.track == "paper_declared_reconstruction"
        else {"n_mels": 128, "weight_decay": 1e-6, "split_protocol": "author_repo_random_file"}
    )
    rows = select_rows(args.manifest.resolve(), args.track)
    started = time.perf_counter()
    batch = preprocess(rows, util, config["n_mels"]).to(device)
    preprocessing_seconds = time.perf_counter() - started
    labels = torch.tensor(
        [LABELS.index(row["native_four_class_label"]) for row in rows], device=device
    )

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = ASTModel(
        input_fdim=1024, input_tdim=256, label_dim=4,
        audioset_pretrain=True, pretrained_path=str(checkpoint),
    ).to(device)
    classifier = deepcopy(model.mlp_head).to(device)
    denoiser = DiffTransformerLayer(d_model=256, num_heads=8, depth=6).to(device)
    model.eval(); classifier.eval(); denoiser.eval()
    logits_parts = []
    denoise_parts = []
    with torch.no_grad():
        for index in range(len(batch)):
            denoised, denoise_logits = denoiser(batch[index:index + 1].squeeze(1))
            features = model(denoised.unsqueeze(1))
            logits_parts.append(classifier(features).float().cpu())
            denoise_parts.append(denoise_logits.float().cpu())
    logits = torch.cat(logits_parts)
    denoise_logits = torch.cat(denoise_parts)

    model.train(); classifier.train(); denoiser.train()
    parameters = list(model.parameters()) + list(classifier.parameters()) + list(denoiser.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=config["weight_decay"])
    smooth_loss = LabelSmoothingLoss(smoothing=0.2).to(device)
    step_times = []
    train_count = args.train_batch_size
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        denoised, train_denoise_logits = denoiser(batch[:train_count].squeeze(1))
        train_logits = classifier(model(denoised.unsqueeze(1)))
        class_loss = torch.nn.functional.cross_entropy(train_logits, labels[:train_count])
        bias_loss = smooth_loss(train_denoise_logits, labels[:train_count])
        loss = 0.5 * bias_loss + 0.5 * class_loss
        loss.backward(); optimizer.step()
        step_times.append(time.perf_counter() - step_start)
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    if not torch.isfinite(loss) or not gradients_finite or not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite ADD-RSC smoke output/loss/gradient")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / f"{args.track}_smoke_8_outputs.npz"
    np.savez_compressed(
        output_npz,
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        label=labels.cpu().numpy(),
        logits=logits.numpy(),
        denoise_logits=denoise_logits.numpy(),
    )
    receipt = {
        "status": "passed", "track": args.track,
        "classification": (
            "bounded paper-declared reconstruction; not official faithful"
            if args.track == "paper_declared_reconstruction"
            else "author-code default execution; target 65.53 not directly comparable"
        ),
        "split_protocol": config["split_protocol"], "n_mels": config["n_mels"],
        "weight_decay": config["weight_decay"], "device": str(device),
        "cycles": 8, "cycle_ids": [row["cycle_id"] for row in rows],
        "labels": [row["native_four_class_label"] for row in rows],
        "input_shape": list(batch.shape), "logits_shape": list(logits.shape),
        "denoise_logits_shape": list(denoise_logits.shape),
        "checkpoint_sha256": sha256(checkpoint),
        "preprocessing": f"16 kHz, 8 s repeat/cyclic pad, {config['n_mels']} fbank bins, resize 1024x256; random SpecAugment reserved for full train loader",
        "alpha_mapping": "paper alpha=0.02 maps to repo AFNO1D.scale=0.02",
        "epsilon_mapping": "paper epsilon=0.2 maps to repo LabelSmoothingLoss(smoothing=0.2)",
        "loss_beta": 0.5, "classification_loss": float(class_loss.item()),
        "bias_denoise_loss": float(bias_loss.item()), "total_loss": float(loss.item()),
        "loss_finite": True, "gradients_finite": gradients_finite,
        "steps": args.steps, "train_batch_size": train_count,
        "step_seconds": step_times, "mean_step_seconds": float(np.mean(step_times)),
        "projected_100_step_seconds": float(np.mean(step_times) * 100),
        "preprocessing_seconds": preprocessing_seconds,
        "process_max_rss_bytes_macos": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    (args.output_dir / f"{args.track}_smoke_8.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
