"""Author-preprocessing and author-model Patch-Mix CL compatibility smoke.

The adapter is intentionally limited to device/path handling and artifact export.
It does not change the split, crop/pad/fbank transform, model, or Patch-Mix math.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
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


def select_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for split in ("train", "test"):
        for label in LABELS:
            selected.append(next(row for row in rows if row["official_split"] == split and row["native_four_class_label"] == label))
    assert len({row["cycle_id"] for row in selected}) == 8
    return selected


def preprocess(rows: list[dict[str, str]], author_repo: Path, args: SimpleNamespace) -> torch.Tensor:
    sys.path.insert(0, str(author_repo))
    from util.augmentation import SpecAugment
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    output = []
    specaugment = SpecAugment(args)
    resize = transforms.Resize(size=(798, 128))
    for row in rows:
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != args.sample_rate:
            waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
        fade_samples = int(args.sample_rate / 16)
        waveform = audio_transforms.Fade(
            fade_in_len=fade_samples,
            fade_out_len=fade_samples,
            fade_shape="linear",
        )(waveform)
        start = int(float(row["cycle_start_s"]) * args.sample_rate)
        end = int(float(row["cycle_end_s"]) * args.sample_rate)
        cycle = waveform[:, min(start, waveform.shape[1]) : min(end, waveform.shape[1])]
        cycle = cut_pad_sample_torchaudio(cycle, args)
        fbank = generate_fbank(cycle, args.sample_rate, n_mels=args.n_mels)
        image = transforms.ToTensor()(fbank)
        image = specaugment(image)
        output.append(resize(image))
    batch = torch.stack(output)
    assert batch.shape == (8, 1, 798, 128)
    assert torch.isfinite(batch).all()
    return batch


def contrastive_loss_device_safe(
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    labels_a: torch.Tensor,
    lam: float,
    index: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    projection1 = torch.nn.functional.normalize(projection1)
    projection2 = torch.nn.functional.normalize(projection2)
    logits = projection2 @ projection1.T / temperature
    mask_a = torch.eye(projection1.shape[0], device=projection1.device)
    mask_b = torch.zeros_like(mask_a)
    mask_b[torch.arange(projection1.shape[0], device=projection1.device), index] = 1
    mask = lam * mask_a + (1 - lam) * mask_b
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    return -((mask * log_prob).sum(dim=1) / mask.sum(dim=1)).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args_cli = parser.parse_args()

    if args_cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args_cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args_cli.device}")

    manifest = args_cli.manifest.resolve()
    author_repo = args_cli.author_repo.resolve()
    checkpoint = args_cli.checkpoint.resolve()
    work_dir = args_cli.work_dir.resolve()
    output_dir = args_cli.output_dir.resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir = work_dir / "pretrained_models"
    pretrained_dir.mkdir(exist_ok=True)
    expected_checkpoint = pretrained_dir / "audioset_10_10_0.4593.pth"
    if expected_checkpoint.exists() or expected_checkpoint.is_symlink():
        expected_checkpoint.unlink()
    expected_checkpoint.symlink_to(checkpoint)

    os.chdir(work_dir)
    sys.path.insert(0, str(author_repo))
    from models import Projector
    from models.ast import ASTModel
    from util.misc import update_moving_average

    protocol_args = SimpleNamespace(
        sample_rate=16000,
        desired_length=8,
        pad_types="repeat",
        n_mels=128,
        specaug_policy="icbhi_ast_sup",
        specaug_mask="mean",
        negative_pair="all",
    )
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    rows = select_rows(manifest)
    start = time.perf_counter()
    batch = preprocess(rows, author_repo, protocol_args)
    preprocessing_seconds = time.perf_counter() - start
    batch = batch.to(device)
    labels = torch.tensor(
        [LABELS.index(row["native_four_class_label"]) for row in rows],
        device=device,
    )

    model = ASTModel(
        label_dim=4,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=798,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
        verbose=False,
        mix_beta=1.0,
    )
    classifier = deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, 768)
    model = model.to(device)
    classifier = classifier.to(device)
    projector = projector.to(device)
    model.train()
    classifier.train()
    projector.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
        lr=5e-5,
        weight_decay=1e-6,
    )
    step_times = []
    for _ in range(args_cli.steps):
        model_before = deepcopy(model.state_dict())
        classifier_before = deepcopy(classifier.state_dict())
        projector_before = deepcopy(projector.state_dict())
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        embedding = model(batch)
        logits = classifier(embedding)
        classification_loss = torch.nn.functional.cross_entropy(logits, labels)
        projection_target = embedding.detach()
        mixed_embedding, y_a, y_b, lam, index = model(batch, labels, patch_mix=True)
        projection_mix = projector(mixed_embedding)
        contrastive_loss = contrastive_loss_device_safe(
            projection_target,
            projection_mix,
            y_a,
            lam,
            index,
            temperature=0.06,
        )
        loss = classification_loss + contrastive_loss
        loss.backward()
        optimizer.step()
        model = update_moving_average(0.5, model, model_before)
        classifier = update_moving_average(0.5, classifier, classifier_before)
        projector = update_moving_average(0.5, projector, projector_before)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - start)

    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    )
    assert torch.isfinite(loss)
    assert gradients_finite
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "patch_mix_cl_smoke_8_outputs.npz",
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        label=labels.detach().cpu().numpy(),
        fbank=batch.detach().cpu().numpy(),
        embedding=embedding.detach().cpu().numpy(),
        logits=logits.detach().cpu().numpy(),
    )
    payload = {
        "status": "passed_with_current_author_hosted_checkpoint; historical_2023_byte_identity_unresolved",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cycles": 8,
        "selection": "first manifest row for each official_split x native_four_class_label combination",
        "cycle_ids": [row["cycle_id"] for row in rows],
        "labels": [row["native_four_class_label"] for row in rows],
        "fbank_shape": list(batch.shape),
        "embedding_shape": list(embedding.shape),
        "logits_shape": list(logits.shape),
        "preprocessing": "author 16 kHz mono, recording fade, cycle crop, 8 s repeat+fade, 128-bin Kaldi fbank, AudioSet normalization, icbhi_ast_sup SpecAugment",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "classification_loss": float(classification_loss.item()),
        "contrastive_loss": float(contrastive_loss.item()),
        "total_loss": float(loss.item()),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "preprocessing_seconds": preprocessing_seconds,
        "steps": args_cli.steps,
        "cpu_training_step_seconds": step_times,
        "mean_cpu_training_step_seconds": float(np.mean(step_times)),
        "projected_100_step_seconds": float(np.mean(step_times) * 100),
        "process_max_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "ema_beta": 0.5,
        "target_type": "grad_block",
        "compatibility_changes": [
            "path symlink to the SHA-verified checkpoint under the author-expected pretrained_models path",
            "device-safe equivalent of PatchMixConLoss masks because author loss hardcodes CUDA",
            "cycle ID and output export only",
        ],
    }
    receipt_name = "patch_mix_cl_smoke_8.json" if args_cli.steps == 1 else f"patch_mix_cl_profile_{args_cli.steps}.json"
    (output_dir / receipt_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
