"""Author-preprocessing SG-SCL device-domain compatibility smoke.

The adapter preserves the author model, preprocessing, labels, and SG-SCL
objective. It only makes MetaCL tensor allocation device-aware and exports a
small auditable artifact so the CUDA-only full command can be validated first.
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
DEVICES = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_name(row: dict[str, str]) -> str:
    name = Path(row["audio_path"]).stem.split("_")[-1]
    if name not in DEVICES:
        raise ValueError(f"unmapped ICBHI device: {name}")
    return name


def select_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for device in DEVICES:
        matches = [
            row for row in rows
            if row["official_split"] == "train" and device_name(row) == device
        ]
        selected.extend(matches[:2])
    if len(selected) != 8 or len({row["cycle_id"] for row in selected}) != 8:
        raise ValueError("could not select two unique official-train cycles per device")
    return selected


def preprocess(rows: list[dict[str, str]], author_repo: Path, args: SimpleNamespace) -> torch.Tensor:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    resize = transforms.Resize(size=(798, 128))
    output = []
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
        fbank = generate_fbank(args, cycle, args.sample_rate, n_mels=args.n_mels)
        output.append(resize(transforms.ToTensor()(fbank)))
    batch = torch.stack(output)
    if batch.shape != (8, 1, 798, 128) or not torch.isfinite(batch).all():
        raise ValueError(f"unexpected fbank batch: {tuple(batch.shape)}")
    return batch


def meta_cl_device_safe(
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    meta_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Exact author MetaCL algebra with allocations on the input device."""
    projection1 = torch.nn.functional.normalize(projection1)
    projection2 = torch.nn.functional.normalize(projection2)
    features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
    batch_size = features.shape[0]
    labels = meta_labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(projection1.device)
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = contrast_feature @ contrast_feature.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = mask.repeat(contrast_count, contrast_count)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * contrast_count, device=projection1.device).view(-1, 1),
        0,
    )
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return -mean_log_prob_pos.view(contrast_count, batch_size).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    cli = parser.parse_args()

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")

    manifest = cli.manifest.resolve()
    author_repo = cli.author_repo.resolve()
    checkpoint = cli.checkpoint.resolve()
    work_dir = cli.work_dir.resolve()
    output_dir = cli.output_dir.resolve()
    pretrained_dir = work_dir / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    expected = pretrained_dir / "audioset_10_10_0.4593.pth"
    if expected.exists() or expected.is_symlink():
        expected.unlink()
    expected.symlink_to(checkpoint)

    os.chdir(work_dir)
    sys.path.insert(0, str(author_repo))
    from models import Projector
    from models.ast import ASTModel
    from util.augmentation import SpecAugment
    from util.misc import update_moving_average

    args = SimpleNamespace(
        sample_rate=16000,
        desired_length=8,
        pad_types="repeat",
        n_mels=128,
        model="ast",
        specaug_policy="icbhi_ast_sup",
        specaug_mask="mean",
        domain_adaptation=False,
        domain_adaptation2=True,
    )
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    rows = select_rows(manifest)
    start = time.perf_counter()
    batch = preprocess(rows, author_repo, args)
    preprocessing_seconds = time.perf_counter() - start
    batch = batch.to(device)
    labels = torch.tensor(
        [LABELS.index(row["native_four_class_label"]) for row in rows], device=device
    )
    devices = torch.tensor(
        [DEVICES.index(device_name(row)) for row in rows], device=device
    )

    # Preserve the repo's set_model dimension order (h=798, w=128). Although
    # unconventional, changing it would alter the executable author protocol.
    model = ASTModel(
        label_dim=4,
        input_fdim=798,
        input_tdim=128,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
        verbose=False,
    ).to(device)
    classifier = deepcopy(model.mlp_head).to(device)
    projector = Projector(model.final_feat_dim, 768).to(device)
    specaugment = SpecAugment(args).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
        lr=5e-5,
        weight_decay=1e-6,
    )
    model.train()
    classifier.train()
    projector.train()
    step_times = []
    profile_start = time.perf_counter()
    for _ in range(cli.steps):
        model_before = deepcopy(model.state_dict())
        classifier_before = deepcopy(classifier.state_dict())
        projector_before = deepcopy(projector.state_dict())
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        feature1 = model(specaugment(batch), args=args, alpha=0.0, training=True)[0]
        logits = classifier(feature1)
        class_loss = torch.nn.functional.cross_entropy(logits, labels)
        feature2 = model(specaugment(batch), args=args, alpha=0.0, training=True)[0]
        projection1 = projector(feature1)
        projection2 = projector(feature2).detach()
        meta_loss = meta_cl_device_safe(projection1, projection2, devices, temperature=0.06)
        loss = class_loss + meta_loss
        loss.backward()
        optimizer.step()
        model = update_moving_average(0.5, model, model_before)
        classifier = update_moving_average(0.5, classifier, classifier_before)
        projector = update_moving_average(0.5, projector, projector_before)
        step_times.append(time.perf_counter() - start)

    parameters = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    if not torch.isfinite(loss) or not gradients_finite:
        raise FloatingPointError("non-finite SG-SCL smoke loss or gradient")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sg_scl_smoke_8_outputs.npz",
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        label=labels.cpu().numpy(),
        device_label=devices.cpu().numpy(),
        fbank=batch.cpu().numpy(),
        embedding=feature1.detach().cpu().numpy(),
        logits=logits.detach().cpu().numpy(),
    )
    payload = {
        "status": "passed_with_current_author_hosted_checkpoint; historical_serialized_byte_identity_unresolved",
        "branch_name": "metadata_device_domain_official_like_reproduction",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cycles": 8,
        "selection": "first two official-train cycles for each author device label",
        "cycle_ids": [row["cycle_id"] for row in rows],
        "class_labels": [row["native_four_class_label"] for row in rows],
        "device_labels": [device_name(row) for row in rows],
        "device_mapping": {name: index for index, name in enumerate(DEVICES)},
        "fbank_shape": list(batch.shape),
        "embedding_shape": list(feature1.shape),
        "projection_shape": list(projection1.shape),
        "logits_shape": list(logits.shape),
        "preprocessing": "author 16 kHz mono, recording fade, cycle crop, 8 s repeat+fade, 128-bin Kaldi fbank, AudioSet normalization, two independent icbhi_ast_sup SpecAugment views",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "classification_loss": float(class_loss.item()),
        "meta_cl_loss": float(meta_loss.item()),
        "total_loss": float(loss.item()),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "preprocessing_seconds": preprocessing_seconds,
        "steps": cli.steps,
        "cpu_training_step_seconds": step_times,
        "mean_cpu_training_step_seconds": float(np.mean(step_times)),
        "projected_100_step_seconds": float(np.mean(step_times) * 100),
        "profile_wall_seconds": time.perf_counter() - profile_start,
        "process_max_rss_bytes_macos": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "temperature": 0.06,
        "alpha": 1.0,
        "ema_beta": 0.5,
        "target_type": "project1_project2block",
        "compatibility_changes": [
            "path symlink to the current author-hosted checkpoint under the author-expected pretrained_models path",
            "device-safe equivalent of MetaCL allocations because author loss hardcodes CUDA",
            "cycle ID, device label, and output export only",
        ],
        "repo_dimension_order_preserved": {"input_fdim": 798, "input_tdim": 128},
    }
    receipt_name = "sg_scl_smoke_8.json" if cli.steps == 1 else f"sg_scl_profile_{cli.steps}.json"
    (output_dir / receipt_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
