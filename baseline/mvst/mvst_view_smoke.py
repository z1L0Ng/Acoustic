"""Run one MVST encoder view on an auditable eight-cycle subset.

Each view runs in a separate process so five AST encoders are never resident at
once. The subset contains two cycles per class from the author's fixed
``random.Random(1)`` recording split and keeps author dataset order.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import importlib
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
VIEWS = {"16": (16, 16), "32": (32, 8), "64": (64, 4), "128": (128, 2), "256": (256, 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cycle_index(cycle_id: str) -> int:
    return int(cycle_id.rsplit("_", 1)[-1])


def select_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recordings = sorted({row["recording_id"] for row in rows})
    indices = list(range(len(recordings)))
    random.Random(1).shuffle(indices)
    train_recordings = {recordings[i] for i in indices[: int(len(indices) * 0.6)]}
    author_order = sorted(
        (row for row in rows if row["recording_id"] in train_recordings),
        key=lambda row: (row["recording_id"], cycle_index(row["cycle_id"])),
    )
    selected_ids: set[str] = set()
    for label in LABELS:
        selected_ids.update(
            row["cycle_id"]
            for row in author_order
            if row["native_four_class_label"] == label
        )
        # Retain only the first two from this class.
        matching = [
            row["cycle_id"]
            for row in author_order
            if row["native_four_class_label"] == label
        ]
        selected_ids.difference_update(matching[2:])
    selected = [row for row in author_order if row["cycle_id"] in selected_ids]
    if len(selected) != 8 or {row["native_four_class_label"] for row in selected} != set(LABELS):
        raise ValueError("failed to select two author-train cycles per class")
    return selected


def preprocess(rows: list[dict[str, str]], view_dir: Path) -> torch.Tensor:
    sys.path.insert(0, str(view_dir))
    util = importlib.import_module("util.icbhi_util")
    args = SimpleNamespace(sample_rate=16000, desired_length=8, pad_types="repeat")
    resize = transforms.Resize(size=(1024, 256), antialias=None)
    output = []
    for row in rows:
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = audio_transforms.Resample(sample_rate, 16000)(waveform)
        fade_samples = 1000
        waveform = audio_transforms.Fade(
            fade_in_len=fade_samples,
            fade_out_len=fade_samples,
            fade_shape="linear",
        )(waveform)
        start = min(int(float(row["cycle_start_s"]) * 16000), waveform.shape[1])
        end = min(int(float(row["cycle_end_s"]) * 16000), waveform.shape[1])
        cycle = util.cut_pad_sample_torchaudio(waveform[:, start:end], args)
        fbank = util.generate_fbank(cycle, 16000, n_mels=128)
        output.append(resize(transforms.ToTensor()(fbank)))
    batch = torch.stack(output)
    if batch.shape != (8, 1, 1024, 256) or not torch.isfinite(batch).all():
        raise ValueError(f"unexpected MVST input: {tuple(batch.shape)}")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=sorted(VIEWS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")

    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    view_dir = (args.author_repo.resolve() / args.view)
    checkpoint = args.checkpoint.resolve()
    work_dir = args.work_dir.resolve()
    pretrained = work_dir / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    expected = pretrained / "audioset_16_16_0.4422.pth"
    if expected.exists() or expected.is_symlink():
        expected.unlink()
    expected.symlink_to(checkpoint)
    os.chdir(work_dir)

    rows = select_rows(manifest)
    preprocessing_start = time.perf_counter()
    batch = preprocess(rows, view_dir)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    sys.path.insert(0, str(view_dir))
    ASTModel = importlib.import_module("models.ast").ASTModel
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    model = ASTModel(
        label_dim=4,
        input_fdim=1024,
        input_tdim=256,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
        verbose=False,
    ).to(device)
    classifier = deepcopy(model.mlp_head).to(device)
    batch = batch.to(device)
    model.eval()
    classifier.eval()
    embeddings = []
    logits = []
    with torch.no_grad():
        for sample in batch:
            feature = model(sample.unsqueeze(0))
            embeddings.append(feature.squeeze(0))
            logits.append(classifier(feature).squeeze(0))
    embeddings_tensor = torch.stack(embeddings)
    logits_tensor = torch.stack(logits)

    # A short batch-size-eight optimizer sequence verifies author train wiring.
    model.train()
    classifier.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=5e-5,
        weight_decay=1e-6,
    )
    labels = torch.tensor(
        [LABELS.index(row["native_four_class_label"]) for row in rows], device=device
    )
    step_times = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        feature = model(batch)
        train_logits = classifier(feature)
        loss = torch.nn.functional.cross_entropy(train_logits, labels)
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - step_start)
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in list(model.parameters()) + list(classifier.parameters())
    )
    if not torch.isfinite(loss) or not gradients_finite:
        raise FloatingPointError("non-finite MVST view loss/gradient")

    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    patch_shape = list(source["module.v.patch_embed.proj.weight"].shape)
    expected_patch_shape = [768, 1, *VIEWS[args.view]]
    patch_projection_loaded = patch_shape == expected_patch_shape

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"mvst_view_{args.view}_smoke_8.npz",
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        label=np.asarray([LABELS.index(row["native_four_class_label"]) for row in rows]),
        embedding=embeddings_tensor.cpu().numpy(),
        logits=logits_tensor.cpu().numpy(),
    )
    receipt = {
        "status": "passed",
        "protocol_name": "author_repo_random_file_split_official_like",
        "device": str(device),
        "view": args.view,
        "patch_kernel_stride": list(VIEWS[args.view]),
        "cycles": 8,
        "cycle_ids": [row["cycle_id"] for row in rows],
        "labels": [row["native_four_class_label"] for row in rows],
        "input_shape": list(batch.shape),
        "embedding_shape": list(embeddings_tensor.shape),
        "logits_shape": list(logits_tensor.shape),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_patch_projection_shape": patch_shape,
        "view_expected_patch_projection_shape": expected_patch_shape,
        "source_patch_projection_loaded_by_author_shape_filter": patch_projection_loaded,
        "source_position_embedding_shape": list(source["module.v.pos_embed"].shape),
        "author_loader_note": "exact key+shape matches load; mismatched view-specific tensors remain initialized by author code",
        "preprocessing_seconds": preprocessing_seconds,
        "train_step_batch_size": 8,
        "steps": args.steps,
        "train_step_seconds": step_times,
        "mean_train_step_seconds": float(np.mean(step_times)),
        "projected_100_step_seconds": float(np.mean(step_times) * 100),
        "train_loss": float(loss.item()),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "process_max_rss_bytes_macos": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    (output_dir / f"mvst_view_{args.view}_smoke_8.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
