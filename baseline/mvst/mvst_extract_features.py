"""Extract ID-bearing MVST view features from a trained author checkpoint."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LABELS = ["normal", "crackle", "wheeze", "both"]


def cycle_index(cycle_id: str) -> int:
    return int(cycle_id.rsplit("_", 1)[-1])


def split_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recordings = sorted({row["recording_id"] for row in rows})
    indices = list(range(len(recordings)))
    random.Random(1).shuffle(indices)
    boundary = int(len(indices) * 0.6)
    selected = indices[:boundary] if split == "train" else indices[boundary:]
    selected_recordings = {recordings[index] for index in selected}
    return sorted(
        (row for row in rows if row["recording_id"] in selected_recordings),
        key=lambda row: (row["recording_id"], cycle_index(row["cycle_id"])),
    )


class CycleDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], util) -> None:
        self.rows = rows
        self.util = util
        self.args = SimpleNamespace(sample_rate=16000, desired_length=8, pad_types="repeat")
        self.resize = transforms.Resize(size=(1024, 256), antialias=None)

    @lru_cache(maxsize=64)
    def load_recording(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = audio_transforms.Resample(sample_rate, 16000)(waveform)
        return audio_transforms.Fade(1000, 1000, fade_shape="linear")(waveform)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        waveform = self.load_recording(row["audio_path"])
        start = min(int(float(row["cycle_start_s"]) * 16000), waveform.shape[1])
        end = min(int(float(row["cycle_end_s"]) * 16000), waveform.shape[1])
        cycle = self.util.cut_pad_sample_torchaudio(waveform[:, start:end], self.args)
        fbank = self.util.generate_fbank(cycle, 16000, n_mels=128)
        image = self.resize(transforms.ToTensor()(fbank))
        return image, LABELS.index(row["native_four_class_label"]), index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["16", "32", "64", "128", "256"], required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--task-checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction requested but CUDA is unavailable")

    manifest = args.manifest.resolve()
    view_dir = args.author_repo.resolve() / args.view
    initial = args.initial_checkpoint.resolve()
    task_checkpoint = args.task_checkpoint.resolve()
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    pretrained = work_dir / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    expected = pretrained / "audioset_16_16_0.4422.pth"
    if expected.exists() or expected.is_symlink():
        expected.unlink()
    expected.symlink_to(initial)
    os.chdir(work_dir)
    sys.path.insert(0, str(view_dir))
    util = importlib.import_module("util.icbhi_util")
    ASTModel = importlib.import_module("models.ast").ASTModel

    rows = split_rows(manifest, args.split)
    expected_count = 4213 if args.split == "train" else 2685
    if len(rows) != expected_count:
        raise ValueError(f"unexpected {args.split} rows: {len(rows)}")
    dataset = CycleDataset(rows, util)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ASTModel(
        label_dim=4, input_fdim=1024, input_tdim=256,
        imagenet_pretrain=True, audioset_pretrain=True,
        model_size="base384", verbose=False,
    )
    checkpoint = torch.load(task_checkpoint, map_location="cpu")
    state = checkpoint["model"]
    cleaned = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    feature_parts, label_parts, index_parts = [], [], []
    with torch.no_grad():
        for images, labels, indices in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model(images.to(device, non_blocking=device.type == "cuda"))
            feature_parts.append(features.float().cpu().numpy())
            label_parts.append(labels.numpy())
            index_parts.append(indices.numpy())
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    indices = np.concatenate(index_parts)
    if not np.array_equal(indices, np.arange(len(rows))) or not np.isfinite(features).all():
        raise ValueError("feature extraction order or finiteness failure")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        label=labels,
        embedding=features,
    )
    print(f"mvst_extract_ok view={args.view} split={args.split} shape={features.shape}")


if __name__ == "__main__":
    main()
