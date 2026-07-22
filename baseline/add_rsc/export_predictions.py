"""Export ID-bearing predictions from one completed ADD-RSC track."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from baseline.common.export_official_predictions import clean_state_dict
from baseline.common.official_reproduction_bootstrap import sha256


LABELS = ["normal", "crackle", "wheeze", "both"]


def find_checkpoint(result_root: Path, track: str, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = list((result_root / "full" / track).rglob("best.pth"))
    if len(paths) != 1:
        raise ValueError(f"expected one {track} best.pth, found {paths}")
    return paths[0].resolve()


def selected_rows(manifest: Path, track: str) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if track == "paper_declared_reconstruction":
        chosen = [row for row in rows if row["official_split"] == "test"]
    else:
        recordings = sorted({row["recording_id"] for row in rows})
        indices = list(range(len(recordings)))
        random.Random(1).shuffle(indices)
        test_ids = {recordings[index] for index in indices[int(len(indices) * 0.6):]}
        chosen = [row for row in rows if row["recording_id"] in test_ids]
    return chosen


def ordered_contract(dataset, manifest: Path, track: str) -> tuple[list[str], np.ndarray]:
    rows = selected_rows(manifest, track)
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_recording.setdefault(row["recording_id"], []).append(row)
    for values in by_recording.values():
        values.sort(key=lambda row: int(row["cycle_id"].rsplit("_", 1)[-1]))
    cycle_ids, labels = [], []
    for recording_id in dataset.filenames:
        expected = by_recording.get(recording_id)
        if expected is None:
            raise ValueError(f"ADD-RSC dataloader recording absent from manifest contract: {recording_id}")
        expected_labels = [LABELS.index(row["native_four_class_label"]) for row in expected]
        observed_labels = list(dataset.filename_to_label[recording_id])
        if expected_labels != observed_labels:
            raise ValueError(f"ADD-RSC cycle label/order mismatch: {recording_id}")
        cycle_ids.extend(row["cycle_id"] for row in expected)
        labels.extend(expected_labels)
    expected_count = 2756 if track == "paper_declared_reconstruction" else 2685
    if len(cycle_ids) != expected_count or len(set(cycle_ids)) != expected_count:
        raise ValueError(f"ADD-RSC test cycle contract mismatch: {len(cycle_ids)}")
    return cycle_ids, np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["paper_declared_reconstruction", "author_repo_default_official_like"], required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    receipt = json.loads((result_root / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != "add_rsc":
        raise ValueError("bootstrap method is not add_rsc")
    source = result_root / "source" / "repo"
    portable = result_root / "portable_run"
    checkpoint_path = find_checkpoint(result_root, args.track, args.trained_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_args = checkpoint["args"]
    training_args.data_folder = str(portable / "data" / "icbhi_dataset")
    training_args.raw_augment = 0

    os.chdir(portable)
    sys.path.insert(0, str(source))
    from models.adapt_diff_denoise import DiffTransformerLayer
    from models.ast import ASTModel
    from util.icbhi_dataset import ICBHIDataset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(1024, 256), antialias=None),
    ])
    dataset = ICBHIDataset(
        train_flag=False, transform=transform, args=training_args, print_flag=True
    )
    model = ASTModel(
        input_fdim=1024, input_tdim=256, label_dim=4,
        audioset_pretrain=True,
        pretrained_path=str(result_root / "checkpoints" / "audioset_10_10_0.4593.pth"),
    )
    classifier = deepcopy(model.mlp_head)
    denoiser = DiffTransformerLayer(d_model=256, num_heads=8, depth=6)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    denoiser.load_state_dict(clean_state_dict(checkpoint["bias_denoise_encoder"]), strict=True)
    classifier.load_state_dict(clean_state_dict(checkpoint["classifier"]), strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is unavailable")
    model.to(device).eval(); classifier.to(device).eval(); denoiser.to(device).eval()
    cycle_ids, expected_labels = ordered_contract(
        dataset, result_root / "manifest" / "icbhi_2017_cycle_manifest.csv", args.track
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    observed_parts, logits_parts = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda").squeeze(1)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                denoised, _ = denoiser(images)
                output = classifier(model(denoised.unsqueeze(1)))
            observed_parts.append(labels.numpy())
            logits_parts.append(output.float().cpu().numpy())
    observed = np.concatenate(observed_parts).astype(np.int64)
    logits = np.concatenate(logits_parts).astype(np.float32)
    if not np.array_equal(observed, expected_labels):
        raise ValueError("ADD-RSC dataloader labels do not match manifest order")
    if logits.shape != (len(cycle_ids), 4) or not np.isfinite(logits).all():
        raise ValueError(f"invalid ADD-RSC logits: {logits.shape}")
    output_dir = result_root / "predictions" / args.track
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_outputs.npz"
    np.savez_compressed(
        output_path, cycle_id=np.asarray(cycle_ids), label=expected_labels, logits=logits
    )
    export_receipt = {
        "status": "exported", "method": "add_rsc", "track": args.track,
        "trained_checkpoint": str(checkpoint_path),
        "trained_checkpoint_sha256": sha256(checkpoint_path),
        "output_npz": str(output_path), "rows": len(cycle_ids),
        "unique_cycle_ids": len(set(cycle_ids)), "logits_shape": list(logits.shape),
        "device": str(device),
        "compatibility_scope": "evaluation-only cycle ID/logit export; no model/data/metric change",
    }
    (output_dir / "export_receipt.json").write_text(
        json.dumps(export_receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(export_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
