"""Export cycle IDs and logits from a completed SG-SCL test-selected run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from baseline.common.export_official_predictions import clean_state_dict, ordered_cycle_contract
from baseline.common.official_reproduction_bootstrap import sha256


def find_checkpoint(result_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = list((result_root / "full").rglob("best.pth"))
    if len(paths) != 1:
        raise ValueError(f"expected one SG-SCL best.pth, found {paths}")
    return paths[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    receipt = json.loads((result_root / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != "sg_scl":
        raise ValueError("bootstrap method is not sg_scl")
    source = result_root / "source" / "repo"
    portable = result_root / "portable_run"
    checkpoint_path = find_checkpoint(result_root, args.trained_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training_args = checkpoint["args"]
    training_args.data_folder = str(portable / "data")
    training_args.test_fold = "official"
    training_args.raw_augment = 0

    os.chdir(portable)
    sys.path.insert(0, str(source))
    from models.ast import ASTModel
    from util.icbhi_dataset import ICBHIDataset

    training_args.h = int(training_args.desired_length * 100 - 2)
    training_args.w = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(int(training_args.h * training_args.resz), int(training_args.w * training_args.resz))),
    ])
    dataset = ICBHIDataset(
        train_flag=False, transform=transform, args=training_args, print_flag=True
    )
    model = ASTModel(
        label_dim=4,
        input_fdim=int(training_args.h * training_args.resz),
        input_tdim=int(training_args.w * training_args.resz),
        imagenet_pretrain=True,
        audioset_pretrain=True,
    )
    classifier = deepcopy(model.mlp_head)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    classifier.load_state_dict(clean_state_dict(checkpoint["classifier"]), strict=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is unavailable")
    model.to(device).eval()
    classifier.to(device).eval()
    cycle_ids, expected_labels = ordered_cycle_contract(
        dataset, result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    observed_labels = []
    logits = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            output = classifier(model(images, args=training_args, training=False))
            observed_labels.append(labels.numpy())
            logits.append(output.float().cpu().numpy())
    observed = np.concatenate(observed_labels).astype(np.int64)
    output_logits = np.concatenate(logits).astype(np.float32)
    if not np.array_equal(observed, expected_labels):
        raise ValueError("SG-SCL author dataloader order/labels do not match manifest")
    if output_logits.shape != (2756, 4) or not np.isfinite(output_logits).all():
        raise ValueError(f"invalid SG-SCL exported logits: {output_logits.shape}")

    output_dir = result_root / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "official_test_outputs.npz"
    np.savez_compressed(
        output_path,
        cycle_id=np.asarray(cycle_ids),
        label=expected_labels,
        logits=output_logits,
    )
    export_receipt = {
        "status": "exported",
        "method": "sg_scl",
        "classification": "metadata-aware official-like, official-test-selected",
        "trained_checkpoint": str(checkpoint_path),
        "trained_checkpoint_sha256": sha256(checkpoint_path),
        "output_npz": str(output_path),
        "rows": 2756,
        "unique_cycle_ids": len(set(cycle_ids)),
        "logits_shape": list(output_logits.shape),
        "device": str(device),
        "compatibility_scope": "evaluation-only cycle ID/logit export; no model/data/metric change",
    }
    (output_dir / "export_receipt.json").write_text(
        json.dumps(export_receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(export_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
