"""Non-semantic cycle-ID/logit export for completed Patch-Mix CL or PAFA runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


LABELS = ["normal", "crackle", "wheeze", "both"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(result_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = list((result_root / "full").glob("*/best.pth"))
    if len(paths) != 1:
        raise ValueError(f"expected one trained best.pth, found {paths}")
    return paths[0].resolve()


def ordered_cycle_contract(dataset, manifest: Path) -> tuple[list[str], np.ndarray]:
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["official_split"] == "test"]
    by_author_recording: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        stable_key = "_".join(row["split_recording_id"].split("_")[:4])
        by_author_recording.setdefault(stable_key, []).append(row)

    cycle_ids: list[str] = []
    labels: list[int] = []
    for recording_id in dataset.filenames:
        stable_key = "_".join(recording_id.split("_")[:4])
        expected = by_author_recording.get(stable_key)
        if expected is None:
            raise ValueError(f"author test recording absent from manifest: {recording_id}")
        author_labels = list(dataset.filename_to_label[recording_id])
        manifest_labels = [LABELS.index(row["native_four_class_label"]) for row in expected]
        if author_labels != manifest_labels:
            raise ValueError(f"cycle label/order mismatch for {recording_id}")
        cycle_ids.extend(row["cycle_id"] for row in expected)
        labels.extend(manifest_labels)
    if len(cycle_ids) != 2756 or len(set(cycle_ids)) != 2756:
        raise ValueError(f"official test cycle contract mismatch: {len(cycle_ids)}")
    return cycle_ids, np.asarray(labels, dtype=np.int64)


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def patch_mix_objects(author_repo: Path, portable: Path, args, base_checkpoint: Path):
    from torchvision import transforms

    sys.path.insert(0, str(author_repo))
    from models.ast import ASTModel
    from util.icbhi_dataset import ICBHIDataset

    expected = portable / "pretrained_models" / "audioset_10_10_0.4593.pth"
    if expected.resolve() != base_checkpoint.resolve():
        raise ValueError("Patch-Mix pretrained checkpoint adapter mismatch")
    args.data_folder = str(portable / "data")
    args.test_fold = "official"
    args.raw_augment = 0
    args.h, args.w = 798, 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz))),
    ])
    dataset = ICBHIDataset(train_flag=False, transform=transform, args=args, print_flag=True)
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
        mix_beta=args.mix_beta,
    )
    classifier = deepcopy(model.mlp_head)
    return dataset, model, classifier


def pafa_objects(author_repo: Path, portable: Path, args, base_checkpoint: Path):
    sys.path.insert(0, str(author_repo))
    from models.beats import BEATsTransferLearningModel
    from util.icbhi_dataset import ICBHIDataset

    args.data_folder = str(portable / "data")
    args.test_fold = "official"
    args.raw_augment = 0
    args.model = "beats"
    args.nospec = True
    dataset = ICBHIDataset(train_flag=False, transform=None, args=args, print_flag=True)
    model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(base_checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    classifier = torch.nn.Linear(model.final_feat_dim, 4)
    return dataset, model, classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["patch_mix_cl", "pafa"])
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args_cli = parser.parse_args()

    result_root = args_cli.result_root.resolve()
    receipt = json.loads((result_root / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != args_cli.method:
        raise ValueError("bootstrap/method mismatch")
    author_repo = result_root / "source" / "repo"
    portable = result_root / "portable_run"
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    base_checkpoint = Path(receipt["checkpoint"]["path"])
    trained_checkpoint = find_checkpoint(result_root, args_cli.trained_checkpoint)
    device = torch.device(args_cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("completed-run export requires an available CUDA device")

    os.chdir(portable)
    checkpoint = torch.load(trained_checkpoint, map_location="cpu")
    training_args = checkpoint["args"]
    if args_cli.method == "patch_mix_cl":
        dataset, model, classifier = patch_mix_objects(
            author_repo, portable, training_args, base_checkpoint
        )
    else:
        dataset, model, classifier = pafa_objects(
            author_repo, portable, training_args, base_checkpoint
        )
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
    classifier.load_state_dict(clean_state_dict(checkpoint["classifier"]), strict=True)
    model.to(device).eval()
    classifier.to(device).eval()

    cycle_ids, expected_labels = ordered_cycle_contract(dataset, manifest)
    loader = DataLoader(
        dataset,
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    observed_labels = []
    logits = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if args_cli.method == "patch_mix_cl":
                class_labels = labels
                output = classifier(model(images))
            else:
                class_labels = labels[0]
                output = classifier(model(images, training=False)).mean(dim=1)
            observed_labels.append(class_labels.numpy())
            logits.append(output.float().cpu().numpy())
    observed = np.concatenate(observed_labels).astype(np.int64)
    output_logits = np.concatenate(logits).astype(np.float32)
    if not np.array_equal(observed, expected_labels):
        raise ValueError("author dataloader labels do not match manifest-derived ordered labels")
    if output_logits.shape != (2756, 4) or not np.isfinite(output_logits).all():
        raise ValueError(f"invalid exported logits: {output_logits.shape}")

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
        "method": args_cli.method,
        "trained_checkpoint": str(trained_checkpoint),
        "trained_checkpoint_sha256": sha256(trained_checkpoint),
        "output_npz": str(output_path),
        "rows": 2756,
        "unique_cycle_ids": len(set(cycle_ids)),
        "logits_shape": list(output_logits.shape),
        "device": str(device),
        "compatibility_scope": "evaluation-only cycle ID and logits export; no model/data/metric change",
    }
    (output_dir / "export_receipt.json").write_text(
        json.dumps(export_receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(export_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
