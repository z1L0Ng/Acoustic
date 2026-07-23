"""Assert five-view identity and run one finite MVST fusion step."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


VIEWS = ["16", "32", "64", "128", "256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fusion smoke requested but CUDA is unavailable")

    archives = []
    for view in VIEWS:
        with np.load(args.smoke_root / view / f"mvst_view_{view}_smoke_8.npz", allow_pickle=False) as archive:
            archives.append({key: archive[key] for key in archive.files})
    reference_ids = archives[0]["cycle_id"].astype(str)
    reference_labels = archives[0]["label"].astype(np.int64)
    for view, archive in zip(VIEWS[1:], archives[1:]):
        if not np.array_equal(reference_ids, archive["cycle_id"].astype(str)):
            raise ValueError(f"cycle order mismatch for view {view}")
        if not np.array_equal(reference_labels, archive["label"].astype(np.int64)):
            raise ValueError(f"label order mismatch for view {view}")

    spec = importlib.util.spec_from_file_location("mvst_gated_fusion", args.author_repo / "gated_fusion.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.new_fuse(num_features=5, num_classes=4, has_logits=True).to(device)
    features = [torch.from_numpy(archive["embedding"]).float().to(device) for archive in archives]
    labels = torch.from_numpy(reference_labels).long().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad(set_to_none=True)
    logits = model(*features)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not torch.isfinite(loss) or not gradients_finite or not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite MVST fusion smoke")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "mvst_fusion_smoke_8_outputs.npz",
        cycle_id=reference_ids,
        label=reference_labels,
        logits=logits.detach().cpu().numpy(),
    )
    receipt = {
        "status": "passed",
        "protocol_name": "author_repo_random_file_split_official_like",
        "device": str(device),
        "view_order": VIEWS,
        "ordered_cycle_ids_identical": True,
        "ordered_labels_identical": True,
        "cycles": len(reference_ids),
        "feature_shapes": {view: list(feature.shape) for view, feature in zip(VIEWS, features)},
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
    }
    (args.output_dir / "mvst_fusion_smoke_8.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
