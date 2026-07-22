"""Train MVST fusion with strict five-view ID checks and export predictions."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


VIEWS = ["16", "32", "64", "128", "256"]


def load_split(root: Path, split: str):
    archives = []
    for view in VIEWS:
        with np.load(root / view / f"{split}.npz", allow_pickle=False) as archive:
            archives.append({key: archive[key] for key in archive.files})
    cycle_ids = archives[0]["cycle_id"].astype(str)
    labels = archives[0]["label"].astype(np.int64)
    for view, archive in zip(VIEWS[1:], archives[1:]):
        if not np.array_equal(cycle_ids, archive["cycle_id"].astype(str)):
            raise ValueError(f"cycle ID/order mismatch for {view}/{split}")
        if not np.array_equal(labels, archive["label"].astype(np.int64)):
            raise ValueError(f"label/order mismatch for {view}/{split}")
    features = [torch.from_numpy(archive["embedding"]).float() for archive in archives]
    return cycle_ids, labels, features


def score(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    normal = labels == 0
    abnormal = ~normal
    sp = float((predictions[normal] == 0).mean())
    se = float((predictions[abnormal] == labels[abnormal]).mean())
    return sp, se, (sp + se) / 2


def evaluate(model, loader, device):
    model.eval()
    logits_parts, labels_parts = [], []
    with torch.no_grad():
        for *features, labels in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*(feature.to(device, non_blocking=device.type == "cuda") for feature in features))
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels)
    return torch.cat(logits_parts), torch.cat(labels_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fusion requested but CUDA is unavailable")
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    train_ids, train_labels, train_features = load_split(args.feature_root, "train")
    test_ids, test_labels, test_features = load_split(args.feature_root, "test")
    train_data = TensorDataset(*train_features, torch.from_numpy(train_labels).long())
    test_data = TensorDataset(*test_features, torch.from_numpy(test_labels).long())
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

    spec = importlib.util.spec_from_file_location("mvst_gated_fusion", args.author_repo / "gated_fusion.py")
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    model = module.new_fuse(num_features=5, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-15)
    class_counts = np.bincount(train_labels, minlength=4)
    # Preserve author fusion.py exactly: frequency-proportional, not inverse, weights.
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32, device=device)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_score = 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); start_epoch = state["epoch"] + 1
        best_score = state["best_score"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for *features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(*(feature.to(device, non_blocking=device.type == "cuda") for feature in features))
                loss = criterion(logits, labels.to(device, non_blocking=device.type == "cuda"))
            loss.backward(); optimizer.step()
        scheduler.step()
        test_logits, observed_labels = evaluate(model, test_loader, device)
        predictions = test_logits.argmax(dim=1).numpy()
        sp, se, current_score = score(observed_labels.numpy(), predictions)
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_score": max(best_score, current_score),
            "protocol": "author_repo_random_file_split_official_like_test_selected",
        }
        torch.save(state, args.output_dir / "last.pth")
        if current_score > best_score and se > 0.05:
            best_score = current_score
            state["best_score"] = best_score
            torch.save(state, args.output_dir / "best.pth")
        print(f"epoch={epoch} sp={sp:.6f} se={se:.6f} score={current_score:.6f} best={best_score:.6f}")

    best = torch.load(args.output_dir / "best.pth", map_location=device)
    model.load_state_dict(best["model"])
    test_logits, observed_labels = evaluate(model, test_loader, device)
    np.savez_compressed(
        args.output_dir / "official_like_test_outputs.npz",
        cycle_id=test_ids,
        label=observed_labels.numpy(),
        logits=test_logits.numpy(),
    )
    print(f"mvst_fusion_ok test_rows={len(test_ids)} best_score={best['best_score']:.6f}")


if __name__ == "__main__":
    main()
