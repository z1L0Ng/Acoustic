"""One-step PAFA full-encoder profile; this does not start a training run."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

from baseline.common.frozen_encoder_target_heads import verify_source_repo
from baseline.patch_mix_cl.frozen_encoder_target_heads.run import (
    LinearTargetHead,
    build_manifest,
    peak_rss_gib,
    write_json,
)

from .run import REPO_COMMIT, make_encoder_builder, preprocess_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("the bounded profile is fixed at batch size 8")
    cache_root = Path(".cache/sprsound_pafa_frozen_encoder_target_heads")
    for variable, relative in (
        ("NUMBA_CACHE_DIR", "runtime/numba"),
        ("MPLCONFIGDIR", "runtime/matplotlib"),
        ("XDG_CACHE_HOME", "runtime/xdg"),
    ):
        path = (cache_root / relative).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    repo = verify_source_repo(args.source_repo, REPO_COMMIT)
    rows = [
        row
        for row in build_manifest(args.dataset_root)
        if row["partition"] == "train" and row["inner_split"] == "subtrain"
    ][: args.batch_size]
    audio = torch.stack([preprocess_recording([row], repo)[0] for row in rows])
    target = torch.tensor(
        [0 if row["raw_label"] == "Normal" else 1 for row in rows],
        dtype=torch.long,
    )
    encoder, _ = make_encoder_builder(
        args.checkpoint,
        "94afaed43a1546af26f9d8d99d2d27329cb8d348fd57cbe142d24310c68ca2b6",
        args.backbone_checkpoint,
    )(repo, torch.device("cpu"))
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    head = LinearTargetHead(2)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-3)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    embeddings = encoder(audio, training=False).mean(dim=1)
    logits = head(embeddings)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    if not torch.isfinite(loss) or not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in [*encoder.parameters(), *head.parameters()]
    ):
        raise RuntimeError("non-finite PAFA full-finetune profile")
    optimizer.step()
    step_seconds = time.perf_counter() - started
    projected_seconds_lower_bound = step_seconds * math.ceil(5_219 / args.batch_size) * 5
    receipt = {
        "method": "pafa",
        "scope": "one optimizer step only; no checkpoint saved; no full training started",
        "task": "representative binary target head",
        "batch_size": args.batch_size,
        "step_seconds": step_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "projected_five_epoch_train_seconds_lower_bound": projected_seconds_lower_bound,
        "projection_excludes": [
            "audio preprocessing",
            "validation",
            "final inter inference",
        ],
        "local_90_minute_gate_passed": False,
        "local_gate_status": (
            "FAIL_EXCEEDS_90_MINUTES_LOWER_BOUND"
            if projected_seconds_lower_bound > 5_400
            else "INDETERMINATE_FAIL_CLOSED_REQUIRED_COSTS_EXCLUDED"
        ),
    }
    write_json(args.result_root / "full_finetune_profile.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
