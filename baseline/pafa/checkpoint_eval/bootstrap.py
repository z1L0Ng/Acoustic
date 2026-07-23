"""Pin PAFA source and verify server checkpoint identities."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from baseline.checkpoint_eval_common.sprsound_inter import sha256_file, write_json


REPO_URL = "https://github.com/wa976/PAFA"
REPO_COMMIT = "e49e294d0db0d6af10ac46290512b9c85d3f71e1"
BACKBONE_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"


def command(parts: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(parts, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-sha256", default=BACKBONE_SHA256)
    parser.add_argument("--expected-epoch", type=int, default=27)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if not root.name.startswith("pafa_sprsound_transfer_"):
        raise ValueError("result root must be result/pafa_sprsound_transfer_<timestamp>")
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "source" / "repo"
    if args.source_repo:
        if repo.exists():
            raise FileExistsError(repo)
        shutil.copytree(args.source_repo.resolve(), repo, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", REPO_URL, str(repo)])
    command(["git", "checkout", "--detach", REPO_COMMIT], cwd=repo)
    if command(["git", "rev-parse", "HEAD"], cwd=repo) != REPO_COMMIT:
        raise RuntimeError("PAFA source commit mismatch")

    task_checkpoint = args.checkpoint.resolve()
    backbone = args.backbone_checkpoint.resolve()
    task_sha = sha256_file(task_checkpoint)
    backbone_sha = sha256_file(backbone)
    if task_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("PAFA task checkpoint SHA mismatch")
    if backbone_sha != args.backbone_sha256.lower():
        raise RuntimeError("PAFA BEATs backbone SHA mismatch")
    state = torch.load(task_checkpoint, map_location="cpu")
    required = {"epoch", "model", "classifier"}
    if not isinstance(state, dict) or not required <= set(state) or int(state["epoch"]) != args.expected_epoch:
        raise RuntimeError(f"PAFA checkpoint completeness/epoch mismatch: {set(state)}")
    if not isinstance(state["model"], dict) or not isinstance(state["classifier"], dict):
        raise RuntimeError("PAFA model/classifier checkpoint states must be mappings")
    receipt = {
        "status": "source_and_checkpoint_structure_verified",
        "method_id": "pafa",
        "source_repo": REPO_URL,
        "source_commit": REPO_COMMIT,
        "license": "no LICENSE in author repository; do not redistribute source/checkpoint without review",
        "task_checkpoint_path": str(task_checkpoint),
        "task_checkpoint_sha256": task_sha,
        "task_checkpoint_size_bytes": task_checkpoint.stat().st_size,
        "task_checkpoint_epoch": int(state["epoch"]),
        "task_checkpoint_keys": sorted(state),
        "model_state_keys": len(state["model"]),
        "classifier_state_keys": len(state["classifier"]),
        "backbone_checkpoint_path": str(backbone),
        "backbone_checkpoint_sha256": backbone_sha,
        "backbone_identity_boundary": "structurally audited BEATs_iter3+ AS2M mirror used by server reproduction",
        "source_numerical_status": {
            "server_best_epoch": 27,
            "specificity": 76.884,
            "sensitivity": 51.402,
            "score": 64.143,
            "paper_five_seed_mean_sd": {"specificity": "82.05+/-1.95", "sensitivity": "47.63+/-2.23", "score": "64.84+/-0.60"},
            "verdict": "one-seed headline Score reasonably aligned (-0.70 pp, about -1.16 SD); specificity lower and sensitivity higher; not componentwise or five-seed aggregate replication",
        },
        "selection": "ICBHI official-test-selected checkpoint",
        "transfer_boundary": "published-model/verified-server-checkpoint exploratory transfer; zero target tuning",
    }
    write_json(root / "receipts" / "bootstrap.json", receipt)
    print(f"pafa_transfer_bootstrap_ok epoch={state['epoch']} sha={task_sha}")


if __name__ == "__main__":
    main()
