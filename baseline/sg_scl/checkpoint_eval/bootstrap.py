"""Pin SG-SCL source and verify the server task-checkpoint contract."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from baseline.checkpoint_eval_common.sprsound_inter import sha256_file, write_json


REPO_URL = "https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning"
REPO_COMMIT = "66564609595090b61540595d3d27764c00553086"


def command(parts: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(parts, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-epoch", type=int, default=27)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if not root.name.startswith("sg_scl_sprsound_transfer_"):
        raise ValueError("result root must be result/sg_scl_sprsound_transfer_<timestamp>")
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "source" / "repo"
    if args.source_repo:
        if repo.exists():
            raise FileExistsError(repo)
        shutil.copytree(
            args.source_repo.resolve(),
            repo,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    elif not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", REPO_URL, str(repo)])
    command(["git", "checkout", "--detach", REPO_COMMIT], cwd=repo)
    if command(["git", "rev-parse", "HEAD"], cwd=repo) != REPO_COMMIT:
        raise RuntimeError("SG-SCL source commit mismatch")

    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("SG-SCL task checkpoint SHA mismatch")
    state = torch.load(checkpoint, map_location="cpu")
    required = {"epoch", "model", "classifier"}
    if not isinstance(state, dict) or not required <= set(state):
        raise RuntimeError("SG-SCL checkpoint is missing predictive state")
    if int(state["epoch"]) != args.expected_epoch:
        raise RuntimeError(f"expected SG-SCL epoch {args.expected_epoch}, got {state['epoch']}")
    if not isinstance(state["model"], dict) or not isinstance(state["classifier"], dict):
        raise RuntimeError("SG-SCL model/classifier checkpoint states must be mappings")
    classifier_weight = state["classifier"].get("1.weight")
    if classifier_weight is None or tuple(classifier_weight.shape) != (4, 768):
        raise RuntimeError("SG-SCL classifier does not match normal/crackle/wheeze/both")
    receipt = {
        "status": "source_and_predictive_checkpoint_structure_verified",
        "method_id": "sg_scl",
        "source_repo": REPO_URL,
        "source_commit": REPO_COMMIT,
        "license": "no LICENSE in author repository; do not redistribute source/checkpoint without review",
        "task_checkpoint_path": str(checkpoint),
        "task_checkpoint_sha256": checkpoint_sha,
        "task_checkpoint_size_bytes": checkpoint.stat().st_size,
        "task_checkpoint_epoch": int(state["epoch"]),
        "task_checkpoint_keys": sorted(state),
        "model_state_keys": len(state["model"]),
        "classifier_state_keys": len(state["classifier"]),
        "classifier_weight_shape": list(classifier_weight.shape),
        "source_numerical_status": {
            "server_best_epoch": 27,
            "specificity": 74.984,
            "sensitivity": 46.984,
            "score": 60.984,
            "paper_five_seed_mean_sd": {
                "specificity": "79.87+/-8.89",
                "sensitivity": "43.55+/-5.93",
                "score": "61.71+/-1.61"
            },
            "verdict": "all components within about 0.6 SD; successful single-seed numerical alignment, not five-seed aggregate reproduction"
        },
        "selection": "ICBHI official-test-selected checkpoint",
        "metadata_boundary": "metadata/device-aware source training; author validation uses only audio features and class classifier",
        "transfer_boundary": "published-model/verified-server-checkpoint exploratory transfer; zero target tuning"
    }
    write_json(root / "receipts" / "bootstrap.json", receipt)
    print(f"sg_scl_transfer_bootstrap_ok epoch={state['epoch']} sha={checkpoint_sha}")


if __name__ == "__main__":
    main()
