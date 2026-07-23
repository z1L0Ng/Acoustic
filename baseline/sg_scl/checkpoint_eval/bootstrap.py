"""Pin SG-SCL source and verify the server task-checkpoint contract."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from acoustic.evaluation.sprsound_inter import sha256_file, write_json


REPO_URL = "https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning"
REPO_COMMIT = "66564609595090b61540595d3d27764c00553086"
ACCEPTED_CHECKPOINT_SHA256 = "8b3652d0dc82b9033251e3aab50ec1b51d328b6d3c836807b7c8de571581c256"
ACCEPTED_CHECKPOINT_SIZE_BYTES = 1_413_896_983
CONTAINER_SAVED_AT_EPOCH = 50
SELECTED_BEST_EPOCH = 27
SOURCE_PREDICTION_CSV_SHA256 = "2ed6abb6fb0a05f97c2325881c2372b6d803bd5290bc2d3f212a0b44b0867b90"
SOURCE_CONFUSION = [
    [1184, 274, 86, 35],
    [217, 400, 11, 21],
    [164, 38, 105, 78],
    [46, 22, 27, 48],
]


def command(parts: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(parts, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def verify_checkpoint_identity(path: Path, caller_sha256: str) -> str:
    if caller_sha256.lower() != ACCEPTED_CHECKPOINT_SHA256:
        raise RuntimeError("caller SG-SCL checkpoint SHA does not match the accepted server artifact")
    if path.stat().st_size != ACCEPTED_CHECKPOINT_SIZE_BYTES:
        raise RuntimeError("SG-SCL task checkpoint size does not match the accepted server artifact")
    checkpoint_sha = sha256_file(path)
    if checkpoint_sha != ACCEPTED_CHECKPOINT_SHA256:
        raise RuntimeError("SG-SCL task checkpoint SHA does not match the accepted server artifact")
    return checkpoint_sha


def _assert_tensor_state_equal(name: str, top_level: object, embedded: object) -> int:
    if not isinstance(top_level, dict) or not isinstance(embedded, dict):
        raise RuntimeError(f"SG-SCL {name} states must be mappings")
    if set(top_level) != set(embedded):
        raise RuntimeError(f"SG-SCL top-level {name} keys differ from embedded best_model")
    for key in top_level:
        top_value, embedded_value = top_level[key], embedded[key]
        if not torch.is_tensor(top_value) or not torch.is_tensor(embedded_value):
            raise RuntimeError(f"SG-SCL {name}.{key} is not a tensor in both states")
        if not torch.equal(top_value, embedded_value):
            raise RuntimeError(f"SG-SCL top-level {name}.{key} differs from embedded best_model")
    return len(top_level)


def verify_checkpoint_state(state: object) -> dict[str, int]:
    required = {"epoch", "model", "classifier", "best_model"}
    if not isinstance(state, dict) or not required <= set(state):
        raise RuntimeError("SG-SCL checkpoint is missing container or predictive best-state fields")
    if int(state["epoch"]) != CONTAINER_SAVED_AT_EPOCH:
        raise RuntimeError(
            f"expected SG-SCL container saved at epoch {CONTAINER_SAVED_AT_EPOCH}, got {state['epoch']}"
        )
    best_model = state["best_model"]
    if not isinstance(best_model, (list, tuple)) or len(best_model) < 2:
        raise RuntimeError("SG-SCL best_model must contain model and classifier states")
    counts = {
        "model_state_keys": _assert_tensor_state_equal("model", state["model"], best_model[0]),
        "classifier_state_keys": _assert_tensor_state_equal(
            "classifier", state["classifier"], best_model[1]
        ),
    }
    classifier_weight = state["classifier"].get("1.weight")
    if classifier_weight is None or tuple(classifier_weight.shape) != (4, 768):
        raise RuntimeError("SG-SCL classifier does not match normal/crackle/wheeze/both")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
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
    checkpoint_sha = verify_checkpoint_identity(checkpoint, args.checkpoint_sha256)
    sys.path.insert(0, str(repo))
    state = torch.load(checkpoint, map_location="cpu")
    state_counts = verify_checkpoint_state(state)
    classifier_weight = state["classifier"].get("1.weight")
    receipt = {
        "status": "source_and_accepted_best_checkpoint_contract_verified",
        "method_id": "sg_scl",
        "source_repo": REPO_URL,
        "source_commit": REPO_COMMIT,
        "license": "no LICENSE in author repository; do not redistribute source/checkpoint without review",
        "task_checkpoint_path": str(checkpoint),
        "task_checkpoint_sha256": checkpoint_sha,
        "task_checkpoint_size_bytes": checkpoint.stat().st_size,
        "container_saved_at_epoch": int(state["epoch"]),
        "selected_best_epoch": SELECTED_BEST_EPOCH,
        "task_checkpoint_keys": sorted(state),
        **state_counts,
        "top_level_predictive_states_equal_embedded_best_model": {
            "model": True,
            "classifier": True,
        },
        "server_audit_final_last_checkpoint_differs": True,
        "classifier_weight_shape": list(classifier_weight.shape),
        "source_numerical_status": {
            "evidence_origin": "accepted_server_audit_not_recomputed_by_transfer_bootstrap",
            "selected_best_epoch": SELECTED_BEST_EPOCH,
            "specificity": 74.9842,
            "sensitivity": 46.9839,
            "score": 60.9840,
            "confusion": SOURCE_CONFUSION,
            "prediction_csv_sha256": SOURCE_PREDICTION_CSV_SHA256,
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
    print(
        "sg_scl_transfer_bootstrap_ok "
        f"container_epoch={state['epoch']} selected_best_epoch={SELECTED_BEST_EPOCH} "
        f"sha={checkpoint_sha}"
    )


if __name__ == "__main__":
    main()
