"""Pin PAFA source and verify server checkpoint identities."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from acoustic.evaluation.sprsound_inter import sha256_file, write_json


REPO_URL = "https://github.com/wa976/PAFA"
REPO_COMMIT = "e49e294d0db0d6af10ac46290512b9c85d3f71e1"
BACKBONE_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
ACCEPTED_CHECKPOINT_SHA256 = "94afaed43a1546af26f9d8d99d2d27329cb8d348fd57cbe142d24310c68ca2b6"
ACCEPTED_CHECKPOINT_SIZE_BYTES = 1_464_382_039
CONTAINER_SAVED_AT_EPOCH = 100
SELECTED_BEST_EPOCH = 27
SOURCE_PREDICTION_CSV_SHA256 = "b4102a572c5ba3a755958c238734837b182d65cb0e50981c5ffa6fafc00c6d4a"
SOURCE_CONFUSION = [
    [1214, 251, 87, 27],
    [232, 401, 6, 10],
    [127, 31, 159, 68],
    [34, 18, 46, 45],
]


def command(parts: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(parts, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def verify_checkpoint_identity(path: Path, caller_sha256: str) -> str:
    if caller_sha256.lower() != ACCEPTED_CHECKPOINT_SHA256:
        raise RuntimeError("caller PAFA checkpoint SHA does not match the accepted server artifact")
    if path.stat().st_size != ACCEPTED_CHECKPOINT_SIZE_BYTES:
        raise RuntimeError("PAFA task checkpoint size does not match the accepted server artifact")
    checkpoint_sha = sha256_file(path)
    if checkpoint_sha != ACCEPTED_CHECKPOINT_SHA256:
        raise RuntimeError("PAFA task checkpoint SHA does not match the accepted server artifact")
    return checkpoint_sha


def _assert_tensor_state_equal(name: str, top_level: object, embedded: object) -> int:
    if not isinstance(top_level, dict) or not isinstance(embedded, dict):
        raise RuntimeError(f"PAFA {name} states must be mappings")
    if set(top_level) != set(embedded):
        raise RuntimeError(f"PAFA top-level {name} keys differ from embedded best_model")
    for key in top_level:
        top_value, embedded_value = top_level[key], embedded[key]
        if not torch.is_tensor(top_value) or not torch.is_tensor(embedded_value):
            raise RuntimeError(f"PAFA {name}.{key} is not a tensor in both states")
        if not torch.equal(top_value, embedded_value):
            raise RuntimeError(f"PAFA top-level {name}.{key} differs from embedded best_model")
    return len(top_level)


def verify_checkpoint_state(state: object) -> dict[str, int]:
    required = {"epoch", "model", "classifier", "projector", "best_model"}
    if not isinstance(state, dict) or not required <= set(state):
        raise RuntimeError("PAFA checkpoint is missing container or predictive best-state fields")
    if int(state["epoch"]) != CONTAINER_SAVED_AT_EPOCH:
        raise RuntimeError(
            f"expected PAFA container saved at epoch {CONTAINER_SAVED_AT_EPOCH}, got {state['epoch']}"
        )
    best_model = state["best_model"]
    if not isinstance(best_model, (list, tuple)) or len(best_model) < 3:
        raise RuntimeError("PAFA best_model must contain model, classifier, and projector states")
    return {
        "model_state_keys": _assert_tensor_state_equal("model", state["model"], best_model[0]),
        "classifier_state_keys": _assert_tensor_state_equal(
            "classifier", state["classifier"], best_model[1]
        ),
        "projector_state_keys": _assert_tensor_state_equal(
            "projector", state["projector"], best_model[2]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-sha256", default=BACKBONE_SHA256)
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
    task_sha = verify_checkpoint_identity(task_checkpoint, args.checkpoint_sha256)
    backbone_sha = sha256_file(backbone)
    if backbone_sha != args.backbone_sha256.lower():
        raise RuntimeError("PAFA BEATs backbone SHA mismatch")
    state = torch.load(task_checkpoint, map_location="cpu")
    state_counts = verify_checkpoint_state(state)
    receipt = {
        "status": "source_and_accepted_best_checkpoint_contract_verified",
        "method_id": "pafa",
        "source_repo": REPO_URL,
        "source_commit": REPO_COMMIT,
        "license": "no LICENSE in author repository; do not redistribute source/checkpoint without review",
        "task_checkpoint_path": str(task_checkpoint),
        "task_checkpoint_sha256": task_sha,
        "task_checkpoint_size_bytes": task_checkpoint.stat().st_size,
        "container_saved_at_epoch": int(state["epoch"]),
        "selected_best_epoch": SELECTED_BEST_EPOCH,
        "task_checkpoint_keys": sorted(state),
        **state_counts,
        "top_level_predictive_states_equal_embedded_best_model": {
            "model": True,
            "classifier": True,
            "projector": True,
        },
        "server_audit_final_last_checkpoint_differs": True,
        "backbone_checkpoint_path": str(backbone),
        "backbone_checkpoint_sha256": backbone_sha,
        "backbone_identity_boundary": "structurally audited BEATs_iter3+ AS2M mirror used by server reproduction",
        "source_numerical_status": {
            "evidence_origin": "accepted_server_audit_not_recomputed_by_transfer_bootstrap",
            "selected_best_epoch": SELECTED_BEST_EPOCH,
            "specificity": 76.8841,
            "sensitivity": 51.4019,
            "score": 64.1430,
            "confusion": SOURCE_CONFUSION,
            "prediction_csv_sha256": SOURCE_PREDICTION_CSV_SHA256,
            "paper_five_seed_mean_sd": {"specificity": "82.05+/-1.95", "sensitivity": "47.63+/-2.23", "score": "64.84+/-0.60"},
            "verdict": "one-seed headline Score reasonably aligned (-0.70 pp, about -1.16 SD); specificity lower and sensitivity higher; not componentwise or five-seed aggregate replication",
        },
        "selection": "ICBHI official-test-selected checkpoint",
        "transfer_boundary": "published-model/verified-server-checkpoint exploratory transfer; zero target tuning",
    }
    write_json(root / "receipts" / "bootstrap.json", receipt)
    print(
        "pafa_transfer_bootstrap_ok "
        f"container_epoch={state['epoch']} selected_best_epoch={SELECTED_BEST_EPOCH} sha={task_sha}"
    )


if __name__ == "__main__":
    main()
