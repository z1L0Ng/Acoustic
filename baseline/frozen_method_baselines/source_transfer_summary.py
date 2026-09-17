"""Aggregate only completed PC-MCL/DCASE source-transfer seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .source_transfer_common import aggregate_seed_metrics, write_json


METHOD_ROOTS = {
    "pcmcl": "result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s",
    "dcase": "result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s",
}
SEEDS = (0, 1, 42)


def summarize(repo_root: Path, method: str) -> dict[str, object]:
    root = repo_root / METHOD_ROOTS[method]
    completed = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "run_summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") not in {
            "complete_test_selected_source_transfer",
            "complete_joint_native_union",
        }:
            continue
        completed.append(payload)
    summary = {
        "method": method,
        "completed_seeds": [int(row["seed"]) for row in completed],
        "n": len(completed),
        "required_seeds": list(SEEDS),
        "aggregate": aggregate_seed_metrics([row["metrics"] for row in completed]),
        "boundary": "only completed independent source-training seeds are included",
    }
    write_json(root / "multiseed_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--method", choices=tuple(METHOD_ROOTS), required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.repo_root.resolve(), args.method), indent=2))


if __name__ == "__main__":
    main()
