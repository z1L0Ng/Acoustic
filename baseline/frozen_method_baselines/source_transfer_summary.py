"""Aggregate only completed PC-MCL/DCASE source-transfer seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pcmcl_numerics import nonfinite_training_issue
from .source_transfer_common import aggregate_seed_metrics, write_json


METHOD_ROOTS = {
    "pcmcl": "result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_400epoch",
    "dcase": "result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s",
}
SEEDS = (0, 1, 42)


def summarize(
    repo_root: Path, method: str, output_root: Path | None = None
) -> dict[str, object]:
    requested_root = output_root or Path(METHOD_ROOTS[method])
    root = requested_root if requested_root.is_absolute() else repo_root / requested_root
    completed = []
    excluded = []
    for seed in SEEDS:
        result_dir = root / f"seed_{seed}"
        if method == "pcmcl":
            issue = nonfinite_training_issue(result_dir)
            if issue is not None:
                excluded.append({"seed": seed, "reason": issue})
                continue
        path = result_dir / "run_summary.json"
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
    if method == "pcmcl":
        summary["excluded_seeds"] = excluded
    write_json(root / "multiseed_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--method", choices=tuple(METHOD_ROOTS), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="explicit result root; relative paths are resolved under repo-root",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.repo_root.resolve(), args.method, args.output_root), indent=2
        )
    )


if __name__ == "__main__":
    main()
