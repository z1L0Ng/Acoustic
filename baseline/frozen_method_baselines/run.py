"""Describe prepared frozen method baseline packages without executing models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import PLANS, SEEDS, asset_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--method", choices=tuple(PLANS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    plan = PLANS[args.method]
    print(
        json.dumps(
            {
                "execution_status": "DESIGN_ONLY_NOT_STARTED",
                "model_seed": args.seed,
                "plan": plan.to_dict(),
                "assets": asset_status(args.repo_root, plan),
                "next_phase": "requires separate implementation/run authorization",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
