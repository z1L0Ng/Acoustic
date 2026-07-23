from __future__ import annotations

import argparse

from .config import CORE_FEATURES
from .run import consolidate_full_results, run_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal ICBHI frozen-feature downstream benchmark")
    parser.add_argument("--backbone", choices=[*CORE_FEATURES, "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tasks", nargs="+", choices=["flat4", "binary"], default=["flat4", "binary"])
    args = parser.parse_args()
    backbones = CORE_FEATURES if args.backbone == "all" else [args.backbone]
    for backbone in backbones:
        run_backbone(backbone, args.mode, tuple(args.tasks))
    if args.mode == "full":
        consolidate_full_results()


if __name__ == "__main__":
    main()
