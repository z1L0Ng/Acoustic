from __future__ import annotations

import argparse

from .config import CORE_FEATURES
from .imbalance import consolidate_imbalance_results, run_imbalance_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal ICBHI flat4 imbalance-loss comparison")
    parser.add_argument("--backbone", choices=[*CORE_FEATURES, "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    backbones = CORE_FEATURES if args.backbone == "all" else [args.backbone]
    for backbone in backbones:
        run_imbalance_backbone(backbone, args.mode)
    if args.mode == "full":
        consolidate_imbalance_results()


if __name__ == "__main__":
    main()
