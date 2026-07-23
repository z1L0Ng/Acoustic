from __future__ import annotations

import argparse

from .extension import (
    audit_extension_inputs, run_extension_architecture, run_extension_imbalance,
    write_six_representation_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal remaining-three baseline extension")
    parser.add_argument("--phase", choices=["audit", "architecture", "imbalance", "all"], default="all")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    audit_extension_inputs()
    if args.phase in {"architecture", "all"}:
        run_extension_architecture(args.mode)
    if args.phase in {"imbalance", "all"}:
        run_extension_imbalance(args.mode)
    if args.mode == "full" and args.phase == "all":
        write_six_representation_receipt()


if __name__ == "__main__":
    main()
