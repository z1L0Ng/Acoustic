from __future__ import annotations

import argparse

from .strict_patient_v3 import run_strict_patient_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal strict-patient BEATs long-tail benchmark v3")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute complete runs instead of safely reusing verified per-run artifacts.",
    )
    args = parser.parse_args()
    rows = run_strict_patient_v3(mode=args.mode, resume=not args.no_resume)
    print(f"strict_patient_v3_complete mode={args.mode} rows={len(rows)}")


if __name__ == "__main__":
    main()
