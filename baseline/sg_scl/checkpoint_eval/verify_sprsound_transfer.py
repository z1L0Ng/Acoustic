"""Independent SG-SCL SPRSound transfer verification wrapper."""

import argparse
from pathlib import Path

from acoustic.evaluation.sprsound_inter import verify_full_result, verify_smoke_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if args.mode == "smoke":
        receipt = verify_smoke_result(root, args.dataset_root, "sg_scl")
        print(f"sg_scl_sprsound_smoke_verification_ok events={receipt['events']} labels=absent metrics=absent")
        return
    receipt = verify_full_result(root, args.dataset_root, "sg_scl")
    print(f"sg_scl_sprsound_full_verification_ok events={receipt['events']}")


if __name__ == "__main__":
    main()
