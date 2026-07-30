"""Independent verifier for the matched PAFA frozen-encoder pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline.common.frozen_encoder_target_heads import verify_package, verify_pilot
from baseline.pafa.checkpoint_eval.bootstrap import (
    ACCEPTED_CHECKPOINT_SHA256,
    BACKBONE_SHA256,
)

from .run import EXPERIMENT_ID, PROTOCOL


PACKAGE_FILES = {
    "baseline/common/frozen_encoder_target_heads.py",
    "baseline/pafa/frozen_encoder_target_heads/__init__.py",
    "baseline/pafa/frozen_encoder_target_heads/README.md",
    "baseline/pafa/frozen_encoder_target_heads/profile_full_finetune.py",
    "baseline/pafa/frozen_encoder_target_heads/protocol.json",
    "baseline/pafa/frozen_encoder_target_heads/run.py",
    "baseline/pafa/frozen_encoder_target_heads/verify.py",
    "experiments/sprsound_pafa_frozen_encoder_target_heads.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["package", "profile", "full"], required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.mode == "package":
        package = Path(__file__).resolve().parent
        verify_package(
            package_dir=package,
            project_root=package.parents[2],
            relative_files=PACKAGE_FILES,
            experiment_id=EXPERIMENT_ID,
        )
        return
    if args.result_root is None or args.cache_root is None:
        parser.error("--result-root and --cache-root are required")
    verify_pilot(
        mode=args.mode,
        result_root=args.result_root,
        cache_root=args.cache_root,
        experiment_id=EXPERIMENT_ID,
        protocol_name=PROTOCOL,
        method_id="pafa",
        task_checkpoint_sha256=ACCEPTED_CHECKPOINT_SHA256,
        backbone_checkpoint_sha256=BACKBONE_SHA256,
    )


if __name__ == "__main__":
    main()
