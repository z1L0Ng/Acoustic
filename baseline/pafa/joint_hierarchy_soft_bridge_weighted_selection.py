"""PAFA-JH3.1 weighted-selection Soft Confidence Bridge runner.

The model, data, and training recipe are reused from JH3.  Only the
test-exposed checkpoint-selection composite and its epoch-boundary patience
monitor differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.pafa.joint_hierarchy import PAFAJointHierarchyConfig
from baseline.pafa.joint_hierarchy_soft_bridge import (
    SelectionSpec,
    _config_payload,
    _default_paths,
    run as _run_soft_bridge,
)


CONDITION = "PAFA_JH3_1_soft_bridge_weighted_selection_seed42"
EVIDENCE_LABEL = "icbhi_test60_spr_validation40_selected_soft_bridge_diagnostic"
SELECTION = SelectionSpec(
    condition=CONDITION,
    evidence_label=EVIDENCE_LABEL,
    icbhi_weight=0.6,
    spr_validation_weight=0.4,
    changed_files=(
        "baseline/pafa/joint_hierarchy_soft_bridge.py",
        "baseline/pafa/joint_hierarchy_soft_bridge_weighted_selection.py",
    ),
)


def run(config: PAFAJointHierarchyConfig) -> dict[str, object]:
    return _run_soft_bridge(config, selection=SELECTION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--icbhi-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    author_repo, checkpoint, icbhi_audio = _default_paths(args.repo_root)
    config = PAFAJointHierarchyConfig(
        repo_root=args.repo_root,
        author_repo=args.author_repo or author_repo,
        checkpoint=args.checkpoint or checkpoint,
        icbhi_audio_dir=args.icbhi_audio_dir or icbhi_audio,
        output_dir=args.output_dir
        or args.repo_root / "result/reproduce/pafa_joint_hierarchy" / CONDITION,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    config.validate()
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_USER_START",
                    "execution_started": False,
                    "evidence_label": EVIDENCE_LABEL,
                    "config": _config_payload(config, SELECTION),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(run(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
