"""Serial three-seed queue for the two approved source-baseline recipes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .pcmcl_numerics import nonfinite_training_issue
from .source_transfer_summary import summarize


SEEDS = (0, 1, 42)


def _call(arguments: list[str], repo_root: Path) -> None:
    subprocess.run(arguments, cwd=repo_root, check=True)


def _seed_state(result_dir: Path) -> tuple[str, Path | None]:
    summary = result_dir / "run_summary.json"
    if summary.is_file() and json.loads(summary.read_text()).get("status") in {
        "complete_test_selected_source_transfer",
        "complete_joint_native_union",
    }:
        return "complete", None
    last = result_dir / "last_checkpoint.pt"
    if last.is_file():
        return "resume", last
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"incomplete seed has no resumable last checkpoint: {result_dir}")
    return "fresh", None


def run_queue(repo_root: Path, methods: tuple[str, ...], device: str) -> dict[str, object]:
    python = sys.executable
    output = {}
    if "pcmcl" in methods:
        config = json.loads(
            (repo_root / "baseline/frozen_method_baselines/pcmcl_source_run.json").read_text()
        )
        for seed in SEEDS:
            result_dir = repo_root / str(config["output_root"]) / f"seed_{seed}"
            issue = nonfinite_training_issue(result_dir)
            if issue is not None:
                raise FloatingPointError(f"refusing to reuse {result_dir}: {issue}")
            state, resume = _seed_state(result_dir)
            if state == "complete":
                continue
            arguments = [
                python,
                "-m",
                "baseline.frozen_method_baselines.pcmcl_source_runner",
                "--repo-root",
                str(repo_root),
                "--seed",
                str(seed),
                "--device",
                device,
                "--run",
            ]
            if resume is not None:
                arguments.extend(("--resume", str(resume)))
            _call(
                arguments,
                repo_root,
            )
        output["pcmcl"] = summarize(repo_root, "pcmcl")
    if "dcase" in methods:
        config = json.loads(
            (repo_root / "baseline/frozen_method_baselines/dcase_joint_union_run.json").read_text()
        )
        cache_root = repo_root / str(config["frame_cache_root"])
        extraction_summary = cache_root / "extraction_summary.json"
        if not extraction_summary.is_file():
            if cache_root.exists() and any(cache_root.iterdir()):
                raise RuntimeError("partial DCASE frame cache has no extraction summary")
            _call(
                [
                    python,
                    "-m",
                    "baseline.frozen_method_baselines.dcase_joint_union_runner",
                    "--repo-root",
                    str(repo_root),
                    "--phase",
                    "extract-frames",
                    "--device",
                    device,
                    "--run",
                ],
                repo_root,
            )
        for seed in SEEDS:
            result_dir = repo_root / str(config["output_root"]) / f"seed_{seed}"
            state, resume = _seed_state(result_dir)
            if state == "complete":
                continue
            arguments = [
                python,
                "-m",
                "baseline.frozen_method_baselines.dcase_joint_union_runner",
                "--repo-root",
                str(repo_root),
                "--phase",
                "train",
                "--seed",
                str(seed),
                "--device",
                device,
                "--run",
            ]
            if resume is not None:
                arguments.extend(("--resume", str(resume)))
            _call(
                arguments,
                repo_root,
            )
        output["dcase"] = summarize(repo_root, "dcase")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--methods", default="pcmcl,dcase", help="comma-separated subset of pcmcl,dcase"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "NOT_RUN",
                    "methods": methods,
                    "seeds": SEEDS,
                    "order": "PC-MCL seeds, DCASE frame extraction, DCASE seeds",
                },
                indent=2,
            )
        )
        return
    print(json.dumps(run_queue(args.repo_root.resolve(), methods, args.device), indent=2))


if __name__ == "__main__":
    main()
