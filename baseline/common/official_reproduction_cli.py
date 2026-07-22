"""Shared fresh-checkout CLI for Patch-Mix CL and PAFA Release 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .official_reproduction_bootstrap import SPECS, bootstrap, verify_bootstrap


SMOKE_MODULES = {
    "patch_mix_cl": "baseline.patch_mix_cl.patch_mix_smoke",
    "pafa": "baseline.pafa.pafa_smoke",
}


def read_bootstrap(method: str, result_root: Path) -> dict:
    path = result_root.resolve() / "receipts" / "bootstrap_receipt.json"
    receipt = json.loads(path.read_text())
    if receipt["method"] != method:
        raise ValueError(f"bootstrap method mismatch: {receipt['method']} != {method}")
    if receipt["minimum_release"] != "official-reproduction-release-1":
        raise ValueError(f"unsupported bootstrap receipt: {receipt['minimum_release']}")
    return receipt


def runtime_environment(receipt: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(receipt["environment"])
    for path in receipt["environment"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return environment


def run_smoke(method: str, result_root: Path, device: str, steps: int, profile: bool) -> None:
    result_root = result_root.resolve()
    receipt = read_bootstrap(method, result_root)
    spec = SPECS[method]
    stage = "profile" if profile else "smoke"
    command = [
        sys.executable, "-m", SMOKE_MODULES[method],
        "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
        "--author-repo", str(result_root / "source" / "repo"),
        "--checkpoint", str(result_root / "checkpoints" / spec.checkpoint_filename),
        "--work-dir", str(result_root / "work" / stage),
        "--output-dir", str(result_root / stage),
        "--steps", str(steps),
        "--device", device,
    ]
    subprocess.run(command, check=True, env=runtime_environment(receipt))


def export_and_verify(
    method: str, result_root: Path, device: str, trained_checkpoint: Path | None,
) -> None:
    export_command = [
        sys.executable, "-m", "baseline.common.export_official_predictions",
        "--method", method,
        "--result-root", str(result_root.resolve()),
        "--device", device,
    ]
    if trained_checkpoint is not None:
        export_command.extend(["--trained-checkpoint", str(trained_checkpoint)])
    receipt = read_bootstrap(method, result_root)
    environment = runtime_environment(receipt)
    subprocess.run(export_command, check=True, env=environment)
    prediction_root = result_root.resolve() / "predictions"
    subprocess.run([
        sys.executable, "-m", "baseline.common.verify_official_predictions",
        "--input-npz", str(prediction_root / "official_test_outputs.npz"),
        "--output-dir", str(prediction_root / "verified"),
        "--manifest", str(result_root.resolve() / "manifest" / "icbhi_2017_cycle_manifest.csv"),
        "--expected-split", "test",
        "--expected-rows", "2756",
    ], check=True, env=environment)


def run_full(
    method: str, project_root: Path, result_root: Path, device: str,
    resume: Path | None,
) -> None:
    result_root = result_root.resolve()
    receipt = read_bootstrap(method, result_root)
    if not device.startswith("cuda"):
        raise ValueError("faithful full entry requires --device cuda or cuda:<index>")
    cuda_index = device.split(":", 1)[1] if ":" in device else "0"
    environment = runtime_environment(receipt)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_index
    if resume is not None:
        resume = resume.resolve()
        try:
            resume.relative_to(result_root)
        except ValueError as error:
            raise ValueError("resume checkpoint must be inside the current result root") from error
        environment["RESUME_CHECKPOINT"] = str(resume)
    script = project_root.resolve() / "baseline" / method / "cuda_full_command.sh"
    subprocess.run(
        ["bash", str(script)],
        cwd=result_root / "portable_run",
        check=True,
        env=environment,
    )
    export_and_verify(method, result_root, device, trained_checkpoint=None)


def add_result_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result-root", type=Path, required=True)


def main_for(method: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Fresh-checkout {method} official-like reproduction entry point"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--dataset-root", type=Path, required=True)
    add_result_root(bootstrap_parser)
    bootstrap_parser.add_argument("--cache-root", type=Path)
    bootstrap_parser.add_argument("--checkpoint-path", type=Path)
    bootstrap_parser.add_argument("--checkpoint-url")
    bootstrap_parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])

    verify_parser = subparsers.add_parser("verify-bootstrap")
    add_result_root(verify_parser)

    smoke_parser = subparsers.add_parser("smoke")
    add_result_root(smoke_parser)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--steps", type=int, default=1)

    profile_parser = subparsers.add_parser("profile")
    add_result_root(profile_parser)
    profile_parser.add_argument("--device", default="cuda")
    profile_parser.add_argument("--steps", type=int, default=100)

    full_parser = subparsers.add_parser("full")
    add_result_root(full_parser)
    full_parser.add_argument("--device", default="cuda")
    full_parser.add_argument("--resume", type=Path)

    export_parser = subparsers.add_parser("export-and-verify")
    add_result_root(export_parser)
    export_parser.add_argument("--device", default="cuda")
    export_parser.add_argument("--trained-checkpoint", type=Path)

    args = parser.parse_args()
    if args.command == "bootstrap":
        result = bootstrap(
            method=method,
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            result_root=args.result_root,
            cache_root=args.cache_root,
            checkpoint_path=args.checkpoint_path,
            checkpoint_url=args.checkpoint_url,
            device=args.device,
        )
    elif args.command == "verify-bootstrap":
        result = verify_bootstrap(method, args.result_root)
    elif args.command == "smoke":
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        run_smoke(method, args.result_root, args.device, args.steps, profile=False)
        return
    elif args.command == "profile":
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        run_smoke(method, args.result_root, args.device, args.steps, profile=True)
        return
    elif args.command == "full":
        run_full(method, args.project_root, args.result_root, args.device, args.resume)
        return
    elif args.command == "export-and-verify":
        export_and_verify(method, args.result_root, args.device, args.trained_checkpoint)
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
