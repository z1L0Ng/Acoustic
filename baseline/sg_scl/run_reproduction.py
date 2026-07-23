"""Fresh-checkout SG-SCL bootstrap, smoke, profile, full, and export CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from baseline.sg_scl.reproduction_bootstrap import (
    METHOD,
    SPEC,
    bootstrap,
    verify_bootstrap,
)


def read_receipt(result_root: Path) -> dict:
    receipt = json.loads((result_root.resolve() / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != METHOD:
        raise ValueError("SG-SCL bootstrap/method mismatch")
    return receipt


def runtime_environment(receipt: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(receipt["environment"])
    for path in receipt["environment"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return environment


def smoke(result_root: Path, device: str, steps: int, profile: bool) -> None:
    result_root = result_root.resolve()
    receipt = read_receipt(result_root)
    stage = "profile" if profile else "smoke"
    command = [
        sys.executable, "-m", "baseline.sg_scl.sg_scl_smoke",
        "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
        "--author-repo", str(result_root / "source" / "repo"),
        "--checkpoint", str(result_root / "checkpoints" / SPEC.checkpoint_filename),
        "--work-dir", str(result_root / "work" / stage),
        "--output-dir", str(result_root / stage),
        "--steps", str(steps),
        "--device", device,
    ]
    subprocess.run(command, check=True, env=runtime_environment(receipt))


def export_and_verify(result_root: Path, device: str, trained_checkpoint: Path | None) -> None:
    result_root = result_root.resolve()
    receipt = read_receipt(result_root)
    command = [
        sys.executable, "-m", "baseline.sg_scl.export_predictions",
        "--result-root", str(result_root), "--device", device,
    ]
    if trained_checkpoint is not None:
        command.extend(["--trained-checkpoint", str(trained_checkpoint.resolve())])
    environment = runtime_environment(receipt)
    subprocess.run(command, check=True, env=environment)
    subprocess.run([
        sys.executable, "-m", "baseline.common.verify_official_predictions",
        "--input-npz", str(result_root / "predictions" / "official_test_outputs.npz"),
        "--output-dir", str(result_root / "predictions" / "verified"),
        "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
        "--expected-split", "test", "--expected-rows", "2756",
    ], check=True, env=environment)


def full(project_root: Path, result_root: Path, device: str, resume: Path | None) -> None:
    result_root = result_root.resolve()
    receipt = read_receipt(result_root)
    if not device.startswith("cuda"):
        raise ValueError("SG-SCL full run requires CUDA")
    environment = runtime_environment(receipt)
    environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if ":" in device else "0"
    if resume is not None:
        resume = resume.resolve()
        try:
            resume.relative_to(result_root)
        except ValueError as error:
            raise ValueError("resume checkpoint must be inside the current result root") from error
        environment["RESUME_CHECKPOINT"] = str(resume)
    subprocess.run(
        ["bash", str(project_root.resolve() / "baseline" / "sg_scl" / "cuda_full_command.sh")],
        cwd=result_root / "portable_run", check=True, env=environment,
    )
    export_and_verify(result_root, device, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="SG-SCL fresh-checkout server package")
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("bootstrap")
    command.add_argument("--project-root", type=Path, default=Path.cwd())
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--cache-root", type=Path)
    command.add_argument("--checkpoint-path", type=Path)
    command.add_argument("--checkpoint-url")
    command.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    for name in ["verify-bootstrap", "smoke", "profile", "full", "export-and-verify"]:
        command = subparsers.add_parser(name)
        command.add_argument("--result-root", type=Path, required=True)
        if name in {"smoke", "profile", "full", "export-and-verify"}:
            command.add_argument("--device", default="cuda" if name != "smoke" else "cpu")
        if name in {"smoke", "profile"}:
            command.add_argument("--steps", type=int, default=100 if name == "profile" else 1)
        if name == "full":
            command.add_argument("--project-root", type=Path, default=Path.cwd())
            command.add_argument("--resume", type=Path)
        if name == "export-and-verify":
            command.add_argument("--trained-checkpoint", type=Path)

    args = parser.parse_args()
    if args.command == "bootstrap":
        result = bootstrap(
            args.project_root, args.dataset_root, args.result_root, args.cache_root,
            args.checkpoint_path, args.checkpoint_url, args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "verify-bootstrap":
        print(json.dumps(verify_bootstrap(args.result_root), indent=2, sort_keys=True))
    elif args.command in {"smoke", "profile"}:
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        smoke(args.result_root, args.device, args.steps, args.command == "profile")
    elif args.command == "full":
        full(args.project_root, args.result_root, args.device, args.resume)
    elif args.command == "export-and-verify":
        export_and_verify(args.result_root, args.device, args.trained_checkpoint)


if __name__ == "__main__":
    main()
