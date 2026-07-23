"""Fresh-checkout ADD-RSC dual-track bootstrap, smoke, profile, and full CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from baseline.add_rsc.reproduction_bootstrap import METHOD, SPEC, TRACKS, bootstrap, verify_bootstrap


def read_receipt(result_root: Path) -> dict:
    receipt = json.loads((result_root.resolve() / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != METHOD:
        raise ValueError("ADD-RSC bootstrap/method mismatch")
    return receipt


def runtime_environment(receipt: dict) -> dict[str, str]:
    environment = os.environ.copy(); environment.update(receipt["environment"])
    for path in receipt["environment"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return environment


def selected_tracks(track: str) -> list[str]:
    return TRACKS if track == "all" else [track]


def smoke(
    result_root: Path, track: str, device: str, steps: int,
    train_batch_size: int, profile: bool,
) -> None:
    result_root = result_root.resolve(); receipt = read_receipt(result_root)
    stage = "profile" if profile else "smoke"
    environment = runtime_environment(receipt)
    for name in selected_tracks(track):
        output = result_root / stage / name
        subprocess.run([
            sys.executable, "-m", "baseline.add_rsc.add_rsc_smoke",
            "--track", name,
            "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
            "--author-repo", str(result_root / "source" / "repo"),
            "--checkpoint", str(result_root / "checkpoints" / SPEC.checkpoint_filename),
            "--work-dir", str(result_root / "work" / stage / name),
            "--output-dir", str(output), "--device", device,
            "--steps", str(steps), "--train-batch-size", str(train_batch_size),
        ], check=True, env=environment)
        subprocess.run([
            sys.executable, "-m", "baseline.common.verify_official_predictions",
            "--input-npz", str(output / f"{name}_smoke_8_outputs.npz"),
            "--output-dir", str(output / "prediction_wiring"), "--expected-rows", "8",
        ], check=True, env=environment)


def checked_resume(result_root: Path, resume: Path) -> str:
    resume = resume.resolve()
    try:
        resume.relative_to(result_root)
    except ValueError as error:
        raise ValueError("resume checkpoint must be inside the current result root") from error
    if not resume.is_file():
        raise FileNotFoundError(resume)
    return str(resume)


def export_and_verify(
    result_root: Path, track: str, device: str, trained_checkpoint: Path | None,
) -> None:
    result_root = result_root.resolve(); receipt = read_receipt(result_root)
    environment = runtime_environment(receipt)
    command = [
        sys.executable, "-m", "baseline.add_rsc.export_predictions",
        "--track", track, "--result-root", str(result_root), "--device", device,
    ]
    if trained_checkpoint is not None:
        command.extend(["--trained-checkpoint", str(trained_checkpoint.resolve())])
    subprocess.run(command, check=True, env=environment)
    output = result_root / "predictions" / track
    verify = [
        sys.executable, "-m", "baseline.common.verify_official_predictions",
        "--input-npz", str(output / "test_outputs.npz"),
        "--output-dir", str(output / "verified"),
        "--expected-rows", "2756" if track == "paper_declared_reconstruction" else "2685",
    ]
    if track == "paper_declared_reconstruction":
        verify.extend([
            "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
            "--expected-split", "test",
        ])
    subprocess.run(verify, check=True, env=environment)


def full(
    project_root: Path, result_root: Path, track: str, device: str, resume: Path | None,
) -> None:
    result_root = result_root.resolve(); receipt = read_receipt(result_root)
    if not device.startswith("cuda"):
        raise ValueError("ADD-RSC full run requires CUDA")
    environment = runtime_environment(receipt)
    environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if ":" in device else "0"
    environment["PROJECT_ROOT"] = str(project_root.resolve())
    environment["RESULT_ROOT"] = str(result_root)
    environment["ADD_RSC_TRACK"] = track
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root.resolve()) + (os.pathsep + existing if existing else "")
    if resume is not None:
        environment["ADD_RSC_RESUME"] = checked_resume(result_root, resume)
    subprocess.run(
        ["bash", str(project_root.resolve() / "baseline" / "add_rsc" / "cuda_full_command.sh")],
        cwd=result_root / "portable_run", check=True, env=environment,
    )
    export_and_verify(result_root, track, device, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADD-RSC bounded dual-track server package")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("bootstrap")
    command.add_argument("--project-root", type=Path, default=Path.cwd())
    command.add_argument("--dataset-root", type=Path, required=True)
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--cache-root", type=Path)
    command.add_argument("--checkpoint-path", type=Path)
    command.add_argument("--checkpoint-url")
    command.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    command = subparsers.add_parser("verify-bootstrap")
    command.add_argument("--result-root", type=Path, required=True)
    for name in ["smoke", "profile"]:
        command = subparsers.add_parser(name)
        command.add_argument("--result-root", type=Path, required=True)
        command.add_argument("--track", choices=[*TRACKS, "all"], default="all" if name == "smoke" else TRACKS[0])
        command.add_argument("--device", default="cpu" if name == "smoke" else "cuda")
        command.add_argument("--steps", type=int, default=1 if name == "smoke" else 100)
        command.add_argument("--train-batch-size", type=int, default=1 if name == "smoke" else 8)
    command = subparsers.add_parser("full")
    command.add_argument("--project-root", type=Path, default=Path.cwd())
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--track", choices=TRACKS, required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--resume", type=Path)
    command = subparsers.add_parser("export-and-verify")
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--track", choices=TRACKS, required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--trained-checkpoint", type=Path)

    args = parser.parse_args()
    if args.command == "bootstrap":
        print(json.dumps(bootstrap(
            args.project_root, args.dataset_root, args.result_root, args.cache_root,
            args.checkpoint_path, args.checkpoint_url, args.device,
        ), indent=2, sort_keys=True))
    elif args.command == "verify-bootstrap":
        print(json.dumps(verify_bootstrap(args.result_root), indent=2, sort_keys=True))
    elif args.command in {"smoke", "profile"}:
        smoke(
            args.result_root, args.track, args.device, args.steps,
            args.train_batch_size, args.command == "profile",
        )
    elif args.command == "full":
        full(args.project_root, args.result_root, args.track, args.device, args.resume)
    elif args.command == "export-and-verify":
        export_and_verify(args.result_root, args.track, args.device, args.trained_checkpoint)


if __name__ == "__main__":
    main()
