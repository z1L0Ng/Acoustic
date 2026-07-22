"""Fresh-checkout MVST bootstrap, five-view smoke, profile, and full CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from baseline.mvst.reproduction_bootstrap import METHOD, SPEC, VIEWS, bootstrap, verify_bootstrap


def read_receipt(result_root: Path) -> dict:
    receipt = json.loads((result_root.resolve() / "receipts" / "bootstrap_receipt.json").read_text())
    if receipt["method"] != METHOD:
        raise ValueError("MVST bootstrap/method mismatch")
    return receipt


def runtime_environment(receipt: dict) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(receipt["environment"])
    for path in receipt["environment"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return environment


def smoke(result_root: Path, device: str, steps: int, profile: bool, views: list[str]) -> None:
    result_root = result_root.resolve()
    receipt = read_receipt(result_root)
    stage = "profile" if profile else "smoke"
    environment = runtime_environment(receipt)
    for view in views:
        subprocess.run([
            sys.executable, "-m", "baseline.mvst.mvst_view_smoke",
            "--view", view,
            "--manifest", str(result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"),
            "--author-repo", str(result_root / "source" / "repo"),
            "--checkpoint", str(result_root / "checkpoints" / SPEC.checkpoint_filename),
            "--work-dir", str(result_root / "work" / stage / view),
            "--output-dir", str(result_root / stage / view),
            "--device", device, "--steps", str(steps),
        ], check=True, env=environment)
    if set(views) == set(VIEWS):
        subprocess.run([
            sys.executable, "-m", "baseline.mvst.mvst_fusion_smoke",
            "--smoke-root", str(result_root / stage),
            "--author-repo", str(result_root / "source" / "repo"),
            "--output-dir", str(result_root / stage / "fusion"),
            "--device", device,
        ], check=True, env=environment)
        subprocess.run([
            sys.executable, "-m", "baseline.common.verify_official_predictions",
            "--input-npz", str(result_root / stage / "fusion" / "mvst_fusion_smoke_8_outputs.npz"),
            "--output-dir", str(result_root / stage / "prediction_wiring"),
            "--expected-rows", "8",
        ], check=True, env=environment)


def checked_resume(result_root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        path.relative_to(result_root)
    except ValueError as error:
        raise ValueError("resume checkpoint must be inside the current result root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def full(
    project_root: Path,
    result_root: Path,
    device: str,
    encoder_resumes: list[str],
    fusion_resume: Path | None,
) -> None:
    result_root = result_root.resolve()
    receipt = read_receipt(result_root)
    if not device.startswith("cuda"):
        raise ValueError("MVST full run requires CUDA")
    environment = runtime_environment(receipt)
    environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1] if ":" in device else "0"
    environment["PROJECT_ROOT"] = str(project_root.resolve())
    environment["RESULT_ROOT"] = str(result_root)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root.resolve()) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    for item in encoder_resumes:
        if "=" not in item:
            raise ValueError("--encoder-resume must be VIEW=PATH")
        view, raw_path = item.split("=", 1)
        if view not in VIEWS:
            raise ValueError(f"invalid MVST resume view: {view}")
        environment[f"MVST_RESUME_{view}"] = checked_resume(result_root, Path(raw_path))
    if fusion_resume is not None:
        environment["MVST_FUSION_RESUME"] = checked_resume(result_root, fusion_resume)
    subprocess.run(
        ["bash", str(project_root.resolve() / "baseline" / "mvst" / "cuda_full_pipeline.sh")],
        cwd=result_root / "portable_run", check=True, env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MVST fresh-checkout server package")
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
        command.add_argument("--device", default="cpu" if name == "smoke" else "cuda")
        command.add_argument("--steps", type=int, default=1 if name == "smoke" else 100)
        command.add_argument("--views", nargs="+", choices=VIEWS, default=VIEWS if name == "smoke" else ["16"])
    command = subparsers.add_parser("full")
    command.add_argument("--project-root", type=Path, default=Path.cwd())
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--encoder-resume", action="append", default=[])
    command.add_argument("--fusion-resume", type=Path)

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
        smoke(args.result_root, args.device, args.steps, args.command == "profile", args.views)
    elif args.command == "full":
        full(
            args.project_root, args.result_root, args.device,
            args.encoder_resume, args.fusion_resume,
        )


if __name__ == "__main__":
    main()
