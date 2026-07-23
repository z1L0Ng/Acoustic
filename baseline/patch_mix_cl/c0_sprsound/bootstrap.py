"""Fresh-checkout source and initialization bootstrap for C0."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import (
    AST_SHA256,
    AST_SIZE,
    AST_URL,
    AUTHOR_REPO_COMMIT,
    AUTHOR_REPO_URL,
    sha256_file,
    update_run_manifest,
    validate_cache_root,
    validate_result_root,
)


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()

    result_root = validate_result_root(args.result_root)
    cache_root = validate_cache_root(args.cache_root)
    result_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    repo = cache_root / "source" / "repo"
    if args.author_repo:
        source = args.author_repo.resolve()
        if repo.exists():
            raise FileExistsError(repo)
        shutil.copytree(source, repo, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", AUTHOR_REPO_URL, str(repo)])
    run(["git", "checkout", "--detach", AUTHOR_REPO_COMMIT], cwd=repo)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo)
    if commit != AUTHOR_REPO_COMMIT:
        raise RuntimeError(f"author commit mismatch: {commit}")
    source_status = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    if source_status:
        raise RuntimeError("author source contains tracked modifications")

    checkpoint = cache_root / "checkpoints" / "audioset_0.4593_runtime_cv4.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_path:
        source_checkpoint = args.checkpoint_path.resolve()
        if source_checkpoint.stat().st_size != AST_SIZE or sha256_file(source_checkpoint) != AST_SHA256:
            raise RuntimeError("explicit AST initialization checkpoint failed identity check")
        if not checkpoint.exists():
            temporary = checkpoint.with_suffix(".pth.tmp")
            shutil.copy2(source_checkpoint, temporary)
            temporary.replace(checkpoint)
    elif not checkpoint.exists():
        temporary = checkpoint.with_suffix(".pth.tmp")
        urllib.request.urlretrieve(AST_URL, temporary)
        if temporary.stat().st_size != AST_SIZE or sha256_file(temporary) != AST_SHA256:
            raise RuntimeError("downloaded AST initialization checkpoint failed identity check")
        temporary.replace(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint.stat().st_size != AST_SIZE or checkpoint_sha != AST_SHA256:
        raise RuntimeError("current author-hosted AST runtime artifact failed identity check")

    now = datetime.now(ZoneInfo("America/Chicago"))
    receipt = {
        "status": "bootstrap_verified",
        "run_start_iso": now.isoformat(),
        "timezone": "America/Chicago",
        "author_repo_url": AUTHOR_REPO_URL,
        "author_repo_commit": commit,
        "author_repo_path": str(repo),
        "checkpoint_url": AST_URL,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_boundary": "current author runtime branch; not historical-byte identity",
    }
    update_run_manifest(result_root, "bootstrap", receipt)
    print(f"c0_bootstrap_ok repo={commit} checkpoint={checkpoint_sha}")


if __name__ == "__main__":
    main()
