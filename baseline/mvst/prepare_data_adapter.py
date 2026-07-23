"""Create MVST's author-expected data/checkpoint layout with symlinks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to replace non-matching path: {path}")
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw_audio = args.raw_audio_dir.resolve()
    adapter = args.adapter_root.resolve()
    data = adapter / "data"
    data.mkdir(parents=True, exist_ok=True)
    ensure_symlink(data / "icbhi_dataset", raw_audio)
    pretrained = adapter / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    ensure_symlink(pretrained / "audioset_16_16_0.4422.pth", checkpoint)
    payload = {
        "status": "ready",
        "adapter_root": str(adapter),
        "raw_audio_target": str(raw_audio),
        "raw_policy": "read-only directory symlink",
        "checkpoint_target": str(checkpoint),
        "wav_count": len(list(raw_audio.glob("*.wav"))),
        "annotation_count": len([p for p in raw_audio.glob("*.txt") if len(p.stem.split("_")) >= 5]),
    }
    if payload["wav_count"] != 920 or payload["annotation_count"] != 920:
        raise ValueError(f"unexpected adapter counts: {payload}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
