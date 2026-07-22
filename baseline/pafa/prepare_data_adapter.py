"""Create PAFA's author-expected paths with read-only ICBHI symlinks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORT_FILES = ["official_split.txt", "metadata.txt", "patient_diagnosis.txt", "patient_list_foldwise.txt"]


def link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to replace non-matching path: {path}")
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    raw_audio = args.raw_audio_dir.resolve()
    source_support = args.author_repo.resolve() / "data"
    root = args.adapter_root.resolve()
    target = root / "data" / "icbhi_dataset"
    target.mkdir(parents=True, exist_ok=True)
    link(target / "audio_test_data", raw_audio)
    for name in SUPPORT_FILES:
        link(target / name, source_support / name)
    checkpoint = args.checkpoint.resolve()
    pretrained = root / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    link(pretrained / "BEATs_iter3_plus_AS2M.pt", checkpoint)

    split = (source_support / "official_split.txt").read_text().splitlines()
    split_counts = {
        "train": sum(line.endswith("\ttrain") for line in split),
        "test": sum(line.endswith("\ttest") for line in split),
    }
    wav_count = len(list(raw_audio.glob("*.wav")))
    annotation_count = len([path for path in raw_audio.glob("*.txt") if len(path.stem.split("_")) >= 5])
    if (wav_count, annotation_count, split_counts) != (920, 920, {"train": 539, "test": 381}):
        raise ValueError("PAFA data adapter count mismatch")
    payload = {
        "status": "ready_with_unverified_checkpoint_identity",
        "adapter_root": str(root),
        "raw_audio_target": str(raw_audio),
        "raw_policy": "read-only symlink; no raw files copied or modified",
        "checkpoint_target": str(checkpoint),
        "checkpoint_identity": "non-Microsoft mirror; smoke/command validation only",
        "wav_count": wav_count,
        "annotation_count": annotation_count,
        "official_split_recording_counts": split_counts,
        "cache_policy": "author relative ./data/training.pt and test.pt resolve inside timestamped result root",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
