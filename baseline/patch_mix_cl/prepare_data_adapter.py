"""Create the author-expected ICBHI layout using read-only symlinks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORT_FILES = [
    "official_split.txt",
    "metadata.txt",
    "patient_diagnosis.txt",
    "patient_list_foldwise.txt",
]


def replace_symlink(path: Path, target: Path, *, allow_repoint: bool = False) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() and path.resolve() != target.resolve() and allow_repoint:
            path.unlink()
        elif not path.is_symlink() or path.resolve() != target.resolve():
            raise FileExistsError(f"refusing to replace non-matching path: {path}")
        else:
            return
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    raw_audio = args.raw_audio_dir.resolve()
    source_support = args.author_repo.resolve() / "data" / "icbhi_dataset"
    target = args.adapter_root.resolve() / "data" / "icbhi_dataset"
    target.mkdir(parents=True, exist_ok=True)
    replace_symlink(target / "audio_test_data", raw_audio)
    for name in SUPPORT_FILES:
        replace_symlink(target / name, source_support / name)
    pretrained = args.adapter_root.resolve() / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    replace_symlink(
        pretrained / "audioset_10_10_0.4593.pth",
        checkpoint,
        allow_repoint=True,
    )

    wav_count = len(list(raw_audio.glob("*.wav")))
    annotation_count = len([
        path for path in raw_audio.glob("*.txt")
        if len(path.stem.split("_")) >= 5
    ])
    split_lines = (source_support / "official_split.txt").read_text().splitlines()
    split_counts = {
        "train": sum(line.endswith("\ttrain") for line in split_lines),
        "test": sum(line.endswith("\ttest") for line in split_lines),
    }
    if wav_count != 920 or annotation_count != 920 or split_counts != {"train": 539, "test": 381}:
        raise ValueError(
            f"unexpected adapter counts: wav={wav_count}, annotations={annotation_count}, split={split_counts}"
        )
    payload = {
        "status": "ready",
        "adapter_root": str(args.adapter_root.resolve()),
        "raw_audio_target": str(raw_audio),
        "raw_policy": "read-only symlink; no raw files copied or modified",
        "checkpoint_target": str(checkpoint),
        "wav_count": wav_count,
        "annotation_count": annotation_count,
        "official_split_recording_counts": split_counts,
        "support_files": {
            name: str((source_support / name).resolve()) for name in SUPPORT_FILES
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
