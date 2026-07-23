"""Safely audit the author-posted Patch-Mix ICBHI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np
import torch

from .restricted_checkpoint import restricted_torch_load


EXPECTED_PICKLE_GLOBALS = {
    "_codecs encode",
    "argparse Namespace",
    "collections OrderedDict",
    "numpy dtype",
    "numpy ndarray",
    "numpy.core.multiarray _reconstruct",
    "torch FloatStorage",
    "torch._utils _rebuild_tensor_v2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pickle_globals(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("best/data.pkl")
    return sorted(
        {
            argument
            for opcode, argument, _ in pickletools.genops(payload)
            if opcode.name == "GLOBAL"
        }
    )


def tensor_state_receipt(state: dict[str, torch.Tensor]) -> dict:
    digest = hashlib.sha256()
    finite = True
    parameters = 0
    shapes = {}
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        finite = finite and bool(torch.isfinite(tensor).all())
        parameters += tensor.numel()
        shapes[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return {
        "key_count": len(state),
        "parameter_count": parameters,
        "finite": finite,
        "canonical_tensor_sha256": digest.hexdigest(),
        "keys_and_shapes": shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    observed_globals = pickle_globals(checkpoint_path)
    if set(observed_globals) != EXPECTED_PICKLE_GLOBALS:
        raise RuntimeError(f"unexpected pickle globals: {observed_globals}")

    checkpoint = restricted_torch_load(checkpoint_path)
    expected_keys = {"args", "optimizer", "model", "epoch", "classifier"}
    if set(checkpoint) != expected_keys:
        raise RuntimeError(f"unexpected top-level keys: {sorted(checkpoint)}")
    if not isinstance(checkpoint["model"], dict) or not isinstance(checkpoint["classifier"], dict):
        raise TypeError("model and classifier entries must be state dictionaries")

    train_args = vars(checkpoint["args"])
    receipt = {
        "status": "verified_author_icbhi_task_checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pickle_globals": observed_globals,
        "restricted_loader": True,
        "top_level_keys": sorted(checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "train_contract": {
            key: train_args.get(key)
            for key in (
                "seed", "epochs", "learning_rate", "weight_decay", "batch_size",
                "class_split", "n_cls", "test_fold", "sample_rate", "desired_length",
                "n_mels", "pad_types", "specaug_policy", "model", "method",
                "ma_update", "ma_beta", "audioset_pretrained", "proj_dim",
                "temperature", "alpha", "target_type", "cls_list",
            )
        },
        "model": tensor_state_receipt(checkpoint["model"]),
        "classifier": tensor_state_receipt(checkpoint["classifier"]),
    }
    if receipt["train_contract"]["cls_list"] != ["normal", "crackle", "wheeze", "both"]:
        raise RuntimeError("unexpected label order")
    if not receipt["model"]["finite"] or not receipt["classifier"]["finite"]:
        raise RuntimeError("non-finite checkpoint tensor")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "checkpoint_sha256", "epoch")}, indent=2))

if __name__ == "__main__":
    main()
