"""Safely receipt the current author-hosted MVST AST16 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("checkpoint is not a tensor state dict")
    payload = {
        "status": "current_author_hosted_artifact_verified",
        "url": "https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1",
        "checkpoint_path": str(checkpoint),
        "http_headers_path": str(args.headers.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "tensor_count": len(state),
        "all_tensors_finite": all(torch.isfinite(value).all() for value in state.values()),
        "canonical_state_dict_sha256": canonical_digest(state),
        "position_embedding_shape": list(state["module.v.pos_embed"].shape),
        "patch_projection_shape": list(state["module.v.patch_embed.proj.weight"].shape),
        "boundary": "Current author-hosted artifact from the URL hardcoded by MVST. No historical paper-run checksum was published.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
