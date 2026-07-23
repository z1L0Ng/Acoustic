"""Record a safe structural signature for a candidate BEATs checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("cfg"), dict) or not isinstance(loaded.get("model"), dict):
        raise TypeError("candidate does not implement the official BEATs cfg/model loading contract")
    state = loaded["model"]
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("BEATs model state contains non-tensor values")
    tensors = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
        for key, value in sorted(state.items())
    }
    canonical = hashlib.sha256()
    for key, item in tensors.items():
        canonical.update(json.dumps({"key": key, **item}, sort_keys=True, separators=(",", ":")).encode())
        canonical.update(b"\n")
    cfg = loaded["cfg"]
    payload = {
        "status": "structurally_valid_candidate; official_identity_unverified",
        "safe_load_policy": "torch.load(weights_only=True, map_location='cpu')",
        "official_source": "https://github.com/microsoft/unilm/tree/master/beats",
        "official_checkpoint_name": "BEATs_iter3+ (AS2M)",
        "official_download_url": "https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf",
        "official_download_probe": "HTTP 403 during local audit; no official checksum available",
        "candidate_source": "https://huggingface.co/mooneyko/BEATs",
        "candidate_source_classification": "non-Microsoft mirror; smoke only until identity is proven",
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "file_sha256": sha256(checkpoint),
        "top_level_keys": sorted(loaded),
        "model_tensor_count": len(state),
        "canonical_sorted_model_state_sha256": canonical.hexdigest(),
        "cfg": cfg,
        "selected_config_signature": {
            key: cfg.get(key)
            for key in [
                "encoder_layers",
                "encoder_embed_dim",
                "encoder_ffn_embed_dim",
                "encoder_attention_heads",
                "input_patch_size",
                "embed_dim",
                "finetuned_model",
                "predictor_class",
            ]
        },
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in [
        "status", "size_bytes", "file_sha256", "model_tensor_count",
        "canonical_sorted_model_state_sha256", "selected_config_signature",
    ]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
