"""Convert the MIT/HF 16x16 AST archive to MVST's expected legacy keys.

This format adapter supports source-gate smoke testing only. The unavailable
Dropbox ``audioset_16_16_0.4422.pth`` has no pinned checksum, so the converted
file must not be represented as the original author artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "audio_spectrogram_transformer"
    converted = {
        "module.v.cls_token": hf[f"{prefix}.embeddings.cls_token"],
        "module.v.dist_token": hf[f"{prefix}.embeddings.distillation_token"],
        "module.v.pos_embed": hf[f"{prefix}.embeddings.position_embeddings"],
        "module.v.patch_embed.proj.weight": hf[f"{prefix}.embeddings.patch_embeddings.projection.weight"],
        "module.v.patch_embed.proj.bias": hf[f"{prefix}.embeddings.patch_embeddings.projection.bias"],
        "module.v.norm.weight": hf[f"{prefix}.layernorm.weight"],
        "module.v.norm.bias": hf[f"{prefix}.layernorm.bias"],
        "module.mlp_head.0.weight": hf["classifier.layernorm.weight"],
        "module.mlp_head.0.bias": hf["classifier.layernorm.bias"],
        "module.mlp_head.1.weight": hf["classifier.dense.weight"],
        "module.mlp_head.1.bias": hf["classifier.dense.bias"],
    }
    for layer in range(12):
        h = f"{prefix}.encoder.layer.{layer}"
        a = f"module.v.blocks.{layer}"
        converted[f"{a}.norm1.weight"] = hf[f"{h}.layernorm_before.weight"]
        converted[f"{a}.norm1.bias"] = hf[f"{h}.layernorm_before.bias"]
        converted[f"{a}.norm2.weight"] = hf[f"{h}.layernorm_after.weight"]
        converted[f"{a}.norm2.bias"] = hf[f"{h}.layernorm_after.bias"]
        converted[f"{a}.attn.qkv.weight"] = torch.cat(
            [
                hf[f"{h}.attention.attention.query.weight"],
                hf[f"{h}.attention.attention.key.weight"],
                hf[f"{h}.attention.attention.value.weight"],
            ],
            dim=0,
        )
        converted[f"{a}.attn.qkv.bias"] = torch.cat(
            [
                hf[f"{h}.attention.attention.query.bias"],
                hf[f"{h}.attention.attention.key.bias"],
                hf[f"{h}.attention.attention.value.bias"],
            ],
            dim=0,
        )
        converted[f"{a}.attn.proj.weight"] = hf[f"{h}.attention.output.dense.weight"]
        converted[f"{a}.attn.proj.bias"] = hf[f"{h}.attention.output.dense.bias"]
        converted[f"{a}.mlp.fc1.weight"] = hf[f"{h}.intermediate.dense.weight"]
        converted[f"{a}.mlp.fc1.bias"] = hf[f"{h}.intermediate.dense.bias"]
        converted[f"{a}.mlp.fc2.weight"] = hf[f"{h}.output.dense.weight"]
        converted[f"{a}.mlp.fc2.bias"] = hf[f"{h}.output.dense.bias"]
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = args.hf_dir.resolve() / "model.safetensors"
    hf = load_file(source)
    converted = convert_state_dict(hf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    payload = {
        "classification": "hf_to_mvst_legacy_format_conversion_for_compatibility_smoke",
        "original_ast_pth_identity_verified": False,
        "identity_gate_status": "unresolved original Dropbox artifact; conversion is not the author pth",
        "author_expected_filename": "audioset_16_16_0.4422.pth",
        "hf_model": "MIT/ast-finetuned-audioset-16-16-0.442",
        "hf_revision": "bf4d03f36ce5904b0daf96023cef4c82b1fe3d26",
        "hf_source_path": str(source),
        "hf_source_sha256": sha256(source),
        "hf_source_size_bytes": source.stat().st_size,
        "converted_tensor_count": len(converted),
        "converted_position_embedding_shape": list(converted["module.v.pos_embed"].shape),
        "converted_patch_projection_shape": list(converted["module.v.patch_embed.proj.weight"].shape),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "mvst_loading_note": "author code loads only exact key+shape matches per view; view-specific receipt must enumerate retained random tensors",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
