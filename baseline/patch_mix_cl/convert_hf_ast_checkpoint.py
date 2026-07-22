"""Convert the MIT HF AST state dict into the author repo's legacy key layout.

This is a format conversion for compatibility testing. It does not establish the
identity of the unavailable original ``audioset_10_10_0.4593.pth`` artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import ASTForAudioClassification


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


def verify_author_load(repo: Path, converted: dict[str, torch.Tensor], hf_dir: Path) -> dict:
    sys.path.insert(0, str(repo))
    from models.ast import ASTModel

    author = ASTModel(
        label_dim=527,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
    )
    wrapped = torch.nn.DataParallel(author)
    incompatible = wrapped.load_state_dict(converted, strict=False)
    expected_missing = {
        "module.v.head.weight",
        "module.v.head.bias",
        "module.v.head_dist.weight",
        "module.v.head_dist.bias",
    }
    assert set(incompatible.missing_keys) == expected_missing
    assert not incompatible.unexpected_keys
    loaded = wrapped.state_dict()
    assert all(torch.equal(loaded[key], value) for key, value in converted.items())

    torch.manual_seed(20260721)
    inputs = torch.randn(1, 1024, 128)
    author.eval()
    hf_model = ASTForAudioClassification.from_pretrained(hf_dir, local_files_only=True)
    hf_model.eval()
    with torch.inference_mode():
        author_embedding = author(inputs.unsqueeze(1))
        author_logits = author.mlp_head(author_embedding)
        hf_logits = hf_model(input_values=inputs).logits
    max_abs_diff = float((author_logits - hf_logits).abs().max().item())
    assert torch.allclose(author_logits, hf_logits, atol=2e-5, rtol=2e-5), max_abs_diff
    return {
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "converted_tensor_count": len(converted),
        "all_converted_tensors_exact_after_author_load": True,
        "author_vs_hf_logits_max_abs_diff": max_abs_diff,
        "author_vs_hf_logits_allclose_atol_rtol": 2e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = args.hf_dir / "model.safetensors"
    hf = load_file(source)
    converted = convert_state_dict(hf)
    verification = verify_author_load(args.repo.resolve(), converted, args.hf_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    payload = {
        "classification": "verified_hf_to_author_format_conversion_for_compatibility_smoke",
        "original_ast_pth_identity_verified": False,
        "identity_gate_status": "unresolved_original_artifact; do not call converted file original checkpoint",
        "hf_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "hf_revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "hf_source_path": str(source),
        "hf_source_sha256": sha256(source),
        "output_path": str(args.output),
        "output_sha256": sha256(args.output),
        "verification": verification,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
