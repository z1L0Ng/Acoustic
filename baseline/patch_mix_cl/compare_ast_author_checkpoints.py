"""Safely compare current author-hosted AST checkpoints and HF conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes()


def tensor_receipt(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sha256": hashlib.sha256(tensor_bytes(tensor)).hexdigest(),
    }


def load_state(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
        loaded = loaded["model"]
    if not isinstance(loaded, dict) or not loaded:
        raise TypeError(f"unsupported checkpoint structure in {path}")
    non_tensors = [key for key, value in loaded.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        raise TypeError(f"non-tensor state values in {path}: {non_tensors[:5]}")
    return dict(loaded)


def canonical_digest(receipts: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(receipts):
        item = receipts[key]
        line = json.dumps(
            {"key": key, "dtype": item["dtype"], "shape": item["shape"], "sha256": item["sha256"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact(path: Path, source_url: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    state = load_state(path)
    tensors = {key: tensor_receipt(value) for key, value in state.items()}
    return {
        "path": str(path.resolve()),
        "source_url": source_url,
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "key_count": len(state),
        "canonical_sorted_state_dict_sha256": canonical_digest(tensors),
        "tensors": tensors,
    }, state


def compare(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    shape_or_dtype_mismatch = []
    tensor_different = []
    max_abs_difference: dict[str, float] = {}
    for key in common:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            shape_or_dtype_mismatch.append(key)
            continue
        if not torch.equal(a, b):
            tensor_different.append(key)
            if a.is_floating_point() or a.is_complex():
                max_abs_difference[key] = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())
    return {
        "common_key_count": len(common),
        "only_left": only_left,
        "only_right": only_right,
        "shape_or_dtype_mismatch": shape_or_dtype_mismatch,
        "tensor_different": tensor_different,
        "bitwise_equal_common_tensors": len(tensor_different) == 0 and len(shape_or_dtype_mismatch) == 0,
        "max_abs_difference_by_tensor": max_abs_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-converted", type=Path, required=True)
    parser.add_argument("--readme-headers", type=Path, required=True)
    parser.add_argument("--runtime-headers", type=Path, required=True)
    parser.add_argument("--conversion-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    readme, readme_state = artifact(
        args.readme_checkpoint.resolve(),
        "https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1",
    )
    runtime, runtime_state = artifact(
        args.runtime_checkpoint.resolve(),
        "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1",
    )
    hf, hf_state = artifact(args.hf_converted.resolve(), "local reverse conversion from pinned MIT/HF safetensors")
    readme_runtime = compare(readme_state, runtime_state)
    readme_hf = compare(readme_state, hf_state)
    runtime_hf = compare(runtime_state, hf_state)
    author_used_keys = sorted(set(hf_state) & set(readme_state) & set(runtime_state))
    if len(hf_state) != 155 or len(author_used_keys) != 155:
        raise ValueError(f"expected 155 HF-converted model-used tensors, got {len(hf_state)} / common {len(author_used_keys)}")
    verdict = (
        "current_author_hosted_state_dicts_tensor_equivalent"
        if readme_runtime["bitwise_equal_common_tensors"]
        and not readme_runtime["only_left"]
        and not readme_runtime["only_right"]
        else "current_author_hosted_state_dicts_differ"
    )
    payload = {
        "safe_load_policy": "torch.load(weights_only=True, map_location='cpu'); tensor-only state required",
        "historical_identity_policy": "comparison verifies only objects served on 2026-07-21; no claim of 2023 byte identity without author confirmation",
        "verdict": verdict,
        "patch_mix_execution_reference": "runtime-loader cv4 Dropbox branch",
        "artifacts": {"readme_ca0b": readme, "runtime_cv4": runtime, "hf_reverse_conversion": hf},
        "comparisons": {
            "readme_ca0b_vs_runtime_cv4": readme_runtime,
            "readme_ca0b_vs_hf_reverse_conversion": readme_hf,
            "runtime_cv4_vs_hf_reverse_conversion": runtime_hf,
        },
        "model_used_tensor_count": len(author_used_keys),
        "model_used_keys": author_used_keys,
        "http_receipts": {
            "readme_ca0b": {
                "path": str(args.readme_headers.resolve()),
                "sha256": file_sha256(args.readme_headers),
                "final_http_status": 200,
                "content_length": readme["size_bytes"],
            },
            "runtime_cv4": {
                "path": str(args.runtime_headers.resolve()),
                "sha256": file_sha256(args.runtime_headers),
                "final_http_status": 200,
                "content_length": runtime["size_bytes"],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    conversion_receipt = json.loads(args.conversion_receipt.read_text())
    conversion_receipt["current_author_artifact_comparison"] = {
        "receipt_path": str(args.output.resolve()),
        "verdict": verdict,
        "author_artifacts_bitwise_file_equal": readme["file_sha256"] == runtime["file_sha256"],
        "author_artifacts_state_dict_tensor_equal": readme_runtime["bitwise_equal_common_tensors"],
        "author_artifact_key_count": readme["key_count"],
        "hf_used_tensor_count": len(hf_state),
        "hf_used_tensors_bitwise_equal_to_both_author_artifacts": (
            readme_hf["bitwise_equal_common_tensors"]
            and runtime_hf["bitwise_equal_common_tensors"]
            and not readme_hf["only_right"]
            and not runtime_hf["only_right"]
        ),
        "historical_identity_verified": False,
        "interpretation": "verified equivalence of currently served author state dicts and HF conversion's 155 used tensors; not proof of 2023 serialized bytes",
    }
    args.conversion_receipt.write_text(json.dumps(conversion_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "readme_size": readme["size_bytes"],
        "runtime_size": runtime["size_bytes"],
        "readme_file_sha256": readme["file_sha256"],
        "runtime_file_sha256": runtime["file_sha256"],
        "readme_canonical": readme["canonical_sorted_state_dict_sha256"],
        "runtime_canonical": runtime["canonical_sorted_state_dict_sha256"],
        "readme_runtime": readme_runtime,
        "readme_hf": readme_hf,
        "runtime_hf": runtime_hf,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
