"""Zero-update L40 adapter preflight; never a training or scoring entrypoint."""

from __future__ import annotations

import argparse
import importlib
import json
from itertools import islice
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from .adapter_factory import (
    AdapterFactoryConfig,
    audit_local_adapter_assets,
    build_production_adapter,
)
from .joint_native import JOINT_LANES, JointNativeProjector
from .preflight import PIPELINE_ENCODERS
from .sliding_window import SlidingWindowBatch, masked_mean_window_embeddings
from .terminal_scoring import audit_terminal_provider_registration
from .window_encoder import ProductionWindowEncoder


def _load_provider(specification: str) -> Callable[[str], Iterable[SlidingWindowBatch]]:
    if ":" not in specification:
        raise ValueError("batch provider must be module:function")
    module_name, function_name = specification.split(":", 1)
    provider = getattr(importlib.import_module(module_name), function_name)
    if not callable(provider):
        raise TypeError("batch provider is not callable")
    return provider


def _require_subtrain(batch: SlidingWindowBatch) -> None:
    batch.validate()
    for lineage in batch.lineage:
        if lineage.get("partition") != "subtrain":
            raise RuntimeError("L40 preflight accepts subtrain units only")
        if str(lineage.get("outer_test_accessed", "false")).lower() != "false":
            raise RuntimeError("outer/test access is forbidden in adapter preflight")


def validate_pipeline_adapter_identity(
    pipeline_id: str, adapter: ProductionWindowEncoder
) -> None:
    expected = PIPELINE_ENCODERS.get(pipeline_id)
    if expected is None or adapter.encoder_identity != expected:
        raise RuntimeError(
            f"pipeline/adapter identity mismatch: {pipeline_id} requires {expected}, "
            f"got {adapter.encoder_identity}"
        )


def _complete_state_snapshot(
    adapter: ProductionWindowEncoder, model: JointNativeProjector
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for prefix, module in (("adapter", adapter), ("model", model)):
        for name, value in module.state_dict().items():
            snapshot[f"{prefix}.state.{name}"] = value.detach().cpu().clone()
        # state_dict intentionally excludes buffers registered with
        # persistent=False.  Compare named_buffers separately so the
        # zero-update claim covers those tensors as well.
        for name, value in module.named_buffers():
            snapshot[f"{prefix}.buffer.{name}"] = value.detach().cpu().clone()
    return snapshot


def run_zero_update_preflight(
    pipeline_id: str,
    adapter: ProductionWindowEncoder,
    batches: Iterable[SlidingWindowBatch],
    *,
    device: torch.device,
) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("L40 preflight requires an available CUDA device")
    if "L40" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError(f"expected an NVIDIA L40-class device, got {torch.cuda.get_device_name(device)}")
    validate_pipeline_adapter_identity(pipeline_id, adapter)
    materialized = list(islice(batches, 3))
    if not 1 <= len(materialized) <= 2:
        raise RuntimeError("L40 preflight requires one or two subtrain batches")
    model = JointNativeProjector().to(device)
    adapter = adapter.to(device)
    before = _complete_state_snapshot(adapter, model)
    seen: set[str] = set()
    batch_receipts = []
    properties = torch.cuda.get_device_properties(device)
    memory_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for batch in materialized:
        _require_subtrain(batch)
        batch = batch.to(device)
        output = adapter(batch)
        for row, lane in enumerate(batch.dataset_ids):
            seen.add(lane)
            values = output.embeddings[row : row + 1]
            mask = output.window_mask[row : row + 1]
            if lane == "HF":
                logits = model(values, lane)["temporal4"]
                total = total + logits[mask].square().mean()
            else:
                pooled = masked_mean_window_embeddings(values, mask)
                for logits in model(pooled, lane).values():
                    total = total + logits.square().mean()
        batch_receipts.append(batch.receipt())
    if seen != set(JOINT_LANES):
        raise RuntimeError(f"preflight batches must cover all four lanes, got {sorted(seen)}")
    total.backward()
    encoder_gradients = [
        name for name, parameter in adapter.backend.named_parameters() if parameter.grad is not None
    ]
    if encoder_gradients:
        raise RuntimeError(f"frozen encoder received gradients: {encoder_gradients}")
    if model.projector.weight.grad is None or not bool(model.projector.weight.grad.abs().sum() > 0):
        raise RuntimeError("shared projector did not receive a gradient")
    after = _complete_state_snapshot(adapter, model)
    changed = [
        name
        for name in sorted(set(before) | set(after))
        if name not in before
        or name not in after
        or not torch.equal(before[name], after[name])
    ]
    if changed:
        raise RuntimeError(f"zero-update preflight changed parameters: {changed}")
    return {
        "status": "L40_zero_update_scope_preflight_passed",
        "pipeline_id": pipeline_id,
        "experiment_result": False,
        "device": torch.cuda.get_device_name(device),
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_total_memory_bytes": int(properties.total_memory),
        },
        "batch_count": len(materialized),
        "covered_lanes": sorted(seen),
        "batch_receipts": batch_receipts,
        "adapter_receipt": adapter.receipt(),
        "frozen_encoder_gradients": 0,
        "shared_projector_gradient_nonzero": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "complete_state_dict_unchanged": True,
        "state_tensors_compared": len(before),
        "memory_allocated_before_bytes": memory_before,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "memory_allocated_after_bytes": int(torch.cuda.memory_allocated(device)),
        "outer_test_accessed": False,
        "performance_scoring": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("P1", "P2", "P3", "P4", "P5"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-provider",
        default=(
            "baseline.multidataset_pipeline.real_subtrain_provider:"
            "build_real_subtrain_preflight_batches"
        ),
        help="module:function; must yield 1-2 real frozen-subtrain batches",
    )
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    device = torch.device(args.device)
    asset_audit = audit_local_adapter_assets(args.repo_root)[args.pipeline]
    if asset_audit.get("asset_status") != "READY_verified_local":
        raise RuntimeError(
            f"{args.pipeline} canonical asset manifest gate is HOLD: "
            f"{asset_audit.get('reason', asset_audit.get('asset_status'))}"
        )
    adapter = build_production_adapter(
        AdapterFactoryConfig(
            pipeline_id=args.pipeline,
            repo_root=args.repo_root,
            device=str(device),
            source_repo=args.source_repo,
            source_revision=args.source_revision,
            checkpoint=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
        )
    )
    provider = _load_provider(args.batch_provider)
    receipt = run_zero_update_preflight(
        args.pipeline, adapter, provider(args.pipeline), device=device
    )
    receipt["adapter_asset_audit"] = asset_audit
    receipt["terminal_provider_readiness"] = audit_terminal_provider_registration(
        args.repo_root
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
