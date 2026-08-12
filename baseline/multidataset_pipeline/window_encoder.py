"""Shared production contract for pooled per-window encoder adapters.

The candidate backbone is frozen.  PANNs and HeAR additionally own one
trainable, dataset-shared dimension adapter before the common biased
``Linear(768, 256)`` projector.  This module never loads data or checkpoints.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .preflight import CandidateDimensionAdapter, SharedWindowEncoderOutput
from .sliding_window import SlidingWindowBatch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_identity(
    path: Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
) -> dict[str, object]:
    """Fail closed unless a local immutable asset matches its frozen identity."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required local checkpoint is missing: {resolved}")
    if len(expected_sha256) != 64:
        raise ValueError("a full 64-character checkpoint SHA256 is required")
    size = resolved.stat().st_size
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise RuntimeError(
            f"checkpoint size mismatch for {resolved}: {size} != {expected_size_bytes}"
        )
    actual = sha256_file(resolved)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"checkpoint SHA256 mismatch for {resolved}: {actual} != {expected_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": actual,
        "size_bytes": size,
        "identity_verified": True,
    }


def require_clean_source_revision(source_repo: Path, expected_revision: str) -> dict[str, object]:
    """Require an exact clean Git source checkout without changing it."""

    resolved = source_repo.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required source checkout is missing: {resolved}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=resolved,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if head != expected_revision or dirty:
        raise RuntimeError(
            f"source revision/cleanliness gate failed: head={head}, "
            f"expected={expected_revision}, dirty={bool(dirty)}"
        )
    return {
        "path": str(resolved),
        "revision": head,
        "tracked_files_clean": True,
    }


def module_device(module: nn.Module) -> torch.device:
    devices = {
        value.device
        for value in (*tuple(module.parameters()), *tuple(module.buffers()))
    }
    if len(devices) != 1:
        raise RuntimeError(f"module must live on exactly one device, got {devices}")
    return next(iter(devices))


def optional_module_device(module: nn.Module) -> torch.device | None:
    devices = {
        value.device
        for value in (*tuple(module.parameters()), *tuple(module.buffers()))
    }
    if len(devices) > 1:
        raise RuntimeError(f"module tensors span multiple devices: {devices}")
    return next(iter(devices)) if devices else None


@dataclass(frozen=True)
class AdapterProvenance:
    encoder_identity: str
    source_url: str
    source_revision: str
    source_license: str
    checkpoint_name: str
    checkpoint_source: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    asset_status: str = "verified_local_asset"
    license_boundary: str = "source_and_model_terms_must_be_reviewed_before_redistribution"

    def as_receipt(self) -> dict[str, object]:
        return {
            "encoder_identity": self.encoder_identity,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "source_license": self.source_license,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "asset_status": self.asset_status,
            "license_boundary": self.license_boundary,
        }


class FrozenWindowBackend(nn.Module):
    """Interface implemented by a frozen candidate-specific waveform encoder."""

    native_dim: int

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ProductionWindowEncoder(nn.Module):
    """Flatten valid B×K windows, encode, adapt to 768, and restore lineage."""

    def __init__(
        self,
        encoder_identity: str,
        backend: FrozenWindowBackend,
        provenance: AdapterProvenance,
        *,
        dimension_adapter: CandidateDimensionAdapter | None = None,
    ) -> None:
        super().__init__()
        self.encoder_identity = encoder_identity
        self.backend = backend.eval()
        self.provenance = provenance
        self.dimension_adapter = dimension_adapter or CandidateDimensionAdapter(
            encoder_identity
        )
        if provenance.encoder_identity != encoder_identity:
            raise ValueError("provenance/encoder identity mismatch")
        if backend.native_dim != self.dimension_adapter.input_dim:
            raise ValueError("backend native dimension and dimension adapter disagree")
        trainable_backend = [
            name for name, parameter in backend.named_parameters() if parameter.requires_grad
        ]
        if trainable_backend:
            raise RuntimeError(f"candidate encoder must be frozen: {trainable_backend}")
        dimension_device = optional_module_device(self.dimension_adapter)
        if dimension_device is not None and module_device(backend) != dimension_device:
            raise RuntimeError("backend and dimension adapter must share one device")

    @property
    def device(self) -> torch.device:
        return module_device(self.backend)

    def forward(self, batch: SlidingWindowBatch) -> SharedWindowEncoderOutput:
        batch.validate()
        if batch.device != self.device:
            raise RuntimeError(
                f"SlidingWindowBatch is on {batch.device}, adapter is on {self.device}; "
                f"call batch.to({self.device!s}) explicitly"
            )
        batch_size, windows, width = batch.waveform_windows.shape
        flat_valid = batch.window_mask.reshape(-1)
        valid_waveforms = batch.waveform_windows.reshape(-1, width)[flat_valid]
        valid_lengths = batch.valid_samples.reshape(-1)[flat_valid]
        with torch.no_grad():
            native = self.backend.encode_valid_windows(valid_waveforms, valid_lengths)
        if native.shape != (int(flat_valid.sum()), self.backend.native_dim):
            raise RuntimeError(
                f"backend output must be [N,{self.backend.native_dim}], got {tuple(native.shape)}"
            )
        if native.device != batch.device or not native.dtype.is_floating_point:
            raise RuntimeError("backend output device/dtype contract failed")
        if not bool(torch.isfinite(native).all()):
            raise RuntimeError("backend output contains non-finite values")
        adapted = self.dimension_adapter(native.unsqueeze(1)).squeeze(1)
        restored = torch.zeros(
            batch_size * windows,
            768,
            dtype=adapted.dtype,
            device=adapted.device,
        )
        restored[flat_valid] = adapted
        output = SharedWindowEncoderOutput(
            embeddings=restored.reshape(batch_size, windows, 768),
            window_mask=batch.window_mask,
            time_map=batch.time_map,
            encoder_identity=self.encoder_identity,
            sample_ids=batch.sample_ids,
            dataset_ids=batch.dataset_ids,
            prediction_units=batch.prediction_units,
        )
        output.validate_against(batch)
        return output

    def receipt(self) -> dict[str, object]:
        dimension = self.dimension_adapter.receipt()
        return {
            "status": "production_adapter_code_ready",
            "experiment_result": False,
            "provenance": self.provenance.as_receipt(),
            "native_embedding_dim": self.backend.native_dim,
            "output_shape": "[B,K,768]",
            "flatten_restore": "valid_B_times_K_only_then_scatter_invalid_to_exact_zero",
            "lineage_preserved": True,
            "window_mask_and_source_time_map_preserved": True,
            "candidate_encoder_scope": "frozen_eval_no_grad",
            "dimension_adapter": dimension,
            "dimension_adapter_scope": (
                "trainable_shared_across_all_four_datasets"
                if dimension["trainable_parameters"]
                else "identity_no_parameters"
            ),
            "downstream_projector": "separate_shared_biased_linear_768_to_256",
            "outer_test_accessed": False,
        }


def missing_asset_receipt(
    encoder_identity: str,
    *,
    required_checkpoint: str,
    required_dependency: str,
    source_url: str,
    source_revision: str,
    license_name: str,
) -> dict[str, object]:
    return {
        "encoder_identity": encoder_identity,
        "code_status": "READY_local_contract_and_synthetic_CPU_smoke",
        "asset_status": "HOLD_missing_local_checkpoint_or_dependency",
        "cpu_real_checkpoint_status": "HOLD",
        "cuda_status": "HOLD_waiting_L40_preflight",
        "required_checkpoint": required_checkpoint,
        "required_dependency": required_dependency,
        "source_url": source_url,
        "source_revision": source_revision,
        "license": license_name,
        "download_performed": False,
        "experiment_result": False,
    }
