"""Canonical HF validation-prediction export from an exact selected checkpoint.

This module is validation-only.  It derives the checkpoint path from the frozen
selection receipt, consumes only the HF validation provider/cache, and delegates
threshold selection and receipt verification to :mod:`hf_thresholds`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .beats_temporal import (
    CHANNEL_ORDER,
    HFTargetPolicy,
    TokenAlignmentPolicy,
    raw_intervals_to_token_supervision,
)
from .embedding_cache import FrozenEmbeddingCache
from .hf_thresholds import (
    HFValidationThresholdBatch,
    _verify_threshold_generation_chain,
    select_and_write_hf_threshold_receipt,
    select_hf_validation_thresholds,
)
from .production_terminal_provider import load_exact_selected_model
from .real_subtrain_provider import build_frozen_provider_index, load_native_window_batch
from .runner_embedding_cache import (
    CachedLanePartition,
    _compute_lane_payload,
    _identity_for_lane,
)
from .train_shared_window import hf_validation_threshold_identity


HF_VALIDATION_PREDICTION_SCHEMA_VERSION = "hf_validation_predictions_v1"
HF_VALIDATION_EXPORT_RECEIPT_SCHEMA_VERSION = "hf_validation_prediction_export_v1"
HF_VALIDATION_EXPORTER_SPECIFICATION = (
    "baseline.multidataset_pipeline.hf_validation_exporter:"
    "export_hf_validation_predictions"
)
HF_VALIDATION_EXPORT_BATCH_SIZE = 64


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_sha256(value: torch.Tensor, dtype: torch.dtype) -> str:
    tensor = value.detach().cpu().to(dtype).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.numpy().dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _read_verified_selection(
    path: Path, expected_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    expected_sha256 = _require_sha256(expected_sha256, "validation selection receipt")
    if not path.is_file() or _sha256_path(path) != expected_sha256:
        raise RuntimeError("validation selection receipt byte SHA256 mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError("validation selection receipt must be a JSON object")
    selected = receipt.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise RuntimeError("validation selection receipt lacks selected checkpoint")
    required = {
        "path",
        "size_bytes",
        "sha256",
        "update",
        "outer_test_accessed",
        "native_metrics_only",
    }
    if required - set(selected):
        raise ValueError("selected checkpoint receipt fields are incomplete")
    checkpoint = Path(str(selected["path"])).resolve()
    checkpoint_sha256 = _require_sha256(selected["sha256"], "selected checkpoint")
    if (
        receipt.get("selected_update") != selected["update"]
        or not isinstance(selected["update"], int)
        or selected["update"] <= 0
        or selected["outer_test_accessed"] is not False
        or selected["native_metrics_only"] is not True
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != int(selected["size_bytes"])
        or _sha256_path(checkpoint) != checkpoint_sha256
    ):
        raise RuntimeError("selected checkpoint path/update/size/SHA isolation gate failed")
    return receipt, {
        "path": checkpoint,
        "size_bytes": int(selected["size_bytes"]),
        "sha256": checkpoint_sha256,
        "update": int(selected["update"]),
    }


def _load_hf_validation_cache(
    *,
    repo_root: Path,
    pipeline_id: str,
    config_identity_sha256: str,
    adapter: torch.nn.Module,
    validation_index: object,
    device: torch.device,
) -> CachedLanePartition:
    units = validation_index.lanes["HF"]
    identity = _identity_for_lane(
        repo_root=repo_root,
        pipeline_id=pipeline_id,
        config_identity_sha256=config_identity_sha256,
        adapter=adapter,
        index=validation_index,
        lane="HF",
    )
    cache = FrozenEmbeddingCache(
        repo_root / ".cache" / "multidataset_pipeline" / "embeddings" / pipeline_id
    )
    payload, cache_receipt = cache.get_or_compute(
        identity,
        adapter,
        lambda: _compute_lane_payload(
            adapter,
            units,
            device=device,
            batch_size=HF_VALIDATION_EXPORT_BATCH_SIZE,
            batch_loader=load_native_window_batch,
        ),
        frontend_deterministic=True,
        augmentation_enabled=False,
    )
    if payload.unit_ids != tuple(unit.sample.sample_id for unit in units):
        raise RuntimeError("HF validation cache ordered unit identity changed")
    return CachedLanePartition(
        partition="validation",
        lane="HF",
        units=units,
        payload=payload,
        receipt={**cache_receipt, "encoder_identity": adapter.encoder_identity},
    )


def _pad_validation_rows(
    rows: Sequence[
        tuple[
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
) -> HFValidationThresholdBatch:
    if not rows:
        raise RuntimeError("HF validation exporter produced no rows")
    maximum = max(row[1].shape[0] for row in rows)
    batch = len(rows)
    probabilities = torch.zeros(batch, maximum, 4, dtype=torch.float64)
    targets = torch.zeros(batch, maximum, 4, dtype=torch.float64)
    window_mask = torch.zeros(batch, maximum, dtype=torch.bool)
    annotation_mask = torch.zeros(batch, maximum, 4, dtype=torch.bool)
    valid_mask = torch.zeros(batch, maximum, 4, dtype=torch.bool)
    time_map = torch.zeros(batch, maximum, 2, dtype=torch.float64)
    identifiers: list[str] = []
    for index, (
        identifier,
        row_probabilities,
        row_targets,
        row_window_mask,
        row_annotation_mask,
        row_valid_mask,
        row_time_map,
    ) in enumerate(rows):
        count = row_probabilities.shape[0]
        identifiers.append(identifier)
        probabilities[index, :count].copy_(row_probabilities.to(torch.float64))
        targets[index, :count].copy_(row_targets.to(torch.float64))
        window_mask[index, :count].copy_(row_window_mask.to(torch.bool))
        annotation_mask[index, :count].copy_(row_annotation_mask.to(torch.bool))
        valid_mask[index, :count].copy_(row_valid_mask.to(torch.bool))
        time_map[index, :count].copy_(row_time_map.to(torch.float64))
    result = HFValidationThresholdBatch(
        prediction_ids=tuple(identifiers),
        probabilities=probabilities,
        targets=targets,
        window_mask=window_mask,
        annotation_mask=annotation_mask,
        valid_mask=valid_mask,
        time_map=time_map,
        partition="validation",
        outer_test_accessed=False,
    )
    result.validate()
    return result


def _write_prediction_artifact(
    path: Path,
    *,
    pipeline_id: str,
    batch: HFValidationThresholdBatch,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HF_VALIDATION_PREDICTION_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "partition": "validation",
        "channel_order": list(CHANNEL_ORDER),
        "prediction_ids": list(batch.prediction_ids),
        "probabilities": batch.probabilities,
        "targets": batch.targets,
        "window_mask": batch.window_mask,
        "annotation_mask": batch.annotation_mask,
        "valid_mask": batch.valid_mask,
        "time_map": batch.time_map,
        "outer_test_accessed": False,
    }
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        torch.save(payload, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"HF validation prediction artifact exists: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256_path(path)}


def _load_prediction_artifact(
    path: Path, expected_sha256: str, pipeline_id: str
) -> HFValidationThresholdBatch:
    if not path.is_file() or _sha256_path(path) != _require_sha256(
        expected_sha256, "HF validation prediction artifact"
    ):
        raise RuntimeError("HF validation prediction artifact byte SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version",
        "pipeline_id",
        "partition",
        "channel_order",
        "prediction_ids",
        "probabilities",
        "targets",
        "window_mask",
        "annotation_mask",
        "valid_mask",
        "time_map",
        "outer_test_accessed",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise RuntimeError("HF validation prediction artifact schema changed")
    if (
        payload["schema_version"] != HF_VALIDATION_PREDICTION_SCHEMA_VERSION
        or payload["pipeline_id"] != pipeline_id
        or payload["partition"] != "validation"
        or payload["channel_order"] != list(CHANNEL_ORDER)
        or payload["outer_test_accessed"] is not False
    ):
        raise RuntimeError("HF validation prediction artifact isolation gate failed")
    batch = HFValidationThresholdBatch(
        prediction_ids=tuple(payload["prediction_ids"]),
        probabilities=payload["probabilities"],
        targets=payload["targets"],
        window_mask=payload["window_mask"],
        annotation_mask=payload["annotation_mask"],
        valid_mask=payload["valid_mask"],
        time_map=payload["time_map"],
        partition="validation",
        outer_test_accessed=False,
    )
    batch.validate()
    return batch


def _write_json_no_overwrite(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(raw)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"HF validation export receipt exists: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256_path(path)}


def export_hf_validation_predictions(
    *,
    repo_root: Path,
    selection_receipt_path: Path,
    expected_selection_receipt_sha256: str,
    full_approval_receipt_path: Path,
    output_root: Path,
    device: str,
) -> dict[str, object]:
    """Export exact-checkpoint HF validation predictions and freeze thresholds."""

    repo_root = repo_root.resolve()
    selection, selected = _read_verified_selection(
        selection_receipt_path, expected_selection_receipt_sha256
    )
    checkpoint = selected["path"]
    config, adapter, model, checkpoint_payload = load_exact_selected_model(checkpoint)
    if config.dataset_root.resolve().parents[1] != repo_root:
        raise RuntimeError("selected checkpoint is not bound to the canonical repository")
    if selection.get("pipeline_id") != config.pipeline_id:
        raise RuntimeError("selection/checkpoint pipeline mismatch")
    validation_index = build_frozen_provider_index(
        config.dataset_root,
        partition="validation",
        kauh_outer_fold=config.kauh_outer_fold,
        enforce_real_counts=True,
    )
    hf_identity = hf_validation_threshold_identity(validation_index)
    if selection.get("hf_validation_threshold_identity") != hf_identity:
        raise RuntimeError("frozen HF validation identity differs from selection receipt")
    if checkpoint_payload.get("data_identity_sha256") != selection.get("data_identity_sha256"):
        raise RuntimeError("checkpoint/selection combined data identity mismatch")
    approval_sha256 = _require_sha256(
        selection.get("full_approval_receipt_sha256"), "full approval receipt"
    )
    _verify_threshold_generation_chain(
        full_approval_receipt_path=full_approval_receipt_path,
        expected_full_approval_receipt_sha256=approval_sha256,
        validation_selection_receipt_path=selection_receipt_path,
        expected_validation_selection_receipt_sha256=expected_selection_receipt_sha256,
        selected_checkpoint_path=checkpoint,
        expected_selected_checkpoint_sha256=selected["sha256"],
        validation_data_identity_sha256=hf_identity["validation_data_identity_sha256"],
        hf_validation_manifest_identity_sha256=hf_identity[
            "hf_validation_manifest_identity_sha256"
        ],
        hf_validation_ordered_prediction_ids_sha256=hf_identity[
            "hf_validation_ordered_prediction_ids_sha256"
        ],
    )

    torch_device = torch.device(device)
    adapter.to(torch_device)
    model.to(torch_device).eval()
    cached = _load_hf_validation_cache(
        repo_root=repo_root,
        pipeline_id=config.pipeline_id,
        config_identity_sha256=config.sha256(),
        adapter=adapter,
        validation_index=validation_index,
        device=torch_device,
    )
    rows = []
    supervision_totals = {
        "constructed_negative_values": 0,
        "positive_values": 0,
        "empty_recordings": 0,
    }
    with torch.no_grad():
        for start in range(0, len(cached.units), HF_VALIDATION_EXPORT_BATCH_SIZE):
            indices = tuple(
                range(start, min(start + HF_VALIDATION_EXPORT_BATCH_SIZE, len(cached.units)))
            )
            batch = cached.batch(indices, device=torch_device)
            logits = model(batch.output.embeddings, "HF")["temporal4"]
            supervision = raw_intervals_to_token_supervision(
                batch.output.time_map,
                batch.output.window_mask,
                batch.hf_intervals,
                batch.hf_recording_states,
                policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
                alignment=TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL,
            )
            if (
                supervision.receipt.get("negative_semantics")
                != "source_task_constructed_not_raw_normal"
                or supervision.receipt.get("raw_explicit_negative_intervals") != 0
            ):
                raise RuntimeError("HF validation negative-semantics contract changed")
            for key in supervision_totals:
                supervision_totals[key] += int(supervision.receipt[key])
            probabilities = torch.sigmoid(logits).detach().cpu()
            for row, identifier in enumerate(batch.output.sample_ids):
                count = int(batch.output.window_mask[row].sum())
                rows.append(
                    (
                        identifier,
                        probabilities[row, :count],
                        supervision.targets[row, :count].detach().cpu(),
                        batch.output.window_mask[row, :count].detach().cpu(),
                        supervision.observation_mask[row, :count].detach().cpu(),
                        supervision.valid_mask[row, :count].detach().cpu(),
                        batch.output.time_map[row, :count].detach().cpu(),
                    )
                )
    validation_batch = _pad_validation_rows(rows)
    if validation_batch.prediction_ids != tuple(
        unit.sample.sample_id for unit in validation_index.lanes["HF"]
    ):
        raise RuntimeError("HF validation exported prediction order changed")

    output_root.mkdir(parents=True, exist_ok=True)
    artifact = _write_prediction_artifact(
        output_root / "hf_validation_predictions.pt",
        pipeline_id=config.pipeline_id,
        batch=validation_batch,
    )
    verified_batch = _load_prediction_artifact(
        Path(str(artifact["path"])), str(artifact["sha256"]), config.pipeline_id
    )
    threshold_result = select_and_write_hf_threshold_receipt(
        output_root / "hf_threshold_receipt.json",
        verified_batch,
        validation_data_identity_sha256=hf_identity["validation_data_identity_sha256"],
        hf_validation_manifest_identity_sha256=hf_identity[
            "hf_validation_manifest_identity_sha256"
        ],
        expected_hf_validation_ordered_prediction_ids_sha256=hf_identity[
            "hf_validation_ordered_prediction_ids_sha256"
        ],
        full_approval_receipt_path=full_approval_receipt_path,
        expected_full_approval_receipt_sha256=approval_sha256,
        validation_selection_receipt_path=selection_receipt_path,
        expected_validation_selection_receipt_sha256=expected_selection_receipt_sha256,
        selected_checkpoint_path=checkpoint,
        expected_selected_checkpoint_sha256=selected["sha256"],
    )
    selected_thresholds = select_hf_validation_thresholds(verified_batch)
    tensor_identities = {
        "probabilities_float64_sha256": _tensor_sha256(verified_batch.probabilities, torch.float64),
        "targets_float64_sha256": _tensor_sha256(verified_batch.targets, torch.float64),
        "window_mask_sha256": _tensor_sha256(verified_batch.window_mask, torch.bool),
        "annotation_mask_sha256": _tensor_sha256(verified_batch.annotation_mask, torch.bool),
        "valid_mask_sha256": _tensor_sha256(verified_batch.valid_mask, torch.bool),
        "time_map_float64_sha256": _tensor_sha256(verified_batch.time_map, torch.float64),
    }
    implementation = Path(__file__).resolve()
    receipt = {
        "schema_version": HF_VALIDATION_EXPORT_RECEIPT_SCHEMA_VERSION,
        "status": "hf_validation_predictions_and_threshold_exported",
        "pipeline_id": config.pipeline_id,
        "partition": "validation",
        "exporter_specification": HF_VALIDATION_EXPORTER_SPECIFICATION,
        "exporter_implementation_sha256": _sha256_path(implementation),
        "selected_checkpoint": {
            "path": str(checkpoint),
            "update": selected["update"],
            "size_bytes": selected["size_bytes"],
            "sha256": selected["sha256"],
        },
        "full_approval_receipt_sha256": approval_sha256,
        "validation_selection_receipt_sha256": expected_selection_receipt_sha256,
        "hf_validation_identity": hf_identity,
        "ordered_prediction_ids_sha256": selected_thresholds.ordered_prediction_ids_sha256,
        "validation_prediction_identity_sha256": selected_thresholds.validation_prediction_identity_sha256,
        "prediction_rows": len(verified_batch.prediction_ids),
        "tensor_shape": list(verified_batch.probabilities.shape),
        "tensor_identities": tensor_identities,
        "prediction_artifact": artifact,
        "threshold_artifact": threshold_result["artifact"],
        "per_channel_selection": threshold_result["per_channel_selection"],
        "supervision_receipt": {
            "policy": HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR.value,
            "alignment": TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL.value,
            "raw_explicit_negative_intervals": 0,
            **supervision_totals,
        },
        "negative_semantics": "source_task_constructed_not_raw_normal",
        "gap_missing_unknown_as_raw_negative": False,
        "constructed_negatives_explicit": True,
        "outer_test_accessed": False,
        "outer_test_waveforms_decoded": 0,
        "terminal_targets_loaded": False,
    }
    receipt_artifact = _write_json_no_overwrite(
        output_root / "validation_export_receipt.json", receipt
    )
    persisted_receipt = json.loads(
        Path(str(receipt_artifact["path"])).read_text(encoding="utf-8")
    )
    if persisted_receipt != receipt or _sha256_path(
        Path(str(receipt_artifact["path"]))
    ) != receipt_artifact["sha256"]:
        raise RuntimeError("HF validation export receipt independent readback failed")
    return {**receipt, "receipt_artifact": receipt_artifact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--full-approval-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = export_hf_validation_predictions(
        repo_root=args.repo_root,
        selection_receipt_path=args.selection_receipt,
        expected_selection_receipt_sha256=args.selection_sha256,
        full_approval_receipt_path=args.full_approval_receipt,
        output_root=args.output_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
