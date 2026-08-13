"""Immutable validation-only HF temporal threshold receipt contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .beats_temporal import CHANNEL_ORDER


HF_THRESHOLD_RECEIPT_SCHEMA_VERSION = "hf_temporal_threshold_receipt_v2"
HF_THRESHOLD_SELECTION_POLICY = (
    "validation_only_per_channel_max_f1;tie=highest_threshold;"
    "threshold_frozen_before_outer_test"
)
HF_THRESHOLD_SERIALIZATION = "ieee754_big_endian_4xfloat64"
HF_THRESHOLD_NATIVE_TASK = "HF_temporal4"
HF_THRESHOLD_SCORER_SCHEMA_VERSION = "shared_window_terminal_scorer_v2"
HF_THRESHOLD_SELECTION_RECEIPT_SCHEMA_VERSION = "validation_selection_v2"
HF_THRESHOLD_RUNNER_SCHEMA_VERSION = "shared_window_training_v5"


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_thresholds(values: Sequence[float]) -> tuple[tuple[float, ...], str]:
    if len(values) != len(CHANNEL_ORDER):
        raise ValueError("HF threshold receipt requires exactly four channel thresholds")
    thresholds = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("HF thresholds must be finite and strictly inside (0,1)")
    return thresholds, struct.pack(">4d", *thresholds).hex()


@dataclass(frozen=True)
class VerifiedHFThresholdReceipt:
    path: Path
    artifact_sha256: str
    size_bytes: int
    payload: Mapping[str, object]
    thresholds: tuple[float, ...]


@dataclass(frozen=True)
class HFValidationThresholdBatch:
    """Validation-only HF predictions used to select terminal thresholds."""

    prediction_ids: tuple[str, ...]
    probabilities: torch.Tensor
    targets: torch.Tensor
    window_mask: torch.Tensor
    annotation_mask: torch.Tensor
    valid_mask: torch.Tensor
    time_map: torch.Tensor
    partition: str = "validation"
    outer_test_accessed: bool = False

    def validate(self) -> None:
        if self.partition != "validation" or self.outer_test_accessed is not False:
            raise PermissionError(
                "HF thresholds may be selected only from validation with outer_test=false"
            )
        probabilities = self.probabilities.detach().cpu()
        targets = self.targets.detach().cpu()
        window_mask = self.window_mask.detach().cpu()
        annotation_mask = self.annotation_mask.detach().cpu()
        valid_mask = self.valid_mask.detach().cpu()
        time_map = self.time_map.detach().cpu()
        if probabilities.ndim != 3 or probabilities.shape[-1] != len(CHANNEL_ORDER):
            raise ValueError("HF validation probabilities must be [B,Nw,4]")
        batch, windows, _ = probabilities.shape
        if len(self.prediction_ids) != batch or not self.prediction_ids:
            raise ValueError("HF validation prediction IDs must align with batch rows")
        if any(not value for value in self.prediction_ids) or len(
            set(self.prediction_ids)
        ) != len(self.prediction_ids):
            raise ValueError("HF validation prediction IDs must be non-empty and unique")
        if targets.shape != probabilities.shape:
            raise ValueError("HF validation targets must match probabilities [B,Nw,4]")
        if window_mask.shape != (batch, windows) or window_mask.dtype != torch.bool:
            raise TypeError("HF validation window_mask must be bool [B,Nw]")
        for name, value in (
            ("annotation_mask", annotation_mask),
            ("valid_mask", valid_mask),
        ):
            if value.shape != probabilities.shape or value.dtype != torch.bool:
                raise TypeError(f"HF validation {name} must be bool [B,Nw,4]")
        if (
            time_map.shape != (batch, windows, 2)
            or time_map.dtype != torch.float64
            or not bool(torch.isfinite(time_map).all())
        ):
            raise TypeError("HF validation time_map must be finite float64 [B,Nw,2]")
        for row in range(batch):
            count = int(window_mask[row].sum())
            if count <= 0 or not bool(window_mask[row, :count].all()):
                raise ValueError("HF validation window_mask must be a non-empty prefix")
            valid_times = time_map[row, :count]
            if bool((valid_times[:, 1] <= valid_times[:, 0]).any()) or (
                count > 1 and bool((valid_times[1:, 0] < valid_times[:-1, 0]).any())
            ):
                raise ValueError("HF validation source-time windows are invalid")
            if bool(torch.count_nonzero(time_map[row, count:])):
                raise ValueError("HF validation padded time_map must be exact zero")
        if not probabilities.dtype.is_floating_point or not bool(
            torch.isfinite(probabilities).all()
        ):
            raise TypeError("HF validation probabilities must be finite floating point")
        if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
            raise ValueError("HF validation probabilities must be in [0,1]")
        if not targets.dtype.is_floating_point or not bool(
            ((targets == 0) | (targets == 1)).all()
        ):
            raise TypeError("HF validation targets must be binary floating point")
        if bool((valid_mask & ~annotation_mask).any()) or bool(
            (valid_mask & ~window_mask.unsqueeze(-1)).any()
        ):
            raise ValueError("HF validation supervision mask exceeds annotation/window scope")
        effective = window_mask.unsqueeze(-1) & annotation_mask & valid_mask
        for index, channel in enumerate(CHANNEL_ORDER):
            selected = targets[..., index][effective[..., index]]
            positives = int((selected == 1).sum())
            negatives = int((selected == 0).sum())
            if positives == 0 or negatives == 0:
                raise RuntimeError(
                    f"HF validation channel {channel} needs positive and negative support"
                )


@dataclass(frozen=True)
class HFThresholdSelection:
    thresholds: tuple[float, ...]
    ordered_prediction_ids_sha256: str
    validation_prediction_identity_sha256: str
    per_channel: tuple[Mapping[str, object], ...]


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_identity_sha256(value: torch.Tensor, *, dtype: torch.dtype) -> str:
    canonical = value.detach().cpu().to(dtype).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(str(canonical.numpy().dtype).encode("ascii"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def select_hf_validation_thresholds(
    batch: HFValidationThresholdBatch,
) -> HFThresholdSelection:
    """Select deterministic validation-only max-F1 thresholds per HF channel.

    Candidate thresholds are the unique predicted probabilities strictly inside
    ``(0,1)``.  If all probabilities are exact endpoints, the sole candidate is
    ``0.5``.  F1 ties are resolved by the highest threshold.
    """

    batch.validate()
    probabilities = batch.probabilities.detach().cpu().to(torch.float64)
    targets = batch.targets.detach().cpu().to(torch.float64)
    window_mask = batch.window_mask.detach().cpu()
    annotation_mask = batch.annotation_mask.detach().cpu()
    valid_mask = batch.valid_mask.detach().cpu()
    effective = window_mask.unsqueeze(-1) & annotation_mask & valid_mask
    selected_thresholds: list[float] = []
    summaries: list[Mapping[str, object]] = []
    for index, channel in enumerate(CHANNEL_ORDER):
        channel_mask = effective[..., index]
        scores = probabilities[..., index][channel_mask]
        labels = targets[..., index][channel_mask].to(torch.int64)
        candidates = sorted(
            {
                float(value)
                for value in scores.tolist()
                if 0.0 < float(value) < 1.0
            }
        )
        if not candidates:
            candidates = [0.5]
        best_threshold: float | None = None
        best_f1: Fraction | None = None
        best_counts: tuple[int, int, int] | None = None
        for threshold in candidates:
            predicted = scores >= threshold
            positive = labels == 1
            tp = int((predicted & positive).sum())
            fp = int((predicted & ~positive).sum())
            fn = int((~predicted & positive).sum())
            denominator = 2 * tp + fp + fn
            f1 = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
            if best_f1 is None or f1 > best_f1 or (
                f1 == best_f1 and threshold > float(best_threshold)
            ):
                best_threshold = threshold
                best_f1 = f1
                best_counts = (tp, fp, fn)
        if best_threshold is None or best_f1 is None or best_counts is None:
            raise RuntimeError(f"HF threshold selection failed for {channel}")
        selected_thresholds.append(best_threshold)
        summaries.append(
            {
                "channel": channel,
                "threshold": best_threshold,
                "max_f1": float(best_f1),
                "candidate_count": len(candidates),
                "valid_count": int(labels.numel()),
                "positive_support": int((labels == 1).sum()),
                "negative_support": int((labels == 0).sum()),
                "tp": best_counts[0],
                "fp": best_counts[1],
                "fn": best_counts[2],
            }
        )
    ordered_ids_sha256 = _canonical_json_sha256(
        {"ordered_prediction_ids": list(batch.prediction_ids)}
    )
    prediction_identity = _canonical_json_sha256(
        {
            "channel_order": list(CHANNEL_ORDER),
            "ordered_prediction_ids_sha256": ordered_ids_sha256,
            "probabilities_float64_sha256": _tensor_identity_sha256(
                probabilities, dtype=torch.float64
            ),
            "targets_float64_sha256": _tensor_identity_sha256(
                targets, dtype=torch.float64
            ),
            "window_mask_sha256": _tensor_identity_sha256(
                window_mask, dtype=torch.bool
            ),
            "annotation_mask_sha256": _tensor_identity_sha256(
                annotation_mask, dtype=torch.bool
            ),
            "valid_mask_sha256": _tensor_identity_sha256(
                valid_mask, dtype=torch.bool
            ),
            "time_map_float64_sha256": _tensor_identity_sha256(
                batch.time_map, dtype=torch.float64
            ),
            "partition": "validation",
            "outer_test_accessed": False,
        }
    )
    return HFThresholdSelection(
        thresholds=tuple(selected_thresholds),
        ordered_prediction_ids_sha256=ordered_ids_sha256,
        validation_prediction_identity_sha256=prediction_identity,
        per_channel=tuple(summaries),
    )


def _normalize_per_channel_selection(
    values: object, thresholds: Sequence[float]
) -> list[dict[str, object]]:
    if not isinstance(values, (list, tuple)) or len(values) != len(CHANNEL_ORDER):
        raise TypeError("HF threshold per-channel selection must contain four rows")
    required = {
        "channel",
        "threshold",
        "max_f1",
        "candidate_count",
        "valid_count",
        "positive_support",
        "negative_support",
        "tp",
        "fp",
        "fn",
    }
    normalized: list[dict[str, object]] = []
    for index, (raw, channel, threshold) in enumerate(
        zip(values, CHANNEL_ORDER, thresholds)
    ):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("HF threshold per-channel selection fields changed")
        row = dict(raw)
        integers = {
            key: row[key]
            for key in (
                "candidate_count",
                "valid_count",
                "positive_support",
                "negative_support",
                "tp",
                "fp",
                "fn",
            )
        }
        if (
            row["channel"] != channel
            or float(row["threshold"]) != float(threshold)
            or not isinstance(row["max_f1"], (int, float))
            or not math.isfinite(float(row["max_f1"]))
            or not 0.0 <= float(row["max_f1"]) <= 1.0
            or any(not isinstance(value, int) or value < 0 for value in integers.values())
            or integers["candidate_count"] <= 0
            or integers["positive_support"] <= 0
            or integers["negative_support"] <= 0
            or integers["valid_count"]
            != integers["positive_support"] + integers["negative_support"]
            or integers["tp"] > integers["positive_support"]
            or integers["fn"] > integers["positive_support"]
            or integers["fp"] > integers["negative_support"]
            or integers["tp"] + integers["fn"] != integers["positive_support"]
        ):
            raise RuntimeError(f"HF threshold per-channel support contract failed at {index}")
        normalized.append(row)
    return normalized


def threshold_receipt_payload(
    *,
    thresholds: Sequence[float],
    validation_data_identity_sha256: str,
    hf_validation_manifest_identity_sha256: str,
    hf_validation_ordered_prediction_ids_sha256: str,
    full_approval_receipt_sha256: str,
    validation_selection_receipt_sha256: str,
    selected_checkpoint_sha256: str,
    validation_prediction_identity_sha256: str,
    per_channel_selection: Sequence[Mapping[str, object]],
    scorer_schema_version: str,
) -> dict[str, object]:
    canonical, threshold_bytes_hex = _canonical_thresholds(thresholds)
    for label, value in (
        ("validation data identity", validation_data_identity_sha256),
        ("HF validation manifest identity", hf_validation_manifest_identity_sha256),
        ("HF validation ordered prediction IDs", hf_validation_ordered_prediction_ids_sha256),
        ("full approval receipt", full_approval_receipt_sha256),
        ("validation selection receipt", validation_selection_receipt_sha256),
        ("selected checkpoint", selected_checkpoint_sha256),
        ("validation prediction identity", validation_prediction_identity_sha256),
    ):
        _require_sha256(value, label)
    if not scorer_schema_version.strip():
        raise ValueError("terminal scorer schema version is empty")
    normalized_per_channel = _normalize_per_channel_selection(
        per_channel_selection, canonical
    )
    return {
        "schema_version": HF_THRESHOLD_RECEIPT_SCHEMA_VERSION,
        "native_task": HF_THRESHOLD_NATIVE_TASK,
        "scorer_schema_version": scorer_schema_version,
        "channel_order": list(CHANNEL_ORDER),
        "thresholds": list(canonical),
        "threshold_dtype": "float64",
        "threshold_serialization": HF_THRESHOLD_SERIALIZATION,
        "threshold_bytes_hex": threshold_bytes_hex,
        "threshold_selection_policy": HF_THRESHOLD_SELECTION_POLICY,
        "selected_on_partition": "validation",
        "validation_data_identity_sha256": validation_data_identity_sha256,
        "hf_validation_manifest_identity_sha256": hf_validation_manifest_identity_sha256,
        "hf_validation_ordered_prediction_ids_sha256": hf_validation_ordered_prediction_ids_sha256,
        "full_approval_receipt_sha256": full_approval_receipt_sha256,
        "validation_selection_receipt_sha256": validation_selection_receipt_sha256,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "validation_prediction_identity_sha256": validation_prediction_identity_sha256,
        "per_channel_selection": normalized_per_channel,
        "negative_semantics": "source_task_constructed_not_raw_normal",
        "shared_label_eligible": False,
        "outer_test_accessed": False,
    }


def write_hf_threshold_receipt(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    """Atomically write a validated receipt; never overwrite an existing artifact."""

    _validate_payload(payload, expected_scorer_schema_version=str(payload.get("scorer_schema_version", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"HF threshold receipt already exists: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _validate_payload(
    payload: Mapping[str, object], *, expected_scorer_schema_version: str
) -> tuple[float, ...]:
    expected_fields = {
        "schema_version",
        "native_task",
        "scorer_schema_version",
        "channel_order",
        "thresholds",
        "threshold_dtype",
        "threshold_serialization",
        "threshold_bytes_hex",
        "threshold_selection_policy",
        "selected_on_partition",
        "validation_data_identity_sha256",
        "hf_validation_manifest_identity_sha256",
        "hf_validation_ordered_prediction_ids_sha256",
        "full_approval_receipt_sha256",
        "validation_selection_receipt_sha256",
        "selected_checkpoint_sha256",
        "validation_prediction_identity_sha256",
        "per_channel_selection",
        "negative_semantics",
        "shared_label_eligible",
        "outer_test_accessed",
    }
    if set(payload) != expected_fields:
        raise ValueError("HF threshold receipt fields changed")
    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, list):
        raise TypeError("HF threshold receipt thresholds must be a JSON list")
    thresholds, threshold_bytes_hex = _canonical_thresholds(raw_thresholds)
    if (
        payload["schema_version"] != HF_THRESHOLD_RECEIPT_SCHEMA_VERSION
        or payload["native_task"] != HF_THRESHOLD_NATIVE_TASK
        or payload["scorer_schema_version"] != expected_scorer_schema_version
        or payload["channel_order"] != list(CHANNEL_ORDER)
        or payload["threshold_dtype"] != "float64"
        or payload["threshold_serialization"] != HF_THRESHOLD_SERIALIZATION
        or payload["threshold_bytes_hex"] != threshold_bytes_hex
        or payload["threshold_selection_policy"] != HF_THRESHOLD_SELECTION_POLICY
        or payload["selected_on_partition"] != "validation"
        or payload["negative_semantics"]
        != "source_task_constructed_not_raw_normal"
        or payload["shared_label_eligible"] is not False
        or payload["outer_test_accessed"] is not False
    ):
        raise RuntimeError("HF threshold semantic/serialization contract failed")
    for label, key in (
        ("validation data identity", "validation_data_identity_sha256"),
        ("HF validation manifest identity", "hf_validation_manifest_identity_sha256"),
        ("HF validation ordered prediction IDs", "hf_validation_ordered_prediction_ids_sha256"),
        ("full approval receipt", "full_approval_receipt_sha256"),
        ("validation selection receipt", "validation_selection_receipt_sha256"),
        ("selected checkpoint", "selected_checkpoint_sha256"),
        ("validation prediction identity", "validation_prediction_identity_sha256"),
    ):
        _require_sha256(payload[key], label)
    _normalize_per_channel_selection(payload["per_channel_selection"], thresholds)
    return thresholds


def load_and_verify_hf_threshold_receipt(
    path: Path,
    expected_artifact_sha256: str,
    *,
    expected_scorer_schema_version: str,
    expected_validation_data_identity_sha256: str,
    expected_hf_validation_manifest_identity_sha256: str,
    expected_hf_validation_ordered_prediction_ids_sha256: str,
    expected_full_approval_receipt_sha256: str,
    expected_validation_selection_receipt_sha256: str,
    expected_selected_checkpoint_sha256: str,
) -> VerifiedHFThresholdReceipt:
    expected_artifact_sha256 = _require_sha256(
        expected_artifact_sha256, "HF threshold receipt artifact"
    )
    if not path.is_file() or _sha256_path(path) != expected_artifact_sha256:
        raise RuntimeError("HF threshold receipt byte SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("HF threshold receipt must be a JSON object")
    thresholds = _validate_payload(
        payload, expected_scorer_schema_version=expected_scorer_schema_version
    )
    expected = {
        "validation_data_identity_sha256": expected_validation_data_identity_sha256,
        "hf_validation_manifest_identity_sha256": expected_hf_validation_manifest_identity_sha256,
        "hf_validation_ordered_prediction_ids_sha256": expected_hf_validation_ordered_prediction_ids_sha256,
        "full_approval_receipt_sha256": expected_full_approval_receipt_sha256,
        "validation_selection_receipt_sha256": expected_validation_selection_receipt_sha256,
        "selected_checkpoint_sha256": expected_selected_checkpoint_sha256,
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise RuntimeError("HF threshold receipt immutable identity chain mismatch")
    return VerifiedHFThresholdReceipt(
        path=path.resolve(),
        artifact_sha256=expected_artifact_sha256,
        size_bytes=path.stat().st_size,
        payload=payload,
        thresholds=thresholds,
    )


def _load_verified_json_artifact(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, object], str]:
    expected_sha256 = _require_sha256(expected_sha256, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual_sha256 = _sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload, actual_sha256


def _verify_threshold_generation_chain(
    *,
    full_approval_receipt_path: Path,
    expected_full_approval_receipt_sha256: str,
    validation_selection_receipt_path: Path,
    expected_validation_selection_receipt_sha256: str,
    selected_checkpoint_path: Path,
    expected_selected_checkpoint_sha256: str,
    validation_data_identity_sha256: str,
    hf_validation_manifest_identity_sha256: str,
    hf_validation_ordered_prediction_ids_sha256: str,
) -> dict[str, str]:
    approval, approval_sha256 = _load_verified_json_artifact(
        full_approval_receipt_path,
        expected_full_approval_receipt_sha256,
        "full approval receipt",
    )
    approval_required = {
        "status",
        "pipeline_id",
        "phase",
        "config_sha256",
        "data_identity_sha256",
        "authorized_by",
        "outer_test_authorized",
    }
    if approval_required - set(approval):
        raise ValueError("full approval receipt fields are incomplete")
    if (
        approval["status"] != "approved"
        or approval["phase"] != "full"
        or approval["outer_test_authorized"] is not False
        or not str(approval["authorized_by"]).strip()
    ):
        raise PermissionError("HF threshold generation requires approved full/outer=false")

    selection, selection_sha256 = _load_verified_json_artifact(
        validation_selection_receipt_path,
        expected_validation_selection_receipt_sha256,
        "validation selection receipt",
    )
    selection_required = {
        "schema_version",
        "runner_schema_version",
        "pipeline_id",
        "config_sha256",
        "data_identity_sha256",
        "full_approval_receipt_sha256",
        "hf_validation_threshold_identity",
        "candidates",
        "selected_checkpoint",
        "outer_test_accessed",
        "reported_as_pooled_performance",
    }
    if selection_required - set(selection):
        raise ValueError("validation selection receipt fields are incomplete")
    expected_hf_identity = {
        "validation_data_identity_sha256": _require_sha256(
            validation_data_identity_sha256, "HF validation data identity"
        ),
        "hf_validation_manifest_identity_sha256": _require_sha256(
            hf_validation_manifest_identity_sha256,
            "HF validation manifest identity",
        ),
        "hf_validation_ordered_prediction_ids_sha256": _require_sha256(
            hf_validation_ordered_prediction_ids_sha256,
            "HF validation ordered prediction IDs",
        ),
    }
    if (
        selection["schema_version"]
        != HF_THRESHOLD_SELECTION_RECEIPT_SCHEMA_VERSION
        or selection["runner_schema_version"] != HF_THRESHOLD_RUNNER_SCHEMA_VERSION
        or selection["pipeline_id"] != approval["pipeline_id"]
        or selection["config_sha256"] != approval["config_sha256"]
        or selection["data_identity_sha256"] != approval["data_identity_sha256"]
        or selection["full_approval_receipt_sha256"] != approval_sha256
        or selection["hf_validation_threshold_identity"] != expected_hf_identity
        or selection["outer_test_accessed"] is not False
        or selection["reported_as_pooled_performance"] is not False
    ):
        raise RuntimeError("HF threshold approval/selection identity chain failed")
    candidates = selection["candidates"]
    selected = selection["selected_checkpoint"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(selected, Mapping)
        or sum(
            isinstance(candidate, Mapping) and dict(candidate) == dict(selected)
            for candidate in candidates
        )
        != 1
    ):
        raise RuntimeError("validation selection does not name one selected checkpoint")
    selected_required = {
        "path",
        "size_bytes",
        "sha256",
        "outer_test_accessed",
        "native_metrics_only",
    }
    if selected_required - set(selected):
        raise ValueError("selected checkpoint receipt fields are incomplete")
    expected_selected_checkpoint_sha256 = _require_sha256(
        expected_selected_checkpoint_sha256, "selected checkpoint"
    )
    if (
        selected["outer_test_accessed"] is not False
        or selected["native_metrics_only"] is not True
        or selected["sha256"] != expected_selected_checkpoint_sha256
        or Path(str(selected["path"])).resolve()
        != selected_checkpoint_path.resolve()
        or not selected_checkpoint_path.is_file()
        or selected_checkpoint_path.stat().st_size != int(selected["size_bytes"])
        or _sha256_path(selected_checkpoint_path)
        != expected_selected_checkpoint_sha256
    ):
        raise RuntimeError("selected checkpoint artifact/outer-test binding failed")
    return {
        "full_approval_receipt_sha256": approval_sha256,
        "validation_selection_receipt_sha256": selection_sha256,
        "selected_checkpoint_sha256": expected_selected_checkpoint_sha256,
    }


def select_and_write_hf_threshold_receipt(
    output_path: Path,
    validation_batch: HFValidationThresholdBatch,
    *,
    validation_data_identity_sha256: str,
    hf_validation_manifest_identity_sha256: str,
    expected_hf_validation_ordered_prediction_ids_sha256: str,
    full_approval_receipt_path: Path,
    expected_full_approval_receipt_sha256: str,
    validation_selection_receipt_path: Path,
    expected_validation_selection_receipt_sha256: str,
    selected_checkpoint_path: Path,
    expected_selected_checkpoint_sha256: str,
    scorer_schema_version: str = HF_THRESHOLD_SCORER_SCHEMA_VERSION,
) -> dict[str, object]:
    """Generate one immutable threshold receipt after a real full selection.

    This function never reads outer/test data.  Its caller must supply the HF
    validation predictions generated from the exact selected checkpoint.
    """

    if scorer_schema_version != HF_THRESHOLD_SCORER_SCHEMA_VERSION:
        raise RuntimeError("HF threshold scorer schema changed")
    selection = select_hf_validation_thresholds(validation_batch)
    expected_ordered_ids = _require_sha256(
        expected_hf_validation_ordered_prediction_ids_sha256,
        "expected HF validation ordered prediction IDs",
    )
    if selection.ordered_prediction_ids_sha256 != expected_ordered_ids:
        raise RuntimeError("HF validation prediction order differs from frozen identity")
    chain = _verify_threshold_generation_chain(
        full_approval_receipt_path=full_approval_receipt_path,
        expected_full_approval_receipt_sha256=expected_full_approval_receipt_sha256,
        validation_selection_receipt_path=validation_selection_receipt_path,
        expected_validation_selection_receipt_sha256=(
            expected_validation_selection_receipt_sha256
        ),
        selected_checkpoint_path=selected_checkpoint_path,
        expected_selected_checkpoint_sha256=expected_selected_checkpoint_sha256,
        validation_data_identity_sha256=validation_data_identity_sha256,
        hf_validation_manifest_identity_sha256=(
            hf_validation_manifest_identity_sha256
        ),
        hf_validation_ordered_prediction_ids_sha256=expected_ordered_ids,
    )
    payload = threshold_receipt_payload(
        thresholds=selection.thresholds,
        validation_data_identity_sha256=validation_data_identity_sha256,
        hf_validation_manifest_identity_sha256=(
            hf_validation_manifest_identity_sha256
        ),
        hf_validation_ordered_prediction_ids_sha256=expected_ordered_ids,
        full_approval_receipt_sha256=chain["full_approval_receipt_sha256"],
        validation_selection_receipt_sha256=chain[
            "validation_selection_receipt_sha256"
        ],
        selected_checkpoint_sha256=chain["selected_checkpoint_sha256"],
        validation_prediction_identity_sha256=(
            selection.validation_prediction_identity_sha256
        ),
        per_channel_selection=selection.per_channel,
        scorer_schema_version=scorer_schema_version,
    )
    artifact = write_hf_threshold_receipt(output_path, payload)
    load_and_verify_hf_threshold_receipt(
        output_path,
        str(artifact["sha256"]),
        expected_scorer_schema_version=scorer_schema_version,
        expected_validation_data_identity_sha256=validation_data_identity_sha256,
        expected_hf_validation_manifest_identity_sha256=(
            hf_validation_manifest_identity_sha256
        ),
        expected_hf_validation_ordered_prediction_ids_sha256=expected_ordered_ids,
        expected_full_approval_receipt_sha256=chain[
            "full_approval_receipt_sha256"
        ],
        expected_validation_selection_receipt_sha256=chain[
            "validation_selection_receipt_sha256"
        ],
        expected_selected_checkpoint_sha256=chain["selected_checkpoint_sha256"],
    )
    return {
        "status": "hf_threshold_artifact_generated_validation_only",
        "policy": HF_THRESHOLD_SELECTION_POLICY,
        "native_task": HF_THRESHOLD_NATIVE_TASK,
        "scorer_schema_version": scorer_schema_version,
        "channel_order": list(CHANNEL_ORDER),
        "thresholds": list(selection.thresholds),
        "threshold_dtype": "float64",
        "threshold_bytes_hex": payload["threshold_bytes_hex"],
        "validation_prediction_identity_sha256": (
            selection.validation_prediction_identity_sha256
        ),
        "per_channel_selection": [dict(value) for value in selection.per_channel],
        "artifact": artifact,
        "outer_test_accessed": False,
    }
