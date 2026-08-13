"""Immutable validation-only HF temporal threshold receipt contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .beats_temporal import CHANNEL_ORDER


HF_THRESHOLD_RECEIPT_SCHEMA_VERSION = "hf_temporal_threshold_receipt_v1"
HF_THRESHOLD_SELECTION_POLICY = (
    "validation_only_per_channel_max_f1;tie=highest_threshold;"
    "threshold_frozen_before_outer_test"
)
HF_THRESHOLD_SERIALIZATION = "ieee754_big_endian_4xfloat64"
HF_THRESHOLD_NATIVE_TASK = "HF_temporal4"


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


def threshold_receipt_payload(
    *,
    thresholds: Sequence[float],
    validation_data_identity_sha256: str,
    hf_validation_manifest_identity_sha256: str,
    hf_validation_ordered_prediction_ids_sha256: str,
    full_approval_receipt_sha256: str,
    validation_selection_receipt_sha256: str,
    selected_checkpoint_sha256: str,
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
    ):
        _require_sha256(value, label)
    if not scorer_schema_version.strip():
        raise ValueError("terminal scorer schema version is empty")
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
    ):
        _require_sha256(payload[key], label)
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
