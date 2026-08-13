"""Production per-native-task terminal metrics for the shared-window runner.

The scorer consumes already-inferred terminal batches from an explicitly loaded
provider.  The runner invokes that provider only after its approval, validation
selection, and exact-checkpoint gates have passed.  No metric is pooled across
datasets and no task name is inferred from arbitrary input.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline.four_dataset_frozen_encoder.data import KAUH_LABELS
from baseline.four_dataset_frozen_encoder.verify import _multiclass_metrics
from baseline.shared_encoder_native_heads.protocol import ICBHI_LABELS, SPR_LABELS

from .beats_temporal import CHANNEL_ORDER
from .hf_thresholds import VerifiedHFThresholdReceipt


TERMINAL_SCORER_SCHEMA_VERSION = "shared_window_terminal_scorer_v1"
TERMINAL_PROVIDER_MANIFEST_SCHEMA_VERSION = "terminal_provider_registration_v1"
TERMINAL_PROVIDER_MANIFEST_RELATIVE_PATH = Path(
    "baseline/multidataset_pipeline/terminal_provider_manifest.json"
)
NATIVE_TASKS = (
    "ICBHI_flat4",
    "SPRSound_binary",
    "SPRSound_raw7",
    "HF_temporal4",
    "KAUH_raw9",
)
MULTICLASS_LABELS = {
    "ICBHI_flat4": tuple(ICBHI_LABELS),
    "SPRSound_binary": ("normal", "adventitious"),
    "SPRSound_raw7": tuple(SPR_LABELS),
    "KAUH_raw9": tuple(KAUH_LABELS),
}
HF_NEGATIVE_SEMANTICS = "source_task_constructed_not_raw_normal"
HF_ALIGNMENT = "window_center_in_interval"
FORBIDDEN_SCORE_KEYS = {
    "pooled_score",
    "global_score",
    "cross_dataset_score",
    "ranking",
}


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _tensor_cpu(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    return value.detach().cpu()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_native(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(child) for child in value]
    return value


@dataclass(frozen=True)
class MulticlassTerminalBatch:
    task: str
    prediction_ids: tuple[str, ...]
    targets: torch.Tensor
    predicted_classes: torch.Tensor

    def validate(self) -> None:
        if self.task not in MULTICLASS_LABELS:
            raise ValueError(f"unsupported multiclass terminal task: {self.task}")
        targets = _tensor_cpu(self.targets, "targets")
        predicted = _tensor_cpu(self.predicted_classes, "predicted_classes")
        if targets.dtype != torch.long or predicted.dtype != torch.long:
            raise TypeError("multiclass targets/predictions must be int64")
        if targets.ndim != 1 or predicted.shape != targets.shape:
            raise ValueError("multiclass targets/predictions must be aligned [N]")
        if len(self.prediction_ids) != targets.numel() or not self.prediction_ids:
            raise ValueError("prediction IDs must align one-to-one with non-empty rows")
        if any(not value for value in self.prediction_ids) or len(set(self.prediction_ids)) != len(
            self.prediction_ids
        ):
            raise ValueError("prediction IDs must be non-empty and unique within a batch")
        classes = len(MULTICLASS_LABELS[self.task])
        if bool((targets < 0).any()) or bool((targets >= classes).any()):
            raise ValueError("multiclass target outside frozen label space")
        if bool((predicted < 0).any()) or bool((predicted >= classes).any()):
            raise ValueError("multiclass prediction outside frozen label space")


@dataclass(frozen=True)
class HFTemporalTerminalBatch:
    prediction_ids: tuple[str, ...]
    probabilities: torch.Tensor
    targets: torch.Tensor
    window_mask: torch.Tensor
    annotation_mask: torch.Tensor
    valid_mask: torch.Tensor
    time_map: torch.Tensor
    thresholds: torch.Tensor
    threshold_receipt_sha256: str
    negative_semantics: str = HF_NEGATIVE_SEMANTICS
    alignment: str = HF_ALIGNMENT
    shared_label_eligible: bool = False
    task: str = "HF_temporal4"

    def validate(self) -> None:
        if self.task != "HF_temporal4":
            raise ValueError("HF terminal task name changed")
        probabilities = _tensor_cpu(self.probabilities, "HF probabilities")
        targets = _tensor_cpu(self.targets, "HF targets")
        window_mask = _tensor_cpu(self.window_mask, "HF window_mask")
        annotation = _tensor_cpu(self.annotation_mask, "HF annotation_mask")
        valid = _tensor_cpu(self.valid_mask, "HF valid_mask")
        time_map = _tensor_cpu(self.time_map, "HF time_map")
        thresholds = _tensor_cpu(self.thresholds, "HF thresholds")
        if probabilities.ndim != 3 or probabilities.shape[-1] != len(CHANNEL_ORDER):
            raise ValueError("HF probabilities must be [B,Nw,4]")
        batch, windows, _ = probabilities.shape
        if not self.prediction_ids or len(self.prediction_ids) != batch:
            raise ValueError("HF prediction IDs must align with [B,Nw,4]")
        if any(not value for value in self.prediction_ids) or len(set(self.prediction_ids)) != batch:
            raise ValueError("HF recording prediction IDs must be non-empty and unique")
        if targets.shape != probabilities.shape:
            raise ValueError("HF targets must match probabilities [B,Nw,4]")
        if window_mask.shape != (batch, windows) or window_mask.dtype != torch.bool:
            raise TypeError("HF window_mask must be bool [B,Nw]")
        if annotation.shape != probabilities.shape or annotation.dtype != torch.bool:
            raise TypeError("HF annotation_mask must be bool [B,Nw,4]")
        if valid.shape != probabilities.shape or valid.dtype != torch.bool:
            raise TypeError("HF valid_mask must be bool [B,Nw,4]")
        if bool((valid & ~annotation).any()):
            raise ValueError("HF valid supervision cannot exist outside annotation policy mask")
        if bool((valid & ~window_mask.unsqueeze(-1)).any()):
            raise ValueError("HF valid supervision cannot exist on a padded window")
        if time_map.shape != (batch, windows, 2) or time_map.dtype != torch.float64:
            raise TypeError("HF time_map must be float64 [B,Nw,2]")
        if not probabilities.dtype.is_floating_point or not bool(torch.isfinite(probabilities).all()):
            raise TypeError("HF probabilities must be finite floating point")
        if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
            raise ValueError("HF probabilities must be in [0,1]")
        if not targets.dtype.is_floating_point or not bool(
            ((targets == 0) | (targets == 1)).all()
        ):
            raise TypeError("HF targets must be binary floating point")
        if thresholds.shape != (4,) or not thresholds.dtype.is_floating_point:
            raise TypeError("HF validation-frozen thresholds must be floating [4]")
        if not bool(torch.isfinite(thresholds).all()) or bool(
            ((thresholds <= 0) | (thresholds >= 1)).any()
        ):
            raise ValueError("HF thresholds must be finite and strictly inside (0,1)")
        _require_sha256(self.threshold_receipt_sha256, "HF threshold receipt")
        if (
            self.negative_semantics != HF_NEGATIVE_SEMANTICS
            or self.alignment != HF_ALIGNMENT
            or self.shared_label_eligible is not False
        ):
            raise RuntimeError("HF source-task semantics/alignment/shared-label boundary changed")
        for row in range(batch):
            count = int(window_mask[row].sum())
            if count <= 0 or not bool(window_mask[row, :count].all()) or bool(
                window_mask[row, count:].any()
            ):
                raise ValueError("HF valid windows must be a non-empty contiguous prefix")
            starts = time_map[row, :count, 0]
            ends = time_map[row, :count, 1]
            if bool((ends <= starts).any()) or bool((starts[1:] <= starts[:-1]).any()):
                raise ValueError("HF source-time windows must be positive and strictly ordered")
            if bool(torch.count_nonzero(time_map[row, count:])):
                raise ValueError("HF padded time_map rows must be exact zero")


TerminalBatch = MulticlassTerminalBatch | HFTemporalTerminalBatch


@dataclass(frozen=True)
class TerminalScoringInput:
    batches: tuple[TerminalBatch, ...]
    expected_prediction_ids_by_task: Mapping[str, tuple[str, ...]]
    data_identity_sha256: str
    provider_identity_sha256: str
    outer_test_accessed: bool = True
    terminal_targets_loaded: bool = True

    def validate(self) -> None:
        _require_sha256(self.data_identity_sha256, "terminal data identity")
        _require_sha256(self.provider_identity_sha256, "terminal provider identity")
        if self.outer_test_accessed is not True or self.terminal_targets_loaded is not True:
            raise RuntimeError("terminal scorer input must explicitly identify terminal target access")
        if set(self.expected_prediction_ids_by_task) != set(NATIVE_TASKS):
            raise RuntimeError("expected terminal prediction IDs must cover exactly native tasks")
        for task, identifiers in self.expected_prediction_ids_by_task.items():
            if not isinstance(identifiers, tuple) or not identifiers:
                raise ValueError(f"{task} expected prediction IDs must be a non-empty tuple")
            if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{task} expected prediction IDs contain missing/duplicates")


TerminalInputProvider = Callable[..., TerminalScoringInput]


def terminal_provider_identity_sha256(
    provider_specification: str, implementation_sha256: str
) -> str:
    """Derive provider identity from implementation bytes and frozen scorer contract."""

    implementation_sha256 = _require_sha256(
        implementation_sha256, "terminal provider implementation"
    )
    if provider_specification.count(":") != 1:
        raise ValueError("terminal provider must use module:function")
    return hashlib.sha256(
        json.dumps(
            {
                "provider_specification": provider_specification,
                "implementation_sha256": implementation_sha256,
                "scorer_schema_version": TERMINAL_SCORER_SCHEMA_VERSION,
                "native_tasks": list(NATIVE_TASKS),
                "outer_test_access_policy": "terminal_score_only_after_exact_selection_checkpoint_and_approval",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def audit_terminal_provider_registration(repo_root: Path) -> dict[str, object]:
    """Validate a provider registration without importing it or reading outer/test."""

    root = repo_root.resolve()
    manifest_path = root / TERMINAL_PROVIDER_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        return {
            "schema_version": TERMINAL_PROVIDER_MANIFEST_SCHEMA_VERSION,
            "status": "HOLD_no_registered_production_terminal_provider",
            "terminal_score_ready": False,
            "required_manifest_path": str(manifest_path),
            "required_scorer_schema_version": TERMINAL_SCORER_SCHEMA_VERSION,
            "required_native_tasks": list(NATIVE_TASKS),
            "provider_imported": False,
            "outer_test_accessed": False,
        }
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    required = {
        "schema_version",
        "status",
        "provider_specification",
        "provider_identity_sha256",
        "implementation_path",
        "implementation_sha256",
        "scorer_schema_version",
        "native_tasks",
        "outer_test_access_policy",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise RuntimeError("terminal provider registration schema changed")
    provider_identity = _require_sha256(
        manifest["provider_identity_sha256"], "registered terminal provider"
    )
    implementation_sha = _require_sha256(
        manifest["implementation_sha256"], "terminal provider implementation"
    )
    specification = manifest["provider_specification"]
    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError("registered terminal provider must use module:function")
    relative = Path(str(manifest["implementation_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("terminal provider implementation path must be repository-relative")
    implementation = (root / relative).resolve()
    try:
        implementation.relative_to(root)
    except ValueError as error:
        raise ValueError("terminal provider implementation escaped repository root") from error
    if not implementation.is_file() or _sha256_path(implementation) != implementation_sha:
        raise RuntimeError("registered terminal provider implementation identity failed")
    if provider_identity != terminal_provider_identity_sha256(
        specification, implementation_sha
    ):
        raise RuntimeError("terminal provider identity is not derived from implementation")
    if (
        manifest["schema_version"] != TERMINAL_PROVIDER_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "registered"
        or manifest["scorer_schema_version"] != TERMINAL_SCORER_SCHEMA_VERSION
        or manifest["native_tasks"] != list(NATIVE_TASKS)
        or manifest["outer_test_access_policy"]
        != "terminal_score_only_after_exact_selection_checkpoint_and_approval"
    ):
        raise RuntimeError("terminal provider registration contract failed")
    return {
        **dict(manifest),
        "provider_identity_sha256": provider_identity,
        "implementation_sha256": implementation_sha,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_score_ready": True,
        "provider_imported": False,
        "outer_test_accessed": False,
    }


def _score_multiclass(
    task: str,
    batches: Sequence[MulticlassTerminalBatch],
    expected_ids: tuple[str, ...],
) -> dict[str, object]:
    actual_ids: list[str] = []
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for batch in batches:
        batch.validate()
        actual_ids.extend(batch.prediction_ids)
        targets.append(batch.targets.detach().cpu())
        predictions.append(batch.predicted_classes.detach().cpu())
    if tuple(actual_ids) != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError(f"{task} prediction IDs are missing, duplicated, or out of order")
    target = torch.cat(targets).numpy()
    predicted = torch.cat(predictions).numpy()
    if target.size == 0:
        raise RuntimeError(f"{task} has empty terminal support")
    helper_task = {
        "ICBHI_flat4": "icbhi_flat4",
        "SPRSound_binary": "spr_binary",
        "SPRSound_raw7": "spr_seven",
        "KAUH_raw9": "kauh_raw9",
    }[task]
    metrics = _json_native(_multiclass_metrics(
        target,
        predicted,
        list(MULTICLASS_LABELS[task]),
        helper_task,
    ))
    return {
        "task": task,
        "prediction_unit_count": len(actual_ids),
        "denominator": int(target.size),
        "labels": list(MULTICLASS_LABELS[task]),
        "metrics": metrics,
    }


def _score_hf(
    batches: Sequence[HFTemporalTerminalBatch],
    expected_ids: tuple[str, ...],
    verified_threshold_receipt: VerifiedHFThresholdReceipt,
) -> dict[str, object]:
    actual_ids: list[str] = []
    values = [[] for _ in CHANNEL_ORDER]
    targets = [[] for _ in CHANNEL_ORDER]
    threshold = torch.tensor(
        verified_threshold_receipt.thresholds, dtype=torch.float64
    )
    valid_window_count = 0
    for batch in batches:
        batch.validate()
        actual_ids.extend(batch.prediction_ids)
        probabilities = batch.probabilities.detach().cpu()
        labels = batch.targets.detach().cpu()
        effective = (
            batch.window_mask.detach().cpu().unsqueeze(-1)
            & batch.annotation_mask.detach().cpu()
            & batch.valid_mask.detach().cpu()
        )
        valid_window_count += int(batch.window_mask.sum())
        current_threshold = batch.thresholds.detach().cpu().to(torch.float64)
        if (
            not torch.equal(threshold, current_threshold)
            or batch.threshold_receipt_sha256
            != verified_threshold_receipt.artifact_sha256
        ):
            raise RuntimeError(
                "HF batch threshold reference differs from verified threshold artifact"
            )
        for index in range(4):
            mask = effective[..., index]
            values[index].append(probabilities[..., index][mask])
            targets[index].append(labels[..., index][mask])
    if tuple(actual_ids) != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("HF prediction IDs are missing, duplicated, or out of order")
    if not batches:
        raise RuntimeError("HF terminal task has no batches")
    per_channel: dict[str, object] = {}
    total_denominator = 0
    for index, channel in enumerate(CHANNEL_ORDER):
        probability = torch.cat(values[index]).numpy()
        target = torch.cat(targets[index]).numpy().astype(np.int64)
        if target.size == 0 or len(np.unique(target)) != 2:
            raise RuntimeError(f"HF {channel} requires non-empty positive and negative support")
        predicted = (probability >= float(threshold[index])).astype(np.int64)
        tn = int(((target == 0) & (predicted == 0)).sum())
        fp = int(((target == 0) & (predicted == 1)).sum())
        fn = int(((target == 1) & (predicted == 0)).sum())
        tp = int(((target == 1) & (predicted == 1)).sum())
        denominator = int(target.size)
        total_denominator += denominator
        per_channel[channel] = {
            "denominator": denominator,
            "positive_support": int(target.sum()),
            "constructed_negative_support": int((target == 0).sum()),
            "threshold": float(threshold[index]),
            "confusion": [[tn, fp], [fn, tp]],
            "accuracy": float((predicted == target).mean()),
            "roc_auc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        }
        if not all(
            value is None or isinstance(value, (str, int, list)) or math.isfinite(float(value))
            for value in per_channel[channel].values()
            if not isinstance(value, list)
        ):
            raise RuntimeError(f"HF {channel} produced non-finite metrics")
    return {
        "task": "HF_temporal4",
        "prediction_unit_count": len(actual_ids),
        "valid_source_windows": valid_window_count,
        "channel_denominator_total": total_denominator,
        "channel_order": list(CHANNEL_ORDER),
        "alignment": HF_ALIGNMENT,
        "negative_semantics": HF_NEGATIVE_SEMANTICS,
        "raw_gap_missing_unknown_not_raw_negative": True,
        "shared_label_eligible": False,
        "threshold_receipt_sha256": verified_threshold_receipt.artifact_sha256,
        "threshold_receipt_path": str(verified_threshold_receipt.path),
        "threshold_selection_policy": verified_threshold_receipt.payload[
            "threshold_selection_policy"
        ],
        "thresholds_selected_on_outer_test": False,
        "per_channel": per_channel,
    }


def _reject_forbidden_keys(value: object, path: str = "result") -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_SCORE_KEYS & set(value)
        if forbidden:
            raise RuntimeError(f"forbidden pooled/ranking keys at {path}: {sorted(forbidden)}")
        for key, child in value.items():
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


class ProductionTerminalScorer:
    """Callable attached to ``terminal_score_gate`` after all immutable gates pass."""

    def __init__(
        self,
        provider: TerminalInputProvider,
        *,
        expected_provider_identity_sha256: str,
        provider_specification: str,
    ) -> None:
        if not callable(provider):
            raise TypeError("terminal input provider must be callable")
        self.provider = provider
        self.expected_provider_identity_sha256 = _require_sha256(
            expected_provider_identity_sha256, "expected terminal provider identity"
        )
        if provider_specification.count(":") != 1:
            raise ValueError("production terminal provider must use module:function")
        self.provider_specification = provider_specification

    def __call__(
        self,
        selected_checkpoint: Path,
        *,
        verified_hf_threshold_receipt: VerifiedHFThresholdReceipt | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(
            verified_hf_threshold_receipt, VerifiedHFThresholdReceipt
        ):
            raise RuntimeError(
                "terminal scorer requires a gate-verified HF threshold receipt"
            )
        signature = inspect.signature(self.provider)
        if "verified_hf_threshold_receipt" in signature.parameters:
            inputs = self.provider(
                selected_checkpoint,
                verified_hf_threshold_receipt=verified_hf_threshold_receipt,
            )
        else:
            # Retain the narrow one-argument seam only for in-memory verifier
            # fixtures.  The registered production provider must accept the
            # gate-verified receipt and cannot discover thresholds elsewhere.
            inputs = self.provider(selected_checkpoint)
        if not isinstance(inputs, TerminalScoringInput):
            raise TypeError("terminal input provider returned the wrong contract type")
        inputs.validate()
        if inputs.provider_identity_sha256 != self.expected_provider_identity_sha256:
            raise RuntimeError("terminal provider identity differs from approved identity")
        grouped: dict[str, list[TerminalBatch]] = {task: [] for task in NATIVE_TASKS}
        for batch in inputs.batches:
            task = batch.task
            if task not in grouped:
                raise RuntimeError(f"terminal provider emitted an extra task: {task}")
            grouped[task].append(batch)
        missing = [task for task, batches in grouped.items() if not batches]
        if missing:
            raise RuntimeError(f"terminal provider omitted native tasks: {missing}")
        tasks: dict[str, object] = {}
        for task in NATIVE_TASKS:
            expected = inputs.expected_prediction_ids_by_task[task]
            if task == "HF_temporal4":
                tasks[task] = _score_hf(
                    grouped[task], expected, verified_hf_threshold_receipt
                )
            else:
                tasks[task] = _score_multiclass(task, grouped[task], expected)
        receipt = {
            "schema_version": TERMINAL_SCORER_SCHEMA_VERSION,
            "status": "terminal_native_tasks_scored",
            "data_identity_sha256": inputs.data_identity_sha256,
            "provider_identity_sha256": inputs.provider_identity_sha256,
            "selected_checkpoint_path": str(selected_checkpoint.resolve()),
            "selected_checkpoint_sha256": _sha256_path(selected_checkpoint),
            "outer_test_accessed": True,
            "terminal_targets_loaded": True,
            "native_task_names": list(NATIVE_TASKS),
            "native_tasks": tasks,
            "cross_dataset_pooling": False,
        }
        _reject_forbidden_keys(receipt)
        return receipt


def load_terminal_input_provider(specification: str) -> TerminalInputProvider:
    """Return a lazy ``module:function`` provider imported only after terminal gates."""

    if ":" not in specification:
        raise ValueError("terminal provider must use module:function syntax")
    module_name, function_name = specification.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("terminal provider module and function must be non-empty")

    def deferred(
        selected_checkpoint: Path,
        *,
        verified_hf_threshold_receipt: VerifiedHFThresholdReceipt,
    ) -> TerminalScoringInput:
        provider = getattr(importlib.import_module(module_name), function_name)
        if not callable(provider):
            raise TypeError("terminal provider target is not callable")
        return provider(
            selected_checkpoint,
            verified_hf_threshold_receipt=verified_hf_threshold_receipt,
        )

    return deferred
