"""Preflight freeze for the four-dataset shared-window encoder queue.

This is an engineering/design contract only.  It never loads outer/test labels,
trains a model, or upgrades a candidate to a scientific result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .beats_temporal import CHANNEL_ORDER
from .contracts import PREDICTION_UNITS
from .joint_native import JOINT_LANES, SEED
from .sliding_window import (
    WINDOW_LENGTH_S,
    WINDOW_POLICY_STATUS,
    WINDOW_STRIDE_S,
    SlidingWindowBatch,
)


PIPELINE_ENCODERS = {
    "P1": "AST",
    "P2": "BEATs",
    "P3": "PANNs_Cnn14",
    "P4": "HeAR",
    "P5": "OPERA_CT",
}
P1_P5_BATCH_SIZE = 8
P1_P5_REFERENCE_EPOCHS = 50
FOUR_DATASET_SUBTRAIN_UNITS = {
    "ICBHI": 3_055,
    "SPRSound": 5_219,
    "HF": 5_322,
    "KAUH": 198,
}
P1_P5_UPDATES_PER_REFERENCE_EPOCH = math.ceil(
    sum(FOUR_DATASET_SUBTRAIN_UNITS.values()) / P1_P5_BATCH_SIZE
)
P1_P5_UPDATE_BUDGET = (
    P1_P5_REFERENCE_EPOCHS * P1_P5_UPDATES_PER_REFERENCE_EPOCH
)
P1_P5_VALIDATION_INTERVAL_UPDATES = P1_P5_UPDATES_PER_REFERENCE_EPOCH
P1_P5_SELECTION_RULE = (
    "minimum_source_proportional_validation_native_loss;"
    "SPRSound=mean(binary_ce,raw7_ce);"
    "HF=equal_channel_masked_window_bce;"
    "tie=earliest_update;outer_test_excluded"
)

HF_TARGET_POLICY = "PAPER_NATIVE_RASTERIZED_OVR"
HF_ALIGNMENT = "window_center_in_interval"
HF_NEGATIVE_SEMANTICS = "source_task_constructed_not_raw_normal"
HF_SHARED_LABEL_ELIGIBLE = False
HF_NATIVE_METRICS = {
    "reporting_unit": "2-second source-time window center",
    "per_channel": [
        "accuracy",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "positive_predictive_value",
        "f1",
    ],
    "threshold_selection": (
        "validation_only_per_channel_max_f1;tie=highest_threshold;"
        "threshold_frozen_before_outer_test"
    ),
    "aggregation": "report_each_I_E_CAS_DAS_channel;optional_equal_channel_macro_within_HF_only",
    "forbidden": "no_cross_dataset_pooled_score",
    "event_metrics_hold": (
        "source-paper event Jaccard/F1/MAPE require a separately verified "
        "postprocessing and event-matching contract"
    ),
}

WINDOW_POLICY = {
    "status": WINDOW_POLICY_STATUS,
    "window_length_s": WINDOW_LENGTH_S,
    "window_stride_s": WINDOW_STRIDE_S,
    "sample_rate": 16_000,
    "tail": "append_unique_end_aligned_window_when_stride_misses_tail",
    "short": "zero_pad_only_with_valid_sample_mask",
    "repeat_pad": False,
    "truncate": False,
    "selection_evidence": "encoder input compatibility only; outer/test not consulted",
    "caveat": (
        "HeAR natively accepts 2 s; AST and OPERA require internal zero-padding "
        "to their fixed frontend grids, so comparisons are package-level"
    ),
}

CANDIDATE_NATIVE_DIMS = {
    "AST": 768,
    "BEATs": 768,
    "PANNs_Cnn14": 2_048,
    "HeAR": 512,
    "OPERA_CT": 768,
}
CANDIDATE_DIMENSION_ADAPTER = {
    "AST": "identity_768",
    "BEATs": "identity_768",
    "PANNs_Cnn14": "trainable_layernorm_2048_plus_bias_free_linear_2048_to_768",
    "HeAR": "trainable_layernorm_512_plus_bias_free_linear_512_to_768",
    "OPERA_CT": "identity_768",
}
CANDIDATE_COMPARISON_CAVEAT = {
    "P1": "AST frontend internally zero-pads each 2 s window to the audited 8 s grid",
    "P2": "BEATs accepts 2 s waveform windows; exact mask/checkpoint binding remains gated",
    "P3": "PANNs emits 2048-d; comparison includes a trainable shared-across-datasets 2048->768 adapter",
    "P4": "HeAR emits 512-d; comparison includes a trainable shared-across-datasets 512->768 adapter",
    "P5": "OPERA uses an 8 s frontend grid and has ICBHI/HF pretraining overlap; provenance gate dominates",
}

PRODUCTION_ADAPTER_STATUS = {
    "AST": {
        "status": "HOLD_missing_shared_window_adapter",
        "required_file": "baseline/multidataset_pipeline/ast_window_encoder.py",
        "legacy_reference": "baseline/shared_encoder_native_heads/protocol.py::preprocess_rows/build_model",
        "gate": "replace repeat-pad/truncate with 2 s source window plus internal zero-pad only",
    },
    "BEATs": {
        "status": "HOLD_missing_checkpoint_loader_binding",
        "required_file": "baseline/multidataset_pipeline/beats_window_encoder.py",
        "partial_ready": "baseline/multidataset_pipeline/beats_temporal.py::BEATsTemporalAdapter",
        "gate": "bind pinned AudioSet checkpoint/source and flatten B*K windows without lineage loss",
    },
    "PANNs_Cnn14": {
        "status": "HOLD_no_production_adapter",
        "required_file": "baseline/multidataset_pipeline/panns_window_encoder.py",
        "gate": "official checkpoint/frontend plus declared trainable 2048->768 dimension adapter",
    },
    "HeAR": {
        "status": "HOLD_no_production_adapter",
        "required_file": "baseline/multidataset_pipeline/hear_window_encoder.py",
        "gate": "official 2 s frontend plus declared trainable 512->768 dimension adapter",
    },
    "OPERA_CT": {
        "status": "HOLD_provenance_overlap_and_adapter",
        "required_file": "baseline/multidataset_pipeline/opera_window_encoder.py",
        "gate": (
            "checkpoint/revision/SHA, 2 s-to-8 s zero-pad adapter, and explicit "
            "ICBHI/HF pretraining-overlap receipt; no clean-generalization claim"
        ),
    },
}


@dataclass(frozen=True)
class SharedWindowEncoderOutput:
    """Candidate adapter output immediately before the shared 768->256 projector."""

    embeddings: torch.Tensor
    window_mask: torch.Tensor
    time_map: torch.Tensor
    encoder_identity: str
    sample_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    prediction_units: tuple[str, ...]

    def validate_against(self, batch: SlidingWindowBatch) -> None:
        batch.validate()
        if self.encoder_identity not in CANDIDATE_NATIVE_DIMS:
            raise ValueError("unknown candidate encoder identity")
        if self.embeddings.shape != (*batch.window_mask.shape, 768):
            raise ValueError("shared-window embeddings must be [B,K,768]")
        if not self.embeddings.dtype.is_floating_point or not torch.isfinite(
            self.embeddings
        ).all():
            raise TypeError("shared-window embeddings must be finite floating point")
        if self.window_mask.dtype != torch.bool or not torch.equal(
            self.window_mask, batch.window_mask
        ):
            raise RuntimeError("encoder window mask changed")
        if self.time_map.dtype != torch.float64 or not torch.equal(
            self.time_map, batch.time_map
        ):
            raise RuntimeError("encoder source-time map changed")
        if any(
            tensor.device != batch.device
            for tensor in (self.embeddings, self.window_mask, self.time_map)
        ):
            raise RuntimeError("encoder output and window batch must share one device")
        if bool(torch.count_nonzero(self.embeddings[~self.window_mask])):
            raise ValueError("invalid padded window slots must have zero embeddings")
        if (
            self.sample_ids != batch.sample_ids
            or self.dataset_ids != batch.dataset_ids
            or self.prediction_units != batch.prediction_units
        ):
            raise RuntimeError("native unit or dataset lineage changed")
        if any(dataset not in JOINT_LANES for dataset in self.dataset_ids):
            raise ValueError("all and only four frozen dataset lanes are supported")


class CandidateDimensionAdapter(nn.Module):
    """Candidate-specific D->768 package adapter shared across all four datasets."""

    def __init__(self, encoder_identity: str) -> None:
        super().__init__()
        if encoder_identity not in CANDIDATE_NATIVE_DIMS:
            raise ValueError("unknown candidate encoder identity")
        self.encoder_identity = encoder_identity
        input_dim = CANDIDATE_NATIVE_DIMS[encoder_identity]
        self.input_dim = input_dim
        self.adapter = (
            nn.Identity()
            if input_dim == 768
            else nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 768, bias=False),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(f"candidate window values must be [B,K,{self.input_dim}]")
        return self.adapter(values)

    def receipt(self) -> dict[str, object]:
        return {
            "encoder_identity": self.encoder_identity,
            "input_dim": self.input_dim,
            "output_dim": 768,
            "architecture": CANDIDATE_DIMENSION_ADAPTER[self.encoder_identity],
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "shared_across_datasets": True,
            "package_level_comparison": self.input_dim != 768,
        }


class P6TokenTemporalHead(nn.Module):
    """Deferred head after shared 768->256 projection of BEATs tokens."""

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        if input_dim != 256:
            raise ValueError("P6 token head input_dim is frozen at projected 256")
        self.classifier = nn.Linear(input_dim, len(CHANNEL_ORDER), bias=True)

    def forward(self, projected_tokens: torch.Tensor) -> torch.Tensor:
        if projected_tokens.ndim != 3 or projected_tokens.shape[-1] != 256:
            raise ValueError("P6 projected temporal tokens must be [B,L,256]")
        return self.classifier(projected_tokens)

    def receipt(self) -> dict[str, object]:
        return {
            "architecture": "shared_projector_output_plus_biased_linear_256_to_4",
            "channel_order": list(CHANNEL_ORDER),
            "role": "deferred_BEATs_token_level_refinement",
        }


def hf_masked_channel_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Equal-channel masked BCE for either common windows or deferred P6 tokens."""

    if logits.ndim != 3 or logits.shape[-1] != len(CHANNEL_ORDER):
        raise ValueError("HF logits must be [B,K,4] or [B,L,4]")
    if targets.shape != logits.shape or not targets.dtype.is_floating_point:
        raise TypeError("HF targets must be floating and match logits")
    if (
        observation_mask.shape != logits.shape
        or valid_mask.shape != logits.shape
        or observation_mask.dtype != torch.bool
        or valid_mask.dtype != torch.bool
    ):
        raise TypeError("HF observation/valid masks must be bool and match logits")
    if any(value.device != logits.device for value in (targets, observation_mask, valid_mask)):
        raise RuntimeError("HF logits, targets, and masks must share one device")
    if bool((valid_mask & ~observation_mask).any()):
        raise ValueError("valid supervision must also be observed")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("HF logits and targets must be finite")
    effective = observation_mask & valid_mask
    denominators = effective.sum(dim=(0, 1))
    if bool((denominators == 0).any()):
        raise RuntimeError("each HF channel needs a nonzero supervised denominator")
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    channel_losses = (elementwise * effective).sum(dim=(0, 1)) / denominators.to(
        logits.dtype
    )
    return channel_losses.mean(), {
        "loss": "equal_channel_mean_of_masked_BCEWithLogits",
        "channel_order": list(CHANNEL_ORDER),
        "denominators": {
            channel: int(denominators[index])
            for index, channel in enumerate(CHANNEL_ORDER)
        },
        "target_policy": HF_TARGET_POLICY,
        "alignment": HF_ALIGNMENT,
        "negative_semantics": HF_NEGATIVE_SEMANTICS,
        "shared_label_eligible": HF_SHARED_LABEL_ELIGIBLE,
    }


@dataclass(frozen=True)
class SharedWindowCoreConfig:
    pipeline_id: str
    encoder_identity: str
    split_digest: str
    update_budget: int = P1_P5_UPDATE_BUDGET
    selection: str = P1_P5_SELECTION_RULE
    seed: int = SEED
    batch_size: int = P1_P5_BATCH_SIZE
    window_length_s: float = WINDOW_LENGTH_S
    window_stride_s: float = WINDOW_STRIDE_S
    projector: str = "shared_biased_linear_768_to_256"
    sampler: str = "four_dataset_source_proportional"
    native_units: tuple[tuple[str, str], ...] = tuple(PREDICTION_UNITS.items())

    def validate(self) -> None:
        if self.pipeline_id not in PIPELINE_ENCODERS:
            raise ValueError("shared-window config pipeline_id must be P1-P5")
        if self.encoder_identity != PIPELINE_ENCODERS[self.pipeline_id]:
            raise ValueError("pipeline/encoder identity mismatch")
        if not self.split_digest:
            raise ValueError("frozen split digest is required")
        if self.seed != SEED or self.batch_size != P1_P5_BATCH_SIZE:
            raise ValueError("seed/batch size changed")
        if (
            self.update_budget != P1_P5_UPDATE_BUDGET
            or self.selection != P1_P5_SELECTION_RULE
        ):
            raise ValueError("update budget/selection changed")
        if (
            self.window_length_s != WINDOW_LENGTH_S
            or self.window_stride_s != WINDOW_STRIDE_S
        ):
            raise ValueError("shared source-time window policy changed")


def assert_p1_p5_matched(configs: Sequence[SharedWindowCoreConfig]) -> None:
    if [config.pipeline_id for config in configs] != list(PIPELINE_ENCODERS):
        raise ValueError("expected ordered P1-P5 configs")
    normalized = []
    for config in configs:
        config.validate()
        values = asdict(config)
        values.pop("pipeline_id")
        values.pop("encoder_identity")
        normalized.append(values)
    if any(values != normalized[0] for values in normalized[1:]):
        raise RuntimeError("P1-P5 matched shared-window config violation")


def source_proportional_validation_selection_loss(
    native_losses: Mapping[str, float],
) -> tuple[float, dict[str, object]]:
    required = {
        "ICBHI_flat4",
        "SPRSound_binary",
        "SPRSound_raw7",
        "HF_temporal4",
        "KAUH_raw9",
    }
    if set(native_losses) != required:
        raise ValueError(f"validation losses must have exactly {sorted(required)}")
    values = {key: float(value) for key, value in native_losses.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("validation native losses must be finite and nonnegative")
    lane_losses = {
        "ICBHI": values["ICBHI_flat4"],
        "SPRSound": (values["SPRSound_binary"] + values["SPRSound_raw7"]) / 2,
        "HF": values["HF_temporal4"],
        "KAUH": values["KAUH_raw9"],
    }
    total = sum(FOUR_DATASET_SUBTRAIN_UNITS.values())
    selection_loss = sum(
        lane_losses[lane] * FOUR_DATASET_SUBTRAIN_UNITS[lane]
        for lane in JOINT_LANES
    ) / total
    return selection_loss, {
        "purpose": "checkpoint_selection_only_not_reported_performance",
        "lane_losses": lane_losses,
        "source_weights": {
            lane: FOUR_DATASET_SUBTRAIN_UNITS[lane] / total for lane in JOINT_LANES
        },
        "outer_test_excluded": True,
        "reported_as_cross_dataset_score": False,
    }


def select_validation_checkpoint(
    candidates: Sequence[tuple[int, float]],
) -> tuple[int, float]:
    if not candidates:
        raise ValueError("at least one validation checkpoint is required")
    normalized = []
    for update, loss in candidates:
        if (
            update <= 0
            or update > P1_P5_UPDATE_BUDGET
            or update % P1_P5_VALIDATION_INTERVAL_UPDATES
        ):
            raise ValueError("candidate update violates the frozen validation cadence")
        if not math.isfinite(loss):
            raise ValueError("validation selection loss must be finite")
        normalized.append((int(update), float(loss)))
    return min(normalized, key=lambda item: (item[1], item[0]))


INDEPENDENT_VERIFIER_REQUIRED_FIELDS = (
    "schema_version",
    "pipeline_id",
    "verifier_identity",
    "verifier_code_commit",
    "subject_code_commit",
    "config_sha256",
    "approval_receipt_sha256",
    "manifest_sha256_by_dataset",
    "split_sha256_by_dataset",
    "checkpoint_sha256_by_component",
    "window_contract_receipt",
    "encoder_adapter_receipt",
    "seed",
    "update_budget",
    "selection_receipt",
    "trainable_scope_receipt",
    "native_metrics_by_dataset_task",
    "outer_test_access_receipt",
    "artifact_sha256",
    "gate_results",
    "warnings",
    "status",
)


def validate_independent_verifier_receipt(receipt: Mapping[str, object]) -> None:
    missing = [key for key in INDEPENDENT_VERIFIER_REQUIRED_FIELDS if key not in receipt]
    if missing:
        raise ValueError(f"independent verifier receipt missing fields: {missing}")
    if receipt["pipeline_id"] not in {"G0", *PIPELINE_ENCODERS}:
        raise ValueError("first-round verifier pipeline_id must be G0 or P1-P5")
    if receipt["seed"] != SEED:
        raise ValueError(f"verifier seed must remain {SEED}")
    if receipt["verifier_identity"] == "archived_baseline_verifier":
        raise ValueError("the archived Baseline verifier cannot verify this queue")
    metrics = receipt["native_metrics_by_dataset_task"]
    if not isinstance(metrics, Mapping):
        raise TypeError("native_metrics_by_dataset_task must be a mapping")
    forbidden = {"pooled_score", "cross_dataset_score", "global_score"}
    if forbidden & set(metrics):
        raise ValueError("cross-dataset pooled Score is forbidden")


def freeze_receipt() -> dict[str, object]:
    return {
        "status": "verified_engineering_shared_window_preflight_contract",
        "experiment_result": False,
        "seed": SEED,
        "window_policy": WINDOW_POLICY,
        "p1_p5": {
            "pipeline_encoders": PIPELINE_ENCODERS,
            "batch_size": P1_P5_BATCH_SIZE,
            "subtrain_units": FOUR_DATASET_SUBTRAIN_UNITS,
            "total_subtrain_units": sum(FOUR_DATASET_SUBTRAIN_UNITS.values()),
            "reference_epochs": P1_P5_REFERENCE_EPOCHS,
            "updates_per_reference_epoch": P1_P5_UPDATES_PER_REFERENCE_EPOCH,
            "update_budget": P1_P5_UPDATE_BUDGET,
            "validation_interval_updates": P1_P5_VALIDATION_INTERVAL_UPDATES,
            "selection": P1_P5_SELECTION_RULE,
            "no_early_stopping": True,
        },
        "architecture": {
            "encoder": "one frozen candidate encoder shared by all four datasets",
            "window_embeddings": ["B", "K", 768],
            "shared_projector": "one trainable biased Linear 768->256 shared by all four datasets",
            "non_hf": "masked window mean then dataset-native heads",
            "hf": "projected window sequence then native temporal4 head",
            "unified_label_task": False,
        },
        "candidate_native_dims": CANDIDATE_NATIVE_DIMS,
        "candidate_dimension_adapter": CANDIDATE_DIMENSION_ADAPTER,
        "candidate_caveats": CANDIDATE_COMPARISON_CAVEAT,
        "production_adapter_status": PRODUCTION_ADAPTER_STATUS,
        "hf": {
            "loss": "equal_channel_mean_of_masked_BCEWithLogits",
            "target_policy": HF_TARGET_POLICY,
            "alignment": HF_ALIGNMENT,
            "negative_semantics": HF_NEGATIVE_SEMANTICS,
            "shared_label_eligible": HF_SHARED_LABEL_ELIGIBLE,
            "metrics": HF_NATIVE_METRICS,
        },
        "p6": {
            "role": "deferred_BEATs_token_level_temporal_refinement",
            "first_round_required": False,
        },
        "queue": {
            "order": ["G0", "P1_and_P2", "P3_and_P4", "P5_gate"],
            "l40_0": ["P1_AST", "P3_PANNs_Cnn14"],
            "l40_1": ["P2_BEATs", "P4_HeAR"],
            "p5": "after provenance/overlap gate; assign one free L40",
            "full_training_authorized": False,
            "outer_test_authorized": False,
        },
    }
