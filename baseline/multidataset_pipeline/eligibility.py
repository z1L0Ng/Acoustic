"""P8 narrow compatible-label routing and masked auxiliary objective."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .contracts import ObservationState
from .joint_native import SEED


SHARED_ATTRIBUTES = ("crackle", "wheeze")
AUXILIARY_LANES = ("ICBHI", "SPRSound")
NATIVE_ONLY_LANES = ("HF", "KAUH")
ICBHI_MAPPING = {
    "normal": (0.0, 0.0),
    "crackle": (1.0, 0.0),
    "wheeze": (0.0, 1.0),
    "both": (1.0, 1.0),
}
SPR_MAPPING = {
    "Normal": (0.0, 0.0),
    "Wheeze": (0.0, 1.0),
    "Coarse Crackle": (1.0, 0.0),
    "Fine Crackle": (1.0, 0.0),
    "Wheeze+Crackle": (1.0, 1.0),
}
SPR_KNOWN_INCOMPATIBLE = {"Rhonchi", "Stridor"}


class ZeroEligibleDenominator(RuntimeError):
    """The auxiliary objective is undefined when no value is eligible."""


@dataclass(frozen=True)
class CompatibleRow:
    dataset_id: str
    native_label: str
    state: ObservationState

    def __post_init__(self) -> None:
        if self.dataset_id not in {*AUXILIARY_LANES, *NATIVE_ONLY_LANES}:
            raise ValueError(f"unknown dataset lane: {self.dataset_id}")
        if not isinstance(self.state, ObservationState):
            raise TypeError("state must be an explicit ObservationState")


@dataclass(frozen=True)
class EligibilityTargets:
    targets: torch.Tensor
    eligible_mask: torch.Tensor
    receipt: dict[str, object]


def _map_observed(row: CompatibleRow) -> tuple[tuple[float, float], bool]:
    if row.dataset_id in NATIVE_ONLY_LANES:
        return (0.0, 0.0), False
    if row.dataset_id == "ICBHI":
        if row.native_label not in ICBHI_MAPPING:
            raise ValueError(f"ICBHI label is outside the narrow mapping: {row.native_label}")
        return ICBHI_MAPPING[row.native_label], True
    if row.dataset_id == "SPRSound":
        if row.native_label in SPR_MAPPING:
            return SPR_MAPPING[row.native_label], True
        if row.native_label in SPR_KNOWN_INCOMPATIBLE:
            return (0.0, 0.0), False
        raise ValueError(
            f"SPRSound label is outside the narrow mapping: {row.native_label}"
        )
    raise ValueError(f"unknown dataset lane: {row.dataset_id}")


def build_eligibility_targets(
    rows: Sequence[CompatibleRow],
) -> EligibilityTargets:
    """Build crackle/wheeze targets only for explicit compatible observations."""

    if not rows:
        raise ValueError("rows must be non-empty")
    targets = torch.zeros(len(rows), len(SHARED_ATTRIBUTES), dtype=torch.float32)
    eligible = torch.zeros_like(targets, dtype=torch.bool)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    state_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        state_counts[row.state.value] += 1
        by_dataset[row.dataset_id]["rows"] += 1
        if row.state is not ObservationState.OBSERVED:
            by_dataset[row.dataset_id]["masked_values"] += len(SHARED_ATTRIBUTES)
            continue
        mapped, is_eligible = _map_observed(row)
        if is_eligible:
            targets[index] = torch.tensor(mapped)
            eligible[index] = True
            by_dataset[row.dataset_id]["eligible_values"] += len(SHARED_ATTRIBUTES)
        else:
            by_dataset[row.dataset_id]["masked_values"] += len(SHARED_ATTRIBUTES)
    for dataset_id in {row.dataset_id for row in rows}:
        counts = by_dataset[dataset_id]
        counts["eligible_values"] += 0
        counts["masked_values"] += (
            counts["rows"] * len(SHARED_ATTRIBUTES)
            - counts["eligible_values"]
            - counts["masked_values"]
        )
    receipt = {
        "attributes": list(SHARED_ATTRIBUTES),
        "auxiliary_lanes": list(AUXILIARY_LANES),
        "native_only_lanes": list(NATIVE_ONLY_LANES),
        "state_counts": dict(sorted(state_counts.items())),
        "by_dataset": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(by_dataset.items())
        },
        "eligible_denominator": int(eligible.sum()),
    }
    return EligibilityTargets(targets, eligible, receipt)


def eligibility_masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Mean BCE over eligible values; masked values have zero denominator/gradient."""

    if logits.shape != targets.shape or logits.shape != eligible_mask.shape:
        raise ValueError("logits, targets, and eligible_mask must have equal shapes")
    if not logits.dtype.is_floating_point or targets.dtype != logits.dtype:
        raise TypeError("logits and targets must use the same floating dtype")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be bool")
    denominator = int(eligible_mask.sum())
    if denominator == 0:
        raise ZeroEligibleDenominator(
            "P8 auxiliary objective has zero eligible denominator"
        )
    per_value = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    loss = (per_value * eligible_mask.to(per_value.dtype)).sum() / denominator
    return loss, {
        "eligible_denominator": denominator,
        "masked_values": int(eligible_mask.numel() - denominator),
        "normalization": "sum_eligible_bce_divide_eligible_denominator",
    }


@dataclass(frozen=True)
class P8ObjectiveConfig:
    comparator: str = "P1"
    auxiliary_lanes: tuple[str, ...] = AUXILIARY_LANES
    native_only_lanes: tuple[str, ...] = NATIVE_ONLY_LANES
    seed: int = SEED
    matched_scope: str = (
        "encoder_projector_native_heads_sampler_scope_budget_seed_selection"
    )

    def validate(self) -> None:
        if self.comparator != "P1":
            raise ValueError("P8 comparator must be P1")
        if self.auxiliary_lanes != AUXILIARY_LANES:
            raise ValueError("P8 auxiliary lanes must be ICBHI+SPRSound only")
        if self.native_only_lanes != NATIVE_ONLY_LANES:
            raise ValueError("HF/KAUH must remain native-only")
        if self.seed != SEED:
            raise ValueError(f"seed must remain {SEED}")


@dataclass(frozen=True)
class GuardrailSchema:
    """Schema only; no metric values are inferred by engineering tests."""

    native_retention_tolerance: float | None = None
    worst_task_floor: float | None = None
    update_budget: int | None = None
    selection: str | None = None

    def require_execution_ready(self) -> None:
        if self.native_retention_tolerance is None:
            raise RuntimeError("native-retention tolerance is not frozen")
        if self.worst_task_floor is None:
            raise RuntimeError("worst-task guardrail is not frozen")
        if self.update_budget is None or self.update_budget <= 0:
            raise RuntimeError("update budget is not frozen")
        if not self.selection:
            raise RuntimeError("checkpoint selection is not frozen")
