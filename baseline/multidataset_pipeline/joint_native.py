"""P1/P2 shared-projector, native-head, and matched-comparator contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


SEED = 20260728
JOINT_LANES = ("ICBHI", "SPRSound", "KAUH")
PROJECTOR_ARCHITECTURE = "minimal_linear_projector"
PROJECTOR_BIAS = True
NATIVE_HEAD_DIMS = {
    "ICBHI": {"flat4": 4},
    "SPRSound": {"binary": 2, "raw7": 7},
    "KAUH": {"raw9": 9},
}


class JointNativeProjector(nn.Module):
    """One minimal biased Linear 768→256 shared by the non-HF native lanes."""

    def __init__(self, input_dim: int = 768, projected_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.projected_dim = projected_dim
        self.projector_architecture = PROJECTOR_ARCHITECTURE
        self.projector_bias = PROJECTOR_BIAS
        self.projector = nn.Linear(
            input_dim, projected_dim, bias=self.projector_bias
        )
        self.heads = nn.ModuleDict(
            {
                f"{lane}:{task}": nn.Linear(projected_dim, classes)
                for lane, tasks in NATIVE_HEAD_DIMS.items()
                for task, classes in tasks.items()
            }
        )

    def architecture_receipt(self) -> dict[str, object]:
        return {
            "architecture": self.projector_architecture,
            "input_dim": self.input_dim,
            "output_dim": self.projected_dim,
            "bias": self.projector_bias,
            "shared_lanes": list(JOINT_LANES),
            "hf_uses_projector": False,
        }

    def project(self, frozen_encoder_output: torch.Tensor) -> torch.Tensor:
        if (
            frozen_encoder_output.ndim != 2
            or frozen_encoder_output.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"expected frozen encoder output [B,{self.input_dim}], "
                f"got {tuple(frozen_encoder_output.shape)}"
            )
        return self.projector(frozen_encoder_output)

    def forward(
        self, frozen_encoder_output: torch.Tensor, lane: str
    ) -> dict[str, torch.Tensor]:
        if lane not in JOINT_LANES:
            raise ValueError(
                f"lane {lane!r} is not routed through the shared projector; "
                "HF uses its fixed temporal reference"
            )
        projected = self.project(frozen_encoder_output)
        return {
            task: self.heads[f"{lane}:{task}"](projected)
            for task in NATIVE_HEAD_DIMS[lane]
        }


def assert_frozen_encoder(encoder: nn.Module) -> None:
    """Fail if a P1/P2 pretrained encoder exposes trainable parameters."""

    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"encoder must be frozen; trainable parameters: {trainable}")


def build_source_proportional_receipt(
    lane_counts: Mapping[str, int],
    draws: int,
    seed: int = SEED,
) -> dict[str, object]:
    """Build a deterministic source-proportional batch-lane receipt."""

    if seed != SEED:
        raise ValueError(f"source-proportional seed must remain {SEED}")
    if set(lane_counts) != set(JOINT_LANES):
        raise ValueError(f"lane_counts must have exactly {JOINT_LANES}")
    ordered_counts = [int(lane_counts[lane]) for lane in JOINT_LANES]
    if any(count <= 0 for count in ordered_counts) or draws <= 0:
        raise ValueError("all source counts and draws must be positive")
    count_tensor = torch.tensor(ordered_counts, dtype=torch.float64)
    probabilities = count_tensor / count_tensor.sum()
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.multinomial(
        probabilities, draws, replacement=True, generator=generator
    ).tolist()
    realized = Counter(JOINT_LANES[index] for index in sampled)
    return {
        "policy": "source_proportional",
        "seed": seed,
        "projector_architecture": PROJECTOR_ARCHITECTURE,
        "projector_bias": PROJECTOR_BIAS,
        "draws": draws,
        "source_counts": dict(zip(JOINT_LANES, ordered_counts)),
        "probabilities": {
            lane: float(probabilities[index])
            for index, lane in enumerate(JOINT_LANES)
        },
        "realized_draws": {lane: int(realized[lane]) for lane in JOINT_LANES},
        "schedule": [JOINT_LANES[index] for index in sampled],
    }


@dataclass(frozen=True)
class CoreConfig:
    """Matched P1/P2 contract; budget and selection intentionally fail closed."""

    pipeline_id: str
    encoder_identity: str
    split_digest: str
    update_budget: int | None = None
    selection: str | None = None
    seed: int = SEED
    projector_input_dim: int = 768
    projector_output_dim: int = 256
    projector_architecture: str = PROJECTOR_ARCHITECTURE
    projector_bias: bool = PROJECTOR_BIAS
    projector_lanes: tuple[str, ...] = JOINT_LANES
    native_head_dims: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = tuple(
        (lane, tuple(tasks.items())) for lane, tasks in NATIVE_HEAD_DIMS.items()
    )
    sampler: str = "source_proportional"
    encoder_scope: str = "frozen_pretrained"
    trainable_scope: str = "shared_projector_plus_dataset_native_heads"
    hf_reference: str = "fixed_native_temporal_independent"

    def validate_static_contract(self) -> None:
        if self.pipeline_id not in {"P1", "P2"}:
            raise ValueError("CoreConfig pipeline_id must be P1 or P2")
        expected_encoder = {"P1": "AST", "P2": "BEATs"}[self.pipeline_id]
        if self.encoder_identity != expected_encoder:
            raise ValueError(
                f"{self.pipeline_id} encoder_identity must be {expected_encoder}"
            )
        if self.seed != SEED:
            raise ValueError(f"seed must remain {SEED}")
        if not self.split_digest:
            raise ValueError("frozen split digest is required")
        if (
            self.projector_architecture != PROJECTOR_ARCHITECTURE
            or self.projector_bias is not PROJECTOR_BIAS
            or self.projector_input_dim != 768
            or self.projector_output_dim != 256
        ):
            raise ValueError(
                "projector must remain minimal_linear_projector 768→256 with bias"
            )

    def require_execution_ready(self) -> None:
        self.validate_static_contract()
        if self.update_budget is None or self.update_budget <= 0:
            raise RuntimeError("update budget is not frozen")
        if not self.selection:
            raise RuntimeError("checkpoint selection is not frozen")


def assert_p1_p2_matched(p1: CoreConfig, p2: CoreConfig) -> None:
    """Require all fields except pipeline/encoder identity to match exactly."""

    if p1.pipeline_id != "P1" or p2.pipeline_id != "P2":
        raise ValueError("expected P1 then P2")
    p1.validate_static_contract()
    p2.validate_static_contract()
    left, right = asdict(p1), asdict(p2)
    for key in ("pipeline_id", "encoder_identity"):
        left.pop(key)
        right.pop(key)
    differences = {
        key: (left[key], right[key]) for key in left if left[key] != right[key]
    }
    if differences:
        raise RuntimeError(f"P1/P2 matched-config violation: {differences}")
