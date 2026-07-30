"""Parameter-matched shared and task-specific residual models."""

from __future__ import annotations

import torch

from baseline.four_dataset_frozen_encoder.train import TASK_SPECS


CONDITIONS = {
    "d2_local_reference": {
        "residual": "none",
        "rank": 0,
        "selective_prior": False,
    },
    "d2_shared_residual_param_matched": {
        "residual": "shared",
        "rank": 96,
        "selective_prior": False,
    },
    "d2_task_residual": {
        "residual": "task",
        "rank": 16,
        "selective_prior": False,
    },
    "d2_task_residual_selective_prior": {
        "residual": "task",
        "rank": 16,
        "selective_prior": True,
    },
}

EXPECTED_PARAMETERS = {
    "d2_local_reference": 205_596,
    "d2_shared_residual_param_matched": 303_900,
    "d2_task_residual": 303_900,
    "d2_task_residual_selective_prior": 303_900,
}


class LowRankResidual(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.down = torch.nn.Linear(input_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, output_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(values)))


class GeneralSpecificModel(torch.nn.Module):
    def __init__(self, condition: str) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        config = CONDITIONS[condition]
        self.condition = condition
        self.input_norm = torch.nn.LayerNorm(768)
        self.general = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.2)
        self.heads = torch.nn.ModuleDict(
            {
                task: torch.nn.Linear(256, len(spec["labels"]))
                for task, spec in TASK_SPECS.items()
            }
        )
        self.shared_residual: LowRankResidual | None = None
        self.task_residuals = torch.nn.ModuleDict()
        if config["residual"] == "shared":
            self.shared_residual = LowRankResidual(768, 256, int(config["rank"]))
        elif config["residual"] == "task":
            self.task_residuals = torch.nn.ModuleDict(
                {
                    task: LowRankResidual(768, 256, int(config["rank"]))
                    for task in TASK_SPECS
                }
            )

    def transform(self, values: torch.Tensor, task: str) -> torch.Tensor:
        normalized = self.input_norm(values)
        general = self.dropout(torch.relu(self.general(normalized)))
        if self.shared_residual is not None:
            general = general + self.shared_residual(normalized)
        elif task in self.task_residuals:
            general = general + self.task_residuals[task](normalized)
        return general

    def forward(self, values: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.transform(values, task))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
