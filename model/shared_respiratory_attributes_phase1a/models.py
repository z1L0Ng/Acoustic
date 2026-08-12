from __future__ import annotations

import math

import torch
from torch import nn


CONDITIONS = ("N", "I16", "S16", "S32")
ICBHI_LABELS = ["normal", "crackle", "wheeze", "both"]
SPR_LABELS = [
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
]
ATTRIBUTES = ("crackle", "wheeze")
INPUT_DIM = 768
HIDDEN_1 = 256
HIDDEN_2 = 128
DROPOUT = 0.3
SOFTPLUS_ONE_RAW = math.log(math.e - 1.0)


class AttributeMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HIDDEN_2, width),
            nn.ReLU(),
            nn.Linear(width, len(ATTRIBUTES)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class Phase1AModel(nn.Module):
    def __init__(self, condition: str) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        self.condition = condition
        self.adapter = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.native_heads = nn.ModuleDict(
            {
                "icbhi": nn.Linear(HIDDEN_2, len(ICBHI_LABELS)),
                "sprsound": nn.Linear(HIDDEN_2, len(SPR_LABELS)),
            }
        )
        self.independent_attr_heads = nn.ModuleDict()
        self.shared_attr_head: AttributeMLP | None = None
        self.shared_temperature_raw: nn.Parameter | None = None
        if condition == "I16":
            self.independent_attr_heads = nn.ModuleDict(
                {
                    "icbhi": AttributeMLP(width=16),
                    "sprsound": AttributeMLP(width=16),
                }
            )
        elif condition == "S16":
            self.shared_attr_head = AttributeMLP(width=16)
        elif condition == "S32":
            self.shared_attr_head = AttributeMLP(width=32)
            self.shared_temperature_raw = nn.Parameter(
                torch.full((len(ATTRIBUTES),), SOFTPLUS_ONE_RAW, dtype=torch.float32)
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

    def native_logits(self, hidden: torch.Tensor, dataset: str) -> torch.Tensor:
        return self.native_heads[dataset](hidden)

    def attr_logits(self, hidden: torch.Tensor, dataset: str) -> torch.Tensor:
        if self.condition == "N":
            raise RuntimeError("N condition has no attribute branch")
        if self.condition == "I16":
            return self.independent_attr_heads[dataset](hidden)
        assert self.shared_attr_head is not None
        logits = self.shared_attr_head(hidden)
        if self.shared_temperature_raw is None:
            return logits
        return logits * torch.nn.functional.softplus(self.shared_temperature_raw)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def condition_parameter_receipt() -> dict[str, dict[str, int]]:
    active_per_sample = {
        "N": 0,
        "I16": 2098,
        "S16": 2098,
        "S32": 4196,
    }
    rows: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        model = Phase1AModel(condition)
        rows[condition] = {
            "trainable_total": trainable_parameter_count(model),
            "parameter_total_including_frozen": total_parameter_count(model),
            "attr_branch_total": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if "attr" in name or "temperature" in name
            ),
            "active_attr_per_sample": active_per_sample[condition],
            "adamw_state_parameters": 2 * trainable_parameter_count(model),
        }
    return rows
