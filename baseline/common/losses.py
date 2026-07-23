from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossSpec:
    name: str
    class_counts: list[int]
    class_weights: list[float] | None
    gamma: float | None = None
    alpha: float | None = None
    beta: float | None = None


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_prob, target, reduction="none")
        probability = torch.exp(-ce)
        return (((1.0 - probability) ** self.gamma) * ce).mean()


def label_counts(y: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.asarray([(y == label).sum() for label in labels], dtype=np.int64)


def balanced_weights(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return (total / (len(counts) * counts)).astype(np.float32)


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts.astype(np.float64)))
    weights /= weights.mean()
    return weights.astype(np.float32)


def build_loss(name: str, y: np.ndarray, labels: list[str], device: torch.device) -> tuple[nn.Module, LossSpec]:
    counts = label_counts(y, labels)
    if np.any(counts <= 0):
        raise ValueError(f"Loss reference labels are missing classes: {counts.tolist()}")
    if name == "unweighted_ce":
        return nn.CrossEntropyLoss(), LossSpec(name, counts.tolist(), None)
    if name == "class_weighted_ce":
        weights = balanced_weights(counts)
        return (
            nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)),
            LossSpec(name, counts.tolist(), weights.tolist()),
        )
    if name == "focal_loss":
        return FocalLoss(gamma=2.0), LossSpec(name, counts.tolist(), None, gamma=2.0, alpha=None)
    if name == "class_balanced_ce":
        weights = effective_number_weights(counts, beta=0.9999)
        return (
            nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device)),
            LossSpec(name, counts.tolist(), weights.tolist(), beta=0.9999),
        )
    raise ValueError(f"Unknown loss: {name}")
