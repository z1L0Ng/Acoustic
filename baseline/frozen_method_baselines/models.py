"""Trainable modules placed on top of frozen 768-d respiratory embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


TASK_DIMS = {
    "icbhi_flat4": 4,
    "spr_binary": 2,
    "hf_cas": 1,
    "kauh_binary": 2,
}


class FrozenTaskAdapter(nn.Module):
    """Common downstream adapter used with a genuinely method-trained encoder."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, output_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(embedding))


class SharedPatientAdapter(nn.Module):
    """Shared adapter so patient auxiliary gradients reach classification features."""

    def __init__(self, task_dims: Mapping[str, int] = TASK_DIMS) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(256, classes) for task, classes in task_dims.items()}
        )
        self.pcmcl_multilabel = nn.Linear(256, 3)
        self.patient_matcher = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.adapter(embedding)

    def classify(self, embedding: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.encode(embedding))

    def multilabel(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.pcmcl_multilabel(self.encode(embedding))

    def patient_match(
        self, first_embedding: torch.Tensor, second_embedding: torch.Tensor
    ) -> torch.Tensor:
        first = self.encode(first_embedding)
        second = self.encode(second_embedding)
        pair = torch.cat(
            (first, second, torch.abs(first - second), first * second), dim=-1
        )
        return self.patient_matcher(pair)


def task_loss(logits: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    if task == "hf_cas":
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), target.to(logits.dtype)
        )
    return F.cross_entropy(logits, target.long())


def pcmcl_inspired_loss(
    *,
    native_logits: torch.Tensor,
    native_target: torch.Tensor,
    task: str,
    multilabel_logits: torch.Tensor | None,
    multilabel_target: torch.Tensor | None,
    multilabel_eligibility: torch.Tensor | None,
    patient_logits: torch.Tensor | None,
    patient_target: torch.Tensor | None,
    patient_weight: float,
) -> dict[str, torch.Tensor]:
    native = task_loss(native_logits, native_target, task)
    if multilabel_logits is not None and multilabel_target is not None:
        if multilabel_eligibility is None:
            raise ValueError("multilabel eligibility is required; unknown is not negative")
        multilabel = masked_multilabel_bce(
            multilabel_logits, multilabel_target, multilabel_eligibility
        )
    else:
        multilabel = native.new_zeros(())
    patient = (
        F.cross_entropy(patient_logits, patient_target.long())
        if patient_logits is not None and patient_target is not None
        else native.new_zeros(())
    )
    main = 0.5 * native + 0.5 * multilabel if multilabel_logits is not None else native
    total = (1.0 - patient_weight) * main + patient_weight * patient
    return {
        "total": total,
        "native": native,
        "multilabel": multilabel,
        "patient": patient,
    }


def masked_multilabel_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    eligibility: torch.Tensor,
) -> torch.Tensor:
    """Mean BCE over explicitly observed label entries only."""

    mask = eligibility.to(dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape or target.shape != logits.shape:
        raise ValueError("logits, target and eligibility must have identical shapes")
    values = F.binary_cross_entropy_with_logits(
        logits, target.to(dtype=logits.dtype, device=logits.device), reduction="none"
    )
    return values[mask].mean() if bool(mask.any()) else logits.sum() * 0.0


def pafa_adapter_regularizer(
    adapted_features: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    pcsl_weight: float,
    gpal_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Optional PAFA-inspired regularizer; input must be the shared adapter output."""

    centroids = []
    within = adapted_features.new_zeros(())
    unique = torch.unique(group_ids)
    if unique.numel() <= 1:
        return within
    for group in unique:
        current = adapted_features[group_ids == group]
        centroid = current.mean(dim=0)
        centroids.append(centroid)
        within = within + ((current - centroid) ** 2).sum()
    centroid_tensor = torch.stack(centroids)
    between = adapted_features.new_zeros(())
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            between = between + ((centroid_tensor[left] - centroid_tensor[right]) ** 2).sum()
    pcsl = within / (between + eps)
    global_centroid = centroid_tensor.mean(dim=0)
    gpal = ((centroid_tensor - global_centroid) ** 2).sum(dim=1).mean()
    return pcsl_weight * pcsl + gpal_weight * gpal
