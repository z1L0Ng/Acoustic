"""Minimal trainable interfaces for the proposed PC-MCL and DCASE adaptations.

This module defines heads and masked objectives only.  It does not load an
encoder, read audio, extract embeddings, build caches, or start training.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import masked_multilabel_bce


PCMCL_LABELS = ("normal", "crackle", "wheeze")
DCASE_RESPIRATORY_LABELS = ("abnormal", "crackle", "wheeze", "other")


def maximum_window_p_w_proxy(window_probabilities: torch.Tensor, wheeze_index: int) -> torch.Tensor:
    """Current Table-1 CAS proxy: weak p_W per 5-s window, then window max."""

    return window_probabilities[..., wheeze_index].max(dim=1).values


def maximum_window_w_or_other_diagnostic(
    window_probabilities: torch.Tensor,
    *,
    wheeze_index: int = 2,
    other_index: int = 3,
) -> torch.Tensor:
    """DCASE-specific W/Rhonchi/Stridor diagnostic, not the main CAS column."""

    per_window = torch.maximum(
        window_probabilities[..., wheeze_index],
        window_probabilities[..., other_index],
    )
    return per_window.max(dim=1).values


def dcase_external_inference_class_mask(
    batch_size: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Fixed output mask for HF inference; it never depends on HF annotations."""

    return torch.ones(
        (batch_size, len(DCASE_RESPIRATORY_LABELS)), dtype=torch.bool, device=device
    )


def compose_pcmcl_pair_targets(
    first_target: torch.Tensor,
    first_eligibility: torch.Tensor,
    second_target: torch.Tensor,
    second_eligibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ternary N/C/W labels for a raw two-unit concatenation.

    A positive observation from either unit makes the composite label positive.
    A composite negative is valid only when both constituent labels are observed
    negatives.  Every other entry remains unknown and is excluded from BCE.
    """

    first_observed = first_eligibility.bool()
    second_observed = second_eligibility.bool()
    positive = (first_observed & first_target.bool()) | (
        second_observed & second_target.bool()
    )
    eligible = positive | (first_observed & second_observed)
    return positive.to(first_target.dtype), eligible


class PCMCLFrozenConcatHeads(nn.Module):
    """Legacy shared head for the superseded frozen-encoder candidate.

    The frozen encoder supplies one pooled 768-d embedding for the complete
    10-s concatenated waveform.  Both objectives update this shared adapter;
    the patient task is therefore not an isolated head with no effect on the
    pathology representation.  Current PC-MCL authority is the full source
    training contract in ``pcmcl_source_transfer.py``.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pathology_head = nn.Linear(256, len(PCMCL_LABELS))
        self.patient_head = nn.Linear(256, 2)

    def forward(self, pooled_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = self.adapter(pooled_embedding)
        return {
            "adapted": adapted,
            "pathology_logits": self.pathology_head(adapted),
            "patient_logits": self.patient_head(adapted),
        }


def pcmcl_frozen_concat_loss(
    *,
    pathology_logits: torch.Tensor,
    pathology_target: torch.Tensor,
    pathology_eligibility: torch.Tensor,
    patient_logits: torch.Tensor,
    patient_target: torch.Tensor,
    patient_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Paper-shaped main plus patient objective with missing-label masking."""

    main = masked_multilabel_bce(
        pathology_logits, pathology_target, pathology_eligibility
    )
    patient = F.cross_entropy(patient_logits, patient_target.long())
    return {
        "total": main + patient_weight * patient,
        "main": main,
        "patient": patient,
    }


class DCASEFrameFusionHead(nn.Module):
    """DCASE-style frame fusion, BiGRU, and class-masked attention readout.

    ``cnn_frames`` are produced by the trainable official-style log-Mel CNN and
    ``beats_frames`` by the frozen BEATs extractor.  This interface deliberately
    keeps frame time; existing mean-pooled 768-d caches are not valid inputs.
    During external inference, ``class_eligibility`` is a fixed model-output
    mask for the whole dataset, never a mask derived from target annotations.
    """

    def __init__(
        self,
        cnn_dim: int = 128,
        beats_dim: int = 768,
        hidden_dim: int = 192,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fusion = nn.Linear(cnn_dim + beats_dim, cnn_dim)
        self.rnn = nn.GRU(
            input_size=cnn_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        output_dim = hidden_dim * 2
        self.classifier = nn.Linear(output_dim, len(DCASE_RESPIRATORY_LABELS))
        self.attention = nn.Linear(output_dim, len(DCASE_RESPIRATORY_LABELS))

    def forward(
        self,
        cnn_frames: torch.Tensor,
        beats_frames: torch.Tensor,
        frame_valid: torch.Tensor,
        class_eligibility: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_frames = cnn_frames.shape[1]
        aligned_beats = F.adaptive_avg_pool1d(
            beats_frames.transpose(1, 2), target_frames
        ).transpose(1, 2)
        fused = self.fusion(torch.cat((cnn_frames, aligned_beats), dim=-1))
        recurrent, _ = self.rnn(self.dropout(fused))
        recurrent = self.dropout(recurrent)

        strong_logits = self.classifier(recurrent)
        strong_probabilities = torch.sigmoid(strong_logits)
        attention_logits = self.attention(recurrent)

        valid = frame_valid.bool().unsqueeze(-1)
        eligible = class_eligibility.bool().unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~valid, -1e30)
        attention_logits = attention_logits.masked_fill(~eligible, -1e30)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = attention_weights.masked_fill(~valid | ~eligible, 0.0)
        strong_probabilities = strong_probabilities.masked_fill(
            ~valid | ~eligible, 0.0
        )
        weak_probabilities = (
            (strong_probabilities * attention_weights).sum(dim=1)
            / attention_weights.sum(dim=1).clamp_min(1e-7)
        )
        weak_probabilities = weak_probabilities.masked_fill(
            ~class_eligibility.bool(), 0.0
        )
        return {
            "strong_logits": strong_logits,
            "strong_probabilities": strong_probabilities,
            "weak_probabilities": weak_probabilities,
        }


def dcase_masked_weak_bce(
    weak_probabilities: torch.Tensor,
    target: torch.Tensor,
    eligibility: torch.Tensor,
) -> torch.Tensor:
    """Mean weak-label BCE over explicitly supported sample/class entries."""

    values = F.binary_cross_entropy(
        weak_probabilities,
        target.to(weak_probabilities.dtype),
        reduction="none",
    )
    mask = eligibility.bool()
    return values[mask].mean() if bool(mask.any()) else weak_probabilities.sum() * 0.0
