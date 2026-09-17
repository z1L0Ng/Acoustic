"""Pure interfaces for the proposed 5-s PC-MCL source-transfer protocol.

No function in this module loads a model or dataset.  The source-training
runner remains intentionally absent until execution is separately authorized.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SAMPLE_RATE = 16_000
SOURCE_INPUT_SECONDS = 5.0
PAIR_CYCLE_SECONDS = 2.5
NCW_ORDER = ("normal", "crackle", "wheeze")
SOURCE_THRESHOLD = 0.5


def compose_icbhi_ncw_targets(
    first_target: torch.Tensor, second_target: torch.Tensor
) -> torch.Tensor:
    """Official additive label: element-wise logical OR over N/C/W."""

    return torch.maximum(first_target, second_target)


def repeat_pad_or_center_crop(
    waveform: torch.Tensor, target_samples: int
) -> torch.Tensor:
    """Apply the frozen source policy to the final waveform dimension."""

    current = waveform.shape[-1]
    if current > target_samples:
        start = (current - target_samples) // 2
        return waveform[..., start : start + target_samples]
    if current == target_samples:
        return waveform
    if current == 0:
        raise ValueError("cannot repeat-pad an empty waveform")
    repeats = [1] * waveform.ndim
    repeats[-1] = math.ceil(target_samples / current)
    return waveform.repeat(*repeats)[..., :target_samples]


def build_pcmcl_pair_5s(
    first_cycle: torch.Tensor,
    second_cycle: torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Normalize two cycles independently to 2.5 s and concatenate to 5 s."""

    half_samples = int(PAIR_CYCLE_SECONDS * sample_rate)
    first = repeat_pad_or_center_crop(first_cycle, half_samples)
    second = repeat_pad_or_center_crop(second_cycle, half_samples)
    return torch.cat((first, second), dim=-1)


def build_pcmcl_single_5s(
    waveform: torch.Tensor, *, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """Prepare one source or target unit for fixed-model 5-s inference."""

    return repeat_pad_or_center_crop(
        waveform, int(SOURCE_INPUT_SECONDS * sample_rate)
    )


class PCMCLSourceHeads(nn.Module):
    """Official-shaped N/C/W and patient heads above a trainable encoder."""

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self.pathology = nn.Linear(feature_dim, len(NCW_ORDER))
        self.patient = nn.Linear(feature_dim, 2)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "pathology_logits": self.pathology(features),
            "patient_logits": self.patient(features),
        }


def pcmcl_source_loss(
    *,
    pathology_logits: torch.Tensor,
    pathology_target: torch.Tensor,
    patient_logits: torch.Tensor,
    patient_target: torch.Tensor,
    patient_eligibility: torch.Tensor,
    patient_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Paper-form main BCE plus alpha times eligible patient CE."""

    main = F.binary_cross_entropy_with_logits(
        pathology_logits, pathology_target.to(pathology_logits.dtype)
    )
    mask = patient_eligibility.bool()
    patient = (
        F.cross_entropy(patient_logits[mask], patient_target[mask].long())
        if bool(mask.any())
        else pathology_logits.sum() * 0.0
    )
    return {"total": main + patient_weight * patient, "main": main, "patient": patient}


def icbhi_flat4_from_ncw(
    probabilities: torch.Tensor, *, threshold: float = SOURCE_THRESHOLD
) -> torch.Tensor:
    """Original fixed C/W-bit conversion: Normal, Crackle, Wheeze, Both."""

    crackle = probabilities[..., 1] >= threshold
    wheeze = probabilities[..., 2] >= threshold
    output = torch.zeros_like(crackle, dtype=torch.long)
    output[crackle & ~wheeze] = 1
    output[~crackle & wheeze] = 2
    output[crackle & wheeze] = 3
    return output


def spr_binary_from_ncw(
    probabilities: torch.Tensor, *, threshold: float = SOURCE_THRESHOLD
) -> torch.Tensor:
    """Fixed source rule: predicted flat4 Normal vs any predicted abnormal class."""

    return (icbhi_flat4_from_ncw(probabilities, threshold=threshold) != 0).long()


def hf_maximum_window_p_w(window_probabilities: torch.Tensor) -> torch.Tensor:
    """Current CAS ranking proxy: maximum p_W over the three fixed windows."""

    return window_probabilities[..., 2].max(dim=1).values


def kauh_patient_from_ncw_views(
    view_probabilities: torch.Tensor, *, threshold: float = SOURCE_THRESHOLD
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average fixed-source abnormal scores across B/D/E, then threshold once."""

    view_abnormal = view_probabilities[..., 1:3].max(dim=-1).values
    patient_probability = view_abnormal.mean(dim=-1)
    return patient_probability, (patient_probability >= threshold).long()
