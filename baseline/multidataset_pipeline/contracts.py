"""Dataset-lineage and zero-padding contracts shared by P1/P2/P6/P8."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, Sequence

import torch


SAMPLE_RATE = 16_000
PREDICTION_UNITS = {
    "ICBHI": "cycle",
    "SPRSound": "event",
    "HF": "recording_15s_with_intervals",
    "KAUH": "recording",
}


class ObservationState(str, Enum):
    """Explicit annotation states; only OBSERVED may create supervision."""

    OBSERVED = "observed"
    MISSING = "missing"
    UNKNOWN = "unknown"
    NOT_ANNOTATED = "not_annotated"
    EMPTY = "empty"


@dataclass(frozen=True)
class WaveformSample:
    """One native prediction unit with source-time and dataset lineage."""

    waveform: torch.Tensor
    sample_id: str
    dataset_id: str
    prediction_unit: str
    source_start_s: float
    source_end_s: float
    sample_rate: int = SAMPLE_RATE
    lineage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz, got {self.sample_rate}")
        if self.dataset_id not in PREDICTION_UNITS:
            raise ValueError(f"unknown dataset lane: {self.dataset_id}")
        expected_unit = PREDICTION_UNITS[self.dataset_id]
        if self.prediction_unit != expected_unit:
            raise ValueError(
                f"{self.dataset_id} requires prediction unit {expected_unit!r}, "
                f"got {self.prediction_unit!r}"
            )
        if self.waveform.ndim != 1 or self.waveform.numel() == 0:
            raise ValueError("waveform must be a non-empty [T] tensor")
        if self.waveform.dtype != torch.float32:
            raise TypeError("waveform must be torch.float32")
        if not bool(torch.isfinite(self.waveform).all()):
            raise ValueError("waveform contains non-finite values")
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        duration_s = self.waveform.numel() / self.sample_rate
        if self.source_start_s < 0 or self.source_end_s <= self.source_start_s:
            raise ValueError("invalid source time interval")
        if abs((self.source_end_s - self.source_start_s) - duration_s) > 1 / self.sample_rate:
            raise ValueError("source interval must match valid waveform samples")
        if self.dataset_id == "HF" and self.waveform.numel() != 15 * self.sample_rate:
            raise ValueError("HF native recording unit must contain exactly 15 seconds")


@dataclass(frozen=True)
class WaveformBatch:
    """Zero-padded waveform batch; padding mask uses True=padding."""

    waveform: torch.Tensor
    waveform_padding_mask: torch.Tensor
    valid_samples: torch.Tensor
    sample_rate: int
    source_start_s: torch.Tensor
    source_end_s: torch.Tensor
    sample_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    prediction_units: tuple[str, ...]
    lineage: tuple[Mapping[str, str], ...]

    @property
    def device(self) -> torch.device:
        return self.waveform.device

    def to(self, device: torch.device | str) -> "WaveformBatch":
        """Return a new batch on one explicit device; lineage stays on the host."""

        target = torch.device(device)
        moved = replace(
            self,
            waveform=self.waveform.to(target),
            waveform_padding_mask=self.waveform_padding_mask.to(target),
            valid_samples=self.valid_samples.to(target),
            source_start_s=self.source_start_s.to(target),
            source_end_s=self.source_end_s.to(target),
        )
        moved.validate()
        return moved

    def validate(self) -> None:
        if self.waveform.ndim != 2 or self.waveform.dtype != torch.float32:
            raise TypeError("waveform must be float32 [B,Tmax]")
        if (
            self.waveform_padding_mask.shape != self.waveform.shape
            or self.waveform_padding_mask.dtype != torch.bool
        ):
            raise TypeError("waveform_padding_mask must be bool [B,Tmax]")
        batch_size, max_samples = self.waveform.shape
        if self.valid_samples.shape != (batch_size,) or self.valid_samples.dtype != torch.long:
            raise TypeError("valid_samples must be int64 [B]")
        if self.source_start_s.shape != (batch_size,) or self.source_end_s.shape != (
            batch_size,
        ):
            raise ValueError("source times must be [B]")
        tensor_devices = {
            self.waveform.device,
            self.waveform_padding_mask.device,
            self.valid_samples.device,
            self.source_start_s.device,
            self.source_end_s.device,
        }
        if len(tensor_devices) != 1:
            raise RuntimeError(
                f"all WaveformBatch tensors must share one device, got {tensor_devices}"
            )
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError("batch sample rate mismatch")
        if any(
            len(values) != batch_size
            for values in (
                self.sample_ids,
                self.dataset_ids,
                self.prediction_units,
                self.lineage,
            )
        ):
            raise ValueError("lineage length mismatch")
        for index, valid in enumerate(self.valid_samples.tolist()):
            if not 0 < valid <= max_samples:
                raise ValueError("invalid valid_samples entry")
            if bool(self.waveform_padding_mask[index, :valid].any()):
                raise ValueError("valid waveform region marked as padding")
            if not bool(self.waveform_padding_mask[index, valid:].all()):
                raise ValueError("padded waveform region marked as valid")
            if bool(torch.count_nonzero(self.waveform[index, valid:])):
                raise ValueError("padding must be zeros; repeat-padding is forbidden")
            duration_s = valid / self.sample_rate
            actual_s = float(self.source_end_s[index] - self.source_start_s[index])
            if abs(actual_s - duration_s) > 1 / self.sample_rate:
                raise ValueError("source time and valid_samples mismatch")


def collate_waveforms(samples: Sequence[WaveformSample]) -> WaveformBatch:
    """Collate variable-length native units with zero padding and no repetition."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_samples = max(sample.waveform.numel() for sample in samples)
    batch_size = len(samples)
    waveform = torch.zeros(batch_size, max_samples, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_samples, dtype=torch.bool)
    valid_samples = torch.empty(batch_size, dtype=torch.long)
    lineage = []
    for index, sample in enumerate(samples):
        valid = sample.waveform.numel()
        waveform[index, :valid].copy_(sample.waveform)
        padding_mask[index, :valid] = False
        valid_samples[index] = valid
        lineage.append(
            {
                "sample_id": sample.sample_id,
                "dataset_id": sample.dataset_id,
                "prediction_unit": sample.prediction_unit,
                **dict(sample.lineage),
            }
        )
    batch = WaveformBatch(
        waveform=waveform,
        waveform_padding_mask=padding_mask,
        valid_samples=valid_samples,
        sample_rate=SAMPLE_RATE,
        source_start_s=torch.tensor(
            [sample.source_start_s for sample in samples], dtype=torch.float64
        ),
        source_end_s=torch.tensor(
            [sample.source_end_s for sample in samples], dtype=torch.float64
        ),
        sample_ids=tuple(sample.sample_id for sample in samples),
        dataset_ids=tuple(sample.dataset_id for sample in samples),
        prediction_units=tuple(sample.prediction_unit for sample in samples),
        lineage=tuple(lineage),
    )
    batch.validate()
    return batch
