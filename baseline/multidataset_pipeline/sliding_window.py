"""Common source-time sliding-window contract for four native dataset lanes.

The 2 s / 1 s policy is a proposed benchmark policy, not a source-paper fact.
Windows never repeat-pad or truncate source audio.  Short native units are
zero-padded, and every valid sample remains identified by masks and time maps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from .beats_temporal import (
    HFRawInterval,
    HFTemporalSupervision,
    HFTargetPolicy,
    TokenAlignmentPolicy,
    raw_intervals_to_token_supervision,
)
from .contracts import ObservationState, SAMPLE_RATE, WaveformSample


WINDOW_POLICY_STATUS = "proposed_benchmark_policy"
WINDOW_LENGTH_S = 2.0
WINDOW_STRIDE_S = 1.0
WINDOW_SAMPLES = int(WINDOW_LENGTH_S * SAMPLE_RATE)
WINDOW_STRIDE_SAMPLES = int(WINDOW_STRIDE_S * SAMPLE_RATE)


def source_window_starts(
    valid_samples: int,
    *,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = WINDOW_STRIDE_SAMPLES,
) -> tuple[int, ...]:
    """Return sorted unique starts, appending one end-aligned tail if needed."""

    if valid_samples <= 0:
        raise ValueError("valid_samples must be positive")
    if window_samples <= 0 or stride_samples <= 0 or stride_samples > window_samples:
        raise ValueError("require 0 < stride_samples <= window_samples")
    if valid_samples <= window_samples:
        return (0,)
    final_start = valid_samples - window_samples
    starts = list(range(0, final_start + 1, stride_samples))
    if starts[-1] != final_start:
        starts.append(final_start)
    if starts != sorted(set(starts)):
        raise RuntimeError("window starts must be sorted and unique")
    return tuple(starts)


@dataclass(frozen=True)
class SlidingWindowBatch:
    """Zero-padded `[B,K,W]` windows with source-time and native-unit lineage."""

    waveform_windows: torch.Tensor
    waveform_padding_mask: torch.Tensor
    valid_samples: torch.Tensor
    window_mask: torch.Tensor
    time_map: torch.Tensor
    sample_rate: int
    window_samples: int
    stride_samples: int
    sample_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    prediction_units: tuple[str, ...]
    lineage: tuple[Mapping[str, str], ...]

    @property
    def device(self) -> torch.device:
        return self.waveform_windows.device

    def to(self, device: torch.device | str) -> "SlidingWindowBatch":
        target = torch.device(device)
        moved = replace(
            self,
            waveform_windows=self.waveform_windows.to(target),
            waveform_padding_mask=self.waveform_padding_mask.to(target),
            valid_samples=self.valid_samples.to(target),
            window_mask=self.window_mask.to(target),
            time_map=self.time_map.to(target),
        )
        moved.validate()
        return moved

    def validate(self) -> None:
        if self.waveform_windows.ndim != 3 or self.waveform_windows.dtype != torch.float32:
            raise TypeError("waveform_windows must be float32 [B,K,W]")
        batch, windows, width = self.waveform_windows.shape
        if width != self.window_samples or self.sample_rate != SAMPLE_RATE:
            raise ValueError("window geometry/sample rate mismatch")
        if self.window_samples <= 0 or not 0 < self.stride_samples <= self.window_samples:
            raise ValueError("invalid window/stride samples")
        if (
            self.waveform_padding_mask.shape != (batch, windows, width)
            or self.waveform_padding_mask.dtype != torch.bool
        ):
            raise TypeError("waveform_padding_mask must be bool [B,K,W]")
        if self.valid_samples.shape != (batch, windows) or self.valid_samples.dtype != torch.long:
            raise TypeError("valid_samples must be int64 [B,K]")
        if self.window_mask.shape != (batch, windows) or self.window_mask.dtype != torch.bool:
            raise TypeError("window_mask must be bool [B,K] with True=valid")
        if self.time_map.shape != (batch, windows, 2) or self.time_map.dtype != torch.float64:
            raise TypeError("time_map must be float64 [B,K,2] source seconds")
        devices = {
            self.waveform_windows.device,
            self.waveform_padding_mask.device,
            self.valid_samples.device,
            self.window_mask.device,
            self.time_map.device,
        }
        if len(devices) != 1:
            raise RuntimeError(f"SlidingWindowBatch tensors must share one device: {devices}")
        if any(
            len(values) != batch
            for values in (
                self.sample_ids,
                self.dataset_ids,
                self.prediction_units,
                self.lineage,
            )
        ):
            raise ValueError("window lineage length mismatch")
        for row in range(batch):
            valid_count = int(self.window_mask[row].sum())
            if valid_count <= 0 or not bool(self.window_mask[row, :valid_count].all()):
                raise ValueError("valid windows must be a non-empty contiguous prefix")
            if bool(self.window_mask[row, valid_count:].any()):
                raise ValueError("invalid window slots cannot appear inside the prefix")
            starts = self.time_map[row, :valid_count, 0]
            ends = self.time_map[row, :valid_count, 1]
            if bool((ends <= starts).any()) or bool((starts[1:] <= starts[:-1]).any()):
                raise ValueError("valid source-time windows must be positive and strictly ordered")
            for index in range(windows):
                valid = int(self.valid_samples[row, index])
                if self.window_mask[row, index]:
                    if not 0 < valid <= width:
                        raise ValueError("valid window has invalid sample count")
                    if bool(self.waveform_padding_mask[row, index, :valid].any()):
                        raise ValueError("valid source samples marked padding")
                    if not bool(self.waveform_padding_mask[row, index, valid:].all()):
                        raise ValueError("zero-padded region marked valid")
                    if bool(torch.count_nonzero(self.waveform_windows[row, index, valid:])):
                        raise ValueError("window padding must be zero; repeat-pad is forbidden")
                    mapped = float(self.time_map[row, index, 1] - self.time_map[row, index, 0])
                    if abs(mapped - valid / SAMPLE_RATE) > 1 / SAMPLE_RATE:
                        raise ValueError("window time_map duration and valid_samples disagree")
                elif (
                    valid != 0
                    or bool(torch.count_nonzero(self.waveform_windows[row, index]))
                    or not bool(self.waveform_padding_mask[row, index].all())
                    or bool(torch.count_nonzero(self.time_map[row, index]))
                ):
                    raise ValueError("invalid window slots must be zero and fully masked")

    def receipt(self) -> dict[str, object]:
        counts = self.window_mask.sum(dim=1).tolist()
        return {
            "status": "shared_sliding_window_contract_passed",
            "policy_status": WINDOW_POLICY_STATUS,
            "sample_rate": self.sample_rate,
            "window_length_s": self.window_samples / self.sample_rate,
            "window_stride_s": self.stride_samples / self.sample_rate,
            "shape": list(self.waveform_windows.shape),
            "window_counts": [int(value) for value in counts],
            "tail_policy": "append_unique_end_aligned_window_when_stride_misses_tail",
            "short_policy": "zero_pad_with_valid_samples_and_source_time_map",
            "repeat_pad": False,
            "truncate": False,
        }


def collate_sliding_windows(
    samples: Sequence[WaveformSample],
    *,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = WINDOW_STRIDE_SAMPLES,
) -> SlidingWindowBatch:
    if not samples:
        raise ValueError("cannot window an empty sample batch")
    starts_by_sample = [
        source_window_starts(
            sample.waveform.numel(),
            window_samples=window_samples,
            stride_samples=stride_samples,
        )
        for sample in samples
    ]
    batch = len(samples)
    max_windows = max(len(starts) for starts in starts_by_sample)
    waveforms = torch.zeros(batch, max_windows, window_samples, dtype=torch.float32)
    padding = torch.ones(batch, max_windows, window_samples, dtype=torch.bool)
    valid_samples = torch.zeros(batch, max_windows, dtype=torch.long)
    window_mask = torch.zeros(batch, max_windows, dtype=torch.bool)
    time_map = torch.zeros(batch, max_windows, 2, dtype=torch.float64)
    lineage = []
    for row, (sample, starts) in enumerate(zip(samples, starts_by_sample)):
        for index, start in enumerate(starts):
            end = min(start + window_samples, sample.waveform.numel())
            valid = end - start
            waveforms[row, index, :valid].copy_(sample.waveform[start:end])
            padding[row, index, :valid] = False
            valid_samples[row, index] = valid
            window_mask[row, index] = True
            source_start = sample.source_start_s + start / SAMPLE_RATE
            time_map[row, index, 0] = source_start
            time_map[row, index, 1] = source_start + valid / SAMPLE_RATE
        lineage.append(
            {
                "sample_id": sample.sample_id,
                "dataset_id": sample.dataset_id,
                "prediction_unit": sample.prediction_unit,
                **dict(sample.lineage),
            }
        )
    output = SlidingWindowBatch(
        waveform_windows=waveforms,
        waveform_padding_mask=padding,
        valid_samples=valid_samples,
        window_mask=window_mask,
        time_map=time_map,
        sample_rate=SAMPLE_RATE,
        window_samples=window_samples,
        stride_samples=stride_samples,
        sample_ids=tuple(sample.sample_id for sample in samples),
        dataset_ids=tuple(sample.dataset_id for sample in samples),
        prediction_units=tuple(sample.prediction_unit for sample in samples),
        lineage=tuple(lineage),
    )
    output.validate()
    return output


def masked_mean_window_embeddings(
    embeddings: torch.Tensor,
    window_mask: torch.Tensor,
    *,
    expected_dim: int = 768,
) -> torch.Tensor:
    """Aggregate non-HF windows without allowing padded slots to contribute."""

    if embeddings.ndim != 3 or embeddings.shape[-1] != expected_dim:
        raise ValueError(f"window embeddings must be [B,K,{expected_dim}]")
    if window_mask.shape != embeddings.shape[:2] or window_mask.dtype != torch.bool:
        raise TypeError("window_mask must be bool [B,K]")
    if embeddings.device != window_mask.device:
        raise RuntimeError("embeddings and window_mask must share one device")
    if not torch.isfinite(embeddings).all():
        raise ValueError("window embeddings must be finite")
    denominator = window_mask.sum(dim=1, keepdim=True)
    if bool((denominator == 0).any()):
        raise RuntimeError("cannot aggregate a native unit without valid windows")
    masked = torch.where(
        window_mask.unsqueeze(-1), embeddings, torch.zeros_like(embeddings)
    )
    return masked.sum(dim=1) / denominator.to(embeddings.dtype)


def hf_window_supervision(
    batch: SlidingWindowBatch,
    intervals: Sequence[Sequence[HFRawInterval]],
    recording_states: Sequence[ObservationState],
    *,
    policy: HFTargetPolicy = HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
) -> HFTemporalSupervision:
    """Align HF raw intervals to source-time window centers with explicit policy."""

    batch.validate()
    if any(dataset != "HF" for dataset in batch.dataset_ids):
        raise ValueError("hf_window_supervision accepts only the HF native lane")
    return raw_intervals_to_token_supervision(
        batch.time_map,
        batch.window_mask,
        intervals,
        recording_states,
        policy=policy,
        alignment=TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL,
    )
