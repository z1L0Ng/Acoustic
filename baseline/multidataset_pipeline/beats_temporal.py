"""Exact BEATs temporal contract for P6.

HF source-task policy reference:
docs/datasets/four_dataset_task_contract_review_2026-07-28.md, sections 5.2-5.3.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

import torch
from torch import nn

from .contracts import ObservationState, SAMPLE_RATE, WaveformBatch


CHANNEL_ORDER = ("I", "E", "CAS", "DAS")
RAW_TOKEN_TO_CHANNEL = {
    "I": "I",
    "E": "E",
    "Wheeze": "CAS",
    "Rhonchi": "CAS",
    "Stridor": "CAS",
    "D": "DAS",
}
SOURCE_TASK_POLICY_REFERENCE = (
    "docs/datasets/four_dataset_task_contract_review_2026-07-28.md"
)


class HFTargetPolicy(str, Enum):
    RAW_CONSERVATIVE_POSITIVE_ONLY = "raw_conservative_positive_only"
    PAPER_NATIVE_RASTERIZED_OVR = "paper_native_rasterized_ovr"


class TokenAlignmentPolicy(str, Enum):
    TOKEN_CENTER_IN_INTERVAL = "token_center_in_interval"
    ANY_OVERLAP_TEST_ONLY = "any_overlap_test_only"


@dataclass(frozen=True)
class BEATsGeometry:
    """Kaldi/BEATs geometry; trailing partial patches are invalid."""

    sample_rate: int = SAMPLE_RATE
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    mel_bins: int = 128
    patch_kernel_time: int = 16
    patch_stride_time: int = 16
    patch_kernel_frequency: int = 16
    patch_stride_frequency: int = 16

    @classmethod
    def from_checkpoint_config(cls, config: object) -> "BEATsGeometry":
        def value(name: str, default: object = None) -> object:
            if isinstance(config, Mapping):
                return config.get(name, default)
            return getattr(config, name, default)

        patch_size = value("input_patch_size")
        if patch_size is None:
            raise ValueError("checkpoint config is missing input_patch_size")
        if isinstance(patch_size, int):
            kernel_time = kernel_frequency = patch_size
        else:
            kernel_time, kernel_frequency = (int(item) for item in patch_size)
        stride = value("input_patch_stride", value("patch_stride", patch_size))
        if isinstance(stride, int):
            stride_time = stride_frequency = stride
        else:
            stride_time, stride_frequency = (int(item) for item in stride)
        return cls(
            patch_kernel_time=int(kernel_time),
            patch_stride_time=int(stride_time),
            patch_kernel_frequency=int(kernel_frequency),
            patch_stride_frequency=int(stride_frequency),
        )

    @property
    def frame_length_samples(self) -> int:
        return round(self.sample_rate * self.frame_length_ms / 1000)

    @property
    def frame_shift_samples(self) -> int:
        return round(self.sample_rate * self.frame_shift_ms / 1000)

    @property
    def receptive_interval_s(self) -> float:
        return (
            (self.patch_kernel_time - 1) * self.frame_shift_samples
            + self.frame_length_samples
        ) / self.sample_rate

    @property
    def temporal_stride_s(self) -> float:
        return (
            self.patch_stride_time * self.frame_shift_samples / self.sample_rate
        )

    def fbank_frames(self, samples: int | torch.Tensor) -> int | torch.Tensor:
        length = self.frame_length_samples
        shift = self.frame_shift_samples
        if isinstance(samples, int):
            return 0 if samples < length else 1 + (samples - length) // shift
        samples = samples.to(dtype=torch.long)
        return torch.where(
            samples < length,
            torch.zeros_like(samples),
            1 + torch.div(samples - length, shift, rounding_mode="floor"),
        )

    @staticmethod
    def full_patch_count(size: int | torch.Tensor, kernel: int, stride: int):
        if isinstance(size, int):
            return 0 if size < kernel else 1 + (size - kernel) // stride
        return torch.where(
            size < kernel,
            torch.zeros_like(size),
            1 + torch.div(size - kernel, stride, rounding_mode="floor"),
        )

    def time_patches(self, samples: int | torch.Tensor):
        return self.full_patch_count(
            self.fbank_frames(samples),
            self.patch_kernel_time,
            self.patch_stride_time,
        )

    def frequency_patches(self) -> int:
        return self.full_patch_count(
            self.mel_bins,
            self.patch_kernel_frequency,
            self.patch_stride_frequency,
        )

    def receipt(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "frame_length_ms": self.frame_length_ms,
            "frame_shift_ms": self.frame_shift_ms,
            "patch_kernel": [
                self.patch_kernel_time,
                self.patch_kernel_frequency,
            ],
            "patch_stride": [
                self.patch_stride_time,
                self.patch_stride_frequency,
            ],
            "temporal_stride_s": self.temporal_stride_s,
            "receptive_interval_s": self.receptive_interval_s,
            "trailing_patch_policy": "complete_valid_patch_only",
            "flatten_order": "time_major_frequency_minor",
        }


def exact_patch_masks(
    valid_samples: torch.Tensor,
    total_samples: int,
    geometry: BEATsGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return temporal-valid and flattened Transformer padding masks."""

    if valid_samples.ndim != 1 or valid_samples.dtype != torch.long:
        raise TypeError("valid_samples must be int64 [B]")
    if total_samples < int(valid_samples.max()):
        raise ValueError("total_samples is shorter than a valid waveform")
    total_time_patches = int(geometry.time_patches(total_samples))
    frequency_patches = geometry.frequency_patches()
    if total_time_patches <= 0 or frequency_patches <= 0:
        raise RuntimeError("BEATs patch grid is empty")
    valid_time_counts = geometry.time_patches(valid_samples)
    if bool((valid_time_counts <= 0).any()):
        raise RuntimeError(
            "complete-valid-patch policy rejects a sample shorter than one patch"
        )
    positions = torch.arange(
        total_time_patches, device=valid_samples.device
    ).unsqueeze(0)
    temporal_valid = positions < valid_time_counts.unsqueeze(1)
    flat_valid = (
        temporal_valid.unsqueeze(-1)
        .expand(-1, -1, frequency_patches)
        .reshape(valid_samples.shape[0], -1)
    )
    return temporal_valid, ~flat_valid


def flatten_patch_grid(features: torch.Tensor) -> torch.Tensor:
    """[B,D,T,F] → [B,T*F,D], with time-major/frequency-minor order."""

    if features.ndim != 4:
        raise ValueError("patch features must be [B,D,T,F]")
    return features.flatten(2).transpose(1, 2).contiguous()


def restore_patch_grid(
    flattened_tokens: torch.Tensor,
    time_patches: int,
    frequency_patches: int,
) -> torch.Tensor:
    """[B,T*F,D] → [B,T,F,D] under the audited flatten order."""

    if flattened_tokens.ndim != 3:
        raise ValueError("flattened tokens must be [B,T*F,D]")
    if flattened_tokens.shape[1] != time_patches * frequency_patches:
        raise ValueError("flattened sequence does not match the patch grid")
    return flattened_tokens.reshape(
        flattened_tokens.shape[0],
        time_patches,
        frequency_patches,
        flattened_tokens.shape[2],
    )


def masked_temporal_mean(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2]:
        raise ValueError("tokens/token_mask shape mismatch")
    if token_mask.dtype != torch.bool:
        raise TypeError("token_mask must be bool with True=valid")
    denominator = token_mask.sum(dim=1, keepdim=True)
    if bool((denominator == 0).any()):
        raise RuntimeError("cannot pool a sample with zero valid temporal tokens")
    return (tokens * token_mask.unsqueeze(-1)).sum(dim=1) / denominator.to(
        tokens.dtype
    )


def build_time_map(
    token_mask: torch.Tensor,
    source_start_s: torch.Tensor,
    geometry: BEATsGeometry,
) -> torch.Tensor:
    """Build source-time half-open receptive intervals for complete patches."""

    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise TypeError("token_mask must be bool [B,L]")
    if source_start_s.shape != (token_mask.shape[0],):
        raise ValueError("source_start_s must be [B]")
    if source_start_s.device != token_mask.device:
        raise RuntimeError("source_start_s and token_mask must share a device")
    starts = (
        source_start_s.to(dtype=torch.float32).unsqueeze(1)
        + torch.arange(
            token_mask.shape[1],
            dtype=torch.float32,
            device=token_mask.device,
        ).unsqueeze(0)
        * geometry.temporal_stride_s
    )
    ends = starts + geometry.receptive_interval_s
    time_map = torch.stack((starts, ends), dim=-1)
    return torch.where(
        token_mask.unsqueeze(-1), time_map, torch.zeros_like(time_map)
    )


@dataclass(frozen=True)
class TemporalEncoderOutput:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    time_map: torch.Tensor
    pooled: torch.Tensor
    channel_order: tuple[str, ...]
    observation_mask: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        if self.tokens.ndim != 3 or not self.tokens.dtype.is_floating_point:
            raise TypeError("tokens must be floating [B,L,D]")
        batch, length, dimension = self.tokens.shape
        if self.token_mask.shape != (batch, length) or self.token_mask.dtype != torch.bool:
            raise TypeError("token_mask must be bool [B,L] with True=valid")
        if self.time_map.shape != (batch, length, 2) or self.time_map.dtype != torch.float32:
            raise TypeError("time_map must be float32 [B,L,2]")
        if self.pooled.shape != (batch, dimension):
            raise ValueError("pooled must be [B,D]")
        if self.channel_order != CHANNEL_ORDER:
            raise ValueError(f"channel_order must be {CHANNEL_ORDER}")
        expected_mask = (batch, length, len(CHANNEL_ORDER))
        if (
            self.observation_mask.shape != expected_mask
            or self.valid_mask.shape != expected_mask
            or self.observation_mask.dtype != torch.bool
            or self.valid_mask.dtype != torch.bool
        ):
            raise TypeError("observation_mask/valid_mask must be bool [B,L,4]")
        if bool((self.valid_mask & ~self.observation_mask).any()):
            raise ValueError("valid supervision must also be observed")
        if bool((self.valid_mask & ~self.token_mask.unsqueeze(-1)).any()):
            raise ValueError("invalid temporal token marked valid for supervision")
        for index in range(batch):
            valid_map = self.time_map[index, self.token_mask[index]]
            if valid_map.numel() and (
                bool((valid_map[:, 1] <= valid_map[:, 0]).any())
                or bool((valid_map[1:, 0] < valid_map[:-1, 0]).any())
            ):
                raise ValueError("time_map must be positive and monotonic")


@dataclass(frozen=True)
class HFInterval:
    """Deprecated low-level explicit-target fixture; not a raw HF adapter row."""

    channel: str
    start_s: float
    end_s: float
    state: ObservationState
    target: bool | None = None

    def __post_init__(self) -> None:
        if self.channel not in CHANNEL_ORDER:
            raise ValueError(f"unknown HF channel: {self.channel}")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("invalid HF interval")
        if self.state is ObservationState.OBSERVED and self.target is None:
            raise ValueError("observed intervals require an explicit target")
        if self.state is not ObservationState.OBSERVED and self.target is not None:
            raise ValueError("unobserved states cannot carry a target")


@dataclass(frozen=True)
class HFRawInterval:
    """One raw positive HF interval; the raw package has no target bool."""

    raw_token: str
    start_s: float
    end_s: float
    state: ObservationState = ObservationState.OBSERVED

    def __post_init__(self) -> None:
        if self.raw_token not in RAW_TOKEN_TO_CHANNEL:
            raise ValueError(f"unknown HF raw token: {self.raw_token}")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("invalid HF raw interval")
        if not isinstance(self.state, ObservationState):
            raise TypeError("state must be an explicit ObservationState")


@dataclass(frozen=True)
class HFTemporalSupervision:
    targets: torch.Tensor
    observation_mask: torch.Tensor
    valid_mask: torch.Tensor
    receipt: Mapping[str, object]


def _interval_membership(
    time_map: torch.Tensor,
    start_s: float,
    end_s: float,
    alignment: TokenAlignmentPolicy,
) -> torch.Tensor:
    if alignment is TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL:
        centers = time_map.mean(dim=-1)
        return (centers >= start_s) & (centers < end_s)
    if alignment is TokenAlignmentPolicy.ANY_OVERLAP_TEST_ONLY:
        return (
            torch.minimum(
                time_map[:, 1],
                torch.tensor(
                    end_s, dtype=time_map.dtype, device=time_map.device
                ),
            )
            - torch.maximum(
                time_map[:, 0],
                torch.tensor(
                    start_s, dtype=time_map.dtype, device=time_map.device
                ),
            )
        ) > 0
    raise ValueError(alignment)


def intervals_to_token_supervision(
    time_map: torch.Tensor,
    token_mask: torch.Tensor,
    intervals: Sequence[Sequence[HFInterval]],
    alignment: TokenAlignmentPolicy = TokenAlignmentPolicy.ANY_OVERLAP_TEST_ONLY,
) -> HFTemporalSupervision:
    """Low-level explicit fixture; never use this as the raw HF execution adapter."""

    if time_map.shape[:2] != token_mask.shape or time_map.shape[2] != 2:
        raise ValueError("time_map/token_mask shape mismatch")
    if len(intervals) != token_mask.shape[0]:
        raise ValueError("interval batch length mismatch")
    shape = (*token_mask.shape, len(CHANNEL_ORDER))
    targets = torch.zeros(shape, dtype=torch.float32, device=time_map.device)
    observed = torch.zeros(shape, dtype=torch.bool, device=time_map.device)
    valid = torch.zeros_like(observed)
    channel_index = {channel: index for index, channel in enumerate(CHANNEL_ORDER)}
    for batch_index, rows in enumerate(intervals):
        for row in rows:
            if row.state is not ObservationState.OBSERVED:
                continue
            index = channel_index[row.channel]
            overlap = _interval_membership(
                time_map[batch_index], row.start_s, row.end_s, alignment
            )
            overlap &= token_mask[batch_index]
            conflict = overlap & observed[batch_index, :, index] & (
                targets[batch_index, :, index] != float(bool(row.target))
            )
            if bool(conflict.any()):
                raise RuntimeError("conflicting observed targets overlap one token")
            targets[batch_index, overlap, index] = float(bool(row.target))
            observed[batch_index, overlap, index] = True
            valid[batch_index, overlap, index] = True
    return HFTemporalSupervision(
        targets,
        observed,
        valid,
        {
            "policy": "explicit_target_fixture_only",
            "alignment": alignment.value,
            "execution_eligible": False,
            "shared_label_eligible": False,
        },
    )


def raw_intervals_to_token_supervision(
    time_map: torch.Tensor,
    token_mask: torch.Tensor,
    intervals: Sequence[Sequence[HFRawInterval]],
    recording_states: Sequence[ObservationState],
    *,
    policy: HFTargetPolicy,
    alignment: TokenAlignmentPolicy = TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL,
) -> HFTemporalSupervision:
    """Build raw-positive-only or source-paper-native one-vs-rest targets."""

    if time_map.shape[:2] != token_mask.shape or time_map.shape[2] != 2:
        raise ValueError("time_map/token_mask shape mismatch")
    if time_map.device != token_mask.device:
        raise RuntimeError("time_map and token_mask must share a device")
    if len(intervals) != token_mask.shape[0] or len(recording_states) != len(
        intervals
    ):
        raise ValueError("HF interval/state batch length mismatch")
    if not isinstance(policy, HFTargetPolicy):
        raise TypeError("policy must be an explicit HFTargetPolicy")
    if not isinstance(alignment, TokenAlignmentPolicy):
        raise TypeError("alignment must be an explicit TokenAlignmentPolicy")

    shape = (*token_mask.shape, len(CHANNEL_ORDER))
    targets = torch.zeros(shape, dtype=torch.float32, device=time_map.device)
    observed = torch.zeros(shape, dtype=torch.bool, device=time_map.device)
    valid = torch.zeros_like(observed)
    channel_index = {channel: index for index, channel in enumerate(CHANNEL_ORDER)}
    empty_recordings = 0
    for batch_index, (rows, recording_state) in enumerate(
        zip(intervals, recording_states)
    ):
        if not isinstance(recording_state, ObservationState):
            raise TypeError("recording state must be an explicit ObservationState")
        if recording_state is ObservationState.EMPTY:
            empty_recordings += 1
            if rows:
                raise RuntimeError("EMPTY recording cannot contain raw intervals")
        elif recording_state is ObservationState.OBSERVED:
            if not rows:
                raise RuntimeError(
                    "recording with no raw intervals must be explicitly EMPTY"
                )
        elif rows:
            raise RuntimeError(
                f"{recording_state.value} recording cannot contain raw intervals"
            )

        if policy is HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR:
            if recording_state not in {
                ObservationState.OBSERVED,
                ObservationState.EMPTY,
            }:
                raise RuntimeError(
                    "paper-native rasterization is undefined for missing, "
                    "unknown, or not_annotated recordings"
                )
            source_task_valid = token_mask[batch_index].unsqueeze(-1).expand(
                -1, len(CHANNEL_ORDER)
            )
            observed[batch_index] = source_task_valid
            valid[batch_index] = source_task_valid

        for row in rows:
            if row.state is not ObservationState.OBSERVED:
                if policy is HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR:
                    raise RuntimeError(
                        "paper-native rasterization requires observed raw rows"
                    )
                continue
            index = channel_index[RAW_TOKEN_TO_CHANNEL[row.raw_token]]
            positive = _interval_membership(
                time_map[batch_index], row.start_s, row.end_s, alignment
            )
            positive &= token_mask[batch_index]
            targets[batch_index, positive, index] = 1.0
            if policy is HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY:
                observed[batch_index, positive, index] = True
                valid[batch_index, positive, index] = True

    constructed_negatives = int(valid.sum() - targets[valid].sum())
    positive_values = int(targets[valid].sum())
    if policy is HFTargetPolicy.RAW_CONSERVATIVE_POSITIVE_ONLY:
        negative_semantics = "none_raw_positive_only"
        detector_closed = False
        if constructed_negatives:
            raise RuntimeError("conservative policy unexpectedly created negatives")
    else:
        negative_semantics = "source_task_constructed_not_raw_normal"
        detector_closed = True
    return HFTemporalSupervision(
        targets,
        observed,
        valid,
        {
            "policy": policy.value,
            "alignment": alignment.value,
            "alignment_execution_default": (
                alignment is TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL
            ),
            "negative_semantics": negative_semantics,
            "raw_explicit_negative_intervals": 0,
            "constructed_negative_values": constructed_negatives,
            "positive_values": positive_values,
            "empty_recordings": empty_recordings,
            "detector_closed": detector_closed,
            "shared_label_eligible": False,
            "source_task_policy_reference": SOURCE_TASK_POLICY_REFERENCE,
            "source_task_policy_sections": ["5.2", "5.3"],
            "raw_token_to_channel": dict(RAW_TOKEN_TO_CHANNEL),
        },
    )


def attach_supervision(
    output: TemporalEncoderOutput, supervision: HFTemporalSupervision
) -> TemporalEncoderOutput:
    updated = replace(
        output,
        observation_mask=supervision.observation_mask,
        valid_mask=supervision.valid_mask,
    )
    updated.validate()
    return updated


def temporalize_transformer_output(
    flattened_tokens: torch.Tensor,
    valid_samples: torch.Tensor,
    total_samples: int,
    source_start_s: torch.Tensor,
    geometry: BEATsGeometry,
    transformer_padding_mask: torch.Tensor | None = None,
) -> TemporalEncoderOutput:
    """Restore Transformer tokens, aggregate frequency patches, and map time."""

    devices = {
        flattened_tokens.device,
        valid_samples.device,
        source_start_s.device,
    }
    if transformer_padding_mask is not None:
        devices.add(transformer_padding_mask.device)
    if len(devices) != 1:
        raise RuntimeError(f"temporal tensors must share one device, got {devices}")
    token_mask, exact_padding = exact_patch_masks(
        valid_samples, total_samples, geometry
    )
    if transformer_padding_mask is not None and not torch.equal(
        transformer_padding_mask, exact_padding
    ):
        raise RuntimeError("Transformer padding mask is not the exact patch mask")
    time_patches = token_mask.shape[1]
    frequency_patches = geometry.frequency_patches()
    grid = restore_patch_grid(
        flattened_tokens, time_patches, frequency_patches
    )
    temporal_tokens = grid.mean(dim=2)
    temporal_tokens = torch.where(
        token_mask.unsqueeze(-1),
        temporal_tokens,
        torch.zeros_like(temporal_tokens),
    )
    time_map = build_time_map(token_mask, source_start_s, geometry)
    pooled = masked_temporal_mean(temporal_tokens, token_mask)
    empty_masks = torch.zeros(
        (*token_mask.shape, len(CHANNEL_ORDER)),
        dtype=torch.bool,
        device=token_mask.device,
    )
    output = TemporalEncoderOutput(
        tokens=temporal_tokens,
        token_mask=token_mask,
        time_map=time_map,
        pooled=pooled,
        channel_order=CHANNEL_ORDER,
        observation_mask=empty_masks,
        valid_mask=empty_masks.clone(),
    )
    output.validate()
    return output


def verify_non_hf_pooled_parity(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Engineering gate for the P6↔P2 non-HF pooled interface."""

    if candidate.shape != reference.shape:
        raise RuntimeError("non-HF pooled parity shape mismatch")
    maximum = float((candidate - reference).abs().max()) if candidate.numel() else 0.0
    if not torch.allclose(candidate, reference, atol=atol, rtol=rtol):
        raise RuntimeError(f"non-HF pooled parity failed: max_abs={maximum}")
    return {
        "status": "pooled_parity_passed",
        "shape": list(candidate.shape),
        "max_abs": maximum,
        "atol": atol,
        "rtol": rtol,
    }


class BEATsTemporalAdapter(nn.Module):
    """Frozen BEATs wrapper that supplies an exact flattened patch mask."""

    def __init__(
        self,
        beats: nn.Module,
        geometry: BEATsGeometry | None = None,
    ) -> None:
        super().__init__()
        trainable = [
            name for name, parameter in beats.named_parameters() if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"BEATs must be frozen before wrapping: {trainable}")
        self.beats = beats.eval()
        self.geometry = geometry or BEATsGeometry.from_checkpoint_config(beats.cfg)
        model_devices = {
            tensor.device
            for tensor in (*tuple(beats.parameters()), *tuple(beats.buffers()))
        }
        if len(model_devices) != 1:
            raise RuntimeError(
                f"BEATs parameters/buffers must share one device, got {model_devices}"
            )
        if not model_devices:
            raise RuntimeError("cannot infer BEATs device from a parameterless model")
        self.model_device = next(iter(model_devices))
        actual_kernel = tuple(int(value) for value in beats.patch_embedding.kernel_size)
        actual_stride = tuple(int(value) for value in beats.patch_embedding.stride)
        expected_kernel = (
            self.geometry.patch_kernel_time,
            self.geometry.patch_kernel_frequency,
        )
        expected_stride = (
            self.geometry.patch_stride_time,
            self.geometry.patch_stride_frequency,
        )
        if actual_kernel != expected_kernel or actual_stride != expected_stride:
            raise RuntimeError(
                "checkpoint geometry and Conv2d disagree: "
                f"kernel {actual_kernel}/{expected_kernel}, "
                f"stride {actual_stride}/{expected_stride}"
            )

    def forward(self, batch: WaveformBatch) -> TemporalEncoderOutput:
        batch.validate()
        if batch.device != self.model_device:
            raise RuntimeError(
                f"WaveformBatch is on {batch.device}, BEATs is on "
                f"{self.model_device}; call batch.to({self.model_device!s}) explicitly"
            )
        with torch.no_grad():
            fbank = self.beats.preprocess(batch.waveform)
            if fbank.ndim != 3 or fbank.shape[2] != self.geometry.mel_bins:
                raise RuntimeError("BEATs preprocess returned unexpected fbank geometry")
            expected_frames = self.geometry.fbank_frames(batch.waveform.shape[1])
            if fbank.shape[1] != expected_frames:
                raise RuntimeError(
                    f"fbank frame count mismatch: {fbank.shape[1]} != {expected_frames}"
                )
            patch_features = self.beats.patch_embedding(fbank.unsqueeze(1))
            flattened = flatten_patch_grid(patch_features)
            temporal_valid, flat_padding = exact_patch_masks(
                batch.valid_samples,
                batch.waveform.shape[1],
                self.geometry,
            )
            expected_grid = (
                temporal_valid.shape[1],
                self.geometry.frequency_patches(),
            )
            if patch_features.shape[2:] != expected_grid:
                raise RuntimeError(
                    f"Conv2d patch grid mismatch: {patch_features.shape[2:]} "
                    f"!= {expected_grid}"
                )
            flattened = self.beats.layer_norm(flattened)
            if self.beats.post_extract_proj is not None:
                flattened = self.beats.post_extract_proj(flattened)
            flattened = self.beats.dropout_input(flattened)
            encoded, _ = self.beats.encoder(
                flattened, padding_mask=flat_padding
            )
        return temporalize_transformer_output(
            encoded,
            batch.valid_samples,
            batch.waveform.shape[1],
            batch.source_start_s,
            self.geometry,
            transformer_padding_mask=flat_padding,
        )
