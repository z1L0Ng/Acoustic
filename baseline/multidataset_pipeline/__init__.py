"""Reusable engineering contracts for the 2026-08-20 multi-dataset core."""

from .contracts import (
    SAMPLE_RATE,
    ObservationState,
    WaveformBatch,
    WaveformSample,
    collate_waveforms,
)
from .hf_data import HFSampleRecord, build_hf_manifest, load_hf_waveform
from .adapter_factory import (
    AdapterFactoryConfig,
    audit_local_adapter_assets,
    build_production_adapter,
)
from .preflight import (
    P1_P5_SELECTION_RULE,
    P1_P5_UPDATE_BUDGET,
    P6TokenTemporalHead,
    SharedWindowEncoderOutput,
    freeze_receipt,
    hf_masked_channel_balanced_bce,
)
from .sliding_window import (
    SlidingWindowBatch,
    collate_sliding_windows,
    hf_window_supervision,
    masked_mean_window_embeddings,
)
from .real_subtrain_provider import (
    FrozenNativeUnit,
    FrozenProviderIndex,
    NativeWindowBatch,
    build_frozen_provider_index,
    build_real_subtrain_preflight_batches,
)

__all__ = [
    "SAMPLE_RATE",
    "ObservationState",
    "WaveformBatch",
    "WaveformSample",
    "collate_waveforms",
    "HFSampleRecord",
    "build_hf_manifest",
    "load_hf_waveform",
    "AdapterFactoryConfig",
    "audit_local_adapter_assets",
    "build_production_adapter",
    "P1_P5_SELECTION_RULE",
    "P1_P5_UPDATE_BUDGET",
    "P6TokenTemporalHead",
    "SharedWindowEncoderOutput",
    "freeze_receipt",
    "hf_masked_channel_balanced_bce",
    "SlidingWindowBatch",
    "collate_sliding_windows",
    "hf_window_supervision",
    "masked_mean_window_embeddings",
    "FrozenNativeUnit",
    "FrozenProviderIndex",
    "NativeWindowBatch",
    "build_frozen_provider_index",
    "build_real_subtrain_preflight_batches",
]
