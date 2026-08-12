"""Reusable engineering contracts for the 2026-08-20 multi-dataset core."""

from .contracts import (
    SAMPLE_RATE,
    ObservationState,
    WaveformBatch,
    WaveformSample,
    collate_waveforms,
)
from .hf_data import HFSampleRecord, build_hf_manifest, load_hf_waveform
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

__all__ = [
    "SAMPLE_RATE",
    "ObservationState",
    "WaveformBatch",
    "WaveformSample",
    "collate_waveforms",
    "HFSampleRecord",
    "build_hf_manifest",
    "load_hf_waveform",
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
]
