"""Reusable engineering contracts for the 2026-08-20 multi-dataset core."""

from .contracts import (
    SAMPLE_RATE,
    ObservationState,
    WaveformBatch,
    WaveformSample,
    collate_waveforms,
)
from .hf_data import HFSampleRecord, build_hf_manifest, load_hf_waveform

__all__ = [
    "SAMPLE_RATE",
    "ObservationState",
    "WaveformBatch",
    "WaveformSample",
    "collate_waveforms",
    "HFSampleRecord",
    "build_hf_manifest",
    "load_hf_waveform",
]
