"""P2 BEATs pooled-window binding reusing the exact patch-mask adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .beats_temporal import BEATsTemporalAdapter
from .contracts import PREDICTION_UNITS, WaveformBatch
from .preflight import CandidateDimensionAdapter
from .window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    require_clean_source_revision,
    require_file_identity,
)


BEATS_IDENTITY = "BEATs"
BEATS_SOURCE_URL = "https://github.com/wa976/PAFA"
BEATS_SOURCE_REVISION = "e49e294d0db0d6af10ac46290512b9c85d3f71e1"
BEATS_SOURCE_LICENSE = "no LICENSE in audited PAFA checkout; vendored BEATs header says MIT"
BEATS_CHECKPOINT_SOURCE = "BEATs_iter3_plus_AS2M AudioSet mirror from accepted PAFA audit"
BEATS_CHECKPOINT_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
BEATS_CHECKPOINT_SIZE_BYTES = 361_499_833
BEATS_WINDOW_SAMPLES = 32_000


class BEATsWindowBackend(FrozenWindowBackend):
    native_dim = 768

    def __init__(self, beats: nn.Module) -> None:
        super().__init__()
        self.temporal = BEATsTemporalAdapter(beats)

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        count, width = waveform_windows.shape
        if width != BEATS_WINDOW_SAMPLES:
            raise ValueError("BEATs window backend requires [N,32000]")
        starts = torch.zeros(count, dtype=torch.float64, device=waveform_windows.device)
        ends = valid_samples.to(torch.float64) / 16_000
        batch = WaveformBatch(
            waveform=waveform_windows,
            waveform_padding_mask=(
                torch.arange(width, device=waveform_windows.device).unsqueeze(0)
                >= valid_samples.unsqueeze(1)
            ),
            valid_samples=valid_samples,
            sample_rate=16_000,
            source_start_s=starts,
            source_end_s=ends,
            sample_ids=tuple(f"window-{index}" for index in range(count)),
            dataset_ids=tuple("ICBHI" for _ in range(count)),
            prediction_units=tuple(PREDICTION_UNITS["ICBHI"] for _ in range(count)),
            lineage=tuple({"internal": "flattened_shared_window"} for _ in range(count)),
        )
        batch.validate()
        return self.temporal(batch).pooled.to(torch.float32)


def _load_beats(source_repo: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    require_clean_source_revision(source_repo, BEATS_SOURCE_REVISION)
    require_file_identity(
        checkpoint,
        BEATS_CHECKPOINT_SHA256,
        expected_size_bytes=BEATS_CHECKPOINT_SIZE_BYTES,
    )
    source = str(source_repo.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from BEATs.BEATs import BEATs, BEATsConfig

    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or set(state) != {"cfg", "model"}:
        raise RuntimeError("BEATs checkpoint must contain exactly cfg/model")
    model = BEATs(BEATsConfig(state["cfg"]))
    loaded = model.load_state_dict(state["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"BEATs checkpoint state mismatch: {loaded}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def build_beats_window_encoder(
    source_repo: Path,
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> ProductionWindowEncoder:
    target = torch.device(device)
    backend = BEATsWindowBackend(_load_beats(source_repo, checkpoint, target)).to(target)
    provenance = AdapterProvenance(
        encoder_identity=BEATS_IDENTITY,
        source_url=BEATS_SOURCE_URL,
        source_revision=BEATS_SOURCE_REVISION,
        source_license=BEATS_SOURCE_LICENSE,
        checkpoint_name=checkpoint.name,
        checkpoint_source=BEATS_CHECKPOINT_SOURCE,
        checkpoint_sha256=BEATS_CHECKPOINT_SHA256,
        checkpoint_size_bytes=BEATS_CHECKPOINT_SIZE_BYTES,
        license_boundary=(
            "audited PAFA repo lacks a repository LICENSE; vendored BEATs source header says "
            "MIT; do not redistribute source/checkpoint until license review"
        ),
    )
    return ProductionWindowEncoder(
        BEATS_IDENTITY,
        backend,
        provenance,
        dimension_adapter=CandidateDimensionAdapter(BEATS_IDENTITY).to(target),
    )
