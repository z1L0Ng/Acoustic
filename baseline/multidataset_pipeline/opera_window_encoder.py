"""P5 official OPERA-CT 8-second frontend production binding.

The matched benchmark still supplies 16 kHz, 2-second source-time windows.
Each valid window is zero-padded exactly once to the official 8-second OPERA
frontend grid.  Padding never changes ``valid_samples`` or the shared source
``time_map``; no repeat padding or truncation is permitted.

OPERA-CT is an overlap-aware reference: its standard pretraining includes
ICBHI and HF Lung.  Whether KAUH entered the published checkpoint is unknown.
It is therefore not a clean cross-dataset generalization candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    require_clean_source_revision,
    require_file_identity,
)


OPERA_IDENTITY = "OPERA_CT"
OPERA_SOURCE_URL = "https://github.com/evelyn0414/OPERA.git"
OPERA_SOURCE_LICENSE = "MIT"
OPERA_SOURCE_REVISION = "3622310e667afb8aa40169050b4dd45de75946a2"
OPERA_CHECKPOINT_NAME = "encoder-operaCT.ckpt"
OPERA_CHECKPOINT_SOURCE = (
    "https://huggingface.co/evelyn0414/OPERA/resolve/"
    "d8de4322870b596f0a6ff6ea907b9a6996cd243a/encoder-operaCT.ckpt"
)
OPERA_CHECKPOINT_LICENSE = "CC-BY-NC-4.0"
OPERA_CHECKPOINT_SHA256 = "83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae"
OPERA_CHECKPOINT_SIZE_BYTES = 355_598_886
OPERA_SOURCE_WINDOW_SAMPLES = 32_000
OPERA_FRONTEND_SAMPLES = 128_000
OPERA_FRONTEND_FRAMES = 251
OPERA_MEL_BINS = 64


def _official_opera_mel_frontend(waveforms: torch.Tensor) -> torch.Tensor:
    """Mirror pinned ``src.util.pre_process_audio_mel_t(..., f_max=8000)``."""

    try:
        import librosa
    except (ImportError, OSError) as error:
        raise RuntimeError("OPERA frontend requires the pinned librosa runtime") from error
    rows = []
    for waveform in waveforms.detach().cpu():
        audio = np.asarray(waveform.numpy(), dtype=np.float32)
        spectrum = librosa.feature.melspectrogram(
            y=audio,
            sr=16_000,
            n_mels=OPERA_MEL_BINS,
            fmin=50,
            fmax=8_000,
            n_fft=1_024,
            hop_length=512,
        )
        spectrum = librosa.power_to_db(spectrum, ref=np.max)
        if spectrum.max() != spectrum.min():
            spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
        mel = np.ascontiguousarray(spectrum.T, dtype=np.float32)
        if mel.shape != (OPERA_FRONTEND_FRAMES, OPERA_MEL_BINS):
            raise RuntimeError(f"OPERA frontend returned {mel.shape}")
        rows.append(torch.from_numpy(mel))
    return torch.stack(rows).to(waveforms.device)


class OPERAWindowBackend(FrozenWindowBackend):
    native_dim = 768

    def __init__(
        self,
        model: nn.Module,
        *,
        frontend: Callable[[torch.Tensor], torch.Tensor] = _official_opera_mel_frontend,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.frontend = frontend
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def zero_pad_to_frontend_grid(
        waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        if waveform_windows.ndim != 2 or waveform_windows.shape[1] != OPERA_SOURCE_WINDOW_SAMPLES:
            raise ValueError("OPERA-CT requires [N,32000] shared source windows")
        if valid_samples.shape != (waveform_windows.shape[0],) or valid_samples.dtype != torch.long:
            raise TypeError("OPERA valid_samples must be int64 [N]")
        if bool((valid_samples <= 0).any()) or bool(
            (valid_samples > OPERA_SOURCE_WINDOW_SAMPLES).any()
        ):
            raise ValueError("invalid OPERA source valid_samples")
        positions = torch.arange(
            OPERA_SOURCE_WINDOW_SAMPLES, device=waveform_windows.device
        ).unsqueeze(0)
        source_valid = positions < valid_samples.unsqueeze(1)
        source = torch.where(source_valid, waveform_windows, torch.zeros_like(waveform_windows))
        padded = torch.zeros(
            waveform_windows.shape[0],
            OPERA_FRONTEND_SAMPLES,
            dtype=waveform_windows.dtype,
            device=waveform_windows.device,
        )
        padded[:, :OPERA_SOURCE_WINDOW_SAMPLES] = source
        return padded

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        padded = self.zero_pad_to_frontend_grid(waveform_windows, valid_samples)
        mel = self.frontend(padded)
        if mel.shape != (
            waveform_windows.shape[0],
            OPERA_FRONTEND_FRAMES,
            OPERA_MEL_BINS,
        ):
            raise RuntimeError(f"OPERA mel frontend returned {tuple(mel.shape)}")
        with torch.inference_mode():
            values = self.model(mel.unsqueeze(1))
        if values.shape != (waveform_windows.shape[0], self.native_dim):
            raise RuntimeError(f"OPERA-CT encoder returned {tuple(values.shape)}")
        return values.to(torch.float32)


def _load_opera(
    source_repo: Path,
    source_revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    device: torch.device,
) -> nn.Module:
    require_clean_source_revision(source_repo, source_revision)
    require_file_identity(
        checkpoint,
        checkpoint_sha256,
        expected_size_bytes=OPERA_CHECKPOINT_SIZE_BYTES,
    )
    source_root = str(source_repo.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from src.model.htsat.htsat import HTSATWrapper
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "OPERA-CT requires pinned official source plus timm/torchlibrosa dependencies"
        ) from error
    model = HTSATWrapper()
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("official OPERA checkpoint must contain a Lightning state_dict")
    prefix = "encoder.encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    if not encoder_state or len(encoder_state) != 200:
        raise RuntimeError("OPERA checkpoint encoder state prefix/count changed")
    loaded = model.load_state_dict(encoder_state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"OPERA checkpoint encoder state mismatch: {loaded}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def build_opera_window_encoder(
    source_repo: Path,
    source_revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    device: torch.device | str = "cpu",
) -> ProductionWindowEncoder:
    target = torch.device(device)
    if source_revision != OPERA_SOURCE_REVISION:
        raise RuntimeError("OPERA source revision differs from the audited revision")
    if checkpoint_sha256 != OPERA_CHECKPOINT_SHA256:
        raise RuntimeError("OPERA checkpoint SHA differs from the audited identity")
    model = _load_opera(
        source_repo, source_revision, checkpoint, checkpoint_sha256, target
    )
    provenance = AdapterProvenance(
        encoder_identity=OPERA_IDENTITY,
        source_url=OPERA_SOURCE_URL,
        source_revision=source_revision,
        source_license=OPERA_SOURCE_LICENSE,
        checkpoint_name=checkpoint.name,
        checkpoint_source=OPERA_CHECKPOINT_SOURCE,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        license_boundary=(
            "checkpoint repository is CC-BY-NC-4.0; no redistribution; "
            "overlap-aware reference only because ICBHI and HF Lung entered standard pretraining; "
            "KAUH checkpoint provenance is unknown"
        ),
    )
    return ProductionWindowEncoder(
        OPERA_IDENTITY,
        OPERAWindowBackend(model).to(target),
        provenance,
    )
