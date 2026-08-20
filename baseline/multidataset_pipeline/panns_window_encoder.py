"""P3 official PANNs Cnn14_16k pooled-window production binding.

The official repository publishes a 16 kHz Cnn14 checkpoint variant.  This
adapter therefore does not resample the frozen 16 kHz source-window contract.
The 2048→768 dimension adapter remains trainable and is part of the P3 package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .preflight import CandidateDimensionAdapter
from .window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    require_clean_source_revision,
    require_file_identity,
)


PANNS_IDENTITY = "PANNs_Cnn14"
PANNS_SOURCE_URL = "https://github.com/qiuqiangkong/audioset_tagging_cnn"
PANNS_SOURCE_LICENSE = "MIT"
PANNS_SOURCE_REVISION = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
PANNS_CHECKPOINT_NAME = "Cnn14_16k_mAP=0.438.pth"
PANNS_CHECKPOINT_SOURCE = "https://zenodo.org/records/3987831"
PANNS_CHECKPOINT_SHA256 = "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
PANNS_CHECKPOINT_SIZE_BYTES = 358_668_570
PANNS_MODEL_CLASS = "Cnn14_16k"
PANNS_WINDOW_SAMPLES = 32_000


class PANNsWindowBackend(FrozenWindowBackend):
    native_dim = 2_048

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        if waveform_windows.ndim != 2 or waveform_windows.shape[1] != PANNS_WINDOW_SAMPLES:
            raise ValueError("PANNs Cnn14_16k requires [N,32000] source windows")
        if bool((valid_samples <= 0).any()) or bool((valid_samples > PANNS_WINDOW_SAMPLES).any()):
            raise ValueError("invalid PANNs valid_samples")
        with torch.inference_mode():
            output = self.model(waveform_windows, None)
        if not isinstance(output, dict) or "embedding" not in output:
            raise RuntimeError("official PANNs output must contain embedding")
        values = output["embedding"]
        if values.shape != (waveform_windows.shape[0], self.native_dim):
            raise RuntimeError(f"PANNs encoder returned {tuple(values.shape)}")
        return values.to(torch.float32)


def _load_panns(
    source_repo: Path,
    source_revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    device: torch.device,
    *,
    verify_historical_identity: bool = True,
) -> nn.Module:
    if verify_historical_identity:
        if len(source_revision) != 40:
            raise ValueError("PANNs requires an exact 40-character source revision")
        require_clean_source_revision(source_repo, source_revision)
        require_file_identity(
            checkpoint,
            checkpoint_sha256,
            expected_size_bytes=PANNS_CHECKPOINT_SIZE_BYTES,
        )
    elif not source_repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("local PANNs source repo or checkpoint is missing")
    pytorch_dir = str((source_repo / "pytorch").resolve())
    utils_dir = str((source_repo / "utils").resolve())
    for path in (utils_dir, pytorch_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from models import Cnn14_16k
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PANNs production binding requires the pinned official source and "
            "its torchlibrosa/librosa dependencies"
        ) from error
    model = Cnn14_16k(
        sample_rate=16_000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8_000,
        classes_num=527,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError("official PANNs checkpoint must contain model state")
    loaded = model.load_state_dict(state["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"PANNs checkpoint state mismatch: {loaded}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def build_panns_window_encoder(
    source_repo: Path,
    source_revision: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    device: torch.device | str = "cpu",
) -> ProductionWindowEncoder:
    target = torch.device(device)
    model = _load_panns(
        source_repo, source_revision, checkpoint, checkpoint_sha256, target
    )
    backend = PANNsWindowBackend(model).to(target)
    provenance = AdapterProvenance(
        encoder_identity=PANNS_IDENTITY,
        source_url=PANNS_SOURCE_URL,
        source_revision=source_revision,
        source_license=PANNS_SOURCE_LICENSE,
        checkpoint_name=checkpoint.name,
        checkpoint_source=PANNS_CHECKPOINT_SOURCE,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        license_boundary="official source is MIT; checkpoint redistribution terms require review",
    )
    return ProductionWindowEncoder(
        PANNS_IDENTITY,
        backend,
        provenance,
        dimension_adapter=CandidateDimensionAdapter(PANNS_IDENTITY).to(target),
    )


def load_local_panns_window_backend(
    source_repo: Path,
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> PANNsWindowBackend:
    """Load official local Cnn14_16k without checksum gates."""

    target = torch.device(device)
    model = _load_panns(
        source_repo,
        "local_official_public_source",
        checkpoint,
        "not_used",
        target,
        verify_historical_identity=False,
    )
    return PANNsWindowBackend(model).to(target).eval()
