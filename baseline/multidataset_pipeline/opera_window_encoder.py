"""Official OPERA-CT pooled-window binding for the 2s source contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .window_encoder import FrozenWindowBackend


OPERA_IDENTITY = "OPERA_CT"
OPERA_SOURCE_URL = "https://github.com/evelyn0414/OPERA"
OPERA_CHECKPOINT_SOURCE = (
    "https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCT.ckpt"
)
SOURCE_WINDOW_SAMPLES = 32_000
PACKAGE_WINDOW_SAMPLES = 128_000


class OPERAWindowBackend(FrozenWindowBackend):
    native_dim = 768

    def __init__(self, model: torch.nn.Module, preprocess) -> None:
        super().__init__()
        self.model = model.eval()
        self.preprocess = preprocess
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        if waveform_windows.ndim != 2 or waveform_windows.shape[1] != SOURCE_WINDOW_SAMPLES:
            raise ValueError("OPERA-CT package requires [N,32000] source windows")
        padded = torch.zeros(
            waveform_windows.shape[0],
            PACKAGE_WINDOW_SAMPLES,
            dtype=torch.float32,
        )
        padded[:, :SOURCE_WINDOW_SAMPLES] = waveform_windows.detach().cpu()
        spectrograms = np.stack(
            [
                self.preprocess(row.numpy(), f_max=8_000)
                for row in padded
            ]
        ).astype(np.float32)
        values = torch.from_numpy(spectrograms).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            embeddings = self.model.extract_feature(values, self.native_dim)
        return embeddings.to(torch.float32)


def load_local_opera_ct_window_backend(
    source_repo: Path,
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> OPERAWindowBackend:
    if not source_repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("official OPERA source/checkpoint is missing")
    source = str(source_repo.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from src.benchmark.model_util import initialize_pretrained_model
    from src.util import pre_process_audio_mel_t

    target = torch.device(device)
    model = initialize_pretrained_model("operaCT")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    return OPERAWindowBackend(model.to(target).eval(), pre_process_audio_mel_t)
