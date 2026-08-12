"""P1 AST pooled-window production binding for the shared 2 s contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .preflight import CandidateDimensionAdapter
from .window_encoder import (
    AdapterProvenance,
    FrozenWindowBackend,
    ProductionWindowEncoder,
    require_clean_source_revision,
    require_file_identity,
)


AST_IDENTITY = "AST"
AST_SOURCE_URL = "https://github.com/raymin0223/patch-mix_contrastive_learning"
AST_SOURCE_REVISION = "836b09fea1b70eb29fe0b25afa481286b56f5104"
AST_SOURCE_LICENSE = "MIT"
AST_CHECKPOINT_SOURCE = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_CHECKPOINT_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
AST_CHECKPOINT_SHA256 = "bc9fe72b1a38b7071db8b606c63f8f2e41bf2cccaf3e80fc0ba5c33094877cb1"
AST_CHECKPOINT_SIZE_BYTES = 346_425_476
AST_INPUT_FRAMES = 798
AST_MEL_BINS = 128
AST_WINDOW_SAMPLES = 32_000


class ASTWindowBackend(FrozenWindowBackend):
    """AudioSet AST with 2 s fbank content zero-padded to the 798-frame grid."""

    native_dim = 768

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def frontend(waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or waveforms.shape[1] != AST_WINDOW_SAMPLES:
            raise ValueError("AST frontend requires [N,32000] source windows at 16 kHz")
        try:
            import torchaudio.compliance.kaldi as ta_kaldi
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "AST production frontend requires a working torchaudio installation"
            ) from error
        rows = []
        for waveform in waveforms:
            fbank = ta_kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16_000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=AST_MEL_BINS,
                dither=0.0,
                frame_shift=10,
            )
            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
            if fbank.shape[0] > AST_INPUT_FRAMES:
                raise RuntimeError("AST frontend would need truncation; contract forbids it")
            rows.append(F.pad(fbank, (0, 0, 0, AST_INPUT_FRAMES - fbank.shape[0])))
        output = torch.stack(rows).unsqueeze(1).to(torch.float32)
        if output.shape != (waveforms.shape[0], 1, AST_INPUT_FRAMES, AST_MEL_BINS):
            raise RuntimeError("AST frontend geometry mismatch")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("AST frontend produced non-finite values")
        return output

    def encode_valid_windows(
        self, waveform_windows: torch.Tensor, valid_samples: torch.Tensor
    ) -> torch.Tensor:
        if bool((valid_samples <= 0).any()) or bool((valid_samples > AST_WINDOW_SAMPLES).any()):
            raise ValueError("invalid AST source valid_samples")
        images = self.frontend(waveform_windows)
        with torch.inference_mode():
            values = self.encoder(images)
        if values.shape != (waveform_windows.shape[0], self.native_dim):
            raise RuntimeError(f"AST encoder returned {tuple(values.shape)}")
        return values.to(torch.float32)


def _load_ast_model(source_repo: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    require_clean_source_revision(source_repo, AST_SOURCE_REVISION)
    require_file_identity(
        checkpoint,
        AST_CHECKPOINT_SHA256,
        expected_size_bytes=AST_CHECKPOINT_SIZE_BYTES,
    )
    source = str(source_repo.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.ast import ASTModel

    model = ASTModel(
        label_dim=527,
        fstride=10,
        tstride=10,
        input_fdim=AST_MEL_BINS,
        input_tdim=1024,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or len(state) != 155:
        raise RuntimeError("AST compatibility checkpoint must contain 155 tensors")
    normalized = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    incompatible = model.load_state_dict(normalized, strict=False)
    allowed_missing = {
        "v.head.weight",
        "v.head.bias",
        "v.head_dist.weight",
        "v.head_dist.bias",
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"AST checkpoint state mismatch: {incompatible}")
    source_positions = model.v.pos_embed[:, 2:, :].detach()
    if source_positions.shape != (1, 1212, 768):
        raise RuntimeError("AST AudioSet positional grid mismatch")
    time_patches = (AST_INPUT_FRAMES - 16) // 10 + 1
    positions = source_positions.transpose(1, 2).reshape(1, 768, 12, 101)
    start = 50 - time_patches // 2
    positions = positions[:, :, :, start : start + time_patches]
    positions = positions.reshape(1, 768, 12 * time_patches).transpose(1, 2)
    model.v.pos_embed = nn.Parameter(
        torch.cat([model.v.pos_embed[:, :2, :].detach(), positions], dim=1),
        requires_grad=False,
    )
    model.v.patch_embed.num_patches = 12 * time_patches
    model.mlp_head = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def build_ast_window_encoder(
    source_repo: Path,
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> ProductionWindowEncoder:
    target = torch.device(device)
    backend = ASTWindowBackend(_load_ast_model(source_repo, checkpoint, target)).to(target)
    provenance = AdapterProvenance(
        encoder_identity=AST_IDENTITY,
        source_url=AST_SOURCE_URL,
        source_revision=AST_SOURCE_REVISION,
        source_license=AST_SOURCE_LICENSE,
        checkpoint_name=checkpoint.name,
        checkpoint_source=f"{AST_CHECKPOINT_SOURCE}@{AST_CHECKPOINT_REVISION}",
        checkpoint_sha256=AST_CHECKPOINT_SHA256,
        checkpoint_size_bytes=AST_CHECKPOINT_SIZE_BYTES,
        license_boundary=(
            "source code is MIT; model source is the pinned Hugging Face AudioSet "
            "artifact converted to legacy keys; review model terms before redistribution"
        ),
    )
    dimension = CandidateDimensionAdapter(AST_IDENTITY).to(target)
    return ProductionWindowEncoder(
        AST_IDENTITY, backend, provenance, dimension_adapter=dimension
    )
