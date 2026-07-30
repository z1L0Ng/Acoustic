"""Audited AudioSet-only and random-init BEATs encoders."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.encoder import (
    BACKBONE_SIZE_BYTES,
    sha256_file,
    verify_source_repo,
)
from baseline.pafa.checkpoint_eval.bootstrap import BACKBONE_SHA256


SEED = 20260728
REPRESENTATIONS = (
    "r1_beats_as2m_audioset_only",
    "r2_beats_random_init_sanity",
)


class FrozenBEATs(torch.nn.Module):
    def __init__(self, beats: torch.nn.Module) -> None:
        super().__init__()
        self.beats = beats

    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask=None,
        training: bool = False,
    ) -> torch.Tensor:
        del training
        values, _ = self.beats.extract_features(waveform, padding_mask)
        return values


def _canonical_tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _config_digest(config: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_representation_encoder(
    representation: str,
    source_repo: Path,
    backbone_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    source_repo = verify_source_repo(source_repo)
    backbone = backbone_path.resolve()
    if (
        backbone.stat().st_size != BACKBONE_SIZE_BYTES
        or sha256_file(backbone) != BACKBONE_SHA256
    ):
        raise RuntimeError("BEATs checkpoint identity mismatch")
    sys.path.insert(0, str(source_repo))
    from BEATs.BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(backbone, map_location="cpu")
    config_values = {
        **checkpoint["cfg"],
        "predictor_class": 4,
        "finetuned_model": False,
        "spec_transform": None,
    }
    config = BEATsConfig(config_values)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="User provided device_type of 'cuda'.*CUDA is not available.*",
            category=UserWarning,
        )
        beats = BEATs(config)
    initial_state = beats.state_dict()
    pretrained_state = checkpoint["model"]
    if set(initial_state) != set(pretrained_state):
        raise RuntimeError("R1/R2 architecture key mismatch")
    if any(
        initial_state[key].shape != pretrained_state[key].shape
        or initial_state[key].dtype != pretrained_state[key].dtype
        for key in initial_state
    ):
        raise RuntimeError("R1/R2 architecture tensor signature mismatch")
    random_digest = _canonical_tensor_digest(initial_state)
    pretrained_digest = _canonical_tensor_digest(pretrained_state)
    loaded_tensors = 0
    if representation == "r1_beats_as2m_audioset_only":
        loaded = beats.load_state_dict(pretrained_state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("AudioSet-only strict load failed")
        loaded_tensors = len(pretrained_state)
    elif random_digest == pretrained_digest:
        raise RuntimeError("random-init state unexpectedly equals AudioSet state")
    del checkpoint
    for parameter in beats.parameters():
        parameter.requires_grad_(False)
    model = FrozenBEATs(beats).to(device).eval()
    state_digest = _canonical_tensor_digest(model.beats.state_dict())
    expected_digest = (
        pretrained_digest
        if representation == "r1_beats_as2m_audioset_only"
        else random_digest
    )
    if state_digest != expected_digest:
        raise RuntimeError("final representation state digest mismatch")
    return model, {
        "representation": representation,
        "architecture": "BEATs_iter3_plus_AS2M config; 12 layers; 768 dim",
        "config_sha256": _config_digest(config_values),
        "architecture_tensor_keys": len(initial_state),
        "loaded_pretrained_tensors": loaded_tensors,
        "beats_checkpoint_sha256": BACKBONE_SHA256,
        "beats_checkpoint_size_bytes": backbone.stat().st_size,
        "pretrained_state_digest": pretrained_digest,
        "random_initial_state_digest": random_digest,
        "final_state_digest": state_digest,
        "encoder_frozen": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "source_task_states_present": False,
        "seed": SEED if representation == "r2_beats_random_init_sanity" else None,
        "claim": (
            "AudioSet-only frozen representation"
            if representation == "r1_beats_as2m_audioset_only"
            else "random-feature sanity floor; no pretrained tensors loaded"
        ),
    }
