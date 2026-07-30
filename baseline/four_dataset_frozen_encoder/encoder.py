"""PAFA encoder loading and deterministic four-dataset embedding extraction."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms

from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.pafa.checkpoint_eval.bootstrap import (
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_CHECKPOINT_SIZE_BYTES,
    BACKBONE_SHA256,
    REPO_COMMIT,
    verify_checkpoint_identity,
    verify_checkpoint_state,
)


BACKBONE_SIZE_BYTES = 361_499_833


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_repo(source_repo: Path) -> Path:
    import subprocess

    repo = source_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != REPO_COMMIT or status:
        raise RuntimeError("PAFA source commit/status gate failed")
    return repo


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def build_encoder(
    source_repo: Path,
    checkpoint_path: Path,
    backbone_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    source_repo = verify_source_repo(source_repo)
    checkpoint = checkpoint_path.resolve()
    backbone = backbone_path.resolve()
    task_sha = verify_checkpoint_identity(checkpoint, ACCEPTED_CHECKPOINT_SHA256)
    if checkpoint.stat().st_size != ACCEPTED_CHECKPOINT_SIZE_BYTES:
        raise RuntimeError("PAFA task checkpoint size mismatch")
    if backbone.stat().st_size != BACKBONE_SIZE_BYTES or sha256_file(backbone) != BACKBONE_SHA256:
        raise RuntimeError("PAFA BEATs checkpoint identity mismatch")
    sys.path.insert(0, str(source_repo))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        from models.beats import BEATsTransferLearningModel

    state = torch.load(checkpoint, map_location="cpu")
    state_counts = verify_checkpoint_state(state)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="User provided device_type of 'cuda'.*CUDA is not available.*",
            category=UserWarning,
        )
        model = BEATsTransferLearningModel(
            num_target_classes=4,
            model_path=str(backbone),
            ft_entire_network=True,
            spec_transform=None,
        )
    loaded = model.load_state_dict(normalize_state(state["model"]), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"PAFA encoder state mismatch: {loaded}")
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device).eval()
    return model, {
        "author_repo_commit": REPO_COMMIT,
        "task_checkpoint_sha256": task_sha,
        "task_checkpoint_size_bytes": checkpoint.stat().st_size,
        "beats_checkpoint_sha256": BACKBONE_SHA256,
        "beats_checkpoint_size_bytes": backbone.stat().st_size,
        "checkpoint_state_verification": state_counts,
        "encoder_frozen": True,
        "discarded_source_states": ["classifier", "projector"],
        "embedding_definition": "mean over PAFA BEATs pre-classifier frame representations",
        "selection_caveat": "ICBHI official-test-selected epoch 27",
    }


def _waveforms_for_sample(
    waveform: torch.Tensor,
    sample_rate: int,
    sample: Sample,
    cut_pad,
) -> list[torch.Tensor]:
    target_rate = 16_000
    if waveform.ndim != 2:
        raise RuntimeError("expected channel-first waveform")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_rate:
        waveform = audio_transforms.Resample(sample_rate, target_rate)(waveform)
    waveform = audio_transforms.Fade(
        fade_in_len=target_rate // 16,
        fade_out_len=target_rate // 16,
        fade_shape="linear",
    )(waveform)
    if sample.crop_start_s is not None and sample.crop_end_s is not None:
        start = int(sample.crop_start_s * target_rate)
        end = int(sample.crop_end_s * target_rate)
        if start < 0 or end <= start or end > waveform.shape[1] + 1:
            raise RuntimeError(f"invalid crop for {sample.sample_id}")
        waveform = waveform[:, start:end]
        outputs = [
            cut_pad(
                waveform,
                SimpleNamespace(
                    sample_rate=target_rate, desired_length=5, pad_types="repeat"
                ),
            )
        ]
    else:
        target_samples = target_rate * 5
        if waveform.shape[1] <= target_samples:
            outputs = [
                cut_pad(
                    waveform,
                    SimpleNamespace(
                        sample_rate=target_rate,
                        desired_length=5,
                        pad_types="repeat",
                    ),
                )
            ]
        else:
            final_start = waveform.shape[1] - target_samples
            starts = list(range(0, final_start + 1, target_samples))
            if starts[-1] != final_start:
                starts.append(final_start)
            outputs = [
                waveform[:, start : start + target_samples] for start in starts
            ]
    squeezed = [output.squeeze(0) for output in outputs]
    if any(
        output.shape != (80_000,) or not torch.isfinite(output).all()
        for output in squeezed
    ):
        raise RuntimeError(f"invalid PAFA inputs for {sample.sample_id}")
    return squeezed


def extract_embeddings(
    samples: list[Sample],
    source_repo: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    guard: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    source_repo = verify_source_repo(source_repo)
    sys.path.insert(0, str(source_repo))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        from util.icbhi_util import cut_pad_sample_torchaudio

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.audio_path].append(sample)
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    pending_waveforms: list[torch.Tensor] = []
    pending_ids: list[str] = []
    window_counts: dict[str, int] = {}
    started = time.perf_counter()

    def flush() -> None:
        if not pending_waveforms:
            return
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.filterwarnings(
                "ignore",
                message="User provided device_type of 'cuda'.*CUDA is not available.*",
                category=UserWarning,
            )
            frames = encoder(
                torch.stack(pending_waveforms).to(device), training=False
            )
            values = frames.mean(dim=1)
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid PAFA embedding batch: {tuple(values.shape)}")
        for sample_id, value in zip(pending_ids, values.cpu().numpy()):
            output[sample_id].append(value.astype(np.float32, copy=False))
        pending_waveforms.clear()
        pending_ids.clear()
        if guard is not None:
            guard()

    for audio_path in sorted(grouped):
        waveform, sample_rate = torchaudio.load(audio_path)
        for sample in grouped[audio_path]:
            windows = _waveforms_for_sample(
                waveform,
                sample_rate,
                sample,
                cut_pad_sample_torchaudio,
            )
            window_counts[sample.sample_id] = len(windows)
            for window in windows:
                pending_waveforms.append(window)
                pending_ids.append(sample.sample_id)
                if len(pending_waveforms) >= batch_size:
                    flush()
    flush()
    ordered = np.stack(
        [
            np.mean(np.stack(output[sample.sample_id]), axis=0, dtype=np.float32)
            for sample in samples
        ]
    )
    if ordered.shape != (len(samples), 768) or not np.isfinite(ordered).all():
        raise RuntimeError("four-dataset embedding coverage gate failed")
    return ordered, {
        "samples": len(samples),
        "unique_audio_files": len(grouped),
        "windows": int(sum(window_counts.values())),
        "window_count_by_dataset": {
            dataset: int(
                sum(
                    window_counts[sample.sample_id]
                    for sample in samples
                    if sample.dataset == dataset
                )
            )
            for dataset in sorted({sample.dataset for sample in samples})
        },
        "window_count_min": min(window_counts.values()),
        "window_count_max": max(window_counts.values()),
        "embedding_shape": list(ordered.shape),
        "finite": True,
        "runtime_seconds": time.perf_counter() - started,
        "input_policy": {
            "event_cycle": "author 5 s repeat/truncate after interval crop",
            "recording": (
                "contiguous non-overlapping 5 s windows plus one end-aligned tail "
                "window when needed; repeat-pad only when <=5 s; frozen frame means "
                "then recording-level mean pool"
            ),
        },
        "cpu_compatibility": (
            "suppresses only PyTorch CUDA-autocast-disabled and legacy librosa "
            "pkg_resources deprecation warnings; model computations are unchanged"
        ),
    }


def save_cache(
    path: Path,
    samples: list[Sample],
    embeddings: np.ndarray,
    receipt: dict[str, object],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_ids=np.asarray([sample.sample_id for sample in samples]),
        embeddings=embeddings.astype(np.float32),
        receipt_json=np.asarray(json.dumps(receipt, sort_keys=True)),
    )
    temporary.replace(path)
    return {**receipt, "cache_path": str(path), "cache_sha256": sha256_file(path)}


def load_cache(path: Path, samples: list[Sample]) -> tuple[np.ndarray, dict[str, object]]:
    archive = np.load(path, allow_pickle=False)
    ids = archive["sample_ids"].astype(str).tolist()
    expected = [sample.sample_id for sample in samples]
    if ids != expected:
        raise RuntimeError("embedding cache sample order mismatch")
    embeddings = archive["embeddings"].astype(np.float32)
    if embeddings.shape != (len(samples), 768) or not np.isfinite(embeddings).all():
        raise RuntimeError("invalid embedding cache")
    receipt = json.loads(str(archive["receipt_json"].item()))
    return embeddings, {**receipt, "cache_path": str(path), "cache_sha256": sha256_file(path)}
