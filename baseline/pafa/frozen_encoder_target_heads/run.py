"""Run matched SPRSound target heads on a frozen PAFA/BEATs encoder."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms

from baseline.common.frozen_encoder_target_heads import run_pilot
from baseline.pafa.checkpoint_eval.bootstrap import (
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_CHECKPOINT_SIZE_BYTES,
    BACKBONE_SHA256,
    REPO_COMMIT,
    verify_checkpoint_identity,
    verify_checkpoint_state,
)
from baseline.patch_mix_cl.frozen_encoder_target_heads.run import sha256_file


EXPERIMENT_ID = "sprsound_pafa_frozen_encoder_target_heads"
PROTOCOL = "pafa_frozen_encoder_target_heads_v1"
BACKBONE_SIZE_BYTES = 361_499_833
SOURCE_PREPROCESSING = (
    "author PAFA 16 kHz mono; full-recording fade; event crop; "
    "5 s repeat/truncate; raw waveform; no SpecAugment"
)


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def preprocess_recording(
    rows: list[dict[str, object]], author_repo: Path
) -> list[torch.Tensor]:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio

    args = SimpleNamespace(sample_rate=16_000, desired_length=5, pad_types="repeat")
    waveform, sample_rate = torchaudio.load(str(rows[0]["audio_path"]))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != args.sample_rate:
        waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
    waveform = audio_transforms.Fade(
        fade_in_len=args.sample_rate // 16,
        fade_out_len=args.sample_rate // 16,
        fade_shape="linear",
    )(waveform)
    duration_ms = waveform.shape[1] / args.sample_rate * 1000
    output = []
    for row in rows:
        if float(row["end_ms"]) > duration_ms + 1:
            raise RuntimeError(f"event exceeds recording: {row['event_id']}")
        start = int(float(row["start_ms"]) / 1000 * args.sample_rate)
        end = int(float(row["end_ms"]) / 1000 * args.sample_rate)
        event = cut_pad_sample_torchaudio(waveform[:, start:end], args).squeeze(0)
        if event.shape != (80_000,) or not torch.isfinite(event).all():
            raise RuntimeError(f"invalid PAFA waveform: {row['event_id']}")
        output.append(event)
    return output


def make_encoder_builder(
    checkpoint_path: Path, checkpoint_sha256: str, backbone_path: Path
):
    def build_encoder(
        author_repo: Path, device: torch.device
    ) -> tuple[torch.nn.Module, dict[str, object]]:
        checkpoint = checkpoint_path.resolve()
        backbone = backbone_path.resolve()
        task_sha = verify_checkpoint_identity(checkpoint, checkpoint_sha256)
        if (
            backbone.stat().st_size != BACKBONE_SIZE_BYTES
            or sha256_file(backbone) != BACKBONE_SHA256
        ):
            raise RuntimeError("PAFA BEATs backbone checkpoint identity mismatch")
        sys.path.insert(0, str(author_repo))
        from models.beats import BEATsTransferLearningModel

        state = torch.load(checkpoint, map_location="cpu")
        state_counts = verify_checkpoint_state(state)
        model = BEATsTransferLearningModel(
            num_target_classes=4,
            model_path=str(backbone),
            ft_entire_network=True,
            spec_transform=None,
        )
        result = model.load_state_dict(normalize_state(state["model"]), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"PAFA encoder state mismatch: {result}")
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to(device).eval(), {
            "task_checkpoint_sha256": task_sha,
            "task_checkpoint_size_bytes": checkpoint.stat().st_size,
            "backbone_checkpoint_sha256": BACKBONE_SHA256,
            "backbone_checkpoint_size_bytes": backbone.stat().st_size,
            "checkpoint_state_verification": state_counts,
            "discarded_source_states": ["classifier", "projector"],
            "embedding_definition": "mean over BEATs pre-classifier frame representations",
        }

    return build_encoder


def extract_embeddings(
    rows: list[dict[str, object]],
    author_repo: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)
    embeddings: dict[str, np.ndarray] = {}
    pending_audio: list[torch.Tensor] = []
    pending_ids: list[str] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_audio:
            return
        with torch.inference_mode():
            frame_features = encoder(torch.stack(pending_audio).to(device), training=False)
            values = frame_features.mean(dim=1)
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid PAFA embedding batch: {tuple(values.shape)}")
        for event_id, value in zip(pending_ids, values.cpu().numpy()):
            embeddings[event_id] = value.astype(np.float32, copy=False)
        pending_audio.clear()
        pending_ids.clear()

    for recording_id in sorted(grouped):
        recording_rows = grouped[recording_id]
        for row, event in zip(
            recording_rows, preprocess_recording(recording_rows, author_repo)
        ):
            pending_ids.append(str(row["event_id"]))
            pending_audio.append(event)
            if len(pending_audio) == batch_size:
                flush()
    flush()
    runtime = time.perf_counter() - started
    ordered = np.stack([embeddings[str(row["event_id"])] for row in rows])
    if ordered.shape != (len(rows), 768) or not np.isfinite(ordered).all():
        raise RuntimeError("PAFA embedding coverage/finite gate failed")
    return ordered, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["profile", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=ACCEPTED_CHECKPOINT_SHA256)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_pilot(
        phase=args.phase,
        dataset_root=args.dataset_root,
        source_repo=args.source_repo,
        result_root=args.result_root,
        cache_root=args.cache_root,
        device_name=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        experiment_id=EXPERIMENT_ID,
        protocol_name=PROTOCOL,
        method_id="pafa",
        author_repo_commit=REPO_COMMIT,
        source_preprocessing=SOURCE_PREPROCESSING,
        direct_transfer_binary_score=55.8209,
        build_encoder=make_encoder_builder(
            args.checkpoint, args.checkpoint_sha256, args.backbone_checkpoint
        ),
        extract_embeddings=extract_embeddings,
    )


if __name__ == "__main__":
    main()
