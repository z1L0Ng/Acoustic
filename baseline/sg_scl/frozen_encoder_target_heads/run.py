"""Run matched SPRSound target heads on a frozen SG-SCL AST encoder."""

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
from torchvision import transforms

from baseline.common.frozen_encoder_target_heads import run_pilot
from baseline.sg_scl.checkpoint_eval.bootstrap import (
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_CHECKPOINT_SIZE_BYTES,
    REPO_COMMIT,
    verify_checkpoint_identity,
    verify_checkpoint_state,
)


EXPERIMENT_ID = "sprsound_sg_scl_frozen_encoder_target_heads"
PROTOCOL = "sg_scl_frozen_encoder_target_heads_v1"
SOURCE_PREPROCESSING = (
    "author SG-SCL 16 kHz mono; full-recording fade; event crop; "
    "8 s repeat/truncate; 128-bin fbank resized to 798x128"
)


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def preprocess_recording(
    rows: list[dict[str, object]], author_repo: Path
) -> list[torch.Tensor]:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    args = SimpleNamespace(
        sample_rate=16_000,
        desired_length=8,
        pad_types="repeat",
        model="ast",
    )
    resize = transforms.Resize(size=(798, 128))
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
        event = cut_pad_sample_torchaudio(waveform[:, start:end], args)
        fbank = generate_fbank(args, event, args.sample_rate, n_mels=128)
        image = resize(transforms.ToTensor()(fbank)).to(torch.float32)
        if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
            raise RuntimeError(f"invalid SG-SCL fbank: {row['event_id']}")
        output.append(image)
    return output


def make_encoder_builder(checkpoint_path: Path, checkpoint_sha256: str):
    def build_encoder(
        author_repo: Path, device: torch.device
    ) -> tuple[torch.nn.Module, dict[str, object]]:
        checkpoint = checkpoint_path.resolve()
        task_sha = verify_checkpoint_identity(checkpoint, checkpoint_sha256)
        sys.path.insert(0, str(author_repo))
        from models.ast import ASTModel

        state = torch.load(checkpoint, map_location="cpu")
        state_counts = verify_checkpoint_state(state)
        model = ASTModel(
            label_dim=4,
            input_fdim=798,
            input_tdim=128,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="base384",
            verbose=False,
        )
        result = model.load_state_dict(normalize_state(state["model"]), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"SG-SCL encoder state mismatch: {result}")
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to(device).eval(), {
            "task_checkpoint_sha256": task_sha,
            "task_checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_state_verification": state_counts,
            "discarded_source_states": [
                "classifier",
                "domain classifier",
                "contrastive projector",
            ],
            "target_device_or_domain_metadata_used": False,
            "embedding_definition": "author validation-path audio embedding",
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
    pending_images: list[torch.Tensor] = []
    pending_ids: list[str] = []
    inference_args = SimpleNamespace(domain_adaptation=False, domain_adaptation2=True)
    started = time.perf_counter()

    def flush() -> None:
        if not pending_images:
            return
        with torch.inference_mode():
            values = encoder(
                torch.stack(pending_images).to(device),
                args=inference_args,
                training=False,
            )
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid SG-SCL embedding batch: {tuple(values.shape)}")
        for event_id, value in zip(pending_ids, values.cpu().numpy()):
            embeddings[event_id] = value.astype(np.float32, copy=False)
        pending_images.clear()
        pending_ids.clear()

    for recording_id in sorted(grouped):
        recording_rows = grouped[recording_id]
        for row, image in zip(
            recording_rows, preprocess_recording(recording_rows, author_repo)
        ):
            pending_ids.append(str(row["event_id"]))
            pending_images.append(image)
            if len(pending_images) == batch_size:
                flush()
    flush()
    runtime = time.perf_counter() - started
    ordered = np.stack([embeddings[str(row["event_id"])] for row in rows])
    if ordered.shape != (len(rows), 768) or not np.isfinite(ordered).all():
        raise RuntimeError("SG-SCL embedding coverage/finite gate failed")
    return ordered, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["profile", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=ACCEPTED_CHECKPOINT_SHA256)
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
        method_id="sg_scl",
        author_repo_commit=REPO_COMMIT,
        source_preprocessing=SOURCE_PREPROCESSING,
        direct_transfer_binary_score=59.9790,
        build_encoder=make_encoder_builder(args.checkpoint, args.checkpoint_sha256),
        extract_embeddings=extract_embeddings,
    )


if __name__ == "__main__":
    main()
